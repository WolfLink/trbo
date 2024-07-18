"""This module implements the NumericalTReductionPass."""
from __future__ import annotations

from typing import Optional

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost import HilbertSchmidtCostGenerator
from bqskit.ir.opt.cost import HilbertSchmidtResidualsGenerator
from bqskit.qis.unitary import UnitaryLike
from bqskit.runtime import get_runtime

import numpy as np

from ntro.clift import circuit_for_rounded_val
from ntro.clift import clifford_gates
from ntro.clift import t_gates
from ntro.clift import rz_gates

from ntro.two_pass_minimization import TwoPassMinimization
from ntro.tcount import RelaxedTCountCostGenerator
from ntro.tcount import get_arr
from ntro.rz_to_t import RzToTPass, RzToT_ScanningBruteForcePass

import logging

_logger = logging.getLogger(__name__)


class NumericalTReductionPass(BasePass):
    def __init__(
        self,
        success_threshold: float = 1e-8,
        full_loops: int = 8,
        **kwargs,
    ) -> None:
        """
        Construct a NumericalTReductionPass

        Args:
            success_threshold (float): The synthesis success threshold.
                (Default: 1e-8)
        """
        self.instantiate_options = {
            'cost_fn_gen': HilbertSchmidtResidualsGenerator(),
            'method' : TwoPassMinimization(),
            'kwargs' : {'pifrac' : 4}
        }
        self.success_threshold = success_threshold
        self.extra_kwargs = kwargs
        self.full_loops = full_loops

        self.acceptable_gates = clifford_gates + t_gates + rz_gates
        self.target_periods = [0.5 * np.pi, 0.25 * np.pi]

    async def attempt_gate_rounding(self, circuit: Circuit, target: UnitaryMatrix, period: float):
        two_pass = TwoPassMinimization(
                pass_w_cost_gen=RelaxedTCountCostGenerator(period=period),
                success_threshold=self.success_threshold,
                num_starts=(128,16),
                **self.extra_kwargs,
                )
        trial_circuit = circuit.copy()
        index = np.argmin(get_arr(circuit.params, period))
        for cycle, op in circuit.operations_with_cycles():
            if len(op.params) != 1:
                continue
            if op.params[0] == circuit.params[index]:
                rounded = circuit_for_rounded_val(op.params[0], period < np.pi * 0.5)
                trial_circuit.replace_gate(
                    (cycle, op.location[0]), rounded, op.location
                )
                break
        trial_circuit.unfold_all()
        result = two_pass.instantiate(trial_circuit, target=target, x0=trial_circuit.params)
        if result is None:
            result = await get_runtime().submit(
                two_pass.multi_start_instantiate_async,
                trial_circuit,
                target=target,
            )

        return result

    async def optimize_for_period(self, circuit: Circuit, target: UnitaryMatrix, period: float):
        trial_circuit = circuit.copy()
        two_pass = TwoPassMinimization(
                pass_w_cost_gen=RelaxedTCountCostGenerator(period=period),
                success_threshold=self.success_threshold,
                num_starts=(128,16),
                **self.extra_kwargs,
                )
        initial_result: Circuit | None = await get_runtime().submit(
                two_pass.multi_start_instantiate_async,
                trial_circuit,
                target=target,
        )
        if initial_result is None:
            return circuit

        trial_circuit = initial_result
        for i in range(circuit.num_params):
            #print(trial_circuit.gate_counts)
            rounded_circuit = await get_runtime().submit(
                    self.attempt_gate_rounding,
                    trial_circuit,
                    target,
                    period,
                    )
            if rounded_circuit is None:
                return trial_circuit
            else:
                trial_circuit = rounded_circuit
        return trial_circuit

    async def optimize_all_periods(self, circuit, target):
        candidate_circuit = circuit
        for period in self.target_periods:
            result = await get_runtime().submit(
                    self.optimize_for_period,
                    candidate_circuit,
                    target,
                    period,
                    )
            if result is not None:
                candidate_circuit = result
        return candidate_circuit


    async def run(self, circuit: Circuit, data: PassData = {}) -> None:
        # Check that circuit has been converrted to Clifford+T+Rz
        if any(g not in self.acceptable_gates for g in circuit.gate_set):
            m = (
                'Circuit must be converted to Clifford+T+Rz before running'
                f' NumericalTReductionPass. Got {circuit.gate_set}.'
            )
            raise ValueError(m)

        utry = circuit.get_unitary()
        best_circuit = circuit
        best_circuit.unfold_all()
        candidate_circuit = best_circuit


        candidate_circuits = await get_runtime().map(
                self.optimize_all_periods,
                [best_circuit] * self.full_loops,
                [utry] * self.full_loops,
                )
        for candidate_circuit in candidate_circuits:
            #print(f"{candidate_circuit.gate_counts}")
            if candidate_circuit.get_unitary().get_distance_from(utry, degree=1) < self.success_threshold:
                curr_rz = sum(candidate_circuit.count(gate) for gate in rz_gates)
                best_rz = sum(best_circuit.count(gate) for gate in rz_gates)
                curr_t = sum(candidate_circuit.count(gate) for gate in t_gates)
                best_t = sum(best_circuit.count(gate) for gate in t_gates)
                
                if curr_rz < best_rz:
                    best_circuit = candidate_circuit
                elif curr_rz == best_rz:
                    if curr_t < best_t:
                        best_circuit = candidate_circuit
                    elif curr_t == best_t:
                        if candidate_circuit.get_unitary().get_distance_from(utry, degree=1) < best_circuit.get_unitary().get_distance_from(utry, degree=1):
                            best_circuit = candidate_circuit

        circuit.become(best_circuit)
