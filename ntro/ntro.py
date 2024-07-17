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

        # trial_circuit will try to round as many parameters as possible to
        # nice values
        for i in range(self.full_loops):
            trial_circuit = circuit.copy()
            candidate_circuit = circuit

            for period in self.target_periods:
                self.instantiate_options["method"] = TwoPassMinimization(
                    pass_2_cost_gen=RelaxedTCountCostGenerator(period=period),
                    success_threshold=self.success_threshold,
                    **self.extra_kwargs,
                )

                for _ in range(circuit.num_params + 1):

                    # check for constant circuits
                    if trial_circuit.num_params < 1:
                        dist = trial_circuit.get_unitary().get_distance_from(utry)
                        if dist < self.success_threshold:
                            candidate_circuit = trial_circuit
                        else:
                            trial_circuit = candidate_circuit.copy()
                        break

                    print(trial_circuit.gate_counts)

                    result: Circuit | None = await get_runtime().submit(
                            self.instantiate_options["method"].multi_start_instantiate_async,
                            trial_circuit,
                            target=utry,
                            num_starts=32,
                            # **self.instantiate_options
                    )

                    if result is None:
                        trial_circuit = candidate_circuit.copy()
                        break

                    cost_calc = HilbertSchmidtCostGenerator().gen_cost(result, utry)
                    # verify that the new result passes the threshold
                    if cost_calc(result.params) < self.success_threshold:
                        candidate_circuit = result
                    else:
                        trial_circuit = candidate_circuit.copy()
                        break

                    trial_circuit: Circuit = candidate_circuit.copy()

                    param_scores = get_arr(trial_circuit.params, period)
                    minval = np.min(param_scores)
                    did_round = False
                    for cycle, op in trial_circuit.operations_with_cycles():
                        if len(op.params) != 1:
                            continue
                        if get_arr(np.array(op.params), period)[0] == minval:
                            rounded = circuit_for_rounded_val(op.params[0], period < np.pi * 0.5)
                            trial_circuit.replace_gate(
                                (cycle, op.location[0]), rounded, op.location
                            )
                            did_round = True
                            break
                    if not did_round:
                        _logger.error(f"Could not find {minval}.")
                    trial_circuit.unfold_all()

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
            else:
                print(f"Result {i} failed the threshold check {self.success_threshold} > {candidate_circuit.get_unitary().get_distance_from(utry, degree=1)}")
            print(f"Best result after loop {i}: dist={best_circuit.get_unitary().get_distance_from(utry, degree=1)} {best_circuit.gate_counts}")

        circuit.become(best_circuit)
