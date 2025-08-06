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

from ntro.multi_start_minimization import MultiStartMinimization
from ntro.tcount import get_arr, get_deviation_arr
from ntro.tcount import SumCostGenerator
from ntro.tcount import RoundSmallestNCostGenerator
from ntro.tcount import SumResidualsGenerator
from ntro.tcount import RoundSmallestNResidualsGenerator
from ntro.clift import better_min_t_count_circuit
from ntro.rz_to_t import RzToTPass, RzToT_ScanningBruteForcePass

from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer

import logging

_logger = logging.getLogger(__name__)


class NumericalTReductionPass(BasePass):
    def __init__(
        self,
        success_threshold: float = 1e-6,
        multistarts: int = 32,
        backup: bool = False,
        target_periods = None,
        target_gates = None,
        **kwargs,
    ) -> None:
        """
        Construct a NumericalTReductionPass

        Args:
            success_threshold (float): The synthesis success threshold.
                (Default: 1e-8)
        """
        self.success_threshold = success_threshold
        self.extra_kwargs = kwargs
        self.backup = backup

        self.acceptable_gates = clifford_gates + t_gates + rz_gates
        if target_periods is None:
            self.target_periods = [0.5 * np.pi, 0.25 * np.pi]
        else:
            self.target_periods = target_periods
        self.multistarts = multistarts


    async def optimize_for_period(self, circuit, target, period, threshold=None):
        trial_circuit = circuit.copy()
        if threshold is None:
            threshold = self.success_threshold
        high = len(circuit.params) + 1
        low = 0

        d_gen = HilbertSchmidtCostGenerator()
        d_res = HilbertSchmidtResidualsGenerator()
        best_params = circuit.params
        best_N = 0
        first_min = CeresMinimizer()
        while high > low:
            N = (low + high) // 2
            n_gen = RoundSmallestNCostGenerator(N, period)
            n_res = RoundSmallestNResidualsGenerator(N, period, smoothed=False)
            sum_gen = SumCostGenerator(d_gen, n_gen)
            sum_res = SumResidualsGenerator(d_res, n_res)
            trial_params = first_min.minimize(sum_res.gen_cost(trial_circuit, target), best_params)
            #score = sum_gen.gen_cost(trial_circuit, target)(trial_params)
            score = n_gen.gen_cost(trial_circuit, target)(trial_params) + trial_circuit.get_unitary(trial_params).get_distance_from(target)
            if score >= threshold:
                miser = MultiStartMinimization(sum_res, self.success_threshold, multistarts=32, minimizer=CeresMinimizer())
                #miser = MultiStartMinimization(sum_gen, self.success_threshold, multistarts=16)
                result = await miser.multi_start_instantiate_async(trial_circuit, target)
                trial_params = result.params
                #score = sum_gen.gen_cost(trial_circuit, target)(trial_params)
                score = n_gen.gen_cost(trial_circuit, target)(trial_params) + trial_circuit.get_unitary(trial_params).get_distance_from(target)

            # The optimization pass we do first is a fast, broad search, but is technically less accurate.
            # If it doesn't quite get us an acceptable result, we try using a more accurate but slower minimization
            # to fine tune the result.
            if score >= threshold:
                trial_params = LBFGSMinimizer().minimize(sum_gen.gen_cost(circuit, target), trial_params)
                score = n_gen.gen_cost(trial_circuit, target)(trial_params) + trial_circuit.get_unitary(trial_params).get_distance_from(target)

                # after all that trying to get a good result, we ultimately have to give up if we still don't have an acceptable result
            if score >= threshold:
                high = N - 1
            else:
                low = N + 1
                best_params = trial_params
                best_N = N
        if best_N == 0:
            #print("best N: 0")
            return
        best_circuit = circuit.copy()
        best_circuit.set_params(best_params)
        best_sum = RoundSmallestNCostGenerator(best_N, period).gen_cost(best_circuit, target)(best_params)
        best_dist = best_circuit.get_unitary().get_distance_from(target)
        #print(f"best N: {best_N} score: {best_sum + best_dist} dist: {best_dist} thresh: {threshold}")
        for i in range(best_N):
            if len(best_circuit.params) < 1:
                break
            trial_circuit = best_circuit
            index = np.argmin(get_deviation_arr(trial_circuit.params, period))
            trial_circuit = best_circuit.copy()
            for cycle, op in trial_circuit.operations_with_cycles():
                if len(op.params) != 1:
                    continue
                if op.params[0] == trial_circuit.params[index]:
                    rounded = circuit_for_rounded_val(op.params[0], period < np.pi * 0.5)
                    trial_circuit.replace_gate(
                        (cycle, op.location[0]), rounded, op.location
                    )
                    break
            if trial_circuit.get_unitary().get_distance_from(target) >= threshold:
                test_params = CeresMinimizer(ftol=5e-16, gtol=1e-15).minimize(HilbertSchmidtResidualsGenerator().gen_cost(trial_circuit, target), trial_circuit.params)
                trial_circuit.set_params(test_params)
            if trial_circuit.get_unitary().get_distance_from(target) >= threshold:
                print(f"deletion {i}\testimate: {(get_deviation_arr(best_circuit.params, period))[index] + best_dist}\tactual: {trial_circuit.get_unitary().get_distance_from(target)} threshold: {threshold}")
                break
            #test_params = CeresMinimizer(ftol=5e-16, gtol=1e-15).minimize(HilbertSchmidtResidualsGenerator().gen_cost(trial_circuit, target), trial_circuit.params)
            #if d_gen.gen_cost(trial_circuit, target)(test_params) >= self.success_threshold:
            #    print(f"Problem with rounding {i}")
            #    break
            #else:
            best_circuit = trial_circuit
        #print(f"opt  N: {best_N} score: {best_sum + best_dist} dist: {best_circuit.get_unitary().get_distance_from(target)}")
        test_params = CeresMinimizer(ftol=5e-16, gtol=1e-15).minimize(HilbertSchmidtResidualsGenerator().gen_cost(best_circuit, target), best_circuit.params)
        if best_circuit.get_unitary(test_params).get_distance_from(target) < best_circuit.get_unitary().get_distance_from(target):
            best_circuit.set_params(test_params)
        if not best_circuit.get_unitary().get_distance_from(target) <= threshold:
            print(f"ERROR got {best_circuit.get_unitary().get_distance_from(target)} > {threshold}")
        return best_circuit

    async def optimize_all_periods(self, circuit, target, x0, threshold=None):
        candidate_circuit = circuit.copy()
        candidate_circuit.set_params(x0)
        for period in self.target_periods:
            result = await get_runtime().submit(
                    self.optimize_for_period,
                    candidate_circuit,
                    target,
                    period,
                    threshold,
                    )
            if result is not None:
                candidate_circuit = result
        candidate_circuit.unfold_all()
        return candidate_circuit


    async def run(self, circuit: Circuit, data: PassData = {}) -> None:
        # Check that circuit has been converrted to Clifford+T+Rz
        if any(g not in self.acceptable_gates for g in circuit.gate_set):
            m = (
                'Circuit must be converted to Clifford+T+Rz before running'
                f' NumericalTReductionPass. Got {circuit.gate_set}.'
            )
            raise ValueError(m)

        if "utry" not in data:
            utry = circuit.get_unitary()
            data["utry"] = utry
        else:
            utry = data["utry"]

        if "adjusted_threshold" in data:
            threshold = data["adjusted_threshold"]
        else:
            threshold = self.success_threshold

        initial_circuit = circuit
        initial_circuit.unfold_all()
        candidate_circuit = initial_circuit
        best_circuit = initial_circuit
        for _ in range(1):
            candidate_circuit = await self.optimize_all_periods(initial_circuit, utry, circuit.params, threshold)
            if better_min_t_count_circuit(best_circuit, candidate_circuit):
                best_circuit = candidate_circuit
        circuit.become(best_circuit)
        return circuit

