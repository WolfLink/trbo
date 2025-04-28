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
from ntro.multi_start_minimization import MultiStartMinimization
from ntro.tcount import RelaxedTCountCostGenerator
from ntro.tcount import get_arr, get_deviation_arr
from ntro.tcount import RelaxedTCount
from ntro.tcount import SumCostGenerator
from ntro.tcount import RoundSmallestNCostGenerator
from ntro.tcount import SumResidualsGenerator
from ntro.tcount import RoundSmallestNResidualsGenerator
from ntro.rz_to_t import RzToTPass, RzToT_ScanningBruteForcePass

from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer

import logging

_logger = logging.getLogger(__name__)


class NumericalTReductionPass(BasePass):
    def __init__(
        self,
        success_threshold: float = 1e-8,
        full_loops: int = 8,
        search_method: str = "greedy",
        multistarts: int = (128, 16),
        backup: bool = False,
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
        self.search_method = search_method
        self.backup = backup

        self.acceptable_gates = clifford_gates + t_gates + rz_gates
        self.target_periods = [0.5 * np.pi, 0.25 * np.pi]
        self.multistarts = multistarts

    async def attempt_gate_rounding(self, circuit: Circuit, target: UnitaryMatrix, period: float, index: int | NoneType = None, threshold=None):
        if threshold is None:
            threshold = self.threshold
        two_pass = TwoPassMinimization(
                pass_w_cost_gen=RelaxedTCountCostGenerator(period=period),
                success_threshold=threshold,
                num_starts=self.multistarts,
                **self.extra_kwargs,
                )
        trial_circuit = circuit.copy()
        if index is None:
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
        result = two_pass.instantiate(trial_circuit, target=target, x0=trial_circuit.params)
        if result is None and self.backup:
            result = await get_runtime().submit(
                two_pass.multi_start_instantiate_async,
                trial_circuit,
                target=target,
            )

        return result

    async def optimize_for_period(self, circuit: Circuit, target: UnitaryMatrix, period: float, threshold: None):
        trial_circuit = circuit.copy()
        if threshold is None:
            threshold = self.success_threshold
        two_pass = TwoPassMinimization(
                pass_w_cost_gen=RelaxedTCountCostGenerator(period=period),
                success_threshold=threshold,
                num_starts=self.multistarts,
                **self.extra_kwargs,
                )
        initial_result: Circuit | None = await get_runtime().submit(
                two_pass.multi_start_instantiate_async,
                trial_circuit,
                target=target,
        )
        if initial_result is not None:
            trial_circuit = initial_result

        for i in range(circuit.num_params):
            #print(trial_circuit.gate_counts)
            rounded_circuit = await get_runtime().submit(
                    self.attempt_gate_rounding,
                    trial_circuit,
                    target,
                    period,
                    threshold=threshold,
                    )
            if rounded_circuit is None:
                break
            else:
                trial_circuit = rounded_circuit
        return trial_circuit

    async def optimize_for_period_n_sum(self, circuit, target, period, threshold=None):
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
                miser = MultiStartMinimization(sum_res, self.success_threshold, multistarts=self.multistarts[0], minimizer=CeresMinimizer())
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

                # after all that trying to get a good result, we ultimately have to 
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



    async def optimize_for_period_greedy_search(self, circuit: Circuit, target: UnitaryMatrix, period: float, threshold=None):
        trial_circuit = circuit.copy()
        if threshold is None:
            threshold = self.success_threshold
        two_pass = TwoPassMinimization(
                pass_w_cost_gen=RelaxedTCountCostGenerator(period=period),
                success_threshold=threshold,
                num_starts=self.multistarts,
                **self.extra_kwargs,
                )
        initial_result: Circuit | None = await get_runtime().submit(
                two_pass.multi_start_instantiate_async,
                trial_circuit,
                target=target,
        )
        if initial_result is not None:
            trial_circuit = initial_result
        
        relaxed_t_count = RelaxedTCount(period)
        candidate_circuit = trial_circuit
        for _ in range(circuit.num_params):
            trial_circuit = candidate_circuit
            candidate_circuit = None
            rounded_circuits = await get_runtime().map(
                    self.attempt_gate_rounding,
                    [trial_circuit] * len(trial_circuit.params),
                    [target] * len(trial_circuit.params),
                    [period] * len(trial_circuit.params),
                    list(range(len(trial_circuit.params))),
                    [threshold] * len(trial_circuit.params),
                    )
            for rounded_circuit in rounded_circuits:
                if candidate_circuit is None:
                    candidate_circuit = rounded_circuit
                elif rounded_circuit is not None and relaxed_t_count.get_cost(rounded_circuit.params) < relaxed_t_count.get_cost(candidate_circuit.params):
                    candidate_circuit = rounded_circuit
            if candidate_circuit is None:
                candidate_circuit = trial_circuit
                break
        return candidate_circuit

    async def optimize_all_periods(self, circuit, target, x0, threshold=None):
        candidate_circuit = circuit.copy()
        candidate_circuit.set_params(x0)
        if self.search_method == "greedy":
            method = self.optimize_for_period_greedy_search
        elif self.search_method == "none" or self.search_method is None:
            method = self.optimize_for_period
        elif self.search_method == "n_sum":
            method = self.optimize_for_period_n_sum
        else:
            raise ValueError
        for period in self.target_periods:
            result = await get_runtime().submit(
                    method,
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

        best_circuit = circuit
        best_circuit.unfold_all()
        candidate_circuit = best_circuit

        param_list = [circuit.params]
        if self.full_loops > 1:
            miser = MultiStartMinimization(HilbertSchmidtResidualsGenerator(), threshold, multistarts=self.multistarts[1], minimizer=CeresMinimizer())
            seed_circuits = await get_runtime().map(
                    miser.multi_start_instantiate_async,
                    [candidate_circuit] * (self.full_loops - 1),
                    [utry] * (self.full_loops - 1),
                    )
            param_list.extend([seed_circuit.params for seed_circuit in seed_circuits])
             

        candidate_circuits = await get_runtime().map(
                self.optimize_all_periods,
                [best_circuit] * self.full_loops,
                [utry] * self.full_loops,
                param_list,
                [threshold] * self.full_loops,
                )
        rz_counts = []
        t_counts = []
        distances = []
        for candidate_circuit in candidate_circuits:
            curr_rz = sum(candidate_circuit.count(gate) for gate in rz_gates)
            curr_t = sum(candidate_circuit.count(gate) for gate in t_gates)
            curr_d = candidate_circuit.get_unitary().get_distance_from(utry)
            rz_counts.append(curr_rz)
            t_counts.append(curr_t)
            distances.append(curr_d)
            if "profiling_mode" in self.extra_kwargs and self.extra_kwargs["profiling_mode"]:
                print(f"Rz: {curr_rz}\tT: {curr_t}\tD: {curr_d}")
            if candidate_circuit.get_unitary().get_distance_from(utry) < threshold:
                best_rz = sum(best_circuit.count(gate) for gate in rz_gates)
                best_t = sum(best_circuit.count(gate) for gate in t_gates)
                
                if curr_rz < best_rz:
                    best_circuit = candidate_circuit
                elif curr_rz == best_rz:
                    if curr_t < best_t:
                        best_circuit = candidate_circuit
                    elif curr_t == best_t:
                        if candidate_circuit.get_unitary().get_distance_from(utry) < best_circuit.get_unitary().get_distance_from(utry):
                            best_circuit = candidate_circuit
        if "distance_list" not in data:
            data["distance_list"] = [best_circuit.get_unitary().get_distance_from(utry)]
        else:
            data["distance_list"].append(best_circuit.get_unitary().get_distance_from(utry))

        if "profiling_mode" in self.extra_kwargs and self.extra_kwargs["profiling_mode"]:
            best_rz = sum(best_circuit.count(gate) for gate in rz_gates)
            best_t = sum(best_circuit.count(gate) for gate in t_gates)
            best_rz_count = 0
            best_t_count = 0
            for i in range(len(rz_counts)):
                if rz_counts[i] == best_rz:
                    best_rz_count += 1
                    if t_counts[i] == best_t:
                        best_t_count += 1
            t_counts = np.array(t_counts)
            rz_counts = np.array(rz_counts)
            print(f"Rz: {np.min(rz_counts)} < {np.mean(rz_counts)} < {np.max(rz_counts)} ({best_rz_count}/{self.full_loops} {100 * best_rz_count / self.full_loops}%)")
            print(f"T: {np.min(t_counts[rz_counts == best_rz])} < {np.mean(t_counts[rz_counts == best_rz])} < {np.max(t_counts[rz_counts == best_rz])} ({best_t_count}/{self.full_loops} {100 * best_t_count / self.full_loops}%)")
            print(f"D: {np.min(np.abs(distances))} < {np.mean(np.abs(distances))} < {np.max(np.abs(distances))}")
            print(best_circuit.gate_counts)
        circuit.become(best_circuit)
