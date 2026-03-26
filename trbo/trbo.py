"""This module implements the TReductionByOptimiationPass."""
from __future__ import annotations

from typing import Optional

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost import HilbertSchmidtCostGenerator
from bqskit.ir.opt.cost import HilbertSchmidtResidualsGenerator
from bqskit.qis.unitary import UnitaryLike
from bqskit.runtime import get_runtime
from bqskit.ir.opt.multistartgens.random import RandomStartGenerator

import numpy as np

from trbo.clift import circuit_for_rounded_val
from trbo.clift import clifford_gates
from trbo.clift import t_gates
from trbo.clift import rz_gates
from trbo.clift import GlobalPhaseGate
from trbo.clift import better_min_t_count_circuit
from trbo.clift import RzAsT 
from trbo.clift import RzAsCliff
from trbo.utils import FutureQueue

from trbo.multi_start_minimization import MultiStartMinimization
from trbo.tcount import get_deviation_arr
from trbo.tcount import SumCostGenerator
from trbo.tcount import RoundSmallestNCostGenerator
from trbo.tcount import SumResidualsGenerator
from trbo.tcount import RoundSmallestNResidualsGenerator
from trbo.tcount import MatrixDistanceCostGenerator

from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer

import logging

_logger = logging.getLogger(__name__)


class TReductionByOptimiationPass(BasePass):
    def __init__(
        self,
        multistarts: int = 64,
        success_threshold: float = 1e-6,
        second_pass_starts: int | None = None,
        rz_disc = None,
        comparator = better_min_t_count_circuit,
        strict_opt = False, # if True it will be slower at a marginal improvement in T count
        **kwargs,
    ) -> None:
        """
        Construct a TReductionByOptimiationPass

        Args:
            success_threshold (float): The synthesis success threshold.
                (Default: 1e-8)
        """
        self.success_threshold = success_threshold
        self.extra_kwargs = kwargs
        self.second_pass_starts = second_pass_starts

        self.acceptable_gates = clifford_gates + t_gates + rz_gates

        if rz_disc is None:
            self.rz_discretizations = [RzAsT(), RzAsCliff()]
        else:
            self.rz_discretizations = rz_disc
        self.multistarts = multistarts
        self.strict_opt = strict_opt
        self.comparator = comparator

    async def validated_optimization(self, circuit, target, N, discretization, threshold, initial_params=[], blacklist=None):
        # compute numbers of starts
        ms1 = self.multistarts
        ms2 = self.second_pass_starts
        if ms2 is None:
            ms2 = ms1 // 2
            ms2 = max(1, ms2)

        # Prepare cost functions
        d_gen = MatrixDistanceCostGenerator()
        d_res = HilbertSchmidtResidualsGenerator()
        n_gen = discretization.cost_generator(N, blacklist)
        n_res = discretization.residuals_generator(N, blacklist)
        sum_gen = SumCostGenerator(d_gen, n_gen)
        sum_res = SumResidualsGenerator(d_res, n_res)

        def get_score(x):
            return sum_gen.gen_cost(circuit, target)(x)

        # Try known good sets of parameters before itrboducing randomness:
        for params in initial_params:
            trial_params = params
            score = get_score(trial_params)
            if score < threshold:
                return score, trial_params

        # When those good parameters don't work, we need to search for new ones.
        # Starting near the "good guesses" is a great place to start.
        if len(initial_params) > 0:
            miser = MultiStartMinimization(sum_res, multistarts=2, minimizer=CeresMinimizer(), second_pass=None, judgement_cost=sum_gen)
            result = await miser.multi_start_instantiate_async(circuit, target, starts=initial_params)
            score = get_score(trial_params)
            if score < threshold:
                return score, trial_params

        # Sometimes its truly necessary to search new territory, so we now use random starting points.
        miser = MultiStartMinimization(sum_res, multistarts=ms1, minimizer=CeresMinimizer(), second_pass=ms2, threshold=threshold, judgement_cost=sum_gen)
        result = await miser.multi_start_instantiate_async(circuit, target)
        trial_params = result.params
        score = get_score(trial_params)
        if score < threshold:
            return score, trial_params

        if score < threshold * 10:
            # try a fine-tuning approach to see if we can squeeze out enough improvement to pass the threshold
            old_score = score
            fine_tuner = LBFGSMinimizer()
            trial_params = fine_tuner.minimize(sum_gen.gen_cost(circuit, target), trial_params)
            score = get_score(trial_params)
        return score, trial_params

    def round_circuit(self, circuit, target, N, discretization, blacklist, threshold):
        indices = np.argsort(discretization.param_distances(circuit.params, blacklist)) + 1
        expectations = discretization.param_distances(circuit.params, blacklist)

        op_index = 0
        for cycle, op in circuit.operations_with_cycles():
            op_index += len(op.params)
            if len(op.params) != 1:
                continue
            if op.gate not in rz_gates:
                continue
            if op_index in indices[:N]:
                if op.gate in rz_gates:
                    rzu = op.get_unitary()
                    rounded = discretization.nearest_gate(op.params[0])
                    rdu = rounded.get_unitary()
                    if rzu.get_distance_from(rdu) > threshold and rzu.get_distance_from(rdu) > expectations[op_index - 1]:
                        print(f"WARNING rounding angle {op.params[0]} at index {op_index - 1}. Expected {expectations[op_index - 1]} but got {rzu.get_distance_from(rdu)}")
                    circuit.replace_gate(
                        (cycle, op.location[0]), rounded, op.location
                    )
                else:
                    raise RuntimeError("Attempted to round unexpected gate type {op.gate}")

        if not circuit.get_unitary().get_distance_from(target) <= threshold:
            test_params = CeresMinimizer(ftol=5e-16, gtol=1e-15).minimize(HilbertSchmidtResidualsGenerator().gen_cost(circuit, target), circuit.params)
            if circuit.get_unitary(test_params).get_distance_from(target) < circuit.get_unitary().get_distance_from(target):
                circuit.set_params(test_params)
        if not circuit.get_unitary().get_distance_from(target) <= threshold:
            print(f"WARNING circuit rounding resulted in {circuit.get_unitary().get_distance_from(target)} > {threshold}.")


    async def optimize_for_discretization(self, circuit, target, discretization, remainders=None, threshold=None, leave_unrounded=0):
        trial_circuit = circuit.copy()
        if threshold is None:
            threshold = self.success_threshold

        def gen_blacklist(circuit):
            blacklisted_indices = []
            op_index = 0
            for cycle, op in circuit.operations_with_cycles():
                if len(op.params) < 1:
                    continue
                if op.gate not in rz_gates:
                    blacklisted_indices.extend([i + op_index for i in range(len(op.params))])
                op_index += len(op.params)
            blacklist = np.zeros_like(circuit.params)
            for i in blacklisted_indices:
                blacklist[i] = 1
            return blacklist, len(blacklisted_indices)


        best_params = circuit.params
        best_N = 0
        first_min = CeresMinimizer()

        blacklist, bll = gen_blacklist(trial_circuit)
        max_N = len(circuit.params) - bll - leave_unrounded
        high = max_N
        low = 0
        while low <= high:
            N = low + (high - low) // 2

            # Two good sets of parameters to try before itrboducing randomness:
            #   1. Previously known good parameters in this loop
            #   2. The original circuit parameters
            # Often one of these sets of parameters will just work, allowing us to skip optimization.
            # (these are passed as initial_params to validated_optimization)
            score, trial_params = await self.validated_optimization(trial_circuit, target, N, discretization, threshold, [circuit.params, best_params], blacklist)

            # if we have remainder discretizations, we have to ensure that the remaining Rz gates can be captured by those
            if N < max_N and score < threshold and remainders is not None and len(remainders) > 0:
                if len(remainders) == 1:
                    test_circuit = trial_circuit.copy()
                    test_circuit.set_params(trial_params)
                    self.round_circuit(test_circuit, target, N, discretization, blacklist, threshold)
                    test_blacklist, _ = gen_blacklist(test_circuit)
                    score, _ = await self.validated_optimization(test_circuit, target, max_N - N, remainders[0], threshold, [test_circuit.params], test_blacklist)


            if score >= threshold:
                high = N - 1
            else:
                low = N + 1
                if N > best_N:
                    best_params = trial_params
                    best_N = N
        if best_N == 0:
            # we failed to find any improvement
            return circuit

        # We have identified a set of parameters that allows N Rz gates to be rounded
        # Now its time to go actually replace those Rz gates with Clifford+T circuits
        best_circuit = circuit.copy()
        best_circuit.set_params(best_params)


        # do the actual rounding of the circuit
        self.round_circuit(best_circuit, target, best_N, discretization, blacklist, threshold)

        # if we have remainders, we are not done yet
        best_circuit.unfold_all()
        if best_N < max_N and remainders is not None and len(remainders) > 0:
            if len(remainders) == 1:
                test_blacklist, _ = gen_blacklist(best_circuit)
                _, test_params = await self.validated_optimization(best_circuit, target, max_N - best_N, remainders[0], threshold, [best_circuit.params], test_blacklist)
                self.round_circuit(best_circuit, target, max_N - best_N, remainders[0], test_blacklist, threshold)
            elif len(remainders) > 1:
                return await self.optimize_for_discretization(best_circuit, target, remainders[0], remainders[1:], threshold, leave_unrounded)

 
        # do some final fine-tuning of the parameters
        test_params = CeresMinimizer(ftol=5e-16, gtol=1e-15).minimize(HilbertSchmidtResidualsGenerator().gen_cost(best_circuit, target), best_circuit.params)
        if best_circuit.get_unitary(test_params).get_distance_from(target) < best_circuit.get_unitary().get_distance_from(target):
            best_circuit.set_params(test_params)
        if not best_circuit.get_unitary().get_distance_from(target) <= threshold:
            print(f"ERROR rounded and post-processed circuit ended up with {best_circuit.get_unitary().get_distance_from(target)} > {threshold}")
        best_circuit.unfold_all()
        return best_circuit

    async def run(self, circuit: Circuit, data: PassData = {}) -> None:
        # Check that circuit has been converrted to Clifford+T+Rz
        if any(g not in self.acceptable_gates for g in circuit.gate_set):
            m = (
                'Circuit must be converted to Clifford+T+Rz before running'
                f' TReductionByOptimiationPass. Got {circuit.gate_set} instead of {self.acceptable_gates}.'
            )
            raise ValueError(m)

        if "utry" not in data:
            utry = circuit.get_unitary()
        else:
            utry = data["utry"]

        if "adjusted_threshold" in data:
            threshold = data["adjusted_threshold"]
        else:
            threshold = self.success_threshold

        initial_circuit = circuit
        initial_circuit.unfold_all()
        candidate_circuit = initial_circuit

        # perform the optimization
        best_circuit = initial_circuit
        leave_unrounded = 0
        for i in range(len(self.rz_discretizations)):
            discretization = self.rz_discretizations[i]
            remainders = self.rz_discretizations[:i]
            candidate_circuit = await self.optimize_for_discretization(initial_circuit, utry, discretization, remainders, threshold, leave_unrounded)
            candidate_circuit.unfold_all()
            if not self.comparator(best_circuit, candidate_circuit):
                break
            best_circuit = candidate_circuit
            rz_count = 0
            for gate in candidate_circuit.gate_counts:
                if gate in rz_gates:
                    rz_count += candidate_circuit.gate_counts[gate]
            leave_unrounded = rz_count
            if leave_unrounded > 0 and not self.strict_opt:
                break

        circuit.become(best_circuit)


# make a shorthand alias for the full pass name
TRbOPass = TReductionByOptimiationPass
