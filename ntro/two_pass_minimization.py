from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Sequence

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.opt import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt import HilbertSchmidtCostGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator as CostGen
from bqskit.ir.opt.instantiater import Instantiater
from bqskit.ir.opt.minimizer import Minimizer
from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.ir.opt.multistartgens.random import RandomStartGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.qis.unitary import RealVector
from bqskit.runtime import get_runtime

from ntro.clift import clifford_gates
from ntro.clift import rz_gates
from ntro.clift import t_gates
from ntro.tcount import RelaxedTCountCostGenerator
from ntro.constrained_minimizer import ConstrainedMinimizer, SLSQPConstrainedMinimizer

import logging

_logger = logging.getLogger(__name__)


class FutureQueue:
    def __init__(self, future, length):
        self.future = future
        self.queue = []
        self.remaining = length

    def __aiter__(self):
        return self

    async def __anext__(self):
        if len(self.queue) > 0:
            self.remaining -= 1
            return self.queue.pop(0)
        elif self.remaining < 1:
            raise StopAsyncIteration
        else:
            try:
                self.queue.extend(await get_runtime().next(self.future))
                return self.queue.pop(0)
            except RuntimeError:
                raise StopAsyncIteration


def run_first_pass_minimization(
    first_pass: Minimizer,
    cost_gen: CostGen,
    threshold_cost_gen: CostGen,
    circuit: Circuit,
    target: UnitaryMatrix | StateVector | StateSystem,
    x0: RealVector,
    target_threshold: float,
    low_quality_threshold: float = 1e-4,
) -> RealVector:
    """Wrapper so we don't pickle the cost function."""
    pass_1_cost = cost_gen.gen_cost(circuit, target)
    threshold_cost = threshold_cost_gen.gen_cost(circuit, target)

    result = first_pass.minimize(pass_1_cost, x0)

    distance = abs(threshold_cost(result))
    if distance > target_threshold and distance < low_quality_threshold:
        # if we get a result that is promising but outside the threshold,
        # try to refine it a little more using higher quality optimizer settings
        ceres = CeresMinimizer(ftol=5e-16, gtol=1e-15)
        result = ceres.minimize(pass_1_cost, result)
        distance = abs(threshold_cost(result))
        
    if distance > target_threshold:
        return None
    return result


def run_second_pass_minimization(
    second_pass: ConstrainedMinimizer,
    pass_2_cost_gen: CostGen,
    pass_2_cstr_gen: CostGen,
    circuit: Circuit,
    target: UnitaryMatrix | StateVector | StateSystem,
    x0: RealVector,
) -> tuple[RealVector, float, float] | None:
    """
    Perform the constrained optimization of the second pass on one input.

    Returns the result, cost, and constraint value if the constraint is met.
    Returns None if the constraint is not met.
    """
    pass_2_cost = pass_2_cost_gen.gen_cost(circuit, target)
    pass_2_cstr = pass_2_cstr_gen.gen_cost(circuit, target)
    result = second_pass.constrained_minimize(pass_2_cost, pass_2_cstr, x0)
    result_cost = pass_2_cost(result)
    result_cstr = pass_2_cstr(result)
    if result_cstr > second_pass.constraint_threshold:
        m = (
            'Rejected a second pass result because the constraint '
            f'value was {result_cstr}'
        )
        _logger.debug(m)
        return None
    
    return result, result_cost, result_cstr


class TwoPassMinimization(Instantiater):

    def __init__(self,
        pass_1_cost_gen: CostGen = HilbertSchmidtResidualsGenerator(),
        pass_2_cost_gen: CostGen = RelaxedTCountCostGenerator(),
        pass_2_cstr_gen: CostGen = HilbertSchmidtCostGenerator(),
        first_pass: Optional[Minimizer] = CeresMinimizer(),
        second_pass: Optional[ConstrainedMinimizer] = None,
        success_threshold: float = 1e-8,
        multistarts = (128, 16),
        **kwargs: dict[str, Any],
    ) -> None:
        """
        """
        if 'success_threshold' in kwargs and success_threshold != 1e-8:
            m = 'Overwritting success_threshold with value from kwargs'
            _logger.debug(m)
            self.threshold = kwargs.get('success_threshold')
        else:
            self.threshold = success_threshold
        # while I am doing everything single-threaded, it makes more sense
        # to do things one at a time IMO
        if 'first_pass_multistarts' in kwargs:
            self.first_pass_multistarts = kwargs.get('first_pass_multistarts')
        else:
            self.first_pass_multistarts = multistarts[0]
        if 'second_pass_multistarts' in kwargs:
            self.second_pass_multistarts = kwargs.get('second_pass_multistarts')
        else:
            self.second_pass_multistarts = multistarts[1]  # TODO: Remove during cleanup

        if second_pass is None:
            second_pass = SLSQPConstrainedMinimizer(self.threshold)

        self.pass_1_cost_gen = pass_1_cost_gen
        self.pass_2_cost_gen = pass_2_cost_gen
        self.pass_2_cstr_gen = pass_2_cstr_gen
        self.first_pass = first_pass
        self.second_pass = second_pass
        # while I am doing everything single-threaded, it makes more sense to do things one at a time IMO
        self.max_first_pass_starts = multistarts[0]
        self.max_second_pass_starts = multistarts[1]

    def is_capable(self, circuit: Circuit) -> bool:
        """
        Return True only if all gates in the circuit are Clifford+T+Rz.
        """
        acceptable_gates = clifford_gates + t_gates + rz_gates
        return all(g in acceptable_gates for g in circuit.gate_set)
    
    def get_violation_report(self, circuit: Circuit) -> str:
        if not self.is_capable(circuit):
            gate_set = circuit.gate_set
            return (
                f'Found gates ({gate_set}) in circuit that are'
                'not Clifford+T+Rz'
            )
        return 'Unknown error'

    def get_method_name(self) -> str:
        return "two-pass-minimization"

    def normalize(self, result: RealVector) -> RealVector:
        return np.mod(result, np.pi * 2)
    
    def is_duplicate_result(
        self,
        candidate: RealVector,
        results: Sequence[RealVector],
    ) -> bool:
        """
        Returns True if candidate is a (near) duplicate of a previous run.
        """
        candidate = self.normalize(candidate)
        for result in results:
            result = self.normalize(result)
            if np.all(np.isclose(result, candidate, 1e-2, 1e-4)):
                return True
        return False
    
    def done_with_first_pass(
        self,
        num_tries: int,
        results: Sequence[RealVector],
        num_starts: int,
    ) -> bool:
        """Whether or not to move on to the second pass."""
        if num_tries >= self.first_pass_retries:
            return True
        if len(results) >= num_starts:
            return True
        return False

    async def two_pass_instantiation_async(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
        num_starts: int,
    ) -> RealVector:


        target = self.check_target(target)
        start_gen = RandomStartGenerator()
        starts = start_gen.gen_starting_points(
            self.first_pass_multistarts, circuit, target
        )

        first_pass_future = get_runtime().map(
                run_first_pass_minimization,
                [self.first_pass] * self.first_pass_multistarts,
                [self.pass_1_cost_gen] * self.first_pass_multistarts,
                [self.pass_2_cstr_gen] * self.first_pass_multistarts,
                [circuit] * self.first_pass_multistarts,
                [target] * self.first_pass_multistarts,
                starts,
                [self.threshold] * self.first_pass_multistarts,
                )

        pass_1_results = []
        second_pass_futures = []
        async for index, result in FutureQueue(first_pass_future, self.first_pass_multistarts):
            if result is None:
                continue
            if self.is_duplicate_result(result, pass_1_results):
                continue
            pass_1_results.append(result)
            assert len(pass_1_results) <= self.second_pass_multistarts
            new_second_pass_future = get_runtime().submit(
                    run_second_pass_minimization,
                    self.second_pass,
                    self.pass_2_cost_gen,
                    self.pass_2_cstr_gen,
                    circuit,
                    target,
                    result,
                    )
            second_pass_futures.append(new_second_pass_future)
            if len(second_pass_futures) >= self.second_pass_multistarts:
                #get_runtime().cancel(first_pass_future)
                break
       # await first_pass_future
        get_runtime().cancel(first_pass_future)
        if len(second_pass_futures) < 1:
            print("No successful first pass results were found.")
            return None

        task_results = [await future for future in second_pass_futures]

        assert len(task_results) == len(second_pass_futures)
        filtered_results = [r for r in task_results if r is not None]
        if len(filtered_results) < 1:
            print("No successful second pass results were found.")
            return None
        results, costs, cstrs = zip(*filtered_results)

        best_result = results[min(zip(costs, cstrs, range(len(costs))))[2]]
        return best_result

    async def multi_start_instantiate_async(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
        num_starts: int,
    ) -> Circuit:
        """
        Run the two-pass minimization multiple times and return best result.

        Args:
            circuit (Circuit): The circuit to optimize.

            target (UnitaryMatrix | StateVector | StateSystem): The target
                unitary matrix or state vector.

            num_starts (int): The number of times to run the two-pass minimization.

        Returns:
            (Circuit) The best circuit found.
        """
        result = await self.two_pass_instantiation_async(circuit, target, num_starts)
        if result is None:
            return None  # TODO: handle this better
        circuit.set_params(result)
        return circuit

    def instantiate(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
        x0: RealVector | NoneType = None, # TODO allow for fine-tuning based on an input vector
    ) -> RealVector:
        return self.two_pass_instantiation(circuit, target)

    def two_pass_instantiation(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
    ) -> RealVector:

        # run the first pass
        pass_1_results = []
        pass_1_cost = self.pass_1_cost_gen.gen_cost(circuit, target)
        pass_2_cstr = self.pass_2_cstr_gen.gen_cost(circuit, target)
        total_tries = 0

        # Ceres may be used to retry promising results that don't
        # quite meet the threshold
        ceres = CeresMinimizer(ftol=5e-16, gtol=1e-15)

        while not self.done_with_first_pass(
            total_tries,
            pass_1_results,
            self.second_pass_multistarts,
        ):
            # run a batch of optimizations
            results = [
                self.first_pass.minimize(
                    pass_1_cost,
                    2 * np.pi * np.random.rand(circuit.num_params),
                ) for _ in range(self.first_pass_multistarts)
            ]

            # filter the results for failures and duplicates
            for result in results:
                if self.is_duplicate_result(result, pass_1_results):
                    continue
                distance = pass_2_cstr(result)
                # this was a promising result and should undergo 
                # higher quality minimization
                if distance > self.threshold and distance < 1e-4:
                    result = ceres.minimize(pass_1_cost, result)
                    distance = pass_2_cstr(result)
                # filter out failures to meet the threshold
                if distance <= self.threshold:
                    pass_1_results.append(result)
            total_tries += 1

        if len(pass_1_results) < 1:
            m = "No successful first pass results were found."
            _logger.warning(m)
            return [0 for _ in range(circuit.num_params)]

        pass_2_cost = self.pass_2_cost_gen.gen_cost(circuit, target)
        best_result = None
        best_cost = None
        best_cstr = None

        for x0 in pass_1_results:
            # TODO: Parallelize
            result = self.second_pass.constrained_minimize(pass_2_cost, pass_2_cstr, x0)
            result_cost = pass_2_cost(result)
            result_cstr = pass_2_cstr(result)
            if result_cstr > self.threshold:
                m = (
                    'Rejected a second pass result because the contraint '
                    f'value was {result_cstr}'
                )
                _logger.debug(m)
                continue

            if best_result is None:
                best_result = result
                best_cost = result_cost
                best_cstr = result_cstr
            elif result_cost < best_cost:
                best_result = result
                best_cost = result_cost
                best_cstr = result_cstr
            elif result_cost == best_cost and result_cstr < best_cstr:
                best_result = result
                best_cost = result_cost
                best_cstr = result_cstr

        return best_result


"""
    # rough outline of what I want to accomplish:
    def two_pass_minimization(
            circuit, # parameterized circuit to optimize
            target,  # target unitary
            cost_fn, # primary cost to be minimized (relaxed T count)
            constraint_fn, # secondary cost to be minimized (HilbertSchmidt)
            first_pass, # minimizer for first pass (whatever is default is fine)
            second_pass, # minimizer for second pass (SLSQP with support for constraints)
            first_pass_multistarts, # size of batch of first passes to run
            second_pass_multistarts, # multistarts for second batch, which also serves as a quota for the first batch
            first_pass_retries, # the first pass is run in batches of first_pass_multistarts until either the quota of second_pass_multistarts is met or until first_pass_retries batches have been run
            ):


        pass_1_results = []
        for _ in range(first_pass_retries):
            results = first_pass.minimize(circuit, target, constraint_fn, first_pass_multistarts)
            pass_1_results.append(results)
            pass_1_results.filter_for_duplicates_and_failures(results, pass_1_results)
            if len(pass_1_results) >= second_pass_multistarts:
            break

        # note that the number of multistarts that second_pass receives could be more or less than second_pass_multistarts
        # it will only be less if the first_pass_retries condition was the limiting factor
        # otherwise it will almost certainly be more
        # its also not necessarily equal to first_pass_multistarts or some multiple because some 1st pass results will get pruned as duplicates or failures

        # also note that second_pass is going to have to use constraint_fn as the constraint threshold-constraint_fn(x).  Idk what will be in charge of that conversion.  Whatever it is, it needs to be aware that the derivative needs to be negated as well.  Also the threshold here isn't necessarily the same threshold as in the first pass.  In fact, it probably shouldn't be, because the second threshold isn't quite guranteed to be met (SLSQP allows small constraint violations)
        pass_2_results = second_pass.minimize(circuit, target, cost_fn, constraint_fn, pass_1_results)
        return pass_2_results.choose_best_result()


    # all of the bonus nuance I can figure out later
    # the primary issues are:
    # 1. What is in charge of organizing multistarts?
    # 2. how can I have the two_pass_minimizer play its role in multi-start selection and running multiple starts?
    #  - note that its not intended for two_pass_minimization to be called in parallel (some number multistarts) times
    #  - should two_pass_minimization be a pass instead of an instantiater?  It really plays the role of "instantiater" but I'm not sure if the instantiater API is powerful enough for it
    # 3. What should be in charge of converting constraint functions to constraints usable by SLSQP, and how should the difference in threshold be expressed?
    # 4. How to get SLSQP?  I can write my own wrapper around scipy, but ideally it would be in rust
"""
