from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Sequence

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.opt import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt import HilbertSchmidtCostGenerator
from bqskit.ir.opt.cost.residual import ResidualsFunction
from bqskit.ir.opt.cost.generator import CostFunctionGenerator as CostGen
from bqskit.ir.opt.instantiater import Instantiater
from bqskit.ir.opt.minimizer import Minimizer
from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.multistartgens.random import RandomStartGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.qis.unitary import RealVector
from bqskit.runtime import get_runtime


import scipy as sp
import scipy.optimize

import logging

from .utils import FutureQueue

_logger = logging.getLogger(__name__)


def run_minimization(
        circuit: Circuit,
        minimizer: Minimizer,
        cost_gen: CostGen,
        target: UnitaryMatrix | StateVector | StateSystem,
        x0: RealVector,
        ):
    cost = cost_gen.gen_cost(circuit, target)
    return minimizer.minimize(cost, x0)

class MultiStartMinimization(Instantiater):

    def __init__(self,
        cost_gen: CostGen = HilbertSchmidtCostGenerator(),
        threshold: float | NoneType = None,
        multistarts: int = 1,
        second_pass: int | NoneType = None,
        minimizer: Minimizer = LBFGSMinimizer(),
        judgement_cost = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        """
        self.cost_gen = cost_gen
        self.threshold = threshold
        self.multistarts = multistarts
        self.minimizer = minimizer
        self.second_pass = second_pass
        self.judgement_cost = None
        self.debug = False

    def is_capable(self, circuit: Circuit) -> bool:
        """
        Return True only if all gates in the circuit are Clifford+T+Rz.
        """
        return True
    
    def get_violation_report(self, circuit: Circuit) -> str:
        if not self.is_capable(circuit):
            gate_set = circuit.gate_set
            return (
                f'Found gates ({gate_set}) in circuit that are'
                'not Clifford+T+Rz'
            )
        return 'Unknown error'

    def get_method_name(self) -> str:
        return "multi-start-minimization"
 
    def check_similarity(self, a, b):
        a = a % (np.pi * 2)
        b = b % (np.pi * 2)
        d = np.sum(np.abs(a - b))
        return False

    async def multi_start_instantiate_async(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
        starts: [RealVector] | NoneType = None,
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
        num_starts = self.multistarts
        if starts is None:
            starts = RandomStartGenerator().gen_starting_points(num_starts, circuit, target)
        elif len(starts) < num_starts:
            starts.extend(RandomStartGenerator().gen_starting_points(num_starts - len(starts), circuit, target))
        elif len(starts) > num_starts:
            num_starts = len(starts)

        # first pass - find solutions that minimize distance
        succeeded = 0
        if self.second_pass is None:
            gathered_results = starts
        else:
            result_future = get_runtime().map(
                    run_minimization,
                    [circuit] * num_starts,
                    [CeresMinimizer()] * num_starts,
                    [HilbertSchmidtResidualsGenerator()] * num_starts,
                    [target] * num_starts,
                    starts
                    )

            gathered_results = []
            first_pass_cost = HilbertSchmidtCostGenerator().gen_cost(circuit, target)
            if self.second_pass >= 1:
                quota = self.second_pass
            else:
                quota = num_starts
            
            queue = FutureQueue(result_future, num_starts)
            async for index, result in queue:
                if self.threshold is not None and first_pass_cost(result) >= self.threshold:
                    continue
                succeeded += 1
                found_match = False
                for other in gathered_results:
                    if self.check_similarity(result, other):
                        found_match = True
                        break
                if found_match:
                    continue
                gathered_results.append(result)
                if len(gathered_results) > quota:
                    queue.cancel()

        # second pass - find solutions that minimize the cost
        num_starts = len(gathered_results)
        if self.debug:
            print(f"{succeeded}/{self.multistarts} -> {num_starts}/{self.second_pass}")
        if num_starts == 0:
            # First pass failed. Generally, this should never happen,
            # Because we know at least one set of parameters that should pass the threshold.
            raise RuntimeWarning("First Pass Failed!")
            return circuit

        result_future = get_runtime().map(
                run_minimization,
                [circuit] * num_starts,
                [self.minimizer] * num_starts,
                [self.cost_gen] * num_starts,
                [target] * num_starts,
                gathered_results,
                )
        
        if self.judgement_cost is None:
            cost = self.cost_gen.gen_cost(circuit, target)
        else:
            cost = self.judgement_cost.gen_cost(circuit, target)
        best_result = circuit.params
        best_cost = cost.get_cost(circuit.params)
        best_cost = None
        async for index, result in FutureQueue(result_future, num_starts):
            distance = cost.get_cost(result)
            if best_cost is None or distance < best_cost:
                best_cost = distance
                best_result = result
            if self.threshold is not None and best_cost < self.threshold:
                break
        get_runtime().cancel(result_future) # BQSKIT BUG causes this to not work TODO
        #if self.threshold is not None and best_cost >= self.threshold:
        #    return None
        circuit.set_params(best_result)
        return circuit

    def instantiate(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
        starts: [RealVector] | None,
    ) -> Circuit:

        raise NotImplementedError

        if num_starts is None:
            num_starts = self.multistarts

        if starts is None:
            starts = RandomStartGenerator().gen_starting_points(num_starts, circuit, target)
        num_starts = len(starts)



        cost = self.cost_gen.gen_cost(circuit, target)
        best_result = None
        best_cost = None
        for start in starts:
            result = run_minimization(self.minimizer, self.cost_gen, target, start)
            distance = cost(result)
            if best_cost is None or distance < best_cost:
                best_cost = distance
                best_result = result
            if self.threshold is not None and best_cost < self.threshold:
                break
        if self.threshold is not None and best_cost >= self.threshold:
            return None
        circuit.set_params(best_result)
        return circuit
