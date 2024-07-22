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

def run_minimization(
        circuit: Circuit,
        minimizer: Minimizer,
        cost_gen: CostGen,
        target: UnitaryMatrix | StateVector | StateSystem,
        x0: RealVector,
        two_pass: bool = True,
        threshold: float | NoneType = None,
        ):
    if two_pass:
        cost = HilbertSchmidtResidualsGenerator().gen_cost(circuit, target)
        x0 = CeresMinimizer().minimize(cost, x0)
        if threshold is not None and HilbertSchmidtCostGenerator().gen_cost(circuit, target)(x0) >= threshold:
            return x0
    cost = cost_gen.gen_cost(circuit, target)
    if isinstance(minimizer, CeresMinimizer):
        return sp.optimize.least_squares(cost.get_residuals, x0, cost.get_grad, method='lm').x
    return minimizer.minimize(cost, x0)

class MultiStartMinimization(Instantiater):

    def __init__(self,
        cost_gen: CostGen = HilbertSchmidtCostGenerator(),
        threshold: float | NoneType = None,
        multistarts: int = 32,
        minimizer: Minimizer = LBFGSMinimizer(),
        **kwargs: dict[str, Any],
    ) -> None:
        """
        """
        self.cost_gen = cost_gen
        self.threshold = threshold
        self.multistarts = multistarts
        self.minimizer = minimizer

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
        num_starts = len(starts)
        num_two_pass = num_starts // 2
        result_future = get_runtime().map(
                run_minimization,
                [circuit] * num_starts,
                [self.minimizer] * num_starts,
                [self.cost_gen] * num_starts,
                [target] * num_starts,
                starts,
                [True] * (num_two_pass) + [False] * (num_starts - num_two_pass),
                [self.threshold] * (num_two_pass) + [None] * (num_starts - num_two_pass),
                )

        cost = self.cost_gen.gen_cost(circuit, target)
        best_result = None
        best_cost = None
        async for index, result in FutureQueue(result_future, num_starts):
            distance = cost(result)
            if isinstance(cost, ResidualsFunction):
                distance = np.sum(np.square(distance))
            if best_cost is None or distance < best_cost:
                best_cost = distance
                best_result = result
            if self.threshold is not None and best_cost < self.threshold:
                break
        get_runtime().cancel(result_future)
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
