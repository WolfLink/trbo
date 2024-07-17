from bqskit.ir.opt.minimizer import Minimizer
from abc import ABC, abstractmethod
from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction
from bqskit.ir.opt.cost.function import CostFunction
from bqskit.qis.unitary import RealVector

from functools import partial

import numpy as np
import numpy.typing as npt

from scipy.optimize import fmin_slsqp


def f_negate(f, threshold, x):
    return threshold - f(x)


def grad_negate(f, x):
    return [-y for y in f(x)]


class ConstrainedMinimizer(ABC):
    @abstractmethod
    def constrained_minimize(
        self,
        cost: CostFunction,
        cstr: CostFunction,
        x0: RealVector
    ) -> RealVector:
        pass


class SLSQPConstrainedMinimizer(ConstrainedMinimizer):
    def __init__(
        self,
        constraint_threshold: float = 1e-8,
        tol: float = 1e-10
    ) -> None:
        """
        Constructor for ConstrainedMinimizer.
        
        Args:
            constraint_threshold (float): The threshold for the constraint.
                Minimizers below this constraint value are considered invalid.
                (Default: 1e-8)
            
            tol (float): The tolerance for the minimizer. (Defalt: 1e-10)
        """
        self.tol = tol
         # make the threshold a little tighter than we really want because 
         # the optimizer doesn't totally respect it
        self.constraint_threshold = constraint_threshold * 0.5

    def constrained_minimize(
        self,
        cost: CostFunction,
        cstr: CostFunction,
        x0: RealVector,
    ) -> npt.NDArray[np.float64]:
        """
        Call constrained minimizer (SLSQP) to minimize the cost function.
        """
        if len(x0) == 0:
            return np.array([])

        if not isinstance(cost, DifferentiableCostFunction):
            m = 'Cost function must be DifferentiableCostFunction, '
            m += f'got {type(cost)}.'
            raise TypeError(m)
        if not isinstance(cstr, DifferentiableCostFunction):
            m = 'Constraint function must be DifferentiableCostFunction, '
            m += f'got {type(cstr)}.'
            raise TypeError(m)

        # Inequality constraints
        f_ineqcons = partial(
            f_negate, cstr.get_cost, self.constraint_threshold
        )
        fprime_ineqcons = partial(grad_negate, cstr.get_grad)

        # Hyperparameters
        num_iters = 5000 * x0.shape[0]
        accuracy = self.constraint_threshold * 1e-4

        result = fmin_slsqp(
            cost,
            x0,
            f_ieqcons=f_ineqcons,
            fprime=cost.get_grad,
            fprime_ieqcons=fprime_ineqcons,
            iter=num_iters,
            acc=accuracy,
            iprint=-1
        )
        return result
