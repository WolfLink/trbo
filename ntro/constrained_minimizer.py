import numpy as np
from scipy.optimize import fmin_slsqp
from functools import partial
from bqskit.ir.opt.minimizer import Minimizer
from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction

def f_negate(f, threshold, x):
    return -f(x)

def grad_negate(f, x):
    return [-y for y in f(x)]

class ConstrainedMinimizer(Minimizer):

    def __init__(self, constraint, constraint_threshold = 1e-6, tol=1e-10):
        self.tol = tol
        self.constraint = constraint
        self.constraint_threshold = constraint_threshold

    def minimize(
            self,
            cost,
            x0,
            ):
        if len(x0) == 0:
            return np.array([])

        constraint = self.constraint
        assert isinstance(cost, DifferentiableCostFunction)
        assert isinstance(self.constraint, DifferentiableCostFunction)
        return fmin_slsqp(cost, x0,
                f_ieqcons=partial(f_negate, self.constraint.get_cost, self.constraint_threshold),
                fprime = cost.get_grad,
                fprime_ieqcons = partial(grad_negate, self.constraint.get_grad),
                iter=5000*x0.shape[0],
                acc = self.constraint_threshold * 1e-4,
                iprint=-1)
