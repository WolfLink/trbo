import numpy as np
from scipy.optimize import fmin_slsqp
from bqskit.ir.opt.minimizer import Minimizer

class ConstrainedMinimizer(Minimizer):

    def __init__(self, constraint, self.constraint_threshold = 1e-8, tol=1e-10):
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
        return fmin_slsqp(cost, x0,
                f_ieqcons=lambda x: self.constraint_threshold - self.constraint(x),
                fprime = TODO,
                fprime_ieqcons = TODO,
                iter=5000*x0.shape[0],
                acc = self.constraint_threshold * 1e-4,
                iprint=-1)
