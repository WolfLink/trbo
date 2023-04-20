from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction
import numpy as np

class RelaxedTCountCostGenerator(CostFunctionGenerator):
    def __init__(self, period=4):
        super().__init__()
        self.period = period

    def gen_cost(self, circuit, target):
        return RelaxedTCount(self.period)



class RelaxedTCount(DifferentiableCostFunction):
    def __init__(self, period):
        super().__init__()
        self.period = period

    def get_cost(self, params):
        period = self.period
        if params.shape[0] < 1:
            return 0
        return (2/period) * np.sum(np.abs(np.mod(params-period/2, period) - period/2))

    def get_arr(self, params):
        period = self.period
        if params.shape[0] < 1:
            return []
        return (2/period)*np.abs(np.mod(params-period/2, period) - period/2)

    def get_grad(self, params):
        period = self.period
        return -(2/period) * np.sign(np.mod(params, period) - period/2)
        

