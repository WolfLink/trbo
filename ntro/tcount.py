from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction
from bqskit.ir.opt.cost import CostFunction
from bqskit.qis.unitary import RealVector
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

import numpy as np
import numpy.typing as npt


class RelaxedTCountCostGenerator(CostFunctionGenerator):
    def __init__(self, period: int=4) -> None:
        """
        Constructor for RelaxedTCountCostGenerator.

        Args:
            period (int): The period of the triangle wave cost function.
                Parameter values will be pushed towards multiple of 
                pi / period. (Default: 4)
        """
        super().__init__()
        self.period = period

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        return RelaxedTCount(self.period)


class RelaxedTCount(DifferentiableCostFunction):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period

    def get_cost(self, params: RealVector) -> float:
        period = self.period
        if params.shape[0] < 1:
            return 0
        deviation = self.get_arr(params)
        cost = np.sum(deviation)
        return cost

    def get_arr(self, params: RealVector) -> npt.NDArray[np.float64]:
        period = self.period
        if params.shape[0] < 1:
            return []
        shifted_params = np.mod(params - period / 2, period)
        deviation = np.abs(shifted_params - period / 2)
        return (2 / period) * deviation

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        period = self.period
        deviation = np.mod(params, period) - period / 2
        signs = np.sign(deviation)
        return -1 * (2 / period) * signs