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
    def __init__(self, period: float=np.pi / 4) -> None:
        """
        Constructor for RelaxedTCountCostGenerator.

        Args:
            period (float): The period of the triangle wave cost function.
                Parameter values will be pushed towards a multiple this
                (Default: np.pi / 4)
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
    def __init__(self, period: float) -> None:
        super().__init__()
        self.period = period

    def get_cost(self, params: RealVector) -> float:
        if len(params) < 1:
            return 0
        if not isinstance(params, np.ndarray):
            params = np.array(params)
        deviation = get_arr(params, self.period)
        cost = np.sum(deviation)
        return float(cost)

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        deviation = np.mod(params, self.period) - self.period / 2
        signs = np.sign(deviation)
        return -1 * (2 / self.period) * signs

def get_arr(params: np.ndarray, period: float) -> npt.NDArray[np.float64]:
    shifted_params = np.mod(params - period / 2, period)
    deviation = np.abs(shifted_params - period / 2)
    return (2 / period) * deviation
