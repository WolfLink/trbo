from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction, DifferentiableResidualsFunction
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
    #return deviation
    return (2 / period) * deviation




class SumCostGenerator(CostFunctionGenerator):
    def __init__(self, A: CostFunctionGenerator, B: CostFunctionGenerator) -> None:
        super().__init__()
        self.A = A
        self.B = B

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        return SumCost(self.A.gen_cost(circuit, target), self.B.gen_cost(circuit, target))

class SumCost(DifferentiableCostFunction):
    def __init__(self, A: DifferentiableCostFunction, B: DifferentiableCostFunction):
        super().__init__()
        self.A = A
        self.B = B

    def get_cost(self, params: RealVector) -> float:
        return self.A.get_cost(params) + self.B.get_cost(params)

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        return self.A.get_grad(params) + self.B.get_grad(params)


class SumResidualsGenerator(CostFunctionGenerator):
    def __init__(self, A: CostFunctionGenerator, B: CostFunctionGenerator) -> None:
        super().__init__()
        self.A = A
        self.B = B

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        return SumResiduals(self.A.gen_cost(circuit, target), self.B.gen_cost(circuit, target), circuit.params)

class SumResiduals(DifferentiableResidualsFunction):
    def __init__(self, A: DifferentiableResidualsFunction, B: DifferentiableResidualsFunction, test_params):
        super().__init__()
        self.A = A
        self.B = B
        self.test_params = test_params

    def get_cost(self, params: RealVector) -> float:
        return np.sum(np.square(self.get_residuals(params)))

    def num_residuals(self):
        try:
            numa = self.A.num_residuals()
        except:
            numa = len(self.A.get_residuals(self.test_params))
        try:
            numb = self.B.num_residuals()
        except:
            numb = len(self.B.get_residuals(self.test_params))
        return numa + numb

    def get_residuals(self, params: RealVector) -> RealVector:
        return np.concatenate((self.A.get_residuals(params), self.B.get_residuals(params)), axis=None)

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        return np.concatenate((self.A.get_grad(params), self.B.get_grad(params)), axis=0)


class RoundSmallestNCostGenerator(CostFunctionGenerator):
    def __init__(self, N: int, period: float=np.pi / 4) -> None:
        """
        Constructor for RelaxedTCountCostGenerator.

        Args:
            period (float): The period of the triangle wave cost function.
                Parameter values will be pushed towards a multiple this
                (Default: np.pi / 4)
        """
        super().__init__()
        self.period = period
        self.N = N

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        return RoundSmallestNCost(self.N, self.period)
# TODO probably need an argsort or something for the grad etc.

def get_deviation_arr(params: np.ndarray, period: float) -> npt.NDArray[np.float64]:
    shifted_params = np.mod(params - period / 2, period)
    deviation = np.abs(shifted_params - period / 2)
    return deviation / 2

def get_deviation_arr_grad(params: np.ndarray, period: float) -> npt.NDArray[np.float64]:
    shifted_params = np.mod(params - period / 2, period)
    return 0.5 * np.sign(shifted_params - period / 2)

def prep_arr(*args):
    return args

class RoundSmallestNCost(DifferentiableCostFunction):
    def __init__(self, N: int, period: float) -> None:
        super().__init__()
        self.period = period
        self.N = N

    def get_cost(self, params: RealVector) -> float:
        if len(params) < 1 or self.N < 1:
            return 0
        if not isinstance(params, np.ndarray):
            params = np.array(params)
        deviation = get_deviation_arr(params, self.period)
        #deviation = 1 - np.sqrt(0.5*(1+np.cos(deviation)))
        #deviation = 0.5 - 0.5 * np.cos(deviation)
        #deviation = deviation / 2
        deviation = np.sort(deviation)

        cost = np.sum(deviation[:self.N])
        return float(cost)

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        if self.N < 1:
            return np.zeros_like(params)
        grad = get_deviation_arr_grad(params, self.period)

        deviation = get_deviation_arr(params, self.period)
        mask = np.zeros_like(params)
        #sort_dev = 1 - np.sqrt(0.5*(1+np.cos(deviation)))
        #sort_dev = 0.5 - 0.5 * np.cos(deviation)
        #sort_dev = deviation / 2
        indices = np.argsort(deviation)
        mask[indices[:self.N]] = 1

        #mult = 0.25 * np.sin(deviation) / np.sqrt(0.5*(1+np.cos(deviation)))
        #mult = 0.5 * np.sin(deviation)
        #mult = sort_dev
        return grad * mask


class RoundSmallestNResidualsGenerator(CostFunctionGenerator):
    def __init__(self, N: int, period: float, smoothed: bool = False) -> None:
        super().__init__()
        self.period = period
        self.N = N
        self.smoothed = smoothed

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        if self.smoothed:
            return RoundSmallestNResidualsSmoothed(self.N, self.period, circuit.dim)
        else:
            return RoundSmallestNResiduals(self.N, self.period, circuit.dim)
# TODO probably need an argsort or something for the grad etc.

class RoundSmallestNResiduals(DifferentiableResidualsFunction):
    def __init__(self, N: int, period: float, dim: int) -> None:
        super().__init__()
        self.period = period
        self.N = N
        self.dim = dim

    def get_cost(self, params: RealVector) -> float:
        return np.sum(np.square(self.get_residuals(params)))

    def num_residuals(self) -> float:
        return self.N

    def get_residuals(self, params: RealVector) -> float:
        if len(params) < 1 or self.N < 1:
            return []
        if not isinstance(params, np.ndarray):
            params = np.array(params)
        deviation = get_deviation_arr(params, self.period)
        #deviation = 1 - np.sqrt(0.5*(1+np.cos(deviation)))
        #deviation = 0.5 - 0.5 * np.cos(deviation)
        deviation = np.sort(deviation)
        #return np.sqrt(deviation[:self.N])
        return self.dim * deviation[:self.N]

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        if self.N < 1:
            return np.zeros((self.N, len(params)))
        grad = get_deviation_arr_grad(params, self.period)

        deviation = get_deviation_arr(params, self.period)
        #sort_dev = 1 - np.sqrt(0.5*(1+np.cos(deviation)))
        #sort_dev = 0.5 - 0.5 * np.cos(deviation)
        sort_dev = deviation
        indices = np.argsort(sort_dev)

        output = np.zeros((self.N, len(params)))
        for i in range(self.N):
            j = indices[i]
            output[i][j] = grad[j]
            #output[i][j] = -1 * (2 / self.period) * signs[j] * 0.5 * np.sin(deviation[j])
            #output[i][j] = -1 * (2 / self.period) * signs[j] * 0.25 * np.sin(deviation[j]) / np.sqrt(0.5 - 0.5 * np.cos(deviation[j]))
        return self.dim * output


class RoundSmallestNResidualsSmoothed(DifferentiableResidualsFunction):
    def __init__(self, N: int, period: float, dim: int) -> None:
        super().__init__()
        self.period = period
        self.N = N
        self.dim = dim

    def get_cost(self, params: RealVector) -> float:
        return np.sum(np.square(self.get_residuals(params)))

    def num_residuals(self) -> float:
        return self.N

    def get_residuals(self, params: RealVector) -> float:
        if len(params) < 1 or self.N < 1:
            return []
        if not isinstance(params, np.ndarray):
            params = np.array(params)
        deviation = get_deviation_arr(params, self.period)
        #deviation = 1 - np.sqrt(0.5*(1+np.cos(deviation)))
        deviation = 0.5 - 0.5 * np.cos(deviation)
        deviation = np.sort(deviation)
        #return np.sqrt(deviation[:self.N])
        return self.dim * deviation[:self.N]

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        if self.N < 1:
            return np.zeros((self.N, len(params)))
        grad = get_deviation_arr_grad(params, self.period)

        deviation = get_deviation_arr(params, self.period)
        #sort_dev = 1 - np.sqrt(0.5*(1+np.cos(deviation)))
        sort_dev = 0.5 - 0.5 * np.cos(deviation)
        sort_dev = deviation
        indices = np.argsort(sort_dev)

        output = np.zeros((self.N, len(params)))
        for i in range(self.N):
            j = indices[i]
            #output[i][j] = grad[j]
            output[i][j] = 0.5 * grad[j] * np.sin(deviation[j])
            #output[i][j] = -1 * (2 / self.period) * signs[j] * 0.25 * np.sin(deviation[j]) / np.sqrt(0.5 - 0.5 * np.cos(deviation[j]))
        return self.dim * output
