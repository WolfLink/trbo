from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction, DifferentiableResidualsFunction
from bqskit.ir.opt.cost import CostFunction
from bqskit.qis.unitary import RealVector
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

NTRORS = False
try:
    from bqskitrs import SumResidualsFunction as NTRORS_SumResidualsFunction
    from bqskitrs import SmallestNResidualsFunction as NTRORS_SmallestNResidualsFunction
    NTRORS = True
except:
    pass


import numpy as np
import numpy.typing as npt

class MatrixDistanceCostGenerator(CostFunctionGenerator):
    def __init__(self, degree=2):
        self.degree = degree

    def gen_cost(self, circuit, target):
        return MatrixDistanceCost(self.degree, circuit, target)

class MatrixDistanceCost(DifferentiableCostFunction):
    def __init__(self, degree, circuit, target):
        self.degree = degree
        self.circuit = circuit
        self.target = target

    def get_cost(self, params):
        mat = self.circuit.get_unitary(params)
        num = np.abs(np.trace(mat.conj().T @ self.target))
        dem = mat.dim
        frac = min(num / dem, 1)
        dist = np.power(1 - (frac ** self.degree), 1.0 / self.degree)
        dist = dist * (dist > 0.0)
        return dist

    def get_grad(self, params):
        U = self.target
        M, J = self.circuit.get_unitary_and_grad(params)

        P = np.multiply(U, np.conj(M))
        S = np.sum(np.array(P))
        JU = np.array([np.multiply(U,np.conj(K)) for K in J])
        JUS = np.sum(JU, axis=(1,2))
        try:
            dem = M.dim
            frac = min(np.abs(S) / dem, 1)

            p1 = -(frac ** (self.degree - 1))
            p2 = np.power(1 - (frac ** self.degree), (1.0 / self.degree) - 1.0)
            p3 = (np.real(S)*np.real(JUS) + np.imag(S)*np.imag(JUS)) / (U.shape[0] * np.abs(S))

            return p1 * p2 * p3
        except (RuntimeError, FloatingPointError, ZeroDivisionError, OverflowError):
            if np.isclose(S, 0):
                return 0 * JUS
            else:
                raise


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
        if NTRORS:
            return NTRORS_SumResidualsFunction(self.A.gen_cost(circuit, target), self.B.gen_cost(circuit, target))
        else:
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
    def __init__(self, N: int, period: float=np.pi / 4, blacklist=None) -> None:
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
        self.blacklist = blacklist

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        return RoundSmallestNCost(self.N, self.period, self.blacklist)

def get_deviation_arr(params: np.ndarray, period: float, blacklist=None) -> npt.NDArray[np.float64]:
    shifted_params = np.mod(params - period / 2, period)
    deviation = np.abs(shifted_params - period / 2)
    if blacklist is not None:
        deviation += np.max(deviation) * blacklist
    return deviation / 2

def get_deviation_arr_grad(params: np.ndarray, period: float) -> npt.NDArray[np.float64]:
    shifted_params = np.mod(params - period / 2, period)
    return 0.5 * np.sign(shifted_params - period / 2)

def prep_arr(*args):
    return args

class RoundSmallestNCost(DifferentiableCostFunction):
    def __init__(self, N: int, period: float, blacklist=None) -> None:
        super().__init__()
        self.period = period
        self.N = N
        self.blacklist = blacklist

    def get_cost(self, params: RealVector) -> float:
        if len(params) < 1 or self.N < 1:
            return 0
        if not isinstance(params, np.ndarray):
            params = np.array(params)
        deviation = get_deviation_arr(params, self.period, self.blacklist)
        deviation = np.sort(deviation)

        cost = np.sum(deviation[:self.N])
        return float(cost)

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        if self.N < 1:
            return np.zeros_like(params)
        grad = get_deviation_arr_grad(params, self.period)

        deviation = get_deviation_arr(params, self.period, self.blacklist)
        mask = np.zeros_like(params)
        indices = np.argsort(deviation)
        mask[indices[:self.N]] = 1

        return grad * mask


class RoundSmallestNResidualsGenerator(CostFunctionGenerator):
    def __init__(self, N: int, period: float = np.pi/4, blacklist=None) -> None:
        super().__init__()
        self.period = period
        self.N = N
        self.blacklist = blacklist

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        if NTRORS and self.blacklist is None:
            return NTRORS_SmallestNResidualsFunction(self.N, self.period, circuit.dim)
        else:
            return RoundSmallestNResiduals(self.N, self.period, circuit.dim, self.blacklist)

class RoundSmallestNResiduals(DifferentiableResidualsFunction):
    def __init__(self, N: int, period: float, dim: int, blacklist=None) -> None:
        super().__init__()
        self.period = period
        self.N = N
        self.dim = dim
        self.blacklist = blacklist

    def get_cost(self, params: RealVector) -> float:
        return np.sum(np.square(self.get_residuals(params)))

    def num_residuals(self) -> float:
        return self.N

    def get_residuals(self, params: RealVector) -> float:
        if len(params) < 1 or self.N < 1:
            return []
        if not isinstance(params, np.ndarray):
            params = np.array(params)
        deviation = get_deviation_arr(params, self.period, self.blacklist)
        deviation = np.sort(deviation)
        return self.dim * deviation[:self.N] # TODO the math suggests taking the square root should be better but I seem to reliably get mariginally better results without taking the square root.  Investigate

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        if self.N < 1:
            return np.zeros((self.N, len(params)))
        grad = get_deviation_arr_grad(params, self.period)

        deviation = get_deviation_arr(params, self.period, self.blacklist)
        sort_dev = deviation
        indices = np.argsort(sort_dev)

        output = np.zeros((self.N, len(params)))
        for i in range(self.N):
            j = indices[i]
            output[i][j] = grad[j]
        return self.dim * output
