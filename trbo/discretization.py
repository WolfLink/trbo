from typing import Optional
from bqskit.ir.opt.cost.generator import CostFunctionGenerator

class RzDiscretization():
    def nearest_gate(self, rz_angle: float) -> CircuitGate:
        raise NotImplementedError

    def param_distances(self, params: [float]) -> [float]:
        raise NotImplementedError

    def cost_generator(self, N: int) -> CostFunctionGenerator:
        raise NotImplementedError

    def residuals_generator(self, N: int, dim: int, blacklist: Optional[[int]] = None) -> CostFunctionGenerator:
        raise NotImplementedError
