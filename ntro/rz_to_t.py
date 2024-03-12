"""This module implements the NumericalTReductionPass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.ir.gates import *

from .clift import *

# Numerical T Reduction and Optimization



class RzToTPass(BasePass):
    def __init__(self,
            rtol=1e-5,
            atol=1e-8,
            conversion_method=None):
        # rtol and atol are the parameters for a numpy.isclose call
        # conversion_method should be a function that maps a Rz angle to a circuit.
        # the value of None does not do any Rz conversion besides rounding sufficiently close Rz gates
        self.rtol = rtol
        self.atol = atol
        self.conversion_method = conversion_method

    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        points = [
                (cycle, op.location[0])
                for cycle, op in circuit.operations_with_cycles()
                if isinstance(op.gate, RZGate)
                ]

        shift = 0
        for point in points:
            point = (point[0] - shift, point[1])

            op = circuit[point]
            og_cycle_count = circuit.num_cycles
            
            val = op.params[0]
            rounded_val = np.round(val * 4 / np.pi) * np.pi / 4
            if np.isclose(val, rounded_val, self.rtol, self.atol):
                circuit.replace_gate(point, circuit_for_rounded_val(val % (np.pi * 2), 0.25), op.location)
            elif self.conversion_method:
                circuit.replace_gate(point, self.conversion_method(val % (np.pi * 2)), op.location)

            shift += og_cycle_count - circuit.num_cycles
