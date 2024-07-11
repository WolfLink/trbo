"""This module implements the NumericalTReductionPass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.ir.gates import *

from .clift import *
from itertools import combinations_with_replacement

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


class RzToT_ScanningBruteForcePass(BasePass):
    def __init__(self, threshold=1e-6, enable_t=True, utry=None, gate_limit=10):
        self.rtol = 1e-5
        self.atol = 1e-8
        self.threshold = threshold
        self.frac = 0.25 if enable_t else 0.5
        self.utry = utry
        self.gate_limit = gate_limit

    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        if circuit.num_params < 1:
            print("Nothing to do - circuit is already CliffT")
            return
        elif circuit.num_params > self.gate_limit:
            print(f"There are {circuit.num_params} params left, which is more than is reasonable to convert by brute force.")
            return
        else:
            print(f"Attempting to round {circuit.num_params} parameters by brute force.")

        utry = self.utry
        if utry is None:
            utry = circuit.get_unitary()

        t_gate_angles = [i * np.pi * self.frac for i in range(int(2  / self.frac))]
        param_lists = combinations_with_replacement(t_gate_angles, circuit.num_params)
        min_distance = 1
        for params in param_lists:
            d = utry.get_distance_from(circuit.get_unitary(params))
            if d < self.threshold:
                circuit.set_params(params)
                break
            else:
                min_distance = min(min_distance, d)

        if not utry.get_distance_from(circuit.get_unitary(params))< self.threshold:
            print(f"Circuit is incompatible with CliffT: {min_distance}")
            return
        else:
            print(f"Circuit successfully converted to CliffT! {utry.get_distance_from(circuit.get_unitary(params))}")

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
