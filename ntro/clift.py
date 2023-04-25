import numpy as np
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates.constant.z import ZGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.sdg import SdgGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.tdg import TdgGate


clifford_gates = ["cx", "h", "s", "sdg", "swap", "sx", "x", "y", "z"]
t_gates = ["t", "tdg"]
rz_gates = ["rz"]


def circuit_for_rounded_val(val, period):
    circuit = Circuit(1)
    if period == 0.5:
        # cliffords
        rounded_val = np.round(val * 2 / np.pi)
        if rounded_val == 1:
            circuit.append_gate(SGate(), 0)
        elif rounded_val == 2: 
            circuit.append_gate(ZGate(), 0)
        elif rounded_val == 3:
            circuit.append_gate(SdgGate(), 0)
    elif period == 0.25:
        rounded_val = np.round(val * 4 / np.pi)
        if rounded_val < 4:
            if rounded_val >= 2:
                circuit.append_gate(SGate(), 0)
            if rounded_val % 2:
                circuit.append_gate(TGate(), 0)
        elif rounded_val > 4:
            if rounded_val <= 6:
                circuit.append_gate(SdgGate(), 0)
            if rounded_val % 2:
                circuit.append_gate(TdgGate(), 0)
        elif rounded_val == 4:
            circuit.append_gate(ZGate(), 0)
    return CircuitGate(circuit)




