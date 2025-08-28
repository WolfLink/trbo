import numpy as np
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate

from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import CZGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import SdgGate
from bqskit.ir.gates import SGate
from bqskit.ir.gates import SwapGate
from bqskit.ir.gates import SqrtXGate
from bqskit.ir.gates import TdgGate
from bqskit.ir.gates import TGate
from bqskit.ir.gates import XGate
from bqskit.ir.gates import YGate
from bqskit.ir.gates import ZGate

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.utils.cachedclass import CachedClass
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class GlobalPhaseGate(QubitGate, DifferentiableUnitary, CachedClass):
    """GlobalPhaseGate is the identity but with a global phase. It doesn't represent a physical gate (or rather, it represents a noop, like the identity), but its inclusion in a circuit allows a matrix distance function that is global phase dependent to find solutions that don't have a global phase match."""
    _num_qudits = 1
    _num_params = 1
    _qasm_name = "identity1"

    def get_unitary(self, params):
        p = np.exp(1j * params[0])
        return UnitaryMatrix([[p, 0],[0,p]])

    def get_grad(self, params):
        dp = 1j * np.exp(1j * params[0])
        return np.array([[[dp, 0],[0,dp]]], dtype='complex128')


# These three lists are used as the definition of what we mean by Clifford, T, and Rz gates throughout the code

clifford_gates = [
    CNOTGate(),
    CZGate(),
    HGate(),
    SdgGate(),
    SGate(),
    SqrtXGate(),
    SwapGate(),
    XGate(),
    YGate(),
    ZGate(),
    GlobalPhaseGate(),
]

t_gates = [TGate(), TdgGate()] 

rz_gates = [RZGate()]




def circuit_for_rounded_val(val: float, enable_t: bool) -> CircuitGate:
    """
    Returns a CircuitGate with in the {Z, S, Sdg, T, Tdg} gateset depending on val.
    
    Args:
        val (float): A parameter value to round.

        period (float): The period of the rounding function. (either pi/2 or pi/4)
    
    Returns:
        (CircuitGate): A CircuitGate containing a Z rotation gate depending
            on val.
    """
    circuit = Circuit(1)
    val = val % (2 * np.pi)
    if not enable_t:
        # cliffords
        rounded_val = np.round(val * 2 / np.pi)
        if rounded_val == 1:
            circuit.append_gate(SGate(), 0)
        elif rounded_val == 2: 
            circuit.append_gate(ZGate(), 0)
        elif rounded_val == 3:
            circuit.append_gate(SdgGate(), 0)
    else:
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


def better_min_t_count_circuit(a, b):
    """Returns True if circuit b is a "better" circuit than circuit a, and False otherwise.

    Circuit b is considered "better" based on the following criteria:
    1. Any circuit is better than None
    2. Fewer gates outside of Clifford+T+Rz is better
    3. Fewer Rz gates is better
    4. Fewer T gates is better
    5. Fewer multi-qubit gates is better
    6. Fewer Clifford gates is better
    7. Ties that persist are given to circuit a (returns False)
    """

    # if one circuit is None and the other isn't, choose the one that isn't None
    if a is None:
        return True
    elif b is None:
        return False

    # If both circuits are not None...
    # choose the circuit that has the least gates that are outside of the cliff+T+Rz set (ideally both circuits have 0)
    agc = 0
    bgc = 0
    for gate in a.gate_counts:
        if gate not in clifford_gates + t_gates + rz_gates:
            agc += a.gate_counts[gate]
    for gate in b.gate_counts:
        if gate not in clifford_gates + t_gates + rz_gates:
            bgc += b.gate_counts[gate]

    if agc > bgc:
        return True
    elif bgc > agc:
        return False


    # If both circuits have the same count of gates outside cliff+T+Rz (ideally both should have 0)...
    # choose the circuit with the least gates following this order of priority:
    # 1. rz_gates 2. t_gates 3. multi-qubit clifford_gates 4. total clifford_gates
    for gate_list in [rz_gates, t_gates, [gate for gate in clifford_gates if gate.num_qudits > 1], clifford_gates]:
        agc = 0
        bgc = 0
        for gate in a.gate_counts:
            if gate in gate_list:
                agc += a.gate_counts[gate]
        for gate in b.gate_counts:
            if gate in gate_list:
                bgc += b.gate_counts[gate]

        if agc > bgc:
            return True
        elif bgc > agc:
            return False


    # If the tie persists this far just pick one
    return False
        
        
