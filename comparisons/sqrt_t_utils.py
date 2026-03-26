from typing import Optional

import numpy as np

import bqskit
from bqskit.ir.gates import ConstantGate, ConstantUnitaryGate, QubitGate, CircuitGate, ZGate, TGate, SGate, TdgGate, CNOTGate, RZGate, SdgGate
from bqskit.ir.opt.cost import CostFunctionGenerator
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.ir.circuit import Circuit
from bqskit.compiler import Compiler
from bqskit.utils.cachedclass import CachedClass

import trbo
from trbo.discretization import RzDiscretization
from trbo.tcount import RoundSmallestNCostGenerator, RoundSmallestNResidualsGenerator, get_deviation_arr
from trbo.clift import clifford_gates, t_gates, rz_gates, RzAsT, RzAsCliff
from trbo.utils import MultistartPass


class SqrtTGate(ConstantGate, QubitGate, CachedClass):
    _num_qudits = 1
    _num_params = 0
    _qasm_name = "pi/8"
    _utry = UnitaryMatrix(
        [
            [1, 0],
            [0, np.exp(1j * np.pi / 8)],
        ],
    )
class SqrtTDaggerGate(ConstantGate, QubitGate, CachedClass):
    _num_qudits = 1
    _num_params = 0
    _qasm_name = "-pi/8"
    _utry = UnitaryMatrix(
        [
            [1, 0],
            [0, np.exp(-1j * np.pi / 8)],
        ],
    )
sqrt_t_gates = [SqrtTGate(), SqrtTDaggerGate()]

class RzAsSqrtT(RzDiscretization):
    def nearest_gate(self, rz_angle: float) -> CircuitGate:
        return circuit_for_rounded_val_sqrt_t(rz_angle)

    def param_distances(self, params: [float], blacklist: Optional[[int]] = None) -> [float]:
        return get_deviation_arr(params, np.pi / 8, blacklist)

    def cost_generator(self, N: int, blacklist: Optional[[int]] = None) -> CostFunctionGenerator:
        return RoundSmallestNCostGenerator(N, period = np.pi / 8, blacklist = blacklist)

    def residuals_generator(self, N: int, blacklist: Optional[[int]] = None) -> CostFunctionGenerator:
        return RoundSmallestNResidualsGenerator(N, np.pi / 8, blacklist)

def circuit_for_rounded_val_sqrt_t(val: float) -> CircuitGate:
    circuit = Circuit(1)
    val = val % (2 * np.pi)
    rounded_val = int(np.round(val * 8 / np.pi)) % 16
    if rounded_val >= 8:
        if rounded_val == 15:
            circuit.append_gate(SqrtTDaggerGate(), 0)
            return CircuitGate(circuit)
        circuit.append_gate(ZGate(), 0)
        rounded_val -= 8
    match rounded_val:
        case 0:
            pass
        case 1:
            circuit.append_gate(SqrtTGate(), 0)
        case 2:
            circuit.append_gate(TGate(), 0)
        case 3:
            circuit.append_gate(SqrtTDaggerGate(), 0)
            circuit.append_gate(SGate(), 0)
        case 4:
            circuit.append_gate(SGate(), 0)
        case 5:
            circuit.append_gate(SGate(), 0)
            circuit.append_gate(SqrtTGate(), 0)
        case 6:
            circuit.append_gate(SGate(), 0)
            circuit.append_gate(TGate(), 0)
        case 7:
            circuit.append_gate(SqrtTDaggerGate(), 0)
            circuit.append_gate(ZGate(), 0)
    return CircuitGate(circuit)

def better_min_sqrt_t_count_circuit(a, b):
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
    if b is None:
        return False
    elif a is None:
        return True

    # If both circuits are not None...
    # choose the circuit that has the least gates that are outside of the cliff+T+Rz+sqrt_t set (ideally both circuits have 0)
    agc = 0
    bgc = 0
    for gate in a.gate_counts:
        if gate not in clifford_gates + t_gates + rz_gates + sqrt_t_gates:
            agc += a.gate_counts[gate]
    for gate in b.gate_counts:
        if gate not in clifford_gates + t_gates + rz_gates + sqrt_t_gates:
            bgc += b.gate_counts[gate]

    if bgc > agc:
        return False
    elif agc > bgc:
        return True


    # If both circuits have the same count of gates outside cliff+T+Rz (ideally both should have 0)...
    # choose the circuit with the least gates following this order of priority:
    # 1. rz_gates 2. sqrt_t_gates 3. t_gates 4. multi-qubit clifford_gates 5. total clifford_gates
    for gate_list in [rz_gates, sqrt_t_gates, t_gates, [gate for gate in clifford_gates if gate.num_qudits > 1], clifford_gates]:
        agc = 0
        bgc = 0
        for gate in a.gate_counts:
            if gate in gate_list:
                agc += a.gate_counts[gate]
        for gate in b.gate_counts:
            if gate in gate_list:
                bgc += b.gate_counts[gate]

        if bgc > agc:
            return False
        elif agc > bgc:
            return True


    # If the tie persists this far just let pick one
    return False
        


def sqrt_t_workflow():
    return trbo.workflows.default(128, strict_opt=True, rz_disc=[RzAsSqrtT(), RzAsT(), RzAsCliff()], comparator=better_min_sqrt_t_count_circuit)


def test_sqrt_t():
    for i in range(32):
        angle = np.pi * i / 8
        rz = RZGate().get_unitary([angle])
        st = circuit_for_rounded_val_sqrt_t(angle).get_unitary()
        assert st.get_distance_from(rz) < 1e-6, f"Large distance {st.get_distance_from(rz)} for angle {angle / np.pi}pi at index {i}"
    for angle in [-1.1241243e-9]:
        rz = RZGate().get_unitary([angle])
        st = circuit_for_rounded_val_sqrt_t(angle).get_unitary()
        assert st.get_distance_from(rz) < 1e-6, f"Large distance {st.get_distance_from(rz)} for angle {angle / np.pi}pi"

