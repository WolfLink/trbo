import pytest
import numpy as np
from bqskit.compiler import Compiler
from bqskit import Circuit
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from trbo.clift import clifford_gates, t_gates, rz_gates
from trbo.workflows import *
from trbo.utils import MultistartPass


def qft(n):
    # this is the qft unitary generator code from qsearch
    root = np.e ** (2j * np.pi / n)
    return np.array(np.fromfunction(lambda x,y: root**(x*y), (n,n)), dtype='complex128') / np.sqrt(n)

def test_stress():
    toffoli_u = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]
    toffoli_u = np.array(toffoli_u, dtype='complex128')
    toffoli = ConstantUnitaryGate(toffoli_u)
    qft2 = ConstantUnitaryGate(qft(4))
    qft3 = ConstantUnitaryGate(qft(8))
    before_circuit = Circuit(7)
    before_circuit.append_gate(qft2, [0,1])
    before_circuit.append_gate(toffoli, [1,3,2])
    before_circuit.append_gate(qft3, [2,4,5])
    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, MultistartPass(default()))

    rz_count = 0
    for gate in after_circuit.gate_counts:
        if gate in rz_gates:
            rz_count += after_circuit.gate_counts[gate]
        else:
            assert gate in clifford_gates + t_gates, f"{after_circuit.gate_counts[gate]} x {gate} not in Clifford + T + Rz"
    assert rz_count == 3, "Unexpected Rz count {rz_count}"


