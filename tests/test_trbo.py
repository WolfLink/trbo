import pytest
from bqskit.compiler import Compiler
from bqskit import Circuit
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from trbo.clift import clifford_gates, t_gates, rz_gates
from trbo.workflows import *

from load_test_file import *

def test_toffoli():
    before_circuit = Circuit.from_file(toffoli_qasm_file)

    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, default())

    assert before_circuit.get_unitary().get_distance_from(after_circuit.get_unitary()) < 1e-6, "Invalid circuit returned"

    t_count = 0
    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates, f"{gate} not in Clifford+T"
        if gate in t_gates:
            t_count += after_circuit.gate_counts[gate]
    assert t_count == 7, f"Unexpected T count {t_count}: {after_circuit.gate_counts}"



def test_qft2():
    before_circuit = Circuit.from_file(qft2_qasm_file)

    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, default())

    assert before_circuit.get_unitary().get_distance_from(after_circuit.get_unitary()) < 1e-6, "Invalid circuit returned"

    t_count = 0
    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates, f"{gate} not in Clifford+T"
        if gate in t_gates:
            t_count += after_circuit.gate_counts[gate]
    assert t_count == 3, f"Unexpected T count {t_count}: {after_circuit.gate_counts}"




def test_qft3():
    before_circuit = Circuit.from_file(qft3_qasm_file)

    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, default())

    assert before_circuit.get_unitary().get_distance_from(after_circuit.get_unitary()) < 1e-6, "Invalid circuit returned"

    rz_count = 0
    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates + rz_gates, f"{gate} not in Clifford+T+Rz"
        if gate in rz_gates:
            rz_count += after_circuit.gate_counts[gate]
    assert rz_count == 3, f"Unexpected Rz count {rz_count}: {after_circuit.gate_counts}"
