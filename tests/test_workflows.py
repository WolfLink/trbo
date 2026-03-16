import pytest
import numpy as np
from bqskit.compiler import Compiler
from bqskit import Circuit
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from trbo.clift import clifford_gates, t_gates, rz_gates
from trbo.workflows import *

from load_test_file import toffoli_qasm_file


def test_sanitize_synthesized():
    before_circuit = Circuit.from_file(toffoli_qasm_file)

    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, sanitize_gateset())

    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates + rz_gates, f"{gate} not in Clifford+T+Rz"

def test_sanitize_needs_synthesis():
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
    before_circuit = Circuit(3)
    toffoli = ConstantUnitaryGate(toffoli_u)
    before_circuit.append_gate(toffoli, [0, 1, 2])
    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, sanitize_gateset())

    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates + rz_gates, f"{gate} not in Clifford+t+Rz"

def test_default():
    before_circuit = Circuit.from_file(toffoli_qasm_file)

    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, default())

    t_count = 0
    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates, f"{gate} not in Clifford+T"
        if gate in t_gates:
            t_count += after_circuit.gate_counts[gate]
    assert t_count == 7, f"Unexpected T count {t_count}"

def test_fast():
    before_circuit = Circuit.from_file(toffoli_qasm_file)

    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, fast())

    t_count = 0
    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates, f"{gate} not in Clifford+T"
        if gate in t_gates:
            t_count += after_circuit.gate_counts[gate]

def test_slow():
    before_circuit = Circuit.from_file(toffoli_qasm_file)

    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, slow())

    t_count = 0
    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates, f"{after_circuit.gate_counts[gate]} x {gate} not in Clifford+T"
        if gate in t_gates:
            t_count += after_circuit.gate_counts[gate]
    assert t_count == 7, f"Unexpected T count {t_count}"

def test_veryslow():
    before_circuit = Circuit.from_file(toffoli_qasm_file)

    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, veryslow())

    t_count = 0
    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates, f"{gate} not in Clifford+T"
        if gate in t_gates:
            t_count += after_circuit.gate_counts[gate]
    assert t_count == 7, f"Unexpected T count {t_count}"

def test_phase_correct():
    before_circuit = Circuit.from_file(toffoli_qasm_file)

    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, default(phase_correct=True))

    t_count = 0
    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates, f"{gate} not in Clifford+T"
        if gate in t_gates:
            t_count += after_circuit.gate_counts[gate]
    assert t_count == 7, f"Unexpected T count {t_count}"

def test_no_paritioning():
    before_circuit = Circuit.from_file(toffoli_qasm_file)

    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, no_partitioning())

    t_count = 0
    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates, f"{gate} not in Clifford+T"
        if gate in t_gates:
            t_count += after_circuit.gate_counts[gate]
    assert t_count == 7, f"Unexpected T count {t_count}"


def test_no_partitioning_target_unitary():
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

    before_circuit = Circuit.from_file(toffoli_qasm_file)

    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, no_partitioning(utry=toffoli_u))

    t_count = 0
    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates, f"{gate} not in Clifford+T"
        if gate in t_gates:
            t_count += after_circuit.gate_counts[gate]
    assert t_count == 7, f"Unexpected T count {t_count}"
