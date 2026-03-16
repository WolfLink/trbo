import pytest
from bqskit.compiler import Compiler
from bqskit import Circuit
from trbo.clift import clifford_gates, t_gates, rz_gates
from trbo.workflows import *


def test_default():
    toffoli_qasm_file = "synthesized_toffoli.qasm"
    before_circuit = Circuit.from_file(toffoli_qasm_file)

    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, default())

    t_count = 0
    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates, f"{gate} not in Clifford+T"
    print(f"Final Gate Counts: {after_circuit.gate_counts}")

