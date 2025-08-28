import pytest
from ntro.gridsynth import GridsynthPass
from ntro.clift import clifford_gates, t_gates, rz_gates
from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit.ir.gates import RZGate
try:
    import pygridsynth
    pygridsynth_installed = True
except:
    pygridsynth_installed = False

@pytest.mark.skipif(not pygridsynth_installed, reason="pygridsynth is not installed")
def test_gridsynth_pass():
    before_circuit = Circuit(1)
    before_circuit.append_gate(RZGate(), 0)
    before_circuit.set_params([1]) # 1 radian corresponds to an RZ Gate that is far from "nice" gates (like Clifford or T gates)
    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, GridsynthPass())
        after_circuit.unfold_all()

    assert after_circuit.get_unitary().get_distance_from(before_circuit.get_unitary()) < 1e-5, f"Insufficient synthesis quality: {after_circuit.get_unitary().get_distance_from(before_circuit.get_unitary())}"
    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates, f"{gate} is not in the Clifford+T set."

