import pytest
from ntro.clift import clifford_gates, t_gates, rz_gates
from ntro import NumericalTReductionPass
from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit.passes import GroupSingleQuditGatePass, ForEachBlockPass, IfThenElsePass, WidthPredicate, ZXZXZDecomposition, UnfoldPass


def test_toffoli():
    toffoli_qasm_file = "synthesized_toffoli.qasm"

    before_circuit = Circuit.from_file(toffoli_qasm_file)
    with Compiler() as compiler:
        after_circuit = compiler.compile(before_circuit, [
            GroupSingleQuditGatePass(),
            ForEachBlockPass([
                IfThenElsePass(
                    WidthPredicate(2),
                    ZXZXZDecomposition(),
                    ),
                ]),
            UnfoldPass(),
            NumericalTReductionPass(),
            ])
        after_circuit.unfold_all()

    assert after_circuit.get_unitary().get_distance_from(before_circuit.get_unitary()) < 1e-5, f"Insufficient synthesis quality: {after_circuit.get_unitary().get_distance_from(before_circuit.get_unitary())}"
    for gate in after_circuit.gate_counts:
        assert gate in clifford_gates + t_gates, f"{gate} is not in the Clifford+T set."

    t_count = 0
    for gate in t_gates:
        if gate in after_circuit.gate_counts:
            t_count += after_circuit.gate_counts[gate]

    assert t_count <= 7, f"Failed to achieve ideal T-Count of 7 for toffoli (got {t_count} instead)."



from sys import argv
from timeit import default_timer as timer
from ntro.utils import *
from ntro.clift import GlobalPhaseGate
import numpy as np
#toffoli_qasm_file = "perfect_toffoli.qasm"
#toffoli_qasm_file = "bad_toffoli.qasm"
toffoli_qasm_file = "synthesized_toffoli.qasm"
before_circuit = Circuit.from_file(toffoli_qasm_file)
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
toffoli_u = np.array(toffoli_u, dtype="complex128") 
print(f"DEBUG DISTANCE: {before_circuit.get_unitary().get_distance_from(before_circuit.get_unitary())}")
print(f"DEBUG DIS2ANCE: {before_circuit.get_unitary().get_distance_from(toffoli_u)}")
#toffoli_u = Circuit.from_file(toffoli_qasm_file).get_unitary()
#target = before_circuit.get_unitary()
#before_circuit.set_params(np.zeros_like(before_circuit.params))
#before_circuit.instantiate(target)

def check_toffoli(circuit, result):
    try:
        new_circuit, data = result
        #assert circuit.get_unitary().get_distance_from(new_circuit.get_unitary()) < 1e-5
        rz = 0
        t = 0
        for gate in new_circuit.gate_counts:
            if gate in t_gates:
                t += new_circuit.gate_counts[gate]
            elif gate in rz_gates:
                rz += new_circuit.gate_counts[gate]
            elif gate not in clifford_gates:
                print(f"{new_circuit.gate_counts[gate]} of {gate} found.")
        #print(f"d: {new_circuit.get_unitary().get_distance_from(toffoli_u)}\trz: {rz}\tt: {t}")
        assert new_circuit.get_unitary().get_distance_from(toffoli_u) < 1e-6
        #print(new_circuit.gate_counts)
        #print(new_circuit.get_unitary().get_distance_from(toffoli_u))
        for gate in new_circuit.gate_counts:
            assert gate in clifford_gates + t_gates
        t_count = 0
        for gate in t_gates:
            if gate in new_circuit.gate_counts:
                t_count += new_circuit.gate_counts[gate]
        assert t_count == 7
        return True
    except:
        return False

if __name__ == "__main__":
    if len(argv) > 1:
        runs = int(argv[1])
    else:
        runs = 10
    start = timer()
    data = {"utry" : toffoli_u}
    with Compiler() as compiler:
        #before_circuit.set_params(np.zeros_like(before_circuit.params))
        after_circuit, data = compiler.compile(before_circuit, [
            SuccessBenchmarkPass([
                GroupSingleQuditGatePass(),
                ForEachBlockPass([
                    IfThenElsePass(
                        WidthPredicate(2),
                        ZXZXZDecomposition(),
                        ),
                    ]),
                UnfoldPass(),
                #AppendGatePass(GlobalPhaseGate()),
                #SetDataPass("utry", toffoli_u),
                NumericalTReductionPass(),
                RemoveGatePass(GlobalPhaseGate()),
                UnfoldPass(),
                ], condition=check_toffoli, runs=runs)
            ], request_data=True, data=data)
    print(f"{data['benchmark_success']['successes']}/{data['benchmark_success']['total']} = {(data['benchmark_success']['successes'] * 10000 // data['benchmark_success']['total']) / 100}%  in {timer() - start}s")
