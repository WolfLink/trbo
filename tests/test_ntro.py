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
from ntro.utils import SuccessBenchmarkPass, MultistartPass

toffoli_qasm_file = "synthesized_toffoli.qasm"
before_circuit = Circuit.from_file(toffoli_qasm_file)

def check_toffoli(circuit, result):
    try:
        new_circuit, data = result
        assert circuit.get_unitary().get_distance_from(new_circuit.get_unitary()) < 1e-5
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
    with Compiler() as compiler:
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
                MultistartPass([
                    NumericalTReductionPass(),
                    UnfoldPass(),
                    ]),
                ], condition=check_toffoli, runs=runs)
            ], request_data=True)
    print(f"{data['benchmark_success']['successes']}/{data['benchmark_success']['total']} = {(data['benchmark_success']['successes'] * 10000 // data['benchmark_success']['total']) / 100}%  in {timer() - start}s")
