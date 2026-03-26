import trbo
from bqskit.compiler import Compiler
from bqskit import Circuit
from bqskit.ir.gates import ControlledGate, SqrtISwapGate, ISwapGate, TGate
from timeit import default_timer as timer
from trbo.utils import *
from trbo.clift import *
from trbo import TRbOPass
from bqskit.passes import UnfoldPass


before_circuit = Circuit(3)
before_circuit.append_gate(ControlledGate(SqrtISwapGate()), [2, 1, 0])


# This is a synthetiq derived circuit, consisting of 20 CNOTs and 10 T gates.
before_circuit = Circuit.from_file("synthetiq/data/output/62/csqrtiswap/12.060000-10-6-5-1127.qasm")

with Compiler() as compiler:
    start = timer()
    after_circuit = compiler.compile(before_circuit, trbo.workflows.sanitize_gateset())
    unopt_circuit = after_circuit
    s2 = timer()
    print(f"synthes took {s2 - start}s")
    after_circuit = compiler.compile(after_circuit, MultistartPass(trbo.workflows.no_partitioning(128)))
    s3 = timer()
    print(f"trbo took {s3 - s2}s")

print(before_circuit.gate_counts)
print("------------------")
print(unopt_circuit.gate_counts)
print("------------------")
print(after_circuit.gate_counts)
after_circuit.save("trbo_sqrtiswap.qasm")


