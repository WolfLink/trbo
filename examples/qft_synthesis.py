from bqskit import compile
from bqskit.compiler import CompilationTask, Compiler
from bqskit.passes import SetModelPass

from bqskit.ir.gates.constant.cx import CNOTGate as CXG
from bqskit.ir.gates.constant.sx import SXGate as SXG
from bqskit.ir.gates.parameterized.rz import RZGate as RZG
from bqskit.ir.gate import Gate
from bqskit.compiler.machine import MachineModel

from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np

import ntro
from ntro.ntro import *
from ntro import gridsynth


def check_grad(circuit, target, cost_gen):
    cstr = cost_gen.gen_cost(circuit, target)
    point = np.random.rand(circuit.num_params) * np.pi * 2

    total_report = 0
    delta = 0.00001
    grad_num = []
    for i in range(circuit.num_params):
        new_point = point + np.array([0] * i + [delta] + [0] * (circuit.num_params - i - 1))
        grad_num.append((np.array(cstr(new_point)) - np.array(cstr(point))) / delta)
    
    grad_an = np.transpose(cstr.get_grad(point))
    total_report = np.sum(np.square(np.array(grad_an) - np.array(grad_num)))
    print(f"Gradient report returned {total_report}")


def qft(n):
    # this is the qft unitary generator code from qsearch
    root = np.e ** (2j * np.pi / n)
    return np.array(np.fromfunction(lambda x,y: root**(x*y), (n,n))) / np.sqrt(n)

# example: qft circuit
q = 3
U = qft(2**q)
U_S = np.array([[1, 0], [0, 1j]], dtype='complex128')
#q = 2
start = timer()
#model = MachineModel
gateset = set([CXG(), SXG(), RZG()])
synthesized_circuit = compile(U, max_synthesis_size = q, model=MachineModel(q, gate_set=gateset))
print(synthesized_circuit.gate_counts)
print(np.shape(synthesized_circuit.get_unitary()))
print(np.shape(U))
print(synthesized_circuit.get_unitary().get_distance_from(U))
print(f"Synthesis took {timer() - start}s")

# run gradient test
from ntro.tcount import RoundSmallestNCostGenerator, SumCostGenerator, RoundSmallestNResidualsGenerator, SumResidualsGenerator
from bqskit.ir.opt.cost import HilbertSchmidtCostGenerator, HilbertSchmidtResidualsGenerator
#check_grad(synthesized_circuit, U, SumResidualsGenerator(RoundSmallestNResidualsGenerator(12, np.pi * 0.5), HilbertSchmidtResidualsGenerator()))
#check_grad(synthesized_circuit, U, HilbertSchmidtCostGenerator())
#check_grad(synthesized_circuit, U, RoundSmallestNResidualsGenerator(len(synthesized_circuit.params) // 2, np.pi * 0.5))
#exit(0)

start = timer()
with Compiler() as compiler:
    synthesized_circuit = compiler.compile(synthesized_circuit, [
    SetModelPass(MachineModel(q, gate_set=gateset)),
    NumericalTReductionPass(full_loops=100, search_method="n_sum", backup=False, profiling_mode=True),
    #RzToT_ScanningBruteForcePass(),
    #gridsynth.GridsynthPass(gridsynth_binary="./gridsynth"),
    ])
synthesized_circuit.unfold_all()
print(f"Optimization took {timer() - start}s")

t_count = 0
rz_count = 0
for gate in synthesized_circuit.gate_set:
    print(f"{gate} Count:", synthesized_circuit.count(gate))
    if f"{gate}" in ["TGate","TdgGate"]:
        t_count += synthesized_circuit.count(gate)
    elif f"{gate}" in ["RZGate"]:
        rz_count += synthesized_circuit.count(gate)
print("")
print(f"Distance: {synthesized_circuit.get_unitary().get_distance_from(U, 1)}")
print(f"T-Count: {t_count}\tRz-Count: {rz_count}")

#from datetime import datetime
#synthesized_circuit.save(f"qasms/T{t_count}Z{rz_count}-{datetime.now().isoformat()}.qasm")

#from notify import notify
#notify(f"T-Count: {t_count}\tRz-Count: {rz_count}", title="QFT4 Finished Optimization")
