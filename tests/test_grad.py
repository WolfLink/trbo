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
import pytest


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
    return total_report

def qft(n):
    # this is the qft unitary generator code from qsearch
    root = np.e ** (2j * np.pi / n)
    return np.array(np.fromfunction(lambda x,y: root**(x*y), (n,n))) / np.sqrt(n)

# example: qft circuit
q = 2
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



from ntro.tcount import MatrixDistanceCost, MatrixDistanceCostGenerator
def test_MatrixDistanceCost_grad():
    report = check_grad(synthesized_circuit, U, MatrixDistanceCostGenerator())
    print(report)
    assert report < 1e-5
