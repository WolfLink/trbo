from bqskit import compile
from bqskit.compiler import CompilationTask, Compiler
from bqskit.passes import SetModelPass

from bqskit.ir.gates.constant.cx import CNOTGate as CXG
from bqskit.ir.gates.constant.sx import SXGate as SXG
from bqskit.ir.gates.parameterized.rz import RZGate as RZG
from bqskit.ir.opt.cost import HilbertSchmidtCostGenerator, HilbertSchmidtResidualsGenerator
from bqskit.ir.gate import Gate
from bqskit.compiler.machine import MachineModel

from timeit import default_timer as timer
import numpy as np

import ntro
from ntro.ntro import *
from ntro.tcount import *
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

toffoli_qasm_file = "synthesized_toffoli.qasm"
toffoli_c = Circuit.from_file(toffoli_qasm_file)

@pytest.mark.parametrize("cost_function_generator", [
    MatrixDistanceCostGenerator(),
    RoundSmallestNCostGenerator(10),
    RoundSmallestNResidualsGenerator(10),
    SumCostGenerator(MatrixDistanceCostGenerator(), RoundSmallestNCostGenerator(10)),
    SumCostGenerator(HilbertSchmidtCostGenerator(), RoundSmallestNCostGenerator(10)),
    SumResidualsGenerator(HilbertSchmidtResidualsGenerator(), RoundSmallestNResidualsGenerator(10)),
    ])
def test_grad(cost_function_generator):
    report = check_grad(toffoli_c, toffoli_u, cost_function_generator)
    assert report < 1e-5

