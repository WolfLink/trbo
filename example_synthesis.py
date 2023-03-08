from bqskit import compile
from bqskit.compiler import CompilationTask, Compiler

from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np

import ntr
from ntr import *

def qft(n):
    # this is the qft unitary generator code from qsearch
    root = np.e ** (2j * np.pi / n)
    return np.array(np.fromfunction(lambda x,y: root**(x*y), (n,n))) / np.sqrt(n)

# example: qft circuit
start = timer()
q = 3
synthesized_circuit = compile(qft(2**q), max_synthesis_size = q)
print(synthesized_circuit.gate_counts)
print(f"Synthesis took {timer() - start}s")


task = CompilationTask(synthesized_circuit, [
    NumericalTReductionPass()
    ])

with Compiler() as compiler:
    synthesized_circuit = compiler.compile(task)

for gate in synthesized_circuit.gate_set:
    print(f"{gate} Count:", synthesized_circuit.count(gate))


