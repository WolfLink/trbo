import os
from timeit import default_timer as timer

import numpy as np

from bqskit import Circuit, MachineModel
from bqskit.compiler import Compiler
from bqskit.passes import *

from ntro.utils import *
from ntro import NumericalTReductionPass
from ntro.gridsynth import GridsynthPass

from synthetiq import run_synthetiq_benchmarks
from run_experiment import sort_inputs_by_qubits


directory = "./unitaries"
files = [os.path.join(directory,filename) for filename in os.listdir(directory)]
#files = [file for file in files if "qft" in file]
files = sort_inputs_by_qubits(files)

def run_unitary_experiment(source_file):
    model = MachineModel(num_qudits=6, coupling_graph=[(i, i+1) for i in range(5)])
    synthesis_passes = [
            MultistartPass(multistarts=2, workflow=[
                IfThenElsePass(WidthPredicate(5),
                               NOOPPass(),
                               SetModelPass(model),
                               ),
                LEAPSynthesisPass(),
                GroupSingleQuditGatePass(),
                ForEachBlockPass([
                    IfThenElsePass(
                        WidthPredicate(2),
                        ZXZXZDecomposition(),
                        ),
                    ]),
                UnfoldPass(),
                NumericalTReductionPass(),
                UnfoldPass(),
                ]),
            LogIntermediateGateCountsPass(), 
            GridsynthPass(gridsynth_binary="../examples/gridsynth"),
            UnfoldPass(),
            ]

    unitary = np.load(source_file)
    with Compiler() as compiler:
        start = timer()
        circuit = compiler.compile(Circuit.from_unitary(unitary), synthesis_passes)
        if circuit.num_qudits > 8:
            print(f"skipped {source_file} because it has {circuit.num_qudits} qubits.")
            return
        time = timer() - start
        print(f"success: {circuit.gate_counts} for {source_file} after {time}s")


#run_synthetiq_benchmarks(files, "./results/unitary/synthetiq/", "./synthetiq")
#for _ in range(10):
#    run_benchmarks(files=files, thresholds=[1e-5], block_sizes=[6], path="./results/unitary/")

[run_unitary_experiment(file) for file in files]
