import os

import numpy as np

from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit.passes import *

from ntro.utils import *
from ntro import NumericalTReductionPass
from ntro.gridsynth import GridsynthPass

from synthetiq import run_synthetiq_benchmarks
from run_experiment import sort_inputs_by_qubits


directory = "./unitaries"
files = [os.path.join(directory,filename) for filename in os.listdir(directory)]
files = sort_inputs_by_qubits(files, max_qubits=3)

def run_unitary_experiment(source_file):
    synthesis_passes = [
            MultistartPass([
                QFASTDecompositionPass(),
                ForEachBlockPass([LEAPSynthesisPass()]),
                UnfoldPass(),
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
        circuit = compiler.compile(Circuit.from_unitary(unitary), synthesis_passes)
        print(f"success: {circuit.gate_counts} for {source_file}")


#run_synthetiq_benchmarks(files, "./results/unitary/synthetiq/", "./synthetiq")
#for _ in range(10):
#    run_benchmarks(files=files, thresholds=[1e-5], block_sizes=[6], path="./results/unitary/")

[run_unitary_experiment(file) for file in files]
