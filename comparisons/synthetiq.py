# This code is meant to be a wrapper around "synthetiq"
# A clone of synthetiq including the built binary must be supplied
# You can clone synthetiq from GitHub at https://github.com/eth-sri/synthetiq


import os
import numpy as np
from subprocess import run, DEVNULL
import shutil
from timeit import default_timer as timer
from qsearch.utils import endian_reverse
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.compiler.basepass import BasePass
from ntro.clift import clifford_gates, t_gates, rz_gates

def unitary_to_synthetiq_input(unitary, input_path):
    unitary = endian_reverse(unitary)
    with open(input_path, "w") as f:
        f.write("matrix\n")
        num_qubits = int(np.log(np.shape(unitary)[0]) / np.log(2))
        f.write(f"{num_qubits}\n")
        for row in unitary:
            for val in row:
                f.write(f"({np.real(val)},{np.imag(val)}) ")
            f.write("\n")
        for row in unitary:
            for val in row:
                f.write(f"1 ")
            f.write("\n")

def select_better_circuit(A, B):
    if A is None:
        return B
    if B is None:
        return A
    gcA = A.gate_counts
    gcB = B.gate_counts

    tcA = 0
    tcB = 0
    ccA = 0
    ccB = 0
    xcA = 0
    xcB = 0
    icA = 0
    icB = 0
    for gate in gcA:
        if gate in t_gates:
            tcA += gcA[gate]
        elif gate in clifford_gates:
            ccA += gcA[gate]
        else:
            icA += gcA[gate]
        if gate in [CNOTGate()]:
            xcA += gcA[gate]
    for gate in gcB:
        if gate in t_gates:
            tcB += gcB[gate]
        elif gate in clifford_gates:
            ccB += gcB[gate]
        else:
            icB += gcB[gate]
        if gate in [CNOTGate()]:
            xcB += gcB[gate]

    if icA > icB:
        return B
    elif icB > icA:
        return A

    if tcA > tcB:
        return B
    elif tcB > tcA:
        return A

    if xcA > xcB:
        return B
    elif xcB > xcA:
        return A

    if ccA > ccB:
        return B
    elif ccB > ccA:
        return A

    return A

def select_best_synthetiq_output(output_path):
    best_circuit = None
    for item in os.listdir(output_path):
        fname, ext = os.path.splitext(item)
        if not ext in [".qasm"]:
            continue
        circuit = Circuit.from_file(os.path.join(output_path, item))
        best_circuit = select_better_circuit(best_circuit, circuit)
    return best_circuit

def synthetiq(unitary, synthetiq_path):
    input_path = os.path.join(synthetiq_path, "data/input/ntro_tmp/")
    output_path = os.path.join(synthetiq_path, "data/output/ntro_tmp/")
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    unitary_to_synthetiq_input(unitary, os.path.join(input_path, "tmp.txt"))
    run(["./bin/main", "ntro_tmp/tmp.txt"], cwd=synthetiq_path, stdout=DEVNULL)
    circuit = select_best_synthetiq_output(os.path.join(output_path, "tmp"))
    shutil.rmtree(input_path)
    shutil.rmtree(output_path)
    return circuit


class SynthetiqPass(BasePass):
    def __init__(self, synthetiq_path, utry=None):
        self.utry = utry
        self.synthetiq_path = synthetiq_path
       
    async def run(self, circuit, data={}):
        if "utry" in data:
            target = data["utry"]
        elif self.utry is not None:
            target = self.utry
        else:
            target = circuit.get_unitary()

        circuit = synthetiq(target, self.synthetiq_path)
        return circuit


def run_synthetiq_benchmarks(benchmarks, output_dir, synthetiq_path):
    synthetiq_tsv = os.path.join(output_dir, "synthetiq.tsv")
    if not os.path.exists(synthetiq_tsv):
        os.makedirs(output_dir, exist_ok=True)
        with open(synthetiq_tsv, "w") as f:
            line = "name"
            line += f"\ttime"
            line += f"\tdist"
            line += f"\trz"
            line += f"\tt"
            line += f"\tcliff"
            line += f"\tother"
            f.write(line + "\n")
    for benchmark in benchmarks:
        benchmark = os.path.normpath(benchmark)
        filename, ext = os.path.splitext(os.path.basename(benchmark))
        circuit = None
        time = None
        print(f"Synthetiq starting: {filename}")
        target = None
        if ext in [".qasm"]:
            og_circuit = circuit.from_file(benchmark)
            target = og_circuit.get_unitary()
            start = timer()
            circuit = synthetiq(og_circuit.get_unitary(), synthetiq_path)
            time = timer() - start
        elif ext in [".npy"]:
            unitary = np.load(benchmark)
            target = unitary
            start = timer()
            circuit = synthetiq(unitary, synthetiq_path)
            time = timer() - start
        if circuit is None:
            with open(synthetiq_tsv, "a") as f:
                line = f"{filename}"
                line += "\t{time}"
                line += "\t-1"
                line += "\t-1"
                line += "\t-1"
                line += "\t-1"
                line += "\t-1"
                line += "\tSynthetiq failure"
                f.write(line + "\n")
                continue

        rz_count = 0
        t_count = 0
        cliff_count = 0
        other_count = 0
        gates = circuit.gate_counts
        for gate in gates:
            if gate in rz_gates:
                rz_count += gates[gate]
            elif gate in t_gates:
                t_count += gates[gate]
            elif gate in clifford_gates:
                cliff_count += gates[gate]
            else:
                other_count += gates[gate]

        print(f"{gates}\t{t_count}")

        dist = circuit.get_unitary().get_distance_from(target)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, filename + ".qasm"), "w") as f:
            f.write(circuit.to("qasm"))
        
        with open(synthetiq_tsv, "a") as f:
            line = f"{filename}"
            line += f"\t{time}"
            line += f"\t{dist}"
            line += f"\t{rz_count}"
            line += f"\t{t_count}"
            line += f"\t{cliff_count}"
            line += f"\t{other_count}"
            f.write(line + "\n")

        print(f"Synthetiq completed: {filename} in {time}s result: {dist}\t{gates}")
