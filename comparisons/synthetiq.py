# This code is meant to be a wrapper around "synthetiq"
# A clone of synthetiq including the built binary must be supplied
# You can clone synthetiq from GitHub at https://github.com/eth-sri/synthetiq
# I recommend building synthetiq inside Docker and copying the contents of synthetiq/bin into the host.


import os
import numpy as np
from subprocess import run, DEVNULL
import shutil
from timeit import default_timer as timer
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, ConstantUnitaryGate
from bqskit.compiler.basepass import BasePass
from trbo.clift import better_min_t_count_circuit
from filelock import FileLock
import json


def endian_reverse(unitary):
    ug = ConstantUnitaryGate(unitary)
    circuit = Circuit(ug.num_qudits)
    circuit.append_gate(ug, list(range(ug.num_qudits-1, -1, -1)))
    return circuit.get_unitary()

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


class TmpID:
    def __init__(self, dirpath):
        self.lockpath = os.path.join(dirpath, "id_reserve.lock")
        self.jsonpath = os.path.join(dirpath, "id_reserve.json")
        self.tmpid = None

    def __enter__(self):
        assert self.tmpid is None
        with FileLock(self.lockpath):
            try:
                with open(self.jsonpath, "r") as f:
                    data = json.load(f)
            except:
                data = []
            tmpid = 0
            tmpmax = 2
            while tmpid in data:
                tmpid = np.random.randint(tmpmax)
                tmpmax *= 2
            data.append(tmpid)
            self.tmpid = tmpid
            with open(self.jsonpath, "w") as f:
                json.dump(data, f)
        return tmpid
    
    def __exit__(self, exc_type, exc_value, traceback):
        with FileLock(self.lockpath):
            with open(self.jsonpath, "r") as f:
                data = json.load(f)
            data.remove(self.tmpid)
            with open(self.jsonpath, "w") as f:
                json.dump(data, f)
            self.tmpid = None

def select_best_synthetiq_output(output_path):
    best_circuit = None
    for item in os.listdir(output_path):
        fname, ext = os.path.splitext(item)
        if not ext in [".qasm"]:
            continue
        circuit = Circuit.from_file(os.path.join(output_path, item))
        if better_min_t_count_circuit(best_circuit, circuit):
            best_circuit = circuit
    return best_circuit

class SynthetiqPass(BasePass):
    def __init__(self, synthetiq_path, utry=None, mode=None, hardfail=False, use_threads=True, time_limit=None):
        self.utry = utry
        self.synthetiq_path = os.path.abspath(synthetiq_path)
        self.mode = mode
        self.hardfail = hardfail
        self.use_threads = use_threads
        self.time_limit = time_limit
       
    async def run(self, circuit, data={}):
        if "utry" in data:
            target = data["utry"]
        elif self.utry is not None:
            target = self.utry
        else:
            target = circuit.get_unitary()

        input_path = os.path.join(self.synthetiq_path, "data/input/trbo_tmp/")
        output_path = os.path.join(self.synthetiq_path, "data/output/trbo_tmp/")
        os.makedirs(input_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        with TmpID(output_path) as tmpid:
            os.makedirs(os.path.join(output_path, f"tmp_{tmpid}"), exist_ok=True)
            xtraargs = []
            if self.use_threads is True:
                xtraargs.append("-h")
                xtraargs.append(f"{os.cpu_count()}")
            elif self.use_threads is not None and self.use_threads is not False:
                xtraargs.append("-h")
                xtraargs.append(f"{self.use_threads}")
            if self.time_limit is None:
                xtraargs.append("-t")
                xtraargs.append("3600")

            if self.time_limit is not None:
                xtraargs.append("-t")
                xtraargs.append(f"{self.time_limit}")

            if self.mode in ["synth", "synthesis", None, "long", "longsynth"]:
                unitary_to_synthetiq_input(target, os.path.join(input_path, f"tmp_{tmpid}.txt"))
                run(["./bin/main", f"trbo_tmp/tmp_{tmpid}.txt"] + xtraargs, cwd=self.synthetiq_path, stdout=DEVNULL)
            elif self.mode in ["resynth", "resynthesis", "simplify", "simplification"]:
                circuit.save(os.path.join(input_path, f"tmp_{tmpid}.qasm"))
                run(["./bin/main_resynth", f"trbo_tmp/tmp_{tmpid}.qasm" + xtraargs], cwd=self.synthetiq_path, stdout=DEVNULL)
            else:
                raise KeyError(f"Unknown synthetiq mode {self.mode}")
            new_circuit = select_best_synthetiq_output(os.path.join(output_path, f"tmp_{tmpid}"))
            os.remove(os.path.join(input_path, f"tmp_{tmpid}.txt"))
            shutil.rmtree(os.path.join(output_path, f"tmp_{tmpid}"))
        if new_circuit is not None or self.hardfail:
            circuit.become(new_circuit)


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
