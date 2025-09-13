#from synthetiq import SynthetiqPass
from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit.passes import QuickPartitioner, UnfoldPass, ForEachBlockPass
from bqskit.ir.gates import BarrierPlaceholder
import qsearch
import os
import ntro
from ntro.clift import t_gates, rz_gates, clifford_gates
import json
from timeit import default_timer as timer
from datetime import datetime
from ntro.utils import *
from ntro import NumericalTReductionPass

import multiprocessing
multiprocessing.set_start_method("fork")


PATHS = {"comparisons" : os.path.dirname(os.path.abspath(__file__))}
PATHS["synthetiq"] = os.path.join(PATHS["comparisons"], "synthetiq")
PATHS["inputs"] = os.path.join(PATHS["comparisons"], "lbnlqasm/qasm")
PATHS["outputs"] = os.path.join(PATHS["comparisons"], "lbnlqasm/outputs")
PATHS["summary"] = os.path.join(PATHS["comparisons"], "lbnlqasm/output_summary.tsv")
PATHS["database"] = os.path.join(PATHS["comparisons"], "benchmark_database.json")



def synthetiq_workflow(partition_size=4):
    if partition_size is None:
        # no partitioning
        return [SynthetiqPass(PATHS["synthetiq"], use_threads=True)]
    else:
        return [QuickPartitioner(partition_size),
                ForEachBlockPass([
                    SynthetiqPass(PATHS["synthetiq"]),
                    ]),
                UnfoldPass(),
                ]

def judge_circuit(circuit, og=None, distlimit=8):
    circuit_results = dict()
    if og is not None and circuit.num_qudits <= distlimit:
        circuit_results["dist"] = og.get_unitary().get_distance_from(circuit.get_unitary())
    rz = 0
    clifford = 0
    t = 0
    unknown = 0
    counts_by_qubit_count = []
    for gate in circuit.gate_counts:
        while gate.num_qudits > len(counts_by_qubit_count):
            counts_by_qubit_count.append(0)
        counts_by_qubit_count[gate.num_qudits - 1] += circuit.gate_counts[gate]
        if gate in rz_gates:
            rz += circuit.gate_counts[gate]
        elif gate in clifford_gates:
            clifford += circuit.gate_counts[gate]
        elif gate in t_gates:
            t += circuit.gate_counts[gate]
        else:
            unknown += circuit.gate_counts[gate]
    circuit_results["rz"] = rz
    circuit_results["clifford" ] = clifford
    circuit_results["t"] = t
    circuit_results["unknown"] = unknown
    circuit_results["qubit_count"] = counts_by_qubit_count
    return circuit_results

def build_benchmark_database():
    benchmark_data = dict()
    benchmark_data["inputs"] = dict()
    benchmark_data["inputs_by_qubit_count"] = []
    for (root, dirs, files) in os.walk(PATHS["inputs"]):
        for file in files:
            try:
                assert os.path.splitext(file)[1] == ".qasm"
            except:
                print(f"{file} is not .qasm")
                continue
            input_path = os.path.join(root, file)
            print(f"Categorizing {input_path}...")
            output_path = os.path.join(PATHS["outputs"], os.path.relpath(PATHS["inputs"], os.path.splitext(input_path)[0]))
            data_entry = {"input" : input_path, "output" : output_path}
            data_entry["name"] = os.path.splitext(file)[0]
            circuit = Circuit.from_file(input_path)
            data_entry["num_qudits"] = circuit.num_qudits
            data_entry["og_data"] = judge_circuit(circuit, circuit)
            
            benchmark_data["inputs"][input_path] = data_entry
            by_count = benchmark_data["inputs_by_qubit_count"]
            while circuit.num_qudits > len(by_count):
                by_count.append([])
            by_count[circuit.num_qudits - 1].append(input_path)
            benchmark_data["inputs_by_qubit_count"] = by_count

    with open("benchmark_database.json", "w") as f:
        json.dump(benchmark_data, f)
    return benchmark_data

def log(text):
    with open(os.path.join(PATHS["comparisons"], "benchmarks.log"), "a") as f:
        f.write(datetime.now().isoformat() + " - " + text + "\n")
        print(text)

def load_benchmarks():
    try:
        with open(PATHS["database"], "r") as f:
            benchmark_data = json.load(f)
    except:
        build_benchmark_database()
        with open(PATHS["database"], "r") as f:
            benchmark_data = json.load(f)
    return benchmark_data


def pprint_ddict(ddict, opname):
    mq = 0
    for i in range(1, len(ddict[opname]['qubit_count'])):
        mq += ddict[opname]['qubit_count'][i]
    output = f"{ddict['name']} -> {opname} in {ddict[opname]['time']}s: {ddict[opname]['rz']} Rz, {ddict[opname]['t']} T, {mq} multiqubit"
    log(output)

def run_synthesis_benchmarks(data, max_synth_size=4):
    summary_dict = dict()
    for i in range(max_synth_size):
        benchmarks = data['inputs_by_qubit_count'][i]
        for benchmark in benchmarks:
            if "heisenberg_4" not in benchmark:
                print(f"Skipping {benchmark} because its not the one we are debugging.")
                continue
            bench_data = data['inputs'][benchmark]

            before_circuit = Circuit.from_file(bench_data["input"])
            invalid_gate = False
            for gate in before_circuit.gate_counts:
                if isinstance(gate, BarrierPlaceholder):
                    invalid_gate = True
                    break
            if invalid_gate:
                log(f"Skipping {benchmark} due to presence of barriers")
                continue
            
            bench_data['og_data']['time'] = 0
            pprint_ddict(bench_data, 'og_data')

            # prep circuit
            before_circuit = Circuit.from_file(bench_data["input"])
            start = timer()
            with Compiler() as compiler:
                before_circuit = compiler.compile(before_circuit, ntro.workflows.sanitize_gateset())
            end = timer()
            print(f"Circuit Sanitization took {end - start}s")

            # run ntro
            #before_circuit = Circuit.from_file(bench_data["input"])
            start = timer()
            with Compiler() as compiler:
                #after_circuit = compiler.compile(before_circuit, ntro.workflows.no_partitioning(32))
                after_circuit = compiler.compile(before_circuit, TimeoutPass(60 * 30, NumericalTReductionPass()))
            end = timer()
            bench_data["ntro"] = judge_circuit(after_circuit, before_circuit)
            bench_data["ntro"]["time"] = end - start
            pprint_ddict(bench_data, "ntro")

            ## run ntro-phase
            #before_circuit = Circuit.from_file(bench_data["input"])
            #start = timer()
            #with Compiler() as compiler:
            #    after_circuit = compiler.compile(before_circuit, ntro.workflows.no_partitioning(32, phase_correct=True))
            #end = timer()
            #bench_data["ntro-phase"] = judge_circuit(after_circuit, before_circuit)
            #bench_data["ntro-phase"]["time"] = end - start
            #pprint_ddict(bench_data, "ntro-phase")

            # run synthetiq
            #before_circuit = Circuit.from_file(bench_data["input"])
            #try:
            #    try:
            #        start = timer()
            #        with Compiler() as compiler:
            #            after_circuit = compiler.compile(before_circuit, [SynthetiqPass(PATHS["synthetiq"], use_threads=True, hardfail=True)])
            #        end = timer()
            #    except:
            #        log(f"Synthetiq failed to quickly solve {benchmark}.")
            #        continue
            #        start = timer()
            #        with Compiler() as compiler:
            #            after_circuit = compiler.compile(before_circuit, [SynthetiqPass(PATHS["synthetiq"], use_threads=True, hardfail=True, mode="longsynth")])
            #        end = timer()
            #    bench_data["synthetiq"] = judge_circuit(after_circuit, before_circuit)
            #    bench_data["synthetiq"]["time"] = end - start
            #    pprint_ddict(bench_data, "synthetiq")
            #except:
            #    log(f"Synthetiq failed to solve {benchmark} at all.")
            #    raise
            #    bench_data["synthetiq"] = "failure"
            
            summary_dict[benchmark] = bench_data
        with open("tmptest.json", "w") as f:
            json.dump(summary_dict, f)


if __name__ == "__main__":
    run_synthesis_benchmarks(load_benchmarks())
