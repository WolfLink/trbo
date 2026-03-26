import os
import trbo
from trbo.clift import t_gates, rz_gates, clifford_gates
import json
from timeit import default_timer as timer
from datetime import datetime
from trbo.utils import *
from bqskit.compiler import Compiler
from bqskit import Circuit
from trbo.gridsynth import GridsynthPass
from parse_quipper import *
from benchmarks import *

def run_quipper_benchmarks(data, max_synth_size=4):
    try:
        with open("quipper_summary.json", "r") as f:
            summary_dict = json.load(f)
    except:
        summary_dict = dict()
    for i in range(max_synth_size + 1, 50):
        benchmarks = data['inputs_by_qubit_count'][i]
        for benchmark in benchmarks:
            if benchmark in summary_dict and False:
                print(f"Skipping {benchmark} because it was already completed.")
                continue
            if "after" not in benchmark or "QFTAdd" not in benchmark:
                continue

            bench_data = data['inputs'][benchmark]
            before_circuit = parse_quipper_file(bench_data["input"])

            rz_count = 0
            for gate in rz_gates:
                if gate in before_circuit.gate_counts:
                    rz_count += before_circuit.gate_counts[gate]
            if rz_count < 1:
                #print(f"Skipping {benchmark} due to lack of Rz gates.")
                continue
            print(f"Starting {benchmark} of size {i + 1}")
            
            bench_data['og_data']['time'] = 0
            pprint_ddict(bench_data, 'og_data')

            # run trbo
            _, result_data = run_benchmark(before_circuit, trbo.workflows.slow(), repeats=1)
            bench_data["trbo"] = result_data
            pprint_ddict(bench_data, "trbo")

            summary_dict[benchmark] = bench_data
            archive_output(summary_dict, "quipper_summary.json")
            break

if __name__ == "__main__":
    PATHS = {"comparisons" : os.path.dirname(os.path.abspath(__file__))}
    PATHS["synthetiq"] = os.path.join(PATHS["comparisons"], "synthetiq")
    PATHS["inputs"] = os.path.join(PATHS["comparisons"], "quipper_circuits/optimizer")
    PATHS["outputs"] = os.path.join(PATHS["comparisons"], "quipper_circuits/outputs")
    PATHS["summary"] = os.path.join(PATHS["comparisons"], "quipper_circuits/output_summary.tsv")
    PATHS["database"] = os.path.join(PATHS["comparisons"], "quipper_benchmark_database.json")
    import resource
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, (1024 * 1024 * 1024 * 124, hard))
    run_quipper_benchmarks(load_benchmarks(PATHS))
