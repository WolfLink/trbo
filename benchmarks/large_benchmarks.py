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

def run_large_benchmarks(data, max_synth_size=4):
    try:
        with open("large_summary.json", "r") as f:
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
            archive_output(summary_dict, "large_summary.json")
            break

if __name__ == "__main__":
    run_large_benchmarks(load_benchmarks())
