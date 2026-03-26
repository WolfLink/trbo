from bqskit import Circuit
from bqskit.passes import QuickPartitioner, UnfoldPass, ForEachBlockPass
from bqskit.ir.gates import BarrierPlaceholder
from time import sleep
import json
import os

from benchmarks import *

blacklist = ["shor_64.qasm", "heisenberg_32.qasm", "shor_32.qasm", "hhl_8.qasm", "qpe_12.qasm", "qpe_14.qasm", "qpe_16.qasm", "qpe_18.qasm"] + [f"qae_{i}.qasm" for i in range(23, 100)]

def run_large_benchmarks(data, max_synth_size=4):
    try:
        with open("large_summary.json", "r") as f:
            summary_dict = json.load(f)
    except:
        summary_dict = dict()
    for i in range(max_synth_size + 1, len(data['inputs_by_qubit_count'])):
        benchmarks = data['inputs_by_qubit_count'][i]
        for benchmark in benchmarks:
            if benchmark in summary_dict:
                print(f"Skipping {benchmark} because it was already completed.")
                continue
            if os.path.basename(benchmark) in blacklist:
                print(f"Skipping {benchmark} because it was blacklisted")
                continue

            print(f"Starting {benchmark}")

            bench_data = data['inputs'][benchmark]

            before_circuit = Circuit.from_file(bench_data["input"])
            
            bench_data['og_data']['time'] = 0
            pprint_ddict(bench_data, 'og_data')

            # prep circuit
            before_circuit, result_data = run_benchmark(before_circuit, ntro.workflows.sanitize_gateset())
            bench_data["sanitization"] = result_data
            pprint_ddict(bench_data, "sanitization")

            # run ntro
            _, result_data = run_benchmark(before_circuit, ntro.workflows.default(sanitize=False), repeats=1)
            bench_data["ntro"] = result_data
            pprint_ddict(bench_data, "ntro")

            # run ntro-phase
            _, result_data = run_benchmark(before_circuit, ntro.workflows.default(sanitize=False, phase_correct=True), repeats=1)
            bench_data["ntro-phase"] = result_data
            pprint_ddict(bench_data, "ntro-phase")

            summary_dict[benchmark] = bench_data
            archive_output(summary_dict, "large_summary.json")

if __name__ == "__main__":
    run_large_benchmarks(load_benchmarks())
