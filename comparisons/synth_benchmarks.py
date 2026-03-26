import os
import trbo
from trbo.clift import t_gates, rz_gates, clifford_gates
import json
from timeit import default_timer as timer
from datetime import datetime
from trbo.utils import *
from bqskit.compiler import Compiler
from bqskit import Circuit
from bqskit.passes import LEAPSynthesisPass
from synthetiq import SynthetiqPass
from benchmarks import *

def run_synth_benchmarks(data, max_synth_size=4):
    try:
        with open("synth_summary.json", "r") as f:
            summary_dict = json.load(f)
    except:
        summary_dict = dict()
    for i in range(1, 4):
        benchmarks = data['inputs_by_qubit_count'][i]
        for benchmark in benchmarks:
            if benchmark in summary_dict:
                print(f"Skipping {benchmark} because it was already completed.")
                continue

            bench_data = data['inputs'][benchmark]
            before_circuit = Circuit.from_file(bench_data["input"])

            print(f"Starting {benchmark} of size {i + 1}")

            bench_data['og_data']['time'] = 0
            pprint_ddict(bench_data, 'og_data')

            # run LEAP
            synth_circuit, result_data = run_benchmark(before_circuit, [LEAPSynthesisPass()] + trbo.workflows.sanitize_gateset())
            bench_data["leap"] = result_data
            pprint_ddict(bench_data, "leap")

            ## run trbo
            _, result_data = run_benchmark(synth_circuit, MultistartPass(trbo.workflows.default()))
            bench_data["trbo"] = result_data
            pprint_ddict(bench_data, "trbo")

            # run Synthetiq
            #_, result_data = run_benchmark(before_circuit, SynthetiqPass(PATHS["synthetiq"]))
            #bench_data["synthetiq"] = result_data
            #pprint_ddict(bench_data, "synthetiq")

            summary_dict[benchmark] = bench_data
            archive_output(summary_dict, "synth_summary.json")

if __name__ == "__main__":
    import resource
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, (1024 * 1024 * 1024 * 124, hard))
    run_synth_benchmarks(load_benchmarks())
