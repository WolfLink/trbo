from bqskit import Circuit
from bqskit.passes import QuickPartitioner, UnfoldPass, ForEachBlockPass
from bqskit.ir.gates import BarrierPlaceholder
from time import sleep

from benchmarks import *

def run_small_benchmarks(data, max_synth_size=4):
    summary_dict = dict()
    for i in range(max_synth_size):
        benchmarks = data['inputs_by_qubit_count'][i]
        for benchmark in benchmarks:
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
            before_circuit, result_data = run_benchmark(before_circuit, trbo.workflows.sanitize_gateset())
            bench_data["sanitization"] = result_data
            pprint_ddict(bench_data, "sanitization")
            if before_circuit.num_params > 600:
                print(f"Skipping {bench_data} because it has {before_circuit.num_params} parameters")
                continue

            # run trbo
            _, result_data = run_benchmark(before_circuit, trbo.workflows.no_partitioning(64), repeats=10)
            bench_data["trbo-phase"] = result_data
            pprint_ddict(bench_data, "trbo-default")

            # run trbo-slow
            _, result_data = run_benchmark(before_circuit, trbo.workflows.no_partitioning(128), repeats=10)
            bench_data["trbo-slow"] = result_data
            pprint_ddict(bench_data, "trbo-slow")
            
            # run trbo-fast
            _, result_data = run_benchmark(before_circuit, trbo.workflows.no_partitioning(32, rz_disc=[RzAsT()]), repeats=10)
            bench_data["trbo-fast"] = result_data
            pprint_ddict(bench_data, "trbo-fast")

            summary_dict[benchmark] = bench_data
            archive_output(summary_dict, "summary-5-20-26.json")


if __name__ == "__main__":
    run_small_benchmarks(load_benchmarks())
