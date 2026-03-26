from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit.passes import QuickPartitioner, UnfoldPass, ForEachBlockPass, LEAPSynthesisPass
from bqskit.ir.gates import BarrierPlaceholder, ConstantUnitaryGate

from benchmarks import *
from synthetiq import SynthetiqPass
from trbo.clift import RzAsT

def run_small_benchmarks(data, max_synth_size=4):
    archive_file = "synth-5-24-26.json"
    summary_dict = dict()
    for i in range(max_synth_size):
        benchmarks = data['inputs_by_qubit_count'][i]
        for benchmark in benchmarks:
            bench_data = data['inputs'][benchmark]
            if benchmark_in_archive(benchmark, archive_file):
                print(f"Skipping {benchmark} due to earlier data")
                continue

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
            target = before_circuit.get_unitary()
            before_circuit = Circuit(before_circuit.num_qudits)
            before_circuit.append_gate(ConstantUnitaryGate(target), list(range(before_circuit.num_qudits)))

            # synth circuit
            start = timer()
            with Compiler() as compiler:
                before_circuit = compiler.compile(before_circuit, [LEAPSynthesisPass()] + trbo.workflows.sanitize_gateset())
            end = timer()
            bench_data["leap"] = judge_circuit(before_circuit, Circuit.from_file(bench_data["input"]))
            bench_data["leap"]["time"] = end - start
            pprint_ddict(bench_data, "leap")

            # run trbo
            start = timer()
            with Compiler() as compiler:
                after_circuit = compiler.compile(before_circuit, MultistartPass(trbo.workflows.no_partitioning(64)))
            end = timer()
            bench_data["leap+trbo"] = judge_circuit(after_circuit, before_circuit)
            bench_data["leap+trbo"]["time"] = end - start
            pprint_ddict(bench_data, "leap+trbo")


            # run trbo-slow
            start = timer()
            with Compiler() as compiler:
                after_circuit = compiler.compile(before_circuit, MultistartPass(trbo.workflows.no_partitioning(128)))
            end = timer()
            bench_data["leap+trbo-slow"] = judge_circuit(after_circuit, before_circuit)
            bench_data["leap+trbo-slow"]["time"] = end - start
            pprint_ddict(bench_data, "leap+trbo-slow")
            
            # run trbo-fast
            start = timer()
            with Compiler() as compiler:
                after_circuit = compiler.compile(before_circuit, MultistartPass(trbo.workflows.no_partitioning(32, rz_disc=[RzAsT()])))
            end = timer()
            bench_data["leap+trbo-fast"] = judge_circuit(after_circuit, before_circuit)
            bench_data["leap+trbo-fast"]["time"] = end - start
            pprint_ddict(bench_data, "leap+trbo-fast")

            # run synthetiq
           # before_circuit = Circuit.from_file(bench_data["input"])
           # try:
           #     start = timer()
           #     with Compiler() as compiler:
           #         after_circuit = compiler.compile(before_circuit, [SynthetiqPass(PATHS["synthetiq"], use_threads=True, hardfail=True, time_limit=60*60)])
           #     end = timer()
           #     bench_data["synthetiq"] = judge_circuit(after_circuit, before_circuit)
           #     bench_data["synthetiq"]["time"] = end - start
           #     pprint_ddict(bench_data, "synthetiq")
           # except:
           #     log(f"Synthetiq failed to solve {benchmark} at all.")
           #     bench_data["synthetiq"] = "failure"
            
            summary_dict[benchmark] = bench_data
            archive_output(summary_dict, archive_file)


if __name__ == "__main__":
    PATHS = {"comparisons" : os.path.dirname(os.path.abspath(__file__))}
    PATHS["synthetiq"] = os.path.join(PATHS["comparisons"], "synthetiq")
    PATHS["inputs"] = os.path.join(PATHS["comparisons"], "synth_benchmarks")
    PATHS["outputs"] = os.path.join(PATHS["comparisons"], "synth_benchmarks/outputs")
    PATHS["summary"] = os.path.join(PATHS["comparisons"], "synth_benchmarks/output_summary.tsv")
    PATHS["database"] = os.path.join(PATHS["comparisons"], "synth_database.json")
    run_small_benchmarks(load_benchmarks(PATHS))
