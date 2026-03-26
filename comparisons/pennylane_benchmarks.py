import os
import trbo
from trbo.clift import t_gates, rz_gates, clifford_gates
import json
from timeit import default_timer as timer
from datetime import datetime
from trbo.utils import *
from bqskit.compiler import Compiler
from bqskit.ir.gates import BarrierPlaceholder
from bqskit import Circuit
from bqskit.passes import LEAPSynthesisPass
from trbo.gridsynth import GridsynthPass
from parse_quipper import *



PATHS = {"comparisons" : os.path.dirname(os.path.abspath(__file__))}
PATHS["synthetiq"] = os.path.join(PATHS["comparisons"], "synthetiq")
PATHS["inputs"] = os.path.join(PATHS["comparisons"], "pennylane/inputs")
PATHS["outputs"] = os.path.join(PATHS["comparisons"], "pennylane/outputs")
PATHS["summary"] = os.path.join(PATHS["comparisons"], "pennylane/output_summary.tsv")
PATHS["database"] = os.path.join(PATHS["comparisons"], "pennylane_benchmark_database.json")



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
    circuit_results["clifford"] = clifford
    circuit_results["t"] = t
    circuit_results["unknown"] = unknown
    circuit_results["qubit_count"] = counts_by_qubit_count

    # Use Gridsynth if the circuit is clifford+t+rz
    if unknown == 0 and rz > 0:
        start = timer()
        with Compiler() as compiler:
            gridsynth_circuit = compiler.compile(circuit, GridsynthPass(1e-3))
        gridsynth_circuit.unfold_all()
        end = timer()
        circuit_results["gridsynth_time"] = end - start
        if circuit.num_qudits <= distlimit:
            if og is not None:
                circuit_results["gridsynth_dist"] = og.get_unitary().get_distance_from(gridsynth_circuit.get_unitary())
            else:
                circuit_results["gridsynth_dist"] = circuit.get_unitary().get_distance_from(gridsynth_circuit.get_unitary())
        gst = 0
        gsc = 0
        gse = 0
        for gate in gridsynth_circuit.gate_counts:
            if gate in clifford_gates:
                gsc += gridsynth_circuit.gate_counts[gate]
            elif gate in t_gates:
                gst += gridsynth_circuit.gate_counts[gate]
            else:
                gse += gridsynth_circuit.gate_counts[gate]
                print(f"Gridsynth circuit had {gridsynth_circuit.gate_counts[gate]} of unexpected gate {gate}")
                raise RuntimeError
        circuit_results["gridsynth_t"] = gst
        circuit_results["gridsynth_clifford"] = gsc
        if gse > 0:
            circuit_results["gridsynth_unknown"] = gse
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
            circuit.remove_all_measurements()
            data_entry["num_qudits"] = circuit.num_qudits
            data_entry["og_data"] = judge_circuit(circuit, circuit)
            
            benchmark_data["inputs"][input_path] = data_entry
            by_count = benchmark_data["inputs_by_qubit_count"]
            while circuit.num_qudits > len(by_count):
                by_count.append([])
            by_count[circuit.num_qudits - 1].append(input_path)
            benchmark_data["inputs_by_qubit_count"] = by_count

    with open(PATHS["database"], "w") as f:
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

def archive_output(data_dict, filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = dict()
    for key in data_dict:
        if key in data:
            data[key].update(data_dict[key])
        else:
            data[key] = data_dict[key]
    with open(filepath, "w") as f:
        json.dump(data, f)

def benchmark_in_archive(benchmark, filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return False
    return benchmark in data


def pprint_ddict(ddict, opname):
    mq = 0
    for i in range(1, len(ddict[opname]['qubit_count'])):
        mq += ddict[opname]['qubit_count'][i]
    output = f"{ddict['name']} -> {opname} in {ddict[opname]['time']}s: {ddict[opname]['rz']} Rz, {ddict[opname]['t']} T, {mq} multiqubit"
    log(output)

def run_benchmark(before_circuit, workflow, repeats=1):
    if repeats == 1:
        start = timer()
        with Compiler() as compiler:
            after_circuit = compiler.compile(before_circuit, workflow)
        end = timer()
        data = judge_circuit(after_circuit, before_circuit)
        data["time"] = end - start
        return after_circuit, data
    elif repeats > 1:
        results = [run_benchmark(before_circuit, workflow) for _ in range(repeats)]
        # find the best result
        best_rz = None
        best_t = None
        best_result = None
        best_circuit = None
        tottime = 0
        for circuit, result in results:
            tottime += result["time"]
            if best_rz is None or result["rz"] < best_rz:
                best_rz = result["rz"]
                best_t = result["t"]
                best_circuit = circuit
                best_result = result
            elif result["rz"] == best_rz and result["t"] < best_t:
                best_t = result["t"]
                best_circuit = circuit
                best_result = result

        # find the success counts
        rz_successes = 0
        t_successes = 0
        for _, result in results:
            if result["rz"] == best_rz:
                rz_successes += 1
                if result["t"] == best_t:
                    t_successes += 1
        # assemble the final report
        best_result["rz_successes"] = rz_successes
        best_result["t_successes"] = t_successes
        best_result["total_attempts"] = repeats
        best_result["avg_time"] = tottime / repeats
        return best_circuit, best_result
    else:
        raise ValueError(f"Repeats should be an integer >= 1. Got {repeats} instead.")

class BenchmarkingMultistartPass(BasePass):
    def __init__(self, workflow, multistarts=10):
        self.workflow = Workflow(workflow)
        self.multistarts = multistarts

    async def run(self, circuit, data={}):
        best_circuit = None
        best_data = None
        best_t = None
        best_rz = None
        futures = get_runtime().map(trbo.utils._run_workflow_on_circuit, [getrandbits(32) for _ in range(self.multistarts)], workflow=self.workflow, circuit=circuit, data=data)
        
        attempts = FutureQueue(futures, self.multistarts)
        results = []
        async for i, result in attempts:
            results.append(result)
            new_circuit, new_data = result
            t_count = sum([new_circuit.gate_counts[gate] for gate in t_gates if gate in new_circuit.gate_counts])
            rz_count = sum([new_circuit.gate_counts[gate] for gate in rz_gates if gate in new_circuit.gate_counts])
            if best_circuit is None or rz_count < best_rz or (rz_count == best_rz and t_count < best_t):
                best_circuit = new_circuit
                best_data = new_data
                best_rz = rz_count
                best_t = t_count

        rz_successes = 0
        t_successes = 0
        for result in results:
            new_circuit, new_data = result
            t_count = sum([new_circuit.gate_counts[gate] for gate in t_gates if gate in new_circuit.gate_counts])
            rz_count = sum([new_circuit.gate_counts[gate] for gate in rz_gates if gate in new_circuit.gate_counts])
            if rz_count == best_rz:
                rz_successes += 1
                if t_count == best_t:
                    t_successes += 1

        circuit.become(best_circuit)
        data.update(best_data)
        data["rz_successes"] = rz_successes
        data["t_successes"] = t_successes
        data["total_attempts"] = self.multistarts

blacklist = ["barenco_tof_3"]
def run_pennylane_benchmarks(data, max_synth_size=4):
    summary_dict = dict()
    archive_file = "pennylane_results.json"
    #for i in range(len(data['inputs_by_qubit_count'])):
    for i in [4, 5, 6]:
        benchmarks = data['inputs_by_qubit_count'][i]
        for benchmark in benchmarks:
            print(f"{benchmark} is of size {i}")
            skip = False
            for keyword in blacklist:
                if keyword in benchmark:
                    print(f"Skipping {benchmark} because it is blacklisted")
                    skip = True
                    break
            if skip:
                continue
            if benchmark_in_archive(benchmark, archive_file):
                print(f"Skipping {benchmark} due to earlier data")
                continue
            bench_data = data['inputs'][benchmark]

            before_circuit = Circuit.from_file(bench_data["input"])
            before_circuit.remove_all_measurements()
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
            before_circuit.remove_all_measurements()

            before_circuit, result_data = run_benchmark(before_circuit, [LEAPSynthesisPass()])
            bench_data["synthesis"] = result_data
            pprint_ddict(bench_data, "synthesis")

            before_circuit, result_data = run_benchmark(before_circuit, trbo.workflows.sanitize_gateset(max_synth_size))
            bench_data["sanitization"] = result_data
            pprint_ddict(bench_data, "sanitization")
            if before_circuit.num_params > 600:
                print(f"Skipping {bench_data} because it has {before_circuit.num_params} parameters")
                continue

            # run trbo-slow
            _, result_data = run_benchmark(before_circuit, trbo.workflows.slow(sanitize=False), repeats=10)
            bench_data["trbo-slow"] = result_data
            pprint_ddict(bench_data, "trbo-slow")
            

            summary_dict[benchmark] = bench_data
            archive_output(summary_dict, archive_file)

if __name__ == "__main__":
    import resource
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, (1024 * 1024 * 1024 * 64, hard))
    run_pennylane_benchmarks(load_benchmarks())
