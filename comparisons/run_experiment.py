import os
import traceback
from timeit import default_timer as timer

import numpy as np

from bqskit.compiler import CompilationTask, Compiler
from bqskit.passes import ForEachBlockPass, QuickPartitioner, LEAPSynthesisPass, ZXZXZDecomposition, GroupSingleQuditGatePass, UnfoldPass

from ntro import NumericalTReductionPass
from ntro.gridsynth import GridsynthPass
from ntro.utils import ComputeErrorThresholdPass, LogIntermediateGateCountsPass, UnwrapForEachPassDown
from ntro.clift import clifford_gates, t_gates, rz_gates

from parse_quipper import parse_quipper_file

def run_experiment(source_file, prefix_passes, pass_lists):
    source_file = os.path.normpath(source_file)
    filename = os.path.basename(source_file)
    name, ext = os.path.splitext(filename)
    data_dict = {"name" : name, "ext" : ext, "filepath" : source_file}

    og_circuit = None
    if ext in [".quipper", ""]:
        try:
            og_circuit = parse_quipper_file(source_file)
            data_dict["ext"] = ".quipper"
        except:
            data_dict["error"] = traceback.format_exc()
            return data_dict
    elif ext in [".qasm"]:
        data_dict["error"] = "I haven't implemented qasm parsing yet.  Just need to let bqskit handle it..."
        return data_dict
    else:
        data_dict["error"] = f"Unknown filetype: {ext}"
        return data_dict

    data_dict["og_gates"] = og_circuit.gate_counts
    data_dict["og_qasm"] = og_circuit.to("qasm")

    with Compiler() as compiler:
        og_circuit, pass_data = compiler.compile(og_circuit, prefix_passes, request_data=True)
    data_dict["num_partitions"] = len(list(og_circuit.operations_with_cycles()))
    
    opt_circuit = og_circuit
    results = []
    for pass_list in pass_lists:
        print("====================================================")
        pass_dict = dict()
        start = timer()
        intermediate_gate_counts = None
        try:
            with Compiler() as compiler:
                result_circuit, pass_data = compiler.compile(og_circuit, pass_list, request_data=True)
                if "ForEachBlockPass_data" in pass_data:
                    for item in pass_data["ForEachBlockPass_data"]:
                        for item_item in item:
                            if "subcircuit_data" in item_item:
                                print(item_item["subcircuit_data"])
                            if "intermediate_gate_counts" in item_item:
                                if intermediate_gate_counts is None:
                                    intermediate_gate_counts = item_item["intermediate_gate_counts"]
                                else:
                                    for key in item_item["intermediate_gate_counts"]:
                                        if key in intermediate_gate_counts:
                                            intermediate_gate_counts[key] += item_item["intermediate_gate_counts"][key]
                                        else:
                                            intermediate_gate_counts[key] = item_item["intermediate_gate_counts"][key]
        except:
            pass_dict["error"] = traceback.format_exc()
            results.append(pass_dict)
            continue
        opt_circuit = result_circuit
        stop = timer()
        result_circuit = opt_circuit.copy()
        result_circuit.unfold_all()
        pass_dict = {"gates" : result_circuit.gate_counts, "qasm" : result_circuit.to("qasm"), "time" : stop - start}
        if intermediate_gate_counts is not None:
            pass_dict["intermediate_gate_counts"] = intermediate_gate_counts
        if opt_circuit.dim > 0 and opt_circuit.dim <= 1024:
            try:
                pass_dict["distance"] = opt_circuit.get_unitary().get_distance_from(og_circuit.get_unitary())
                pass_dict["distance_type"] = "exact"
            except:
                print(f"Found an error with dim: {opt_circuit.dim}")
                raise
        elif "distances" in pass_data:
            pass_dict["distance"] = np.sum(pass_data["distance_list"])
            pass_dict["distance_type"] = "upper_bound"
        elif "error" in pass_data:
            pass_dict["distance"] = pass_data["error"]
            pass_dict["distance_type"] = "upper_bound"

        results.append(pass_dict)
    data_dict["results"] = results


    return data_dict

def gate_count_str(name, pdict):
    pass_rz = np.sum([pdict["gates"][gate] for gate in pdict["gates"] if gate in rz_gates])
    pass_t = np.sum([pdict["gates"][gate] for gate in pdict["gates"] if gate in t_gates])
    pass_cliff = np.sum([pdict["gates"][gate] for gate in pdict["gates"] if gate in clifford_gates])
    pass_other = np.sum([pdict["gates"][gate] for gate in pdict["gates"] if gate not in rz_gates + clifford_gates + t_gates + rz_gates])
    printstr = f"{name}:\tRz: {pass_rz}\tT: {pass_t}\tCliff: {pass_cliff}\tOther: {pass_other}"
    if "time" in pdict:
        printstr += f"({pdict['time']}s)"
    if "distance" in pdict:
        printstr += f"\tDist: {pdict['distance']}"
    return printstr

def pprint_ddict(ddict, title=None, pass_titles = None):
    print("="*10)
    if title is not None:
        print(title)
    if "error" in ddict:
        print(ddict["error"])
        print("="*10)
        return
    og_rz = np.sum([ddict["og_gates"][gate] for gate in ddict["og_gates"] if gate in rz_gates])
    og_t = np.sum([ddict["og_gates"][gate] for gate in ddict["og_gates"] if gate in t_gates])
    og_cliff = np.sum([ddict["og_gates"][gate] for gate in ddict["og_gates"] if gate in clifford_gates])
    og_other = np.sum([ddict["og_gates"][gate] for gate in ddict["og_gates"] if gate not in rz_gates + clifford_gates + t_gates + rz_gates])
    print(f"Start:\tRz: {og_rz}\tT: {og_t}\tCliff: {og_cliff}\tOther: {og_other} Partitions: {ddict['num_partitions']}")
    for i, pdict in enumerate(ddict["results"]):
        if "error" in pdict:
            print(pdict["error"])
            continue
        if pass_titles:
            name = pass_titles[i]
        else:
            name = f"Pass {i}"
        if "intermediate_gate_counts" in pdict:
            print(gate_count_str(name+"-I", {"gates" : pdict["intermediate_gate_counts"]}))
        print(gate_count_str(name, pdict))

    print("="*10)



if __name__ == "__main__":
    threshold = 1e-12
    prefix_passes = [
            QuickPartitioner(4),
            ]
    ntro_passes = [
            ComputeErrorThresholdPass(target_threshold=1e-10),
            ForEachBlockPass([
                UnwrapForEachPassDown(),
                #NumericalTReductionPass(full_loops=1, search_method="n_sum", backup=False, profiling_mode=False, success_threshold=threshold),
                #LogIntermediateGateCountsPass(),
                GridsynthPass(gridsynth_binary="../examples/gridsynth", threshold=threshold)
                ], calculate_error_bound=True),
            ]
    gridsynth_passes = [
            ComputeErrorThresholdPass(target_threshold=1e-10),
            ForEachBlockPass([
                UnwrapForEachPassDown(),
                GridsynthPass(gridsynth_binary="../examples/gridsynth", threshold=threshold)
                ], calculate_error_bound=True),
            ]
   
    #passes = [ntro_passes, gridsynth_passes]
    passes = [gridsynth_passes, gridsynth_passes]
    names = ["no opt", "NTRO"]
    #before = run_experiment("quipper_circuits/optimizer/QFT_and_Adders/QFT8_before", [gridsynth_passes, ntro_passes])
    #pprint_ddict(before, "Before", names)

    after = run_experiment("quipper_circuits/optimizer/QFT_and_Adders/QFT8_after", prefix_passes, [gridsynth_passes, ntro_passes])
    pprint_ddict(after, "After", names)
    print(f"Threshold: {threshold} or {np.sqrt(threshold)}")
