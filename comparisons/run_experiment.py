import os
import traceback
import json
from timeit import default_timer as timer

import numpy as np

from bqskit.compiler import CompilationTask, Compiler
from bqskit.passes import ForEachBlockPass, QuickPartitioner, LEAPSynthesisPass, ZXZXZDecomposition, GroupSingleQuditGatePass, UnfoldPass, NOOPPass

from ntro import NumericalTReductionPass
from ntro.gridsynth import GridsynthPass
#from ntro.utils import ComputeErrorThresholdPass, LogIntermediateGateCountsPass, UnwrapForEachPassDown
from ntro.utils import *
from ntro.clift import clifford_gates, t_gates, rz_gates

from parse_quipper import parse_quipper_file

from data_collecting import log_ddict_to_tsv

def run_experiment(source_file, prefix_passes, pass_lists, threshold=None, block_size=None):
    source_file = os.path.normpath(source_file)
    filename = os.path.basename(source_file)
    name, ext = os.path.splitext(filename)
    data_dict = {"name" : name, "ext" : ext, "filepath" : source_file}


    print_title(f"{name} - parse ({block_size}, {threshold})")
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

    # validate the circuits gates
    # we want only circuits in the Clifford + T + Rz gate set
    # any specifically with at least one Rz gate for us to optimize
    found_rz = False
    found_invalid = False
    for gate in og_circuit.gate_counts:
        if gate in rz_gates:
            found_rz = True
        elif gate not in clifford_gates + t_gates:
            found_invalid = True
    
    if found_invalid:
        print(f"Skipping circuit {name} because it contains a gate that isn't Clifford+T+Rz")
        return None
    elif not found_rz:
        print(f"Skipping circuit {name} because it is alreay in Clifford+T (nothing to optimize!)")

    data_dict["og_gates"] = og_circuit.gate_counts
    data_dict["og_qasm"] = og_circuit.to("qasm")

    print_title(f"{name} - prefix ({block_size}, {threshold})")
    with Compiler() as compiler:
        og_circuit, pass_data = compiler.compile(og_circuit, prefix_passes, request_data=True)
    data_dict["num_partitions"] = len(list(og_circuit.operations_with_cycles()))
    
    opt_circuit = og_circuit
    results = []
    for i, pass_list in enumerate(pass_lists):
        if i == 0:
            pass_name = "gridsynth"
        elif i == 1:
            pass_name = "ntro"
        else:
            pass_name = "UNKNOWN"
        print_title(f"{name} - {pass_name} ({block_size}, {threshold})")

        pass_dict = dict()
        start = timer()
        intermediate_gate_counts = None
        gridsynth_stats = None
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
                            if "gridsynth_stats" in item_item:
                                if gridsynth_stats is None:
                                    gridsynth_stats = {"rz" : [], "t" : [], "e" : [], "d" : [], "thresh" : []}
                                stats = item_item["gridsynth_stats"]
                                gridsynth_stats["rz"].append(stats["rz"])
                                gridsynth_stats["t"].append(stats["t"])
                                gridsynth_stats["e"].append(stats["e"])
                                gridsynth_stats["d"].append(stats["d"])
                                gridsynth_stats["thresh"].append(stats["thresh"])


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
        if gridsynth_stats is not None:
            pass_dict["gridsynth_stats"] = gridsynth_stats
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
    if threshold is not None:
        data_dict["threshold"] = threshold
    if block_size is not None:
        data_dict["block_size"] = block_size
    try:
        with Compiler() as compiler:
            noop_passes = [ForEachBlockPass([NOOPPass()], calculate_error_bound=True)]
            _, pass_data = compiler.compile(og_circuit, noop_passes, request_data=True)
            if "error" in pass_data:
                data_dict["control_dist"] = pass_data["error"]
    except:
        pass


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
        printstr += f"\tDist: {pdict['distance']} ({pdict['distance_type']})"
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
        if "gridsynth_stats" in pdict and False:
            rz = np.array(pdict["gridsynth_stats"]["rz"])
            t = np.array(pdict["gridsynth_stats"]["t"])
            d = np.array(pdict["gridsynth_stats"]["d"])
            e = np.array(pdict["gridsynth_stats"]["e"])
            thresh = np.array(pdict["gridsynth_stats"]["thresh"])
            print(f"{name}-gridsynth Rz: {np.min(rz)}<{np.mean(rz):.3}<{np.max(rz)} ({np.sum(rz)})\tT: {np.min(t)}<{np.mean(t):.3}<{np.max(t)} ({np.sum(t)})\td: {np.min(d):.3}<{np.mean(d):.3}<{np.max(d):.3}\tthresh: {np.min(thresh):.3}<{np.mean(thresh):.3}<{np.max(thresh):.3}\te: {np.min(e):.3}<{np.mean(e):.3}<{np.max(e):.3}")
        print(gate_count_str(name, pdict))

    print("="*10)

            


def qasm_from_ddict(ddict, pass_titles, base_path=None):
    if base_path is None:
        base_path = "./"

    for i, pdict in enumerate(ddict["results"]):
        path = os.path.join(base_path, f"{pass_titles[i]}.qasm")
        print(f"writing to {path}")
        with open(path, "w+") as f:
            f.write(pdict["qasm"])

def print_title(text, length=80):
    titlestr = " " + text + " "
    before_len = (length - len(titlestr)) // 2
    after_len = (length - len(titlestr) - before_len)
    print("=" * before_len + titlestr + "=" * after_len)



#if __name__ == "__main__":
#for threshold in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
def run_benchmarks(files=[], thresholds=[1e-5,1e-3,1e-7, 1e-4, 1e-6, 1e-8], block_sizes=[4, 3, 5, 6, 7], path=None):
    checkpoint = None
    try:
        with open("checkpoint.json", "r") as f:
            checkpoint = json.load(f)
        assert checkpoint["files"]["data"] == files[checkpoint["files"]["index"]]
        assert checkpoint["thresholds"]["data"] == thresholds[checkpoint["thresholds"]["index"]]
        assert checkpoint["block_sizes"]["data"] == block_sizes[checkpoint["block_sizes"]["index"]]
    except:
        checkpoint = None

    for indb, block_size in enumerate(block_sizes):
        for indt, threshold in enumerate(thresholds):
            for indf, file in enumerate(files):
                if checkpoint is not None:
                    if indb < checkpoint["block_sizes"]["index"]:
                        continue
                    elif indt < checkpoint["thresholds"]["index"]:
                        continue
                    elif indf < checkpoint["files"]["index"]:
                        continue
                    elif indb == checkpoint["block_sizes"]["index"] and indt == checkpoint["thresholds"]["index"] and indf == checkpoint["files"]["index"]:
                        checkpoint = None
                        continue
                blacklist = ["2048", "1024"]
                triggered = False
                for trigger in blacklist:
                    if trigger in file:
                        triggered = True
                        break
                if triggered:
                    continue
                prefix_passes = [
                        QuickPartitioner(block_size),
                        ]
                ntro_passes = [
                        ComputeErrorThresholdPass(target_threshold=threshold),
                        ForEachBlockPass([
                            UnwrapForEachPassDown(),
                            NumericalTReductionPass(full_loops=1, search_method="n_sum", backup=False, profiling_mode=False, success_threshold=threshold),
                            LogIntermediateGateCountsPass(),
                            GridsynthPass(gridsynth_binary="../examples/gridsynth", threshold=threshold)
                            ], calculate_error_bound=True),
                        #LogErrorPass("after_ntro"),
                        ]
                gridsynth_passes = [
                        ComputeErrorThresholdPass(target_threshold=threshold),
                        ForEachBlockPass([
                            UnwrapForEachPassDown(),
                            GridsynthPass(gridsynth_binary="../examples/gridsynth", threshold=threshold)
                            ], calculate_error_bound=True),
                        #LogErrorPass("after_gridsynth"),
                        ]
               
                passes = [gridsynth_passes, ntro_passes]
                ddict = run_experiment(file, prefix_passes, passes, threshold, block_size)
                pprint_ddict(ddict, os.path.basename(file), ["gridsynth", "ntro"])
                log_ddict_to_tsv(os.path.basename(file), ddict, path)
                new_checkpoint = {
                        "files" : {"index" : indf, "data" : file},
                        "block_sizes" : {"index" : indb, "data" : block_size},
                        "thresholds" : {"index" : indt, "data" : threshold},
                        }
                with open("checkpoint.json", "w") as f:
                    json.dump(new_checkpoint, f)

    from notify import notify
    notify("Completed the big compilation experiment!")


def sort_benchmarks(files):
    return list(reversed(sorted(files))) # TODO - sort first by size (qubits) then alphabetically


directory = "./quipper_circuits/optimizer/QFT_and_Adders/"
files = [directory + filename for filename in os.listdir(directory)]
files = [directory + filename for filename in ["QFTAdd8_before", "QFTAdd8_after"]]

run_benchmarks(files=sort_benchmarks(files))
