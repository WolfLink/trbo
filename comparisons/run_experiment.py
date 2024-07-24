import os
import traceback
from timeit import default_timer as timer

import numpy as np

from bqskit.compiler import CompilationTask, Compiler
from bqskit.passes import ForEachBlockPass, QuickPartitioner, LEAPSynthesisPass, ZXZXZDecomposition, GroupSingleQuditGatePass, UnfoldPass

from ntro import NumericalTReductionPass
from ntro.clift import clifford_gates, t_gates, rz_gates

from parse_quipper import parse_quipper_file

def run_experiment(source_file, pass_lists):
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
    
    opt_circuit = og_circuit
    results = []
    for pass_list in pass_lists:
        pass_dict = dict()
        start = timer()
        try:
            with Compiler() as compiler:
                result_circuit, pass_data = compiler.compile(og_circuit, pass_list, request_data=True)
        except:
            pass_dict["error"] = traceback.format_exc()
            results.append(pass_dict)
            continue
        opt_circuit = result_circuit
        stop = timer()
        result_circuit = opt_circuit.copy()
        result_circuit.unfold_all()
        pass_dict = {"gates" : result_circuit.gate_counts, "qasm" : result_circuit.to("qasm"), "time" : stop - start}
        if opt_circuit.dim <= 1024:
            pass_dict["distance"] = opt_circuit.get_unitary().get_distance_from(og_circuit.get_unitary())
            pass_dict["distance_type"] = "exact"
        elif "distances" in pass_data:
            pass_dict["distance"] = np.sum(pass_data["distance_list"])
            pass_dict["distance_type"] = "sum"
        else:
            pass_dict["distance"] = None
            pass_dict["distance_type"] = "none"

        results.append(pass_dict)
    data_dict["results"] = results


    return data_dict

def pprint_ddict(ddict, title=None):
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
    print(f"Start:\tRz: {og_rz}\tT: {og_t}\tCliff: {og_cliff}\tOther: {og_other}")
    for i, pdict in enumerate(ddict["results"]):
        if "error" in pdict:
            print(pdict["error"])
            continue
        pass_rz = np.sum([pdict["gates"][gate] for gate in pdict["gates"] if gate in rz_gates])
        pass_t = np.sum([pdict["gates"][gate] for gate in pdict["gates"] if gate in t_gates])
        pass_cliff = np.sum([pdict["gates"][gate] for gate in pdict["gates"] if gate in clifford_gates])
        pass_other = np.sum([pdict["gates"][gate] for gate in pdict["gates"] if gate not in rz_gates + clifford_gates + t_gates + rz_gates])
        print(f"Pass {i}:\tRz: {pass_rz}\tT: {pass_t}\tCliff: {pass_cliff}\tOther: {pass_other} ({pdict['time']}s)")
    print("="*10)



if __name__ == "__main__":
    ntro_passes = [
            QuickPartitioner(5),
            ForEachBlockPass([
                NumericalTReductionPass(full_loops=1, search_method="n_sum", backup=True, profiling_mode=False),
                ]),
            ]
    
    before = run_experiment("quipper_circuits/optimizer/QFT_and_Adders/QFT256_before", [ntro_passes])
    pprint_ddict(before, "Before")

    after = run_experiment("quipper_circuits/optimizer/QFT_and_Adders/QFT256_after", [ntro_passes])
    pprint_ddict(after, "After")
