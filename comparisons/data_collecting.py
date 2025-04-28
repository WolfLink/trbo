import os
from ntro.clift import clifford_gates, t_gates, rz_gates
import datetime


def log_error_message(message, gate_base_name, gate_suffix, parsed_data, path):
    # first column is gate name and suffix
    dataline = f"{gate_base_name}\t{gate_suffix}"

    # next add the data from the parsed_data dict
    for key in parsed_data:
        dataline += f"\t{parsed_data[key]}"

    # add a timestamp
    timestr = datetime.datetime.now().isoformat()
    dataline += f"\t{timestr}"

    # finally append the error message
    dataline += f"\t{message}"

    with open(path, "a") as f:
        f.write(dataline + "\n")

# expected: two pass dicts with the first one being just gridsynth and the second being ntro
def log_ddict_to_tsv(gate_name, ddict, path=None):
    key_blacklist = ["qasm", "gridsynth_stats", "gates", "intermediate_gate_counts"]
    parsed_data = {
            "block_size" : -1,
            "partitions" : -1,
            "ntro_partitions" : -1,
            "threshold" : -1,
            "gridsynth_time" : -1,
            "ntro_time" : -1,
            "control_dist" : -1,
            "gridsynth_dist" : -1,
            "ntro_dist" : -1,
            "og_t" : -1,
            "og_cliff" : -1,
            "og_rz" : -1,
            "gridsynth_t" : -1,
            "gridsynth_cliff" : -1,
            "int_rz" : -1,
            "opt_t" : -1,
            "opt_cliff" : -1,
            }
    if path is None:
        path = "./"

    path = os.path.join(path, "summary.tsv")

    # separate gate base name from suffix
    if "_after" in gate_name:
        gate_base_name, gate_suffix = gate_name.split("_after")
        gate_suffix = "after" + gate_suffix
    elif "_before" in gate_name:
        gate_base_name, gate_suffix = gate_name.split("_before")
        gate_suffix = "before" + gate_suffix
    else:
        gate_base_name = gate_name
        gate_suffix = ""

    if "threshold" in ddict:
        parsed_data["threshold"] = ddict["threshold"]
    if "block_size" in ddict:
        parsed_data["block_size"] = ddict["block_size"]
    if "num_partitions" in ddict:
        parsed_data["partitions"] = ddict["num_partitions"]

    if "error" in ddict:
        log_error_message(ddict["error"], gate_base_name, gate_suffix, parsed_data, path)
        return

    all_gates = []
    for gate in ddict["og_gates"]:
        if gate not in all_gates:
            all_gates.append(gate)
    for pdict in ddict["results"]:
        for key in ["gates", "intermediate_gate_counts"]:
            if key in pdict:
                for gate in pdict[key]:
                    if gate not in all_gates:
                        all_gates.append(gate)

    for gate in ddict["og_gates"]:
        if gate not in clifford_gates + rz_gates + t_gates:
            print(f"WARNING: unexpected gate {gate} not in Cliff+T+Rz")

    og_gate_summary = {"cliff" : 0, "rz" : 0, "t" : 0}
    for gate in all_gates:
        if gate in ddict["og_gates"]:
            value = ddict["og_gates"][gate]
            if gate in t_gates:
                og_gate_summary["t"] = og_gate_summary["t"] + value
            elif gate in clifford_gates:
                og_gate_summary["cliff"] = og_gate_summary["cliff"] + value
            elif gate in rz_gates:
                og_gate_summary["rz"] = og_gate_summary["rz"] + value

    parsed_data["og_t"] = og_gate_summary["t"]
    parsed_data["og_cliff"] = og_gate_summary["cliff"]
    parsed_data["og_rz"] = og_gate_summary["rz"]


    if "control_dist" in ddict:
        parsed_data["control_dist"] = ddict["control_dist"]

    if not os.path.isfile(path):
        print(f"creating {path}")
        with open(path, "w+") as f:
            pdict = ddict["results"][0]
            # first column is gate name and suffix
            dataline = "gate_name\tgate_suffix"

            # next add the data from the parsed_data dict
            for key in parsed_data:
                dataline += f"\t{key}"

            # finally add a timestamp
            dataline += "\ttimestamp"

            f.write(dataline + "\n")

    for i, pdict in enumerate(ddict["results"]):
        # print the column titles line
        # next columns are the extra data from the pdict
        opt_cliff = 0
        int_rz = 0
        opt_t = 0
        for gate in all_gates:
            if gate in pdict["gates"]:
                if gate in t_gates:
                    opt_t += pdict["gates"][gate]
                elif gate in clifford_gates:
                    opt_cliff += pdict["gates"][gate]
        if "intermediate_gate_counts" in pdict:
            for gate in all_gates:
                if gate in pdict["intermediate_gate_counts"] and gate in rz_gates:
                    int_rz += pdict["intermediate_gate_counts"][gate]
        else:
            int_rz = -1
        if "intermediate_block_count" in pdict:
            parsed_data["ntro_partitions"] = pdict["intermediate_block_count"]

        if i == 0:
            # assume gridsynth
            parsed_data["gridsynth_t"] = opt_t
            parsed_data["gridsynth_cliff"] = opt_cliff
            parsed_data["gridsynth_dist"] = pdict["distance"]
            parsed_data["gridsynth_time"] = pdict["time"]
        elif i == 1:
            # assume ntro
            parsed_data["int_rz"] = int_rz
            parsed_data["opt_t"] = opt_t
            parsed_data["opt_cliff"] = opt_cliff
            parsed_data["ntro_dist"] = pdict["distance"]
            parsed_data["ntro_time"] = pdict["time"]
        else:
            print("Surprised by more than 2 pass dicts!")
            continue


    # first column is gate name and suffix
    dataline = f"{gate_base_name}\t{gate_suffix}"

    # next add the data from the parsed_data dict
    for key in parsed_data:
        dataline += f"\t{parsed_data[key]}"

    # finally add a timestamp
    timestr = datetime.datetime.now().isoformat()
    dataline += f"\t{timestr}"
    with open(path, "a") as f:
        f.write(dataline + "\n")

