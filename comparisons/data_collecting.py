import os
from ntro.clift import clifford_gates, t_gates, rz_gates


# expected: two pass dicts with the first one being just gridsynth and the second being ntro
def log_ddict_to_tsv(gate_name, ddict, path=None):
    key_blacklist = ["qasm", "gridsynth_stats", "gates", "intermediate_gate_counts"]
    if path is None:
        path = "./results.tsv"

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

    if not os.path.isfile(path):
        print(f"creating {path}")
        with open(path, "w+") as f:
            pdict = ddict["results"][0]
            # first column is gate name, second is pass title
            dataline = "gate_name\tpass_title"

            # next columns are the extra data from the pdict
            for key in pdict:
                if key in key_blacklist:
                    continue
                dataline += f"\t{key}"

            # lastly we add titles for gate analysis
            dataline += f"\tog_cliff\tog_rz\tog_t\topt_cliff\tint_rz\topt_t"
            f.write(dataline + "\n")

    for i, pdict in enumerate(ddict["results"]):
        # print the column titles line

        # first column is gate name, second is pass title
        dataline = f"{gate_name}\t{pass_titles[i]}"

        # next columns are the extra data from the pdict
        for key in pdict:
            if key in key_blacklist:
                continue
            dataline += f"\t{pdict[key]}"

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

        # lastly we add gate analysis
        dataline += f"\t{og_gate_summary['cliff']}\t{og_gate_summary['rz']}\t{og_gate_summary['t']}\t{opt_cliff}\t{int_rz}\t{opt_t}"
        with open(path, "a") as f:
            f.write(dataline + "\n")

