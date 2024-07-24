import json
from sys import argv



try:
    with open(argv[1], "r") as f:
        data = json.load(f)
        name = data["shared_prefix"]
        reference = data["before"]["before_rz"]
        quipper = data["after"]["before_rz"]
        ntro = data["before"]["after_rz"]
        both = data["after"]["after_rz"]
        time_ntro = data["before"]["time"]
        time_both = data["after"]["time"]

        print(f"{name}\t{reference}\t{ntro}\t{quipper}\t{both}\t\t{time_ntro}\t{time_both}")
        
except:
    raise
