from parse_quipper import *
from sys import argv
import os

filepath = argv[1]

success = 0
failure = 0
for root, dirs, files in os.walk(filepath):
    dirpath = os.path.join(argv[2], root)
    os.makedirs(dirpath, exist_ok=True)
    for file in files:
        try:
            circuit = parse_quipper_file(os.path.join(root,file))
        except:
            print(f"Couldn't convert {file}")
            failure += 1
            continue
        circuit.save(os.path.join(dirpath, f"{file}.qasm"))
        success += 1
        print(f"Converted {file}")


print(f"Converted {success}/{success+failure} = {int(success * 100 / (success + failure))}%")
