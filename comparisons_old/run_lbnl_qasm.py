import os

from run_experiment import run_benchmarks


files = []
for (root, dirs, files) in os.walk('lbnlqasm/qasm'):
    for file in files:
        if os.path.splitext(file)[1] == '.qasm':
            path = os.path.join(root, file)
            files.append(path)

run_benchmarks(files=files, threshold=[1e-5], block_sizes=[4], path="./lbmlqasm")
