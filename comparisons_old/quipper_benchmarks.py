from run_experiment import run_benchmarks

directory = "./quipper_circuits/optimizer/QFT_and_Adders/"
files = [directory + "QFT8_before", directory + "QFT8_after"]

run_benchmarks(files=files, thresholds=[1e-5], block_sizes=[3,4,5,6,7,8], path="./qft8_block_size_sweep")
