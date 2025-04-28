from bqskit.compiler.basepass import BasePass
from bqskit.passes.control.foreach import ForEachBlockPass

class ComputeErrorThresholdPass(BasePass):
    def __init__(self, target_threshold):
        self.target_threshold = target_threshold

    async def run(self, circuit, data = {}):
        num_blocks = self.get_num_blocks(circuit)
        output_threshold = self.get_threshold(circuit)
        data[ForEachBlockPass.pass_down_key_prefix + "adjusted_threshold"] = output_threshold
        data[ForEachBlockPass.pass_down_key_prefix + "num_blocks"] = num_blocks
        #for key in data:
        #    if key.startswith(ForEachBlockPass.pass_down_key_prefix):
        #        print(f"{key} was a hit")

    def get_threshold(self, circuit):
        num_blocks = self.get_num_blocks(circuit)
        output_threshold = self.target_threshold / num_blocks
        return output_threshold

    def get_num_blocks(self, circuit):
        num_blocks = len(list(circuit.operations_with_cycles()))
        return num_blocks


class UnwrapForEachPassDown(BasePass):
    async def run(self, circuit, data={}):
        keys_to_unwrap = []
        for key in data:
            if key.startswith(ForEachBlockPass.pass_down_key_prefix):
                keys_to_unwrap.append(key)
        for key in keys_to_unwrap:
            new_key = key.removeprefix(ForEachBlockPass.pass_down_key_prefix)
            data[new_key] = data[key]


class LogIntermediateGateCountsPass(BasePass):
    async def run(self, circuit, data):
        data["intermediate_block_count"] = len(list(circuit.operations_with_cycles()))
        test_circuit = circuit.copy()
        test_circuit.unfold_all()
        data["intermediate_gate_counts"] = test_circuit.gate_counts


class SaveQasmPass(BasePass):
    def __init__(self, filepath):
        self.filepath = filepath

    async def run(self, circuit, data):
        test_circuit = circuit.copy()
        test_circuit.unfold_all()
        with open(self.filepath, "w") as f:
            f.write(test_circuit.to("qasm"))

class LogErrorPass(BasePass):
    def __init__(self, title):
        self.title = title
    async def run(self, circuit, data):
        print(f"{self.title}: {data.error}")
