from bqskit.compiler.basepass import BasePass
from bqskit.passes.control.foreach import ForEachBlockPass

class ComputeErrorThresholdPass(BasePass):
    def __init__(self, target_threshold):
        self.target_threshold = target_threshold

    async def run(self, circuit, data = {}):
        num_blocks = len(list(circuit.operations_with_cycles()))
        output_threshold = self.target_threshold / num_blocks
        data[ForEachBlockPass.pass_down_key_prefix + "adjusted_threshold"] = output_threshold
        data[ForEachBlockPass.pass_down_key_prefix + "num_blocks"] = num_blocks
        #for key in data:
        #    if key.startswith(ForEachBlockPass.pass_down_key_prefix):
        #        print(f"{key} was a hit")


class UnwrapForEachPassDown(BasePass):
    async def run(self, circuit, data={}):
        keys_to_unwrap = []
        for key in data:
            if key.startswith(ForEachBlockPass.pass_down_key_prefix):
                keys_to_unwrap.append(key)
        for key in keys_to_unwrap:
            new_key = key.removeprefix(ForEachBlockPass.pass_down_key_prefix)
            #print(f"{data[new_key]} changed to {data[key]}")
            data[new_key] = data[key]


class LogIntermediateGateCountsPass(BasePass):
    async def run(self, circuit, data):
        data["intermediate_gate_counts"] = circuit.gate_counts

class LogErrorPass(BasePass):
    def __init__(self, title):
        self.title = title
    async def run(self, circuit, data):
        print(f"{self.title}: {data.error}")
