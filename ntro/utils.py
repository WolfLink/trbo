from bqskit.compiler.basepass import BasePass
from bqskit.compiler import Workflow
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.utils.random import seed_random_sources
from bqskit.runtime import get_runtime
import numpy as np
from random import getrandbits
from .clift import best_min_t_count_circuit


class FutureQueue:
    def __init__(self, future, length):
        self.future = future
        self.queue = []
        self.remaining = length

    def __aiter__(self):
        return self

    async def __anext__(self):
        if len(self.queue) > 0:
            self.remaining -= 1
            return self.queue.pop(0)
        elif self.remaining < 1:
            raise StopAsyncIteration
        else:
            try:
                self.queue.extend(await get_runtime().next(self.future))
                return self.queue.pop(0)
            except RuntimeError:
                raise StopAsyncIteration


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

async def _run_workflow_on_circuit(seed, workflow, circuit, data):
    data.seed = seed
    await workflow.run(circuit, data)
    return (circuit, data)

class MultistartPass(BasePass):
    def __init__(self, workflow, multistarts=10, circuit_comparator=best_min_t_count_circuit):
        self.workflow = Workflow(workflow)
        self.multistarts = multistarts
        self.comparator = circuit_comparator

    async def run(self, circuit, data={}):
        best_circuit = None
        best_data = None
        futures = get_runtime().map(_run_workflow_on_circuit, [getrandbits(32) for _ in range(self.multistarts)], workflow=self.workflow, circuit=circuit, data=data)
        random_numbers = []
        async for i, result in FutureQueue(futures, self.multistarts):
            new_circuit, new_data = result
            if self.comparator(best_circuit, new_circuit):
                best_circuit = new_circuit
                best_data = new_data
        circuit.become(best_circuit)
        data.update(best_data)


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
