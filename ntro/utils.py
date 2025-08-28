from bqskit.compiler.basepass import BasePass
from bqskit.compiler import Workflow
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.utils.random import seed_random_sources
from bqskit.runtime import get_runtime
from bqskit.compiler.workflow import Workflow
from bqskit.passes.control.predicate import PassPredicate
import numpy as np
from random import getrandbits
from .clift import better_min_t_count_circuit


class FutureQueue:
    def __init__(self, future, length):
        self._cancelled = False
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
                self.remaining -= 1
                return self.queue.pop(0)
            except RuntimeError:
                raise StopAsyncIteration

    def cancel(self):
        if not self._cancelled:
            self._cancelled = True
            get_runtime().cancel(self.future)

class HasGateSetPredicate(PassPredicate):
    """Predicate that returns true if all gates are in the specified gateset."""

    def __init__(self, gateset):
        self.gateset = gateset

    def get_truth_value(self, circuit, data):
        return all(g in self.gateset for g in circuit.gate_set)


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
    workflow = Workflow(workflow)
    data.seed = seed
    await workflow.run(circuit, data)
    return (circuit, data)

class MultistartPass(BasePass):
    def __init__(self, workflow, multistarts=10, circuit_comparator=better_min_t_count_circuit, goal_condition=None):
        self.workflow = Workflow(workflow)
        self.multistarts = multistarts
        self.comparator = circuit_comparator
        self.goal_condition = goal_condition

    async def run(self, circuit, data={}):
        best_circuit = None
        best_data = None
        futures = get_runtime().map(_run_workflow_on_circuit, [getrandbits(32) for _ in range(self.multistarts)], workflow=self.workflow, circuit=circuit, data=data)
        random_numbers = []
        
        attempts = FutureQueue(futures, self.multistarts)
        async for i, result in attempts:
            new_circuit, new_data = result
            if self.comparator(best_circuit, new_circuit):
                best_circuit = new_circuit
                best_data = new_data
                if self.goal_condition is not None and self.goal_condition(best_circuit):
                    attempts.cancel()
        circuit.become(best_circuit)
        data.update(best_data)

class SuccessBenchmarkPass(BasePass):
    def __init__(self, workflow, condition, key="benchmark_success", runs=100):
        self.workflow = workflow
        self.condition = condition
        self.runs = runs
        self.key = key

    async def run(self, circuit, data={}):
        futures = get_runtime().map(_run_workflow_on_circuit, [getrandbits(32) for _ in range(self.runs)], workflow=self.workflow, circuit=circuit, data=data)
        successes = 0
        failures = 0
        async for i, result in FutureQueue(futures, self.runs):
            try:
                if self.condition(circuit, result):
                    successes += 1
                else:
                    failures += 1
            except:
                failures += 1

        if successes + failures != self.runs:
            print(f"UNEXPECTED: {successes} successes + {failures} failures != {self.runs} expected total.")

        data[self.key] = {
                "successes" : successes,
                "failures" : failures,
                "total" : self.runs,
                }

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

class SetDataPass(BasePass):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    async def run(self, circuit, data):
        data[self.key] = self.value

class AppendGatePass(BasePass):
    def __init__(self, gate, location=None):
        self.gate = gate
        if location is None:
            location = [0]
        self.location = location

    async def run(self, circuit, data):
        circuit.append_gate(self.gate, self.location)
        return circuit

class RemoveGatePass(BasePass):
    def __init__(self, target_gate_type):
        self.target_gate_type = target_gate_type

    async def run(self, circuit, data):
        circuit.remove_all(self.target_gate_type)
        return circuit

