import bqskit
import ntro
from timeit import default_timer as timer
import numpy as np
import time
from tqdm import tqdm

start = timer()
before_circuit = bqskit.Circuit.from_file("lbnlqasm/qasm/heisenberg/heisenberg_4.qasm")
with bqskit.compiler.Compiler() as compiler:
    before_circuit = compiler.compile(before_circuit, ntro.workflows.sanitize_gateset())

target = before_circuit.get_unitary()
end = timer()
print(f"Loaded and sanitized circuit in {end - start}s")

print(f"before: {before_circuit.gate_counts}")
start = timer()


from ntro.multi_start_minimization import *
from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost import HilbertSchmidtResidualsGenerator
from ntro.tcount import *
from ntro.clift import *
from ntro.multi_start_minimization import *
class DebugPass(bqskit.compiler.basepass.BasePass):
    #async def run(self, circuit, data = {}):
    #    n_res = ntro.tcount.RoundSmallestNCostGenerator(500, np.pi * 0.25)
    #    d_res = ntro.tcount.MatrixDistanceCostGenerator()
    #    d_res = bqskit.ir.opt.cost.HilbertSchmidtCostGenerator()
    #    s_res = ntro.tcount.SumCostGenerator(n_res, d_res)
    #    futures = bqskit.runtime.get_runtime().map(run_minimization, [circuit] * 500, [LBFGSMinimizer()] * 500, [d_res] * 500, [target] * 500, [circuit.params] * 500)
    #    queue = ntro.utils.FutureQueue(futures, 500)
    #    t = tqdm(total=500)
    #    async for index, result in queue:
    #        t.update()
    #    t.close()

    #async def run(self, circuit, data = {}):
    #    n_res = ntro.tcount.RoundSmallestNResidualsGenerator(90, np.pi * 0.25)
    #    d_res = HilbertSchmidtResidualsGenerator()
    #    s_res = ntro.tcount.SumResidualsGenerator(n_res, d_res)
    #    futures = bqskit.runtime.get_runtime().map(run_minimization, [circuit] * 1000, [CeresMinimizer()] * 1000, [s_res] * 1000, [target] * 1000, [circuit.params] * 1000)
    #    queue = ntro.utils.FutureQueue(futures, 1000)
    #    t = tqdm(total=1000)
    #    async for index, result in queue:
    #        t.update()
    #    t.close()
    #   # async for _ in tqdm(range(1000), total=1000):
    #   #     #run_minimization(circuit, CeresMinimizer(), s_res, target, np.random.rand(*circuit.params.shape))
    #   #     run_minimization(circuit, CeresMinimizer(), s_res, target, circuit.params)

    async def run2(self, circuit, data={}):
        n_res = ntro.tcount.RoundSmallestNCostGenerator(500, np.pi * 0.25)
        d_res = ntro.tcount.MatrixDistanceCostGenerator()
        s_res = ntro.tcount.SumCostGenerator(n_res, d_res)

        period = np.pi * 0.25
        threshold = 1e-5
        best_circuit = circuit.copy()
        best_params = run_minimization(circuit, LBFGSMinimizer(), s_res, target, circuit.params)
        best_circuit.set_params(best_params)
        best_N = 500
        best_sum = RoundSmallestNCostGenerator(best_N, period).gen_cost(best_circuit, target)(best_params)
        best_dist = best_circuit.get_unitary().get_distance_from(target)
        for i in range(best_N):
            if len(best_circuit.params) < 1:
                break
            trial_circuit = best_circuit
            index = np.argmin(get_deviation_arr(trial_circuit.params, period))
            trial_circuit = best_circuit.copy()
            op_index = 0
            for cycle, op in trial_circuit.operations_with_cycles():
                op_index += len(op.params)
                if len(op.params) != 1:
                    continue
                if op.gate not in rz_gates:
                    continue
                if op_index > index:
                    if op.gate in rz_gates:
                        rounded = circuit_for_rounded_val(op.params[0], period < np.pi * 0.5)
                        trial_circuit.replace_gate(
                            (cycle, op.location[0]), rounded, op.location
                        )
                        break
                    else:
                        raise RuntimeError("Attempted to round unexpected gate type {op.gate}")
            if trial_circuit.get_unitary().get_distance_from(target) >= threshold:
                #test_params = CeresMinimizer(ftol=5e-16, gtol=1e-15).minimize(HilbertSchmidtResidualsGenerator().gen_cost(trial_circuit, target), trial_circuit.params)
                test_params = LBFGSMinimizer().minimize(s_res.gen_cost(trial_circuit, target), trial_circuit.params)
                trial_circuit.set_params(test_params)
            if trial_circuit.get_unitary().get_distance_from(target) >= threshold:
                # we failed to round as much as expected
                # generally if this happens, its indicative of a bug
                raise RuntimeWarning("Failed to round as many gates as expected.")
                break
            best_circuit = trial_circuit
        test_params = CeresMinimizer(ftol=5e-16, gtol=1e-15).minimize(HilbertSchmidtResidualsGenerator().gen_cost(best_circuit, target), best_circuit.params)
        if best_circuit.get_unitary(test_params).get_distance_from(target) < best_circuit.get_unitary().get_distance_from(target):
            best_circuit.set_params(test_params)
        if not best_circuit.get_unitary().get_distance_from(target) <= threshold:
            print(f"ERROR got {best_circuit.get_unitary().get_distance_from(target)} > {threshold}")
        best_circuit.unfold_all()
        circuit.become(best_circuit)

    async def run(self, circuit, data={}):
        best_N = 600
        period = np.pi * 0.25
        n_cost = ntro.tcount.RoundSmallestNCostGenerator(best_N, period)
        d_cost = ntro.tcount.MatrixDistanceCostGenerator()
        s_cost = ntro.tcount.SumCostGenerator(n_cost, d_cost)
        n_res = ntro.tcount.RoundSmallestNResidualsGenerator(best_N, period)
        d_res = HilbertSchmidtResidualsGenerator()
        s_res = ntro.tcount.SumResidualsGenerator(n_res, d_res)

        threshold = 1e-5
        best_circuit = circuit.copy()
        start = timer()
        print("start of opt")
        best_params = run_minimization(circuit, CeresMinimizer(), s_res, target, circuit.params)
        result = await MultiStartMinimization(s_res, multistarts=128, minimizer=CeresMinimizer(), second_pass=32, threshold=threshold, judgement_cost=s_cost).multi_start_instantiate_async(circuit, target, starts=[circuit.params])
        assert s_cost.gen_cost(circuit, target)(best_params) < threshold
        end = timer()
        print(f"first opt in {end - start}s")
        start = end
        best_circuit.set_params(best_params)
        best_sum = RoundSmallestNCostGenerator(best_N, period).gen_cost(best_circuit, target)(best_params)
        best_dist = best_circuit.get_unitary().get_distance_from(target)

        indices = np.argsort(get_deviation_arr(best_params, period)) + 1
        op_index = 0
        t = tqdm(total=best_N)
        for cycle, op in best_circuit.operations_with_cycles():
            op_index += len(op.params)
            if len(op.params) != 1:
                continue
            if op.gate not in rz_gates:
                continue
            if op_index in indices[:best_N]:
                if op.gate in rz_gates:
                    t.update()
                    rounded = circuit_for_rounded_val(op.params[0], period < np.pi * 0.5)
                    best_circuit.replace_gate(
                        (cycle, op.location[0]), rounded, op.location
                    )
                else:
                    raise RuntimeError("Attempted to round unexpected gate type {op.gate}")
        t.close()
        end = timer()
        print(f"replacement in {end - start}s")
        start = end
        print(f"current dist: {best_circuit.get_unitary(best_circuit.params).get_distance_from(target)}")
        #test_params = CeresMinimizer(ftol=5e-16, gtol=1e-15).minimize(HilbertSchmidtResidualsGenerator().gen_cost(best_circuit, target), best_circuit.params)
        result = await MultiStartMinimization(d_res, multistarts=32, minimizer=CeresMinimizer(), second_pass=None).multi_start_instantiate_async(best_circuit, target, starts=[best_circuit.params])
        test_params = result.params
        end = timer()
        print(f"final opt in {end - start}s")
        if best_circuit.get_unitary(test_params).get_distance_from(target) < best_circuit.get_unitary().get_distance_from(target):
            best_circuit.set_params(test_params)
        else:
            print(f"Opt gave: {best_circuit.get_unitary(test_params).get_distance_from(target)}")
        if not best_circuit.get_unitary().get_distance_from(target) <= threshold:
            print(f"ERROR got {best_circuit.get_unitary().get_distance_from(target)} > {threshold}")
        best_circuit.unfold_all()
        circuit.become(best_circuit)

with bqskit.compiler.Compiler() as compiler:
    after_circuit = compiler.compile(before_circuit, DebugPass())
end = timer()
print(f"Optimized circuit for one round in {end - start}s")
print(f"After circuit: {after_circuit.gate_counts}")

