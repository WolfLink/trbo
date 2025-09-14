import bqskit
import ntro
from timeit import default_timer as timer
import numpy as np
import time
from tqdm import tqdm

PassData = dict

start = timer()
#before_circuit = bqskit.Circuit.from_file("N_570.qasm")
before_circuit = bqskit.Circuit.from_file("/home/marc/Documents/quantum/ntro/comparisons/lbnlqasm/qasm/heisenberg/heisenberg_4.qasm")
with bqskit.compiler.Compiler() as compiler:
    before_circuit = compiler.compile(before_circuit, ntro.workflows.sanitize_gateset())
TRUEN = 570
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
from ntro import NumericalTReductionPass
import psutil
from datetime import datetime
class DebugPass(bqskit.compiler.basepass.BasePass):
    def __init__(
        self,
        success_threshold: float = 1e-6,
        multistarts: int = 32,
        second_pass_starts: int | None = None,
        target_periods = None,
        target_gates = None,
        **kwargs,
    ) -> None:
        """
        Construct a NumericalTReductionPass

        Args:
            success_threshold (float): The synthesis success threshold.
                (Default: 1e-8)
        """
        self.success_threshold = success_threshold
        self.extra_kwargs = kwargs
        self.second_pass_starts = second_pass_starts

        self.acceptable_gates = clifford_gates + t_gates + rz_gates
        if target_periods is None:
            self.target_periods = [0.5 * np.pi, 0.25 * np.pi]
        else:
            self.target_periods = target_periods
        self.multistarts = multistarts


    async def optimize_for_period(self, circuit, target, period, threshold=None):
        trial_circuit = circuit.copy()
        if threshold is None:
            threshold = self.success_threshold

        def gen_blacklist(circuit):
            blacklisted_indices = []
            op_index = 0
            for cycle, op in circuit.operations_with_cycles():
                if len(op.params) < 1:
                    continue
                if op.gate not in rz_gates:
                    blacklisted_indices.extend([i + op_index for i in range(len(op.params))])
                op_index += len(op.params)
            blacklist = np.zeros_like(circuit.params)
            for i in blacklisted_indices:
                blacklist[i] = 1
            return blacklist

        ms1 = self.multistarts
        ms2 = self.second_pass_starts
        if ms2 is None:
            ms2 = ms1 // 2
            ms2 = max(1, ms2)

        d_gen = MatrixDistanceCostGenerator()
        d_res = HilbertSchmidtResidualsGenerator()
        best_params = circuit.params
        best_N = 0
        first_min = CeresMinimizer()

        high = len(circuit.params)
        low = 0
        while low <= high:
            N = low + (high - low) // 2
            blacklist = gen_blacklist(trial_circuit)
            n_gen = RoundSmallestNCostGenerator(N, period, blacklist=blacklist)
            n_res = RoundSmallestNResidualsGenerator(N, period, blacklist=blacklist)
            sum_gen = SumCostGenerator(d_gen, n_gen)
            sum_res = SumResidualsGenerator(d_res, n_res)
            def get_score(x):
                return sum_gen.gen_cost(trial_circuit, target)(x)

            # Two good sets of parameters to try before introducing randomness:
            #   1. Previously known good parameters in this loop
            #   2. The original circuit parameters
            # Often one of these sets of parameters will just work, allowing us to skip optimization.

            trial_params = best_params
            score = get_score(trial_params)
            if score >= threshold:
                trial_params = circuit.params
                score = get_score(trial_params)

            # When those good parameters don't work, we need to search for new ones.
            # Starting near the "good guesses" is a great place to start.
            if score >= threshold:
                miser = MultiStartMinimization(sum_res, multistarts=2, minimizer=CeresMinimizer(), second_pass=None, judgement_cost=sum_gen)
                result = await miser.multi_start_instantiate_async(trial_circuit, target, starts=[best_params, circuit.params])
                score = get_score(trial_params)

            # Sometimes its truly necessary to search new territory, so we now use random starting points.
            if score >= threshold:
                miser = MultiStartMinimization(sum_res, multistarts=ms1, minimizer=CeresMinimizer(), second_pass=ms2, threshold=threshold, judgement_cost=sum_gen)
                result = await miser.multi_start_instantiate_async(trial_circuit, target)
                trial_params = result.params
                score = get_score(trial_params)

            if score >= threshold and score < threshold * 10:
                # try a fine-tuning approach to see if we can squeeze out enough improvement to pass the threshold
                old_score = score
                fine_tuner = LBFGSMinimizer()
                trial_params = fine_tuner.minimize(sum_gen.gen_cost(circuit, target), trial_params)
                score = get_score(trial_params)

                # after all that trying to get a good result, we ultimately have to give up if we still don't have an acceptable result
            with open("pslog.log", "a") as f:
                text = f"{datetime.now().isoformat()}\t{low}\t{N}\t{high}\t{psutil.virtual_memory().percent}"
                f.write(text + "\n")
                print(text)
            if score >= threshold:
                high = N - 1
            else:
                low = N + 1
                if N > best_N:
                    best_params = trial_params
                    best_N = N
 
        if best_N == 0:
            # we failed to find any improvement
            return

        # We have identified a set of parameters that allows N Rz gates to be rounded
        # Now its time to go actually replace those Rz gates with Clifford+T circuits
        best_circuit = circuit.copy()
        best_circuit.set_params(best_params)
        best_sum = RoundSmallestNCostGenerator(best_N, period, blacklist=gen_blacklist(best_circuit)).gen_cost(best_circuit, target)(best_params)
        best_dist = best_circuit.get_unitary().get_distance_from(target)

        indices = np.argsort(get_deviation_arr(best_circuit.params, period, gen_blacklist(best_circuit))) + 1
        op_index = 0
        for cycle, op in best_circuit.operations_with_cycles():
            op_index += len(op.params)
            if len(op.params) != 1:
                continue
            if op.gate not in rz_gates:
                continue
            if op_index in indices[:best_N]:
                if op.gate in rz_gates:
                    rounded = circuit_for_rounded_val(op.params[0], period < np.pi * 0.5)
                    best_circuit.replace_gate(
                        (cycle, op.location[0]), rounded, op.location
                    )
                else:
                    raise RuntimeError("Attempted to round unexpected gate type {op.gate}")
        test_params = CeresMinimizer(ftol=5e-16, gtol=1e-15).minimize(HilbertSchmidtResidualsGenerator().gen_cost(best_circuit, target), best_circuit.params)
        if best_circuit.get_unitary(test_params).get_distance_from(target) < best_circuit.get_unitary().get_distance_from(target):
            best_circuit.set_params(test_params)
        if not best_circuit.get_unitary().get_distance_from(target) <= threshold:
            print(f"ERROR got {best_circuit.get_unitary().get_distance_from(target)} > {threshold}")
        return best_circuit

    async def optimize_all_periods(self, circuit, target, x0, threshold=None):
        candidate_circuit = circuit.copy()
        candidate_circuit.set_params(x0)
        for period in self.target_periods:
            result = await get_runtime().submit(
                    self.optimize_for_period,
                    candidate_circuit,
                    target,
                    period,
                    threshold,
                    )
            if result is not None:
                result.unfold_all()
                candidate_circuit = result
        return candidate_circuit


    async def run_real(self, circuit: Circuit, data: PassData = {}) -> None:
        # Check that circuit has been converrted to Clifford+T+Rz
        if any(g not in self.acceptable_gates for g in circuit.gate_set):
            m = (
                'Circuit must be converted to Clifford+T+Rz before running'
                f' NumericalTReductionPass. Got {circuit.gate_set}.'
            )
            raise ValueError(m)

        if "utry" not in data:
            utry = circuit.get_unitary()
        else:
            utry = data["utry"]

        if "adjusted_threshold" in data:
            threshold = data["adjusted_threshold"]
        else:
            threshold = self.success_threshold

        initial_circuit = circuit
        initial_circuit.unfold_all()
        candidate_circuit = initial_circuit

        # perform the optimization
        candidate_circuit = await self.optimize_all_periods(initial_circuit, utry, circuit.params, threshold)

        # verify that the optimization improved the circuit before accepting the new circuit
        if better_min_t_count_circuit(initial_circuit, candidate_circuit):
            circuit.become(candidate_circuit)
        else:
            circuit.become(initial_circuit)
        return circuit

    async def run(self, circuit, data=None):
        for _ in tqdm(list(range(100))):
            future = get_runtime().map(
                    do_a_ceres,
                    [np.random.rand(circuit.num_params) * np.pi * 2 for _ in range(128)],
                    )
            time.sleep(1)
            get_runtime().cancel(future)


def do_a_ceres(x0):
    mizer = CeresMinimizer()
    circuit = before_circuit
    d_res = HilbertSchmidtResidualsGenerator()
    n_res = RoundSmallestNResidualsGenerator(500, 0.5 * np.pi)
    sum_res = SumResidualsGenerator(d_res, n_res)
    cost = sum_res.gen_cost(circuit, target)
    try:
        mizer.minimize(cost, x0)
    except GeneratorExit:
        print("Got a generator exit!")
        raise

with bqskit.compiler.Compiler() as compiler:
    after_circuit = compiler.compile(before_circuit, DebugPass())
    #after_circuit = compiler.compile(before_circuit, NumericalTReductionPass(periods=[0.25 * np.pi]))

end = timer()
print(f"Optimized circuit for one round in {end - start}s")
print(f"After circuit: {after_circuit.gate_counts}")

