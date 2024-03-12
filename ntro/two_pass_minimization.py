from __future__ import annotations

from typing import Any

import numpy as np


from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.opt.instantiater import Instantiater
from bqskit.ir.opt.minimizer import Minimizer
from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.compiler.basepass import BasePass


from .clift import *
from .tcount import *
from .constrained_minimizer import *



class TwoPassMinimization(Instantiater):
    def __init__(self,
            pass_1_cost_gen: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
            pass_2_cost_gen: CostFunctionGenerator = RelaxedTCountCostGenerator(),
            pass_2_cstr_gen: CostFunctionGenerator = HilbertSchmidtCostGenerator(),
            first_pass: Minimizer | None = None,
            second_pass: Minimizer | None = None,
            **kwargs: dict[str, Any],
            ) -> None:

        if first_pass is None:
            first_pass = CeresMinimizer()

        if "success_threshold" in kwargs:
            self.threshold = kwargs["success_threshold"]
        else:
            self.threshold = 1e-6
        if second_pass is None:
            second_pass = ConstrainedMinimizer(None, constraint_threshold=self.threshold)

        self.pass_1_cost_gen = pass_1_cost_gen
        self.pass_2_cost_gen = pass_2_cost_gen
        self.pass_2_cstr_gen = pass_2_cstr_gen
        self.first_pass = first_pass
        self.second_pass = second_pass
        # while I am doing everything single-threaded, it makes more sense to do things one at a time IMO
        self.first_pass_multistarts = 1
        self.first_pass_retries = 16
        self.second_pass_multistarts = 8

    def is_capable(self, circuit):
        for cycle, op in circuit.operations_with_cycles():
            if op.gate.qasm_name not in clifford_gates + t_gates + rz_gates:
                return False
        return True

    def get_violation_report(self, circuit):
        for cycle, op in circuit.operations_with_cycles():
            if op.gate.qasm_name not in clifford_gates + t_gates + rz_gates:
                return f"Found gate {op.gate.qasm_name} which is not in {clifford_gates + t_gates + rz_gates}"

        raise ValueError("I am not sure what I am supposed to do here so I'll leave this as a TODO for later")

    def get_method_name(self):
        return "two-pass-minimization"


    def instantiate(self, circuit, target, x0):
        # how to do multistarts?
        #print("INSTANTIATE WAS CALLED")
        return self.multi_start_instantiation(circuit, target)

        # multi_start_instantiation(self, circuit, target)
        # lets my code handle multi starts itself

    def normalize(self, result):
        return np.mod(result, np.pi * 2)

    def multi_start_instantiation(self, circuit, target):
        #print("MULTISTART INSTATIATION WAS CALLED")

        # run the first pass
        pass_1_results = []
        pass_1_cost = self.pass_1_cost_gen.gen_cost(circuit, target)
        pass_2_cstr = self.pass_2_cstr_gen.gen_cost(circuit, target)
        self.second_pass.constraint = pass_2_cstr
        best_1st_pass_result = 1
        total_tries = 0

        while total_tries < self.first_pass_retries:
            total_tries += self.first_pass_multistarts
            # run a batch of optimizations
            results = [self.first_pass.minimize(pass_1_cost, np.random.rand(circuit.num_params) * np.pi * 2) for _ in range(self.first_pass_multistarts)]

            # filter the results for failures and duplicates
            for result in results:
                # filter out failures to meet the threshold
                if pass_2_cstr(result) < best_1st_pass_result:
                    best_1st_pass_result = pass_2_cstr(result)
                if pass_2_cstr(result) > self.threshold:
                    if pass_2_cstr(result) < 1e-4: # this was a promising result and should undergo higher quality minimization
                        result2 = CeresMinimizer(ftol=5e-16, gtol=1e-15).minimize(pass_1_cost, result)

                        if pass_2_cstr(result2) > self.threshold:
                            continue # if its still not an acceptable result, reject it
                        else:
                            result = result2
                    else:
                        continue

                # normalize the parameters to make comparison simpler
                normalized_result = self.normalize(result)
                # filter out duplicates
                hit = False
                for pass_1_result in pass_1_results:
                    if np.all(np.isclose(normalized_result, pass_1_result, 1e-2, 1e-4)):
                        hit = True
                        break
                if not hit:
                    pass_1_results.append(normalized_result)

            # if we have met the quota, we can stop retrying
            if len(pass_1_results) >= self.second_pass_multistarts:
                break

        if len(pass_1_results) < 1:
            return [0 for _ in range(circuit.num_params)]
        pass_2_cost = self.pass_2_cost_gen.gen_cost(circuit, target)
        best_result = None
        best_cost = None
        best_cstr = None
        for x0 in pass_1_results:
            result = self.second_pass.minimize(pass_2_cost, x0)
            result_cost = pass_2_cost(result)
            result_cstr = pass_2_cstr(result)
            #print(f"Finished a second pass with cost {result_cost} and cstr {result_cstr}")
            if result_cstr > self.threshold:
                print(f"Rejected a second pass result because the contraint value was {result_cstr}")
                continue
            if best_result is None:
                best_result = result
                best_cost = result_cost
                best_cstr = result_cstr
            else:
                if result_cost < best_cost:
                    best_result = result
                    best_cost = result_cost
                    best_cstr = result_cstr
                elif result_cost == best_cost and result_cstr < best_cstr:
                    best_result = result
                    best_cost = result_cost
                    best_cstr = result_cstr

        #print(f"chose a result with {pass_2_cost(best_result)} & {pass_2_cstr(best_result)}")
        return best_result



"""
    # rough outline of what I want to accomplish:
    def two_pass_minimization(
            circuit, # parameterized circuit to optimize
            target,  # target unitary
            cost_fn, # primary cost to be minimized (relaxed T count)
            constraint_fn, # secondary cost to be minimized (HilbertSchmidt)
            first_pass, # minimizer for first pass (whatever is default is fine)
            second_pass, # minimizer for second pass (SLSQP with support for constraints)
            first_pass_multistarts, # size of batch of first passes to run
            second_pass_multistarts, # multistarts for second batch, which also serves as a quota for the first batch
            first_pass_retries, # the first pass is run in batches of first_pass_multistarts until either the quota of second_pass_multistarts is met or until first_pass_retries batches have been run
            ):


        pass_1_results = []
        for _ in range(first_pass_retries):
            results = first_pass.minimize(circuit, target, constraint_fn, first_pass_multistarts)
            pass_1_results.append(results)
            pass_1_results.filter_for_duplicates_and_failures(results, pass_1_results)
            if len(pass_1_results) >= second_pass_multistarts:
            break

        # note that the number of multistarts that second_pass receives could be more or less than second_pass_multistarts
        # it will only be less if the first_pass_retries condition was the limiting factor
        # otherwise it will almost certainly be more
        # its also not necessarily equal to first_pass_multistarts or some multiple because some 1st pass results will get pruned as duplicates or failures

        # also note that second_pass is going to have to use constraint_fn as the constraint threshold-constraint_fn(x).  Idk what will be in charge of that conversion.  Whatever it is, it needs to be aware that the derivative needs to be negated as well.  Also the threshold here isn't necessarily the same threshold as in the first pass.  In fact, it probably shouldn't be, because the second threshold isn't quite guranteed to be met (SLSQP allows small constraint violations)
        pass_2_results = second_pass.minimize(circuit, target, cost_fn, constraint_fn, pass_1_results)
        return pass_2_results.choose_best_result()


    # all of the bonus nuance I can figure out later
    # the primary issues are:
    # 1. What is in charge of organizing multistarts?
    # 2. how can I have the two_pass_minimizer play its role in multi-start selection and running multiple starts?
    #  - note that its not intended for two_pass_minimization to be called in parallel (some number multistarts) times
    #  - should two_pass_minimization be a pass instead of an instantiater?  It really plays the role of "instantiater" but I'm not sure if the instantiater API is powerful enough for it
    # 3. What should be in charge of converting constraint functions to constraints usable by SLSQP, and how should the difference in threshold be expressed?
    # 4. How to get SLSQP?  I can write my own wrapper around scipy, but ideally it would be in rust

"""
