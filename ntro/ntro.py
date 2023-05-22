"""This module implements the NumericalTReductionPass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import *
from bqskit.runtime import get_runtime

from .two_pass_minimization import *
from .clift import *

# Numerical T Reduction and Optimization



class NumericalTReductionPass(BasePass):
    def __init__(
            self,
            utry=None,
            success_threshold: float = 1e-6
            ) -> None:
        """
        Construct a NumericalTReductionPass
        """
        self.instantiate_options = {
                'cost_fn_gen': HilbertSchmidtResidualsGenerator(),
                'method' : TwoPassMinimization(),
                'kwargs' : {
                    'pifrac' : 4,
                    }
                }
        self.utry = utry
        self.success_threshold = success_threshold



    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        best_circuit = circuit
        utry = self.utry
        if utry is None:
            utry = circuit.get_unitary()


        # look at bqskit.compiler.compile._get_single_qudit_gate_rebase_pass

        ZXZXZCirc = Circuit(1)
        ZXZXZCirc.append_gate(RZGate(), 0)
        ZXZXZCirc.append_gate(SXGate(), 0)
        ZXZXZCirc.append_gate(RZGate(), 0)
        ZXZXZCirc.append_gate(SXGate(), 0)
        ZXZXZCirc.append_gate(RZGate(), 0)

        ZXZXZ = CircuitGate(ZXZXZCirc)
        # first step is to convert to ZXZXZ
        for cycle, op in circuit.operations_with_cycles():
            if op.gate.qasm_name in clifford_gates + t_gates + rz_gates:
                continue
            elif len(op.location) == 1:
                best_circuit.replace_gate((cycle, op.location[0]), ZXZXZ, op.location) # what is the difference between replace and replace_gate?
                
            else:
                print(f"Unable to convert {op.gate.qasm_name} to Clifford+Rz.  This warning may be replaced by an error or by a synthesis pass in the future.")

        best_circuit.unfold_all()


        # run 1st pass minimization
        #circuit.instantiate(target=utry, **self.instantiate_options)
        best_result = circuit
        trial_circuit = circuit.copy()
        for period in [0.5, 0.25]:
            for _ in range(circuit.num_params+1):
                # check for constant circuits
                if trial_circuit.num_params < 1:
                    if trial_circuit.get_unitary().get_distance_from(utry) < self.success_threshold:
                        best_result = trial_circuit
                    else:
                        trial_circuit = best_result.copy()
                    break
                self.instantiate_options["method"] = TwoPassMinimization(pass_2_cost_gen = RelaxedTCountCostGenerator(period=period * np.pi), success_threshold=self.success_threshold)
                relaxedTCount = RelaxedTCountCostGenerator(period=period * np.pi).gen_cost(best_result, utry)
                result = await get_runtime().submit(
                        Circuit.instantiate,
                        trial_circuit,
                        target=utry,
                        **self.instantiate_options
                )
                if result is None:
                    break
                cost_calc = HilbertSchmidtCostGenerator().gen_cost(result, utry)
                #print(f"cost array is {relaxedTCount.get_arr(result.params)}")
                #print(f"cost array length is {len(result.params)}")
                #print(f"distance is {cost_calc(result.params)}")
                #print(f"distance is {result.get_unitary().get_distance_from(utry)}")
                # verify that the new result passes the threshold
                if cost_calc(result.params) < self.success_threshold:
                    best_result = result
                else:
                    trial_circuit = best_result.copy()
                    break

                trial_circuit = best_result.copy()

                param_scores = relaxedTCount.get_arr(trial_circuit.params)
                minval = np.min(param_scores)
                did_round = False
                for cycle, op in trial_circuit.operations_with_cycles():
                    if len(op.params) != 1:
                        continue
                    if relaxedTCount.get_arr(np.array(op.params))[0] == minval:
                        trial_circuit.replace_gate((cycle, op.location[0]), circuit_for_rounded_val(op.params[0], period), op.location)
                        #print(f"Rounded out {op.params[0]} to {repr(circuit_for_rounded_val(op.params[0], period))}")
                        did_round = True
                        break
                if not did_round:
                    print(f"Could not find {minval}")
                    exit(1)
                trial_circuit.unfold_all()
                #print(repr(best_result))






        circuit.become(best_result)
        # bqskit.ir.circuit helper functions to convert between parameter index and gate index
        return
