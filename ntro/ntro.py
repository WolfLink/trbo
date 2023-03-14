"""This module implements the NumericalTReductionPass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import *
from bqskit.runtime import get_runtime

from two_pass_minimization import *
from clift import *

# Numerical T Reduction and Optimization

class NumericalTReductionPass(BasePass):
    def __init__(
            self,
            utry=None,
            success_threshold: float = 1e-10
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
        result = await get_runtime().submit(
                Circuit.instantiate,
                circuit,
                target=utry,
                **self.instantiate_options
        )

        # bqskit.ir.circuit helper functions to convert between parameter index and gate index
        return
