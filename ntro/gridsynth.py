# This code is meant to be a wrapper around "gridsynth", also known as "newsynth" from Selinger and Ross.
# A binary of gridsynth/newsynth must be provided.
# You can find downloads of gridsynth/newsynth in both binary and source code form from:
# https://www.mathstat.dal.ca/~selinger/newsynth/#downloading

from os import environ
from subprocess import run
import numpy as np

from bqskit.compiler.basepass import BasePass

from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.h import HGate


def set_gridsynth_binary(binary):
    environ["gridsynth"] = binary

def get_gridsynth_binary():
    try:
        return environ["gridsynth"]
    except:
        return None

def gridsynth(angle, e=1e-10, pi=False, gridsynth_binary=None):
    angle = angle % 2 if pi else angle % (2 * np.pi)
    anglestr = f"pi*{angle}" if pi else f"{angle}"
    if gridsynth_binary is None:
        gridsynth_binary = get_gridsynth_binary()

    resultstr = run([gridsynth_binary, anglestr, "-p", "-e", f"{e}"], capture_output=True, encoding='utf-8').stdout

    str_to_gate = {
            "H" : HGate(),
            "S" : SGate(),
            "T" : TGate(),
            "X" : XGate(),
            }

    circuit = Circuit(1)
    for c in resultstr:
        if c in str_to_gate:
            circuit.append_gate(str_to_gate[c], 0)
    return CircuitGate(circuit)


class GridsynthPass(BasePass):
    def __init__(self, threshold=1e-6, utry=None, gridsynth_binary=None):
        self.threshold = threshold
        self.utry = utry
        self.gridsynth_binary = gridsynth_binary
       
    async def run(self, circuit, data={}):
        if circuit.num_params < 1:
            return
        if self.gridsynth_binary is None and get_gridsynth_binary() is None:
            print("A gridsynth binary must be provided to run GridsynthPass.")
            return
        target = self.utry if self.utry is not None else circuit.get_unitary()

        # as a first step, lets see if we can tune the parameters any better (that will give us more room for gridsynth error)

        cost_func = HilbertSchmidtResidualsGenerator().gen_cost(circuit, target)
        compare_func = HilbertSchmidtCostGenerator().gen_cost(circuit, target)
        print(f"Initial Cost: {compare_func(circuit.params)}")
        result = CeresMinimizer(ftol=5e-16, gtol=1e-15).minimize(cost_func, circuit.params)
        print(f"Optimized Cost: {compare_func(result)}")
        circuit.set_params(result)

        trial_e = np.sqrt(self.threshold-compare_func(result)) / circuit.num_params
        best_circuit = circuit.copy()
        for cycle, op in best_circuit.operations_with_cycles():
            if len(op.params) != 1:
                continue
            best_circuit.replace_gate((cycle, op.location[0]), gridsynth(op.params[0], e=trial_e, gridsynth_binary=self.gridsynth_binary), op.location)
        best_dist = HilbertSchmidtCostGenerator().gen_cost(best_circuit, target)(best_circuit.params)
        iterations = 0
        max_iter = 10
        print(f"Initial Test: {best_dist} using {trial_e} threshold:{self.threshold}")
        if best_dist >= self.threshold:
            while best_dist >= self.threshold and iterations < max_iter:
                trial_e /= 2
                best_circuit = circuit.copy()
                for cycle, op in best_circuit.operations_with_cycles():
                    if len(op.params) != 1:
                        continue
                    best_circuit.replace_gate((cycle, op.location[0]), gridsynth(op.params[0], e=trial_e, gridsynth_binary=self.gridsynth_binary), op.location)
                best_dist = HilbertSchmidtCostGenerator().gen_cost(best_circuit, target)(best_circuit.params)
                print(f"tried {trial_e} got {best_dist}")
                iterations += 1
        else:
            trial_dist = best_dist
            trial_circuit = best_circuit
            while trial_dist < self.threshold and iterations < max_iter:
                trial_e *= 2
                best_circuit = trial_circuit
                best_dist = trial_dist
                trial_circuit = circuit.copy()
                for cycle, op in trial_circuit.operations_with_cycles():
                    if len(op.params) != 1:
                        continue
                    trial_circuit.replace_gate((cycle, op.location[0]), gridsynth(op.params[0], e=trial_e, gridsynth_binary=self.gridsynth_binary), op.location)
                trial_dist = HilbertSchmidtCostGenerator().gen_cost(trial_circuit, target)(trial_circuit.params)
                print(f"tried {trial_e} got {trial_dist}")
                iterations += 1
            trial_e /= 2


        circuit.become(best_circuit)

        circuit.unfold_all()
        print(f"Final Cost: {HilbertSchmidtCostGenerator().gen_cost(circuit, target)(circuit.params)} e={trial_e} iter={iterations}")

