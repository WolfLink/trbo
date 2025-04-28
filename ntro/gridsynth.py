# This code is meant to be a wrapper around "gridsynth", also known as "newsynth" from Selinger and Ross.
# A binary of gridsynth/newsynth must be provided.
# You can find downloads of gridsynth/newsynth in both binary and source code form from:
# https://www.mathstat.dal.ca/~selinger/newsynth/#downloading

from os import environ
from subprocess import run
import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.ir import Operation

from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.h import HGate
import logging

from .clift import clifford_gates, t_gates, rz_gates

_logger = logging.getLogger(__name__)


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

class GridsynthSweeper:
    def __init__(self, circuit, target, gridsynth_binary):
        self.points = []
        self.ops = []
        self.gridsynth_binary = gridsynth_binary
        self.target = target
        for cycle,op in circuit.operations_with_cycles():
            if isinstance(op.gate, RZGate):
                self.points.append((cycle, op.location[0]))
                self.ops.append((op.params[0], op.location))

    def resynthesize(self, circuit, e):
        trial_circuit = circuit.copy()
        new_gates = [Operation(
            gridsynth(ops[0], e=e, gridsynth_binary=self.gridsynth_binary),
            ops[1],
            [])
            for ops in self.ops
            ]
        trial_circuit.batch_replace(self.points, new_gates)
        distance = trial_circuit.get_unitary().get_unitary().get_distance_from(self.target)
        return trial_circuit, distance

class GridsynthPass(BasePass):
    def __init__(self, threshold=1e-6, utry=None, gridsynth_binary=None, retries=4, preoptimize=False):
        self.threshold = threshold
        self.utry = utry
        self.gridsynth_binary = gridsynth_binary
        if gridsynth_binary is None:
            raise FileNotFoundError("A gridsynth binary must be provided to run GridsynthPass.")
        self.retries = retries
        self.preoptimize = preoptimize
       
    async def run(self, circuit, data={}):
        if circuit.num_params < 1:
            return

        preoptimize = self.preoptimize
        target_type = None
        if "utry" in data:
            target = data["utry"]
            target_type = "data"
        elif self.utry is not None:
            target = self.utry
            target_type = "self"
        else:
            target = circuit.get_unitary()
            preoptimize = False
            target_type = "getu"


        if "adjusted_threshold" in data:
            threshold = data["adjusted_threshold"]
        else:
            threshold = self.threshold
        log_params = circuit.params
        # as a first step, lets see if we can tune the parameters any better (that will give us more room for gridsynth error)
        
        if preoptimize:
            cost_func = HilbertSchmidtResidualsGenerator().gen_cost(circuit, target)
            result = CeresMinimizer(ftol=1e-17, gtol=1e-17).minimize(cost_func, circuit.params)
            if circuit.get_unitary(result).get_distance_from(target) < circuit.get_unitary().get_distance_from(target):
                circuit.set_params(result)

        d = circuit.get_unitary().get_distance_from(target)
        if d >= threshold:
            print(f"Gridsynth failed because the initial circuit is not close enough to the target circuit ({d} > {threshold}) target_type: {target_type
                  }")
            print(f"WTFDIST: {circuit.get_unitary().get_distance_from(circuit.get_unitary())}")
            return

        min_e = (threshold-d) / circuit.num_params
        best_circuit = circuit.copy()
        gridsynth = GridsynthSweeper(circuit, target, self.gridsynth_binary)
        max_iter = 100
        delta = threshold * 0.1

        min_c, min_d = gridsynth.resynthesize(circuit, min_e)
        max_e = min_e
        max_d = min_d

        iterations = 1
        GSLOG = [(min_e, min_d)]
        while min_d >= threshold and iterations < max_iter:
            max_e = min_e
            max_d = min_d
            min_e /= 10
            min_c, min_d = gridsynth.resynthesize(circuit, min_e)
            iterations += 1
            GSLOG.append((min_e, min_d))

        if min_d >= threshold:
            for entry in GSLOG:
                print(f"e: {entry[0]}\t->\td: {entry[1]}")
            print(f"Gridsynth failed because gridsynth could not find a good enough solution even at low e values ({min_d} > {threshold})")
            return
        best_circuit = min_c
        best_dist = min_d


        while max_d < threshold and iterations < max_iter:
            max_e *= 10
            _, max_d = gridsynth.resynthesize(circuit, max_e)
            iterations += 1
            GSLOG.append((max_e, max_d))

        while max_d - min_d > delta and max_e - min_e > threshold and iterations < max_iter:
            e = (min_e + max_e) / 2
            c, d = gridsynth.resynthesize(circuit, e)
            r = self.retries
            while d >= threshold and r > 0:
                r -= 1
                c, d = gridsynth.resynthesize(circuit, e)
            GSLOG.append((e, d))
            iterations += 1
            if d < threshold:
                best_circuit = c
                best_dist = d
                min_e = e
                min_d = d
            else:
                max_e = e
                max_d = d

        if best_dist < threshold and best_circuit.num_params == 0:
            if iterations > 90:
                print(f"GRIDSYNTH RESULTS: {min_d} < {threshold} at {min_e} after {iterations}")
            circuit.become(best_circuit)
        else:
            _logger.info(f"Gridsynth failed to find a valid circuit.  This likely indicates a bug in bqskit or ntro.")
            print(f"GRIDSYNTH FAILURE at END: {repr(best_dist)} > {repr(threshold)} after {iterations} and {min_e}")
            for entry in GSLOG:
                print(f"e: {entry[0]}\t->\td: {entry[1]}")
        #T_counts = np.sum([circuit.gate_counts[gate] for gate in circuit.gate_counts if gate in t_gates])
        #Rz_counts = circuit.num_params
        #data["gridsynth_stats"] = {"rz" : Rz_counts, "t" : T_counts, "e" : trial_e, "thresh" : threshold, "d" : best_circuit.get_unitary().get_distance_from(target)}
        #data["subcircuit_data"] = f"keys: {[key for key in data]}"
        #data["subcircuit_data"] = f"Circuit {data['subnumbering']}, d is {HilbertSchmidtCostGenerator().gen_cost(circuit, target)(circuit.params)} T: {T_counts} iter: {iterations} {min_e}/{min_d} < {max_e}/{max_d}"
        #data["subcircuit_data"] = f"Circuit {data['subnumbering']}\t{iterations}\t{best_dist}\t<\t{threshold}\t{min_e}/{min_d}\t<\t{max_e}/{max_d}"
        #data["subcircuit_data"] = f"Circuit {data['subnumbering']}, 1iter: {iter_1} iter: {iterations} {min_e} < {e} < {max_e}, {log_params}"

