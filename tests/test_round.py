from trbo.clift import *
import numpy as np
from bqskit.ir.gates import RZGate

specific_problem_values = [
        4.678530403675978,
        -0.033858576708710696,
        0.03385857670871165,
        1.5369377500861858,
        0.03550044926502359,
        0.03385857670871071,
        4.746247557093401,
        1.6046549035036073,
        ]

def test_cliff():
    disc = RzAsCliff()
    for _ in range(1000):
        p = np.random.rand() * np.pi * 4 - 2 * np.pi
        U = RZGate().get_unitary([p])
        G = disc.nearest_gate(p).get_unitary()
        d = U.get_distance_from(G)
        for cliff in clifford_gates:
            if cliff.num_qudits > 1 or cliff.num_params > 0:
                continue
            assert d <= U.get_distance_from(cliff.get_unitary()), f"Chose {G} for {p} when {cliff} was a better choice"

    for p in specific_problem_values:
        U = RZGate().get_unitary([p])
        G = disc.nearest_gate(p).get_unitary()
        d = U.get_distance_from(G)
        for cliff in clifford_gates:
            if cliff.num_qudits > 1 or cliff.num_params > 0:
                continue
            assert d <= U.get_distance_from(cliff.get_unitary()), f"Chose {G} for {p} when {cliff} was a better choice"




def test_t():
    disc = RzAsT()
    for _ in range(1000):
        p = np.random.rand() * np.pi * 4 - 2 * np.pi
        U = RZGate().get_unitary([p])
        G = disc.nearest_gate(p).get_unitary()
        d = U.get_distance_from(G)
        for clift in clifford_gates + t_gates:
            if clift.num_qudits > 1 or clift.num_params > 0:
                continue
            assert d <= U.get_distance_from(clift.get_unitary()), f"Chose {G} for {p} when {clift} was a better choice"

    for p in specific_problem_values:
        U = RZGate().get_unitary([p])
        G = disc.nearest_gate(p).get_unitary()
        d = U.get_distance_from(G)
        for clift in clifford_gates + t_gates:
            if clift.num_qudits > 1 or clift.num_params > 0:
                continue
            assert d <= U.get_distance_from(clift.get_unitary()), f"Chose {G} for {p} when {clift} was a better choice"
