"""This module contains definitions of common workflows for convenience."""
from bqskit.passes import GroupSingleQuditGatePass, ForEachBlockPass, IfThenElsePass, WidthPredicate, ZXZXZDecomposition, UnfoldPass, NOOPPass, QuickPartitioner, QSearchSynthesisPass
from .clift import clifford_gates, t_gates, rz_gates, GlobalPhaseGate
from .utils import HasGateSetPredicate, AppendGatePass, RemoveGatePass, SetDataPass
from .ntro import NumericalTReductionPass


def sanitize_gateset():
    """Converts a circuit to Clifford + T + Rz."""
    return [
            UnfoldPass(),
            GroupSingleQuditGatePass(),
            ForEachBlockPass([
                IfThenElsePass(
                    WidthPredicate(2),
                    ZXZXZDecomposition(),
                    ),
                ]),
            UnfoldPass(),
            IfThenElsePass(
                HasGateSetPredicate(clifford_gates + t_gates + rz_gates),
                NOOPPass(),
                [QuickPartitioner(3),
                 ForEachBlockPass([
                     IfThenElsePass(
                         HasGateSetPredicate(clifford_gates + t_gates + rz_gates),
                         NOOPPass(),
                         [QSearchSynthesisPass(),
                          GroupSingleQuditGatePass(),
                          ForEachBlockPass([
                              IfThenElsePass(
                                  WidthPredicate(2),
                                  ZXZXZDecomposition(),
                                  ),
                              ]),
                          ]),
                         ]),
                 UnfoldPass(),
                 ]),
            ]

def no_partitioning(multistarts=32, sanitize=True, phase_correct=False, utry=None):
    passes = []
    if utry is not None:
        passes += [SetDataPass("utry", utry)]
        phase_correct = True

    if sanitize:
        passes += sanitize_gateset()

    if phase_correct:
        passes += [AppendGatePass(GlobalPhaseGate()),
                   NumericalTReductionPass(multistarts=multistarts),
                   RemoveGatePass(GlobalPhaseGate())]
    else:
        passes += [NumericalTReductionPass(multistarts=multistarts)]
    return passes

def default(multistarts=32, partition_size=4, sanitize=True, phase_correct=False):
    if phase_correct:
        passes = [QuickPartitioner(partition_size), 
                  ForEachBlockPass([
                      AppendGatePass(GlobalPhaseGate()),
                      NumericalTReductionPass(multistarts=multistarts),
                      RemoveGatePass(GlobalPhaseGate()),
                      ]), 
                  UnfoldPass()]
    else:
        passes = [QuickPartitioner(partition_size),
                  ForEachBlockPass(NumericalTReductionPass(multistarts=multistarts)),
                  UnfoldPass()]
    if sanitize:
        passes = sanitize_gateset() + passes
    return passes

def fast():
    return default(16, 4)

def slow():
    return default(64, 6)

def veryslow():
    return default(128, 7)
