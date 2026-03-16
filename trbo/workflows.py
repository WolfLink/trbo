"""This module contains definitions of common workflows for convenience."""
from bqskit.passes import GroupSingleQuditGatePass, ForEachBlockPass, IfThenElsePass, WidthPredicate, ZXZXZDecomposition, UnfoldPass, NOOPPass, QuickPartitioner, QSearchSynthesisPass
from .clift import clifford_gates, t_gates, rz_gates, GlobalPhaseGate, RzAsT
from .utils import HasGateSetPredicate, AppendGatePass, RemoveGatePass, SetDataPass
from .trbo import TRbOPass


def sanitize_gateset(synthesize_size=3):
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
                [QuickPartitioner(synthesize_size),
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

def no_partitioning(multistarts=32, sanitize=True, phase_correct=False, utry=None, strict_opt=False, rz_disc=None):
    passes = []
    if utry is not None:
        passes += [SetDataPass("utry", utry)]
        phase_correct = True

    if sanitize:
        passes += sanitize_gateset()

    if phase_correct:
        passes += [AppendGatePass(GlobalPhaseGate()),
                   TRbOPass(multistarts=multistarts, strict_opt=strict_opt, rz_discretizations=rz_disc),
                   RemoveGatePass(GlobalPhaseGate())]
    else:
        passes += [TRbOPass(multistarts=multistarts)]
    return passes

def default(multistarts=32, partition_size=4, sanitize=True, phase_correct=True, strict_opt=False, rz_disc=None):
    if phase_correct:
        passes = [QuickPartitioner(partition_size), 
                  ForEachBlockPass([
                      AppendGatePass(GlobalPhaseGate()),
                      TRbOPass(multistarts=multistarts, strict_opt=strict_opt, rz_discretizations=rz_disc),
                      RemoveGatePass(GlobalPhaseGate()),
                      ]), 
                  UnfoldPass()]
    else:
        passes = [QuickPartitioner(partition_size),
                  ForEachBlockPass(TRbOPass(multistarts=multistarts)),
                  UnfoldPass()]
    if sanitize:
        passes = sanitize_gateset() + passes
    return passes

def fast():
    return default(16, 4, phase_correct=False, rz_disc=[RzAsT()])

def slow():
    return default(64, 6, phase_correct=True, strict_opt=True)

def veryslow():
    return default(128, 7, phase_correct=True, strict_opt=True)
