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

def no_partitioning(multistarts=64, sanitize=True, utry=None, strict_opt=False, rz_disc=None):
    passes = []
    if utry is not None:
        passes += [SetDataPass("utry", utry)]
        phase_correct = True

    if sanitize:
        passes += sanitize_gateset()

    passes += [AppendGatePass(GlobalPhaseGate()),
               TRbOPass(multistarts=multistarts, strict_opt=strict_opt, rz_discretizations=rz_disc),
               RemoveGatePass(GlobalPhaseGate())]
    return passes

def default(multistarts=64, partition_size=4, sanitize=True, strict_opt=False, rz_disc=None):
    passes = [QuickPartitioner(partition_size), 
              ForEachBlockPass([
                  AppendGatePass(GlobalPhaseGate()),
                  TRbOPass(multistarts=multistarts, strict_opt=strict_opt, rz_discretizations=rz_disc),
                  RemoveGatePass(GlobalPhaseGate()),
                  ]), 
              UnfoldPass()]
    if sanitize:
        passes = sanitize_gateset() + passes
    return passes

def fast():
    # Doesn't prefer Clifford gates over T gates
    # This pass will run quickly and reduce the need to use gridsynth
    # but it won't be able to achieve optimal T-counts on small circuits
    return default(32, 4, rz_disc=[RzAsT()])

def slow():
    # Uses more mutlistarts than default
    # Uses strict_opt which will attempt to convert use Clifford instead of T gates
    # even when there will be some leftover Rz gates (this usually is a large compute
    # cost for a small benefit).
    return default(128, 6, strict_opt=True)
