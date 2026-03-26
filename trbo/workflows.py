"""This module contains definitions of common workflows for convenience."""
from bqskit.passes import GroupSingleQuditGatePass, ForEachBlockPass, IfThenElsePass, WidthPredicate, ZXZXZDecomposition, UnfoldPass, NOOPPass, QuickPartitioner, LEAPSynthesisPass
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
                    IfThenElsePass(
                        HasGateSetPredicate(clifford_gates + t_gates + rz_gates),
                        NOOPPass(),
                        ZXZXZDecomposition(),
                        ),
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
                         [LEAPSynthesisPass(),
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

def no_partitioning(*args, sanitize=True, utry=None, **kwargs):
    passes = []
    if utry is not None:
        passes += [SetDataPass("utry", utry)]
        phase_correct = True

    if sanitize:
        passes += sanitize_gateset()

    passes += [AppendGatePass(GlobalPhaseGate()),
               TRbOPass(*args, **kwargs),
               RemoveGatePass(GlobalPhaseGate())]
    return passes

def default(*args, partition_size=4, sanitize=True, **kwargs):
    passes = [QuickPartitioner(partition_size), 
              ForEachBlockPass([
                  AppendGatePass(GlobalPhaseGate()),
                  TRbOPass(*args, **kwargs),
                  RemoveGatePass(GlobalPhaseGate()),
                  ]), 
              UnfoldPass()]
    if sanitize:
        passes = sanitize_gateset() + passes
    return passes

def fast(*args, **kwargs):
    # Doesn't prefer Clifford gates over T gates
    # This pass will run quickly and reduce the need to use gridsynth
    # but it won't be able to achieve optimal T-counts on small circuits
    return default(32, *args, partition_size=4, rz_disc=[RzAsT()], **kwargs)

def slow(*args, **kwargs):
    # Uses more mutlistarts than default
    # Uses strict_opt which will attempt to convert use Clifford instead of T gates
    # even when there will be some leftover Rz gates (this usually is a large compute
    # cost for a small benefit).
    return default(128, *args, partition_size=6, strict_opt=True, **kwargs)
