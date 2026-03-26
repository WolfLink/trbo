from bqskit import compile
from bqskit.compiler import CompilationTask, Compiler
from bqskit.passes import SetModelPass, PassGroup, IfThenElsePass, NotPredicate, SinglePhysicalPredicate, GroupSingleQuditGatePass, ForEachBlockPass, ZXZXZDecomposition, UnfoldPass, QuickPartitioner, MultiPhysicalPredicate, QSearchSynthesisPass

from bqskit.ir import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate as CXG
from bqskit.ir.gates import XGate, YGate, ZGate, HGate, TGate, SGate, SXGate, SdgGate
from bqskit.ir.gates import TGate, TdgGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gate import Gate
from bqskit.compiler.machine import MachineModel


from tqdm import tqdm
import numpy as np
from timeit import default_timer as timer

import ntro
from ntro import *

cliff1q = [XGate(), YGate(), ZGate(), HGate(), SGate(), SdgGate(), SXGate()]

gateset = set([CXG(), RZGate(), TGate(), TdgGate()] + cliff1q)

def optimize_qasm(qasmfile):
    # parse the qasm into 
    original_circuit = Circuit.from_file(qasmfile)
    return optimize_circuit(original_circuit)

def check_needs_resynthesis(original_circuit):
    for gate in original_circuit.gate_set:
        if gate not in gateset:
            print(f"Gate {gate} was not in the gateset")
            return True
    return False


def resynthesize_if_necessary(original_circuit):
    needs_resynthesis = check_needs_resynthesis(original_circuit)
    max_qudits = 1
    for gate in original_circuit.gate_set:
        max_qudits = max(gate.num_qudits, max_qudits)

    if not needs_resynthesis:
        return original_circuit, False
    #print(f"Found a gate of size {max_qudits}")
    resynthesis_pass_list = [
            QuickPartitioner(3),
            ForEachBlockPass([
                IfThenElsePass(
                    NotPredicate(MultiPhysicalPredicate()),
                        [QSearchSynthesisPass(),
                        GroupSingleQuditGatePass(),
                        ForEachBlockPass(ZXZXZDecomposition()),
                        UnfoldPass(),
                        NumericalTReductionPass(),
                        RzToTPass()]
                    )
                ])
            ]
    if max_qudits > 3:
        resynthesis_pass_list = [
                QuickPartitioner(5),
                ForEachBlockPass([
                    IfThenElsePass(
                        NotPredicate(MultiPhysicalPredicate()),
                        [LEAPSynthesisPass(),
                        GroupSingleQuditGatePass(),
                        ForEachBlockPass(ZXZXZDecomposition()),
                        UnfoldPass(),
                        NumericalTReductionPass(),
                        RzToTPass()]
                        )
                    ])
                ]
    if max_qudits > 5:
        resynthesis_pass_list = [
                QuickPartitioner(7),
                ForEachBlockPass([
                    IfThenElsePass(
                        NotPredicate(MultiPhysicalPredicate()), [
                        QFASTDecompositionPass(),
                        ForEachBlockPass([
                            [LEAPSynthesisPass(),
                            GroupSingleQuditGatePass(),
                            ForEachBlockPass(ZXZXZDecomposition()),
                            UnfoldPass(),
                            NumericalTReductionPass(),
                            RzToTPass()]
                            ])
                        ])
                    ])
                ]

    resynthesize_if_necessary_pass = PassGroup([
        UnfoldPass(),
        IfThenElsePass(
            NotPredicate(MultiPhysicalPredicate()), resynthesis_pass_list)
        ])
    task = CompilationTask(original_circuit, [resynthesize_if_necessary_pass, UnfoldPass()])
    with Compiler() as compiler:
        start = timer()
        new_circuit = compiler.compile(task)
        time = timer() - start
        print(f"Resynthesis took {time}s")
    return new_circuit, True

def optimize_circuit(original_circuit, partition_size=4, pass1=16, pass2=8, success_threshold=1e-6):
    # resynthesize any mutli-qubit gates that are not CNOT

    # convert any single qubit gates that are not clifford
    rebase_pass = PassGroup([
        UnfoldPass(),
        IfThenElsePass(
            NotPredicate(SinglePhysicalPredicate()),
            [
                GroupSingleQuditGatePass(),
                ForEachBlockPass([
                    IfThenElsePass(
                        NotPredicate(SinglePhysicalPredicate()),
                        ZXZXZDecomposition(),
                    ),
                ]),
                UnfoldPass(),
            ],
        ),
    ])


    # create the list of passes we wish to perform
    task = CompilationTask(original_circuit, [
        SetModelPass(MachineModel(original_circuit.num_qudits, gate_set=gateset)), # IMO SetModelPass should perform a rebase operation, or there should be a rebasepass that will do it
        rebase_pass,
        QuickPartitioner(partition_size),
        ForEachBlockPass([NumericalTReductionPass(success_threshold=success_threshold, first_pass_retries=pass1, second_pass_multistarts=pass2), RzToTPass()]),
        UnfoldPass(),
        ])

    with Compiler() as compiler:
        synthesized_circuit = compiler.compile(task)

    return synthesized_circuit
