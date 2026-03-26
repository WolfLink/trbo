from sqrt_t_utils import *
from bqskit.compiler import Compiler


def qft(n):
    # this is the qft unitary generator code from qsearch
    root = np.e ** (2j * np.pi / n)
    return np.array(np.fromfunction(lambda x,y: root**(x*y), (n,n)), dtype='complex128') / np.sqrt(n)

test_sqrt_t()
print(f"All tests passed!")
qftsize = 3
try:
    print(f"Loading circuit from qft{qftsize}.qasm...")
    before_circuit = Circuit.from_file(f"qft{qftsize}.qasm")
except:
    print(f"Synthesizing qft{qftsize}...")
    before_circuit = bqskit.compile(qft(2**qftsize), max_synthesis_size=4)
    print("Synthesis complete!")
    before_circuit.save(f"qft{qftsize}.qasm")
    print(f"Circuit saved to qft{qftsize}.qasm")

print("Running TRbO...")
with Compiler() as compiler:
    after_circuit = compiler.compile(before_circuit, MultistartPass(sqrt_t_workflow(), 100, better_min_sqrt_t_count_circuit))
cnots = 0
rz = 0
t = 0
sqrt_t = 0
cliff = 0
other = 0
print(after_circuit.gate_counts)
for gate in after_circuit.gate_counts:
    if gate in [CNOTGate()]:
        cnots += after_circuit.gate_counts[gate]
    if gate in clifford_gates:
        cliff += after_circuit.gate_counts[gate]
    elif gate in rz_gates:
        rz += after_circuit.gate_counts[gate]
    elif gate in t_gates:
        t += after_circuit.gate_counts[gate]
    elif gate.__class__ is SqrtTGate:
        sqrt_t += after_circuit.gate_counts[gate]
    elif gate.__class__ is SqrtTDaggerGate:
        sqrt_t += after_circuit.gate_counts[gate]
    else:
        assert gate not in sqrt_t_gates
        other += after_circuit.gate_counts[gate]
print(f"Rz: {rz}\tsqrtT: {sqrt_t}\tT: {t}\tcliff: {cliff}\tCX: {cnots}")
if other > 0:
    print(f"Unrecognized: {other}")

