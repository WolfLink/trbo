import numpy as np
from bqskit.compiler import Compiler
from bqskit import Circuit
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
import trbo


# TRbO is a BQSKit Pass, so to start we need a BQSKit Circuit to optimize

toffoli_u = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
], dtype='complex128')
before_circuit = Circuit(3)
before_circuit.append_gate(ConstantUnitaryGate(toffoli_u), [0,1,2])

# We have now made a circuit from the Toffoli unitary.
# Another common way to get a circuit to optimize is to import it from qasm:

# before_circuit = Circuit.from_file(qasm_file_path)


# Now we will run the default TRbO optimization pass on it:
with Compiler() as compiler:
    after_circuit = compiler.compile(before_circuit, trbo.workflows.default())

# Now lets print the gate counts to see how well it did:
print(after_circuit.gate_counts)

# We can also export to qasm:
after_circuit.save("trbo_toffoli.qasm")

