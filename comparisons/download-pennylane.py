import os
import pennylane

data = pennylane.data.load("other", name="op-t-mize")
circuits = data[0].circuits
names = data[0].circuit_names

os.makedirs("pennylane", exist_ok=True)
for i in range(len(circuits)):
    with open(f"pennylane/{names[i]}.qasm", "w") as f:
        f.write(pennylane.to_openqasm(circuits[i]))
