# T Reduction by Optimization BQSKit Pass

This software implements a technique developed by Marc Davis to use numerical optimization to reduce the number of T gates in a quantum circuit. This is implemented as a BQSKit pass.

## Prerequisites
- [BQSKit](https://github.com/BQSKit/bqskit) `pip install bqskit`
- [gridsynth (optional)](https://www.mathstat.dal.ca/~selinger/newsynth/) `pip install pygridsynth`

## Installation

This is available for Python 3.10+ on Linux, macOS, and Windows.

```sh
git clone https://github.com/WolfLink/trbo
pip install ./trbo
```

## Basic Usage

TRbO provides tools to be used in a `BQSKit` workflow to convert circuits to the Clifford+T+Rz gate set with a minimized Rz and T count.

```python
from bqskit import Circuit
from bqskit.compiler import Compiler
import trbo

before_circuit = Circuit.from_file("input.qasm")
with Compiler() as compiler:
    after_circuit = compiler.compile(before_circuit, trbo.workflows.default())
print(after_circuit.gate_counts)
after_circuit.save("output.qasm")
```

If the resulting circuit after TRbO still has Rz gates, they can be converted using Gridsynth. You can also do this in one step just by adding it to the workflow:

```python
from trbo.gridsynth import GridsynthPass
with Compiler() as compiler:
    after_circuit = compiler.compile(before_circuit, trbo.workflows.default() + [GridsynthPass()])
```
We also provide `trbo.workflows.fast()` which provides lower-quality results at a faster runtime, and `trbo.workflows.slow()` which will more often find optimal results but at a slower runtime. You can further tune TRbO by controlling the following parameters passed to `trbo.workflows.default()`:

- `mutistarts` (Default: `64`) The number of random starting points to for numerical optimization. Larger values may offer better results at the cost of increased runtime.
- `success_threshold` (Default: `1e-6`) The maximum allowed synthesis error using BQSKit's matrix distance (`circuit.get_unitary().get_distance_from(original_circuit.get_unitary())`)
- `strict_opt` (Default: `False`) Small quality increase at a significant performance cost. When the minimum Rz count that TRbO can find is not zero, the number of T gates added by gridsynth is expected to be large compared to the number of remaining T gates that could be removed, so TRbO normally skips the second step of removing T gates. Setting this value to `True` will always perform T gate reduction even when the final Rz count is not zero.
- `partition_size` (Default: `4`) The maximum number of qubits to include in a block when partitioning a large circuit. Larger values may offer better results at the cost of increased runtime.

If you want to try tweaking settings we recommend starting with changin `multistarts`. You can also use `trbo.utils.MultiStartPass` to run the entire `trbo` routine multiple times and get the best result, which is sometimes more effective than just increasing the `multistarts` value.

If you are experienced with BQSKit, the main Pass is `trbo.trbo.TRbOPass`, and we recommend looking at `trbo/workflows.py` for suggestions on how to use it.

For more details on input, output, and working with quantum circuits in BQSKit, see [the BQSKit documentation](https://github.com/BQSKit/bqskit).
