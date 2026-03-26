from bqskit.compiler.basepass import BasePass
from bqskit.ir.region import CircuitRegion

class GroupGatePass(BasePass):
    """
    The GroupSingleQuditGatePass Pass.

    This pass groups together consecutive single-qudit gates.
    """

    async def run(self, circuit, data) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Go through each qudit individually
        for q in range(circuit.num_qudits):

            regions = []
            region_start = None

            for c in range(circuit.num_cycles):
                if circuit.is_point_idle((c, q)):
                    continue

                op = circuit[c, q]
                region = CircuitRegion({q: (c, c)})
                regions.append(region)

            for region in reversed(regions):
                circuit.fold(region)
