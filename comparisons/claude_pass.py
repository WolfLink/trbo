"""BQSKit synthesis pass wrapping the Synthetiq C++ synthesis tool."""
from __future__ import annotations

import asyncio
import glob
import os
import shutil
import subprocess
import tempfile
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from bqskit.compiler.passdata import PassData
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SynthetiqSynthesisPass(SynthesisPass):
    """
    A synthesis pass that uses the Synthetiq C++ tool to synthesize circuits.

    Synthetiq uses simulated annealing to find quantum circuits implementing
    a given unitary specification over finite gate sets (e.g., Clifford+T).

    The target unitary is written to a temporary file, Synthetiq is invoked
    as a subprocess, and the best output circuit is parsed and returned.

    References:
        Paradis, Anouk, et al. "Synthetiq: Fast and Versatile Quantum
        Circuit Synthesis." Proc. ACM Program. Lang. 8, OOPSLA1, 2024.
    """

    def __init__(
        self,
        synthetiq_path: str | None = None,
        gateset_path: str | None = None,
        time_limit: float = 100.0,
        num_circuits: int = 10,
        num_threads: int = 1,
        epsilon: float = 1e-6,
        start_temp: float = 0.1,
        iterations_factor: int = 40,
        enable_permutations: bool = False,
        do_resynth: bool = True,
        selection_strategy: str = 'lowest_cost',
    ) -> None:
        """
        Construct a SynthetiqSynthesisPass.

        Args:
            synthetiq_path: Absolute path to the Synthetiq ``bin/main``
                binary. If None, defaults to ``<project>/synthetiq/bin/main``
                relative to this package.

            gateset_path: Absolute path to a Synthetiq gate set directory.
                If None, uses Synthetiq's default CliffordT gate set.

            time_limit: Maximum seconds for Synthetiq to search.

            num_circuits: Number of circuits for Synthetiq to find before
                stopping.

            num_threads: Number of parallel threads for Synthetiq.

            epsilon: Tolerance for unitary equality checking.

            start_temp: Initial temperature for simulated annealing.

            iterations_factor: MCMC iterations per qubit factor.

            enable_permutations: Whether to allow qubit permutations in
                the search. Disabled by default so the returned circuit
                exactly matches the target unitary. Enable for potentially
                shorter circuits at the cost of permuted qubit ordering.

            do_resynth: Whether to run Synthetiq's resynthesis
                simplification pass on found circuits.

            selection_strategy: How to select the best circuit from
                Synthetiq's output. One of ``'lowest_cost'``,
                ``'lowest_t_count'``, or ``'fewest_gates'``.

        Raises:
            FileNotFoundError: If the Synthetiq binary cannot be found.
            ValueError: If an invalid selection_strategy is provided.
        """
        if synthetiq_path is None:
            # Default: <wrapper_project>/synthetiq/bin/main
            package_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(package_dir)
            synthetiq_path = os.path.join(
                project_dir, 'synthetiq', 'bin', 'main',
            )

        if not os.path.isfile(synthetiq_path):
            raise FileNotFoundError(
                f'Synthetiq binary not found at {synthetiq_path}. '
                f'Build it with: make -C <synthetiq_dir>',
            )

        valid_strategies = ('lowest_cost', 'lowest_t_count', 'fewest_gates')
        if selection_strategy not in valid_strategies:
            raise ValueError(
                f'Invalid selection_strategy {selection_strategy!r}. '
                f'Must be one of {valid_strategies}.',
            )

        self.synthetiq_path = synthetiq_path
        self.gateset_path = gateset_path
        self.time_limit = time_limit
        self.num_circuits = num_circuits
        self.num_threads = num_threads
        self.epsilon = epsilon
        self.start_temp = start_temp
        self.iterations_factor = iterations_factor
        self.enable_permutations = enable_permutations
        self.do_resynth = do_resynth
        self.selection_strategy = selection_strategy

        # Derive Synthetiq install root (parent of bin/)
        self._synthetiq_root = os.path.dirname(
            os.path.dirname(self.synthetiq_path),
        )

    async def synthesize(
        self,
        target: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """Synthesize a unitary into a circuit using Synthetiq."""
        from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

        if not isinstance(target, UnitaryMatrix):
            raise TypeError(
                f'SynthetiqSynthesisPass only supports UnitaryMatrix targets, '
                f'got {type(target).__name__}.',
            )

        if not target.is_qubit_only():
            raise ValueError(
                'SynthetiqSynthesisPass only supports qubit-based unitaries '
                '(radix 2).',
            )

        # Create temp dir inside Synthetiq root so relative paths work
        # (Synthetiq's createOutputFolder doesn't handle absolute paths)
        tmpdir = tempfile.mkdtemp(
            prefix='.synthetiq_tmp_', dir=self._synthetiq_root,
        )
        try:
            # Write input specification
            input_path = os.path.join(tmpdir, 'input.txt')
            input_text = self._unitary_to_synthetiq_input(target)
            with open(input_path, 'w') as f:
                f.write(input_text)

            # Output dir as relative path from Synthetiq root
            rel_tmpdir = os.path.relpath(tmpdir, self._synthetiq_root)
            output_rel = os.path.join(rel_tmpdir, 'output')

            # Build command
            cmd = self._build_command(input_path, output_rel)

            # Run Synthetiq in a thread executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                partial(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self._synthetiq_root,
                ),
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f'Synthetiq exited with code {result.returncode}.\n'
                    f'stderr: {result.stderr}',
                )

            # Select and load best circuit
            output_dir = os.path.join(self._synthetiq_root, output_rel)
            best_qasm_path = self._select_best_qasm(output_dir)
            with open(best_qasm_path, 'r') as f:
                qasm_source = f.read()

            circuit = OPENQASM2Language().decode(qasm_source)
            return circuit

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _build_command(
        self,
        input_path: str,
        output_rel: str,
    ) -> list[str]:
        """Build the Synthetiq CLI command."""
        cmd = [
            self.synthetiq_path,
            input_path,
            '--absolute-input',
            '--absolute-output',
            '--output', output_rel,
            '--time', str(self.time_limit),
            '--circuits', str(self.num_circuits),
            '--threads', str(self.num_threads),
            '--epsilon', str(self.epsilon),
            '--start-temp', str(self.start_temp),
            '--iterations-factor', str(self.iterations_factor),
        ]

        if self.gateset_path is not None:
            cmd.extend([
                '--gate-set', self.gateset_path,
                '--absolute-gates',
            ])

        if not self.enable_permutations:
            cmd.append('--no-perms')

        if not self.do_resynth:
            cmd.append('--no-resynth')

        return cmd

    @staticmethod
    def _reverse_qubit_order(matrix: np.ndarray, n: int) -> np.ndarray:
        """Reverse qubit ordering to convert between BQSKit and Synthetiq.

        BQSKit uses qubit 0 = MSB, Synthetiq uses qubit 0 = LSB.
        This permutes rows and columns by bit-reversing the indices.
        """
        dim = 2 ** n
        perm = np.zeros(dim, dtype=int)
        for i in range(dim):
            perm[i] = int(bin(i)[2:].zfill(n)[::-1], 2)
        return matrix[np.ix_(perm, perm)]

    @staticmethod
    def _unitary_to_synthetiq_input(utry: UnitaryMatrix) -> str:
        """Convert a UnitaryMatrix to Synthetiq's input text format."""
        num_qudits = utry.num_qudits
        # Reverse qubit ordering: BQSKit (q0=MSB) -> Synthetiq (q0=LSB)
        matrix = SynthetiqSynthesisPass._reverse_qubit_order(
            utry.numpy, num_qudits,
        )
        dim = matrix.shape[0]

        lines: list[str] = []
        lines.append('matrix')
        lines.append(str(num_qudits))

        # Matrix rows: (re,im) (re,im) ...
        for row in matrix:
            entries = []
            for z in row:
                entries.append(f'({z.real:.17g},{z.imag:.17g})')
            lines.append(' '.join(entries))

        # Cover matrix: all ones (fully specified)
        cover_row = ' '.join(['1'] * dim)
        for _ in range(dim):
            lines.append(cover_row)

        return '\n'.join(lines) + '\n'

    def _select_best_qasm(self, output_dir: str) -> str:
        """Select the best QASM file from Synthetiq's output directory."""
        qasm_files = glob.glob(os.path.join(output_dir, '**', '*.qasm'), recursive=True)

        if not qasm_files:
            raise RuntimeError(
                'Synthetiq did not produce any output circuits. '
                'Try increasing time_limit or num_circuits.',
            )

        if len(qasm_files) == 1:
            return qasm_files[0]

        if self.selection_strategy == 'fewest_gates':
            # Need to count lines (gates) in each file
            best_path = None
            best_count = float('inf')
            for path in qasm_files:
                with open(path, 'r') as f:
                    # Count non-header lines (skip OPENQASM, include, qreg,
                    # and empty lines)
                    count = sum(
                        1 for line in f
                        if line.strip()
                        and not line.startswith('OPENQASM')
                        and not line.startswith('include')
                        and not line.startswith('qreg')
                    )
                if count < best_count:
                    best_count = count
                    best_path = path
            return best_path

        # Parse filename metadata for cost-based strategies
        parsed = []
        for path in qasm_files:
            filename = os.path.basename(path)
            name = filename.removesuffix('.qasm')
            parts = name.split('-')
            if len(parts) >= 3:
                try:
                    total_cost = float(parts[0])
                    t_count = int(parts[1])
                    t_depth = int(parts[2])
                    parsed.append((total_cost, t_count, t_depth, path))
                except (ValueError, IndexError):
                    continue

        if not parsed:
            # Fallback: return first file if filenames can't be parsed
            return qasm_files[0]

        if self.selection_strategy == 'lowest_cost':
            parsed.sort(key=lambda x: x[0])
        elif self.selection_strategy == 'lowest_t_count':
            parsed.sort(key=lambda x: (x[1], x[0]))

        return parsed[0][3]
