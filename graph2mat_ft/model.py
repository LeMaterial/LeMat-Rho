"""Graph2MatModel -- wrapper around Graph2Mat coefficient prediction.

Single-call interface ``coefficients = model(atoms)`` so the
Graph2Mat arm slots into the same evaluation pipeline as ChargE3Net
/ DeepDFT / SALTED.

Stub mode (``ckpt_path=None``) returns deterministic
position-and-species-dependent coefficients. This is what powers
the unit tests and the end-to-end pipeline plumbing tests in D5;
PR zeta-gamma-prime (D6 train-script follow-up) wires in the real
Graph2Mat backbone.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import ase
import numpy as np

from salted_ft.basis import BasisSpec
from salted_ft.projection import reconstruct_grid_from_basis


class Graph2MatModel:
    """Predict atom-centered basis coefficients for a structure.

    Parameters
    ----------
    basis_spec :
        Basis the coefficients are defined against. Must match the
        spec the trained checkpoint was trained on.
    ckpt_path :
        Path to a Graph2Mat checkpoint. If ``None`` (default), the
        model runs in stub mode: deterministic, position-dependent
        fake coefficients useful for testing the surrounding pipeline.
    """

    def __init__(
        self, basis_spec: BasisSpec, ckpt_path: str | Path | None = None
    ) -> None:
        self.basis_spec = basis_spec
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
        self._g2m_model = None  # populated when the real forward lands in D6

    def __call__(self, atoms: ase.Atoms) -> np.ndarray:
        """Predict coefficients for ``atoms``.

        Returns
        -------
        np.ndarray of shape ``(n_atoms, basis_spec.n_coeffs_per_atom)``,
        float64, deterministic, finite.
        """
        if self.ckpt_path is None:
            return self._stub_predict(atoms)
        return self._g2m_predict(atoms)

    def reconstruct_density(
        self, atoms: ase.Atoms, grid_shape: tuple[int, int, int]
    ) -> np.ndarray:
        """Predict coefficients, then reconstruct the real-space density.

        Equivalent to::

            c = model(atoms)
            reconstruct_grid_from_basis(c, atoms, grid_shape, basis_spec)
        """
        coeffs = self(atoms)
        return reconstruct_grid_from_basis(coeffs, atoms, grid_shape, self.basis_spec)

    def _stub_predict(self, atoms: ase.Atoms) -> np.ndarray:
        """Deterministic position-dependent coefficients without Graph2Mat.

        Seeded RNG keyed off positions + numbers + basis spec, so
        same atoms in -> same coefficients out. Output magnitude is
        kept small (factor 1e-3) so reconstructed densities stay in
        the metric-test range.
        """
        n_atoms = len(atoms)
        n_coeffs = self.basis_spec.n_coeffs_per_atom
        positions = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()

        # Hash every byte: int.from_bytes(...[:16]) would discard atoms
        # past index 0 and silently collapse different structures into
        # the same seed.
        digest = hashlib.blake2b(
            positions.astype(np.float64).tobytes()
            + numbers.astype(np.int64).tobytes()
            + str(self.basis_spec).encode("utf-8"),
            digest_size=16,
        ).digest()
        seed_int = int.from_bytes(digest, byteorder="little", signed=False)
        rng = np.random.default_rng(seed_int)
        return rng.standard_normal((n_atoms, n_coeffs), dtype=np.float64) * 1e-3

    def _g2m_predict(self, atoms: ase.Atoms) -> np.ndarray:
        """Real Graph2Mat forward pass. Lands with D6 training driver."""
        raise NotImplementedError(
            "Real Graph2Mat forward pass is deferred to D6. "
            "Construct Graph2MatModel with ckpt_path=None for stub mode."
        )
