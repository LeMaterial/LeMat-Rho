"""SALTEDModel — wrapper around rholearn's basis-coefficient prediction.

The wrapper exposes a single-call interface
``coefficients = model(atoms)`` so the SALTED arm slots into the same
evaluation pipeline as ChargE3Net / DeepDFT: predict, reconstruct on
the VASP FFT grid, compare against the converged density via NMAPE
and friends.

When constructed with ``ckpt_path=None`` the model is in **stub mode**:
it returns deterministic, position-dependent coefficients without
requiring a trained rholearn checkpoint. This is what powers the
unit tests and the end-to-end pipeline plumbing tests during PR
gamma; PR gamma-prime (a follow-up) will swap in real rholearn
forward calls.

When ``ckpt_path`` points at a real rholearn checkpoint the model
delegates to rholearn. The rholearn sibling repo is expected at
``../rholearn/`` relative to the LeMat-Rho clone (same pattern as
``charge3net`` for ChargE3Net and ``DeepDFT`` for DeepDFT).
"""

from __future__ import annotations

import sys
from pathlib import Path

import ase
import numpy as np

from salted_ft.basis import BasisSpec
from salted_ft.projection import reconstruct_grid_from_basis

# rholearn sibling-repo discovery follows the same pattern as
# charge3net_ft/model.py and deepdft_ft/runner.py. Resolution is lazy:
# we only insist on the sibling repo when ckpt_path is provided.
_RHOLEARN_ROOT = Path(__file__).resolve().parent.parent.parent / "rholearn"


def _ensure_rholearn_importable() -> None:
    """Make ``rholearn`` importable; only called when ckpt_path is set."""
    if not _RHOLEARN_ROOT.exists():
        raise RuntimeError(
            f"rholearn repo not found at {_RHOLEARN_ROOT}.\n"
            "Clone it with: git clone https://github.com/lab-cosmo/rholearn "
            f"{_RHOLEARN_ROOT}\n"
            "Note: the metatensor.torch.atomistic -> metatomic.torch namespace "
            "patch in rholearn/utils/system.py may also be required."
        )
    if str(_RHOLEARN_ROOT) not in sys.path:
        sys.path.insert(0, str(_RHOLEARN_ROOT))


class SALTEDModel:
    """Predict atom-centered basis coefficients for a structure.

    Parameters
    ----------
    basis_spec :
        The basis the coefficients are defined against. Must match the
        spec the trained checkpoint was trained on.
    ckpt_path :
        Path to a rholearn checkpoint. If ``None`` (default), the model
        runs in stub mode: deterministic, position-dependent fake
        coefficients useful for testing the surrounding pipeline.
    """

    def __init__(
        self, basis_spec: BasisSpec, ckpt_path: str | Path | None = None
    ) -> None:
        self.basis_spec = basis_spec
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
        if self.ckpt_path is not None:
            _ensure_rholearn_importable()
            # Lazy import; defer the heavy load to inference call sites.
            self._rholearn_model = None
        else:
            self._rholearn_model = None

    def __call__(self, atoms: ase.Atoms) -> np.ndarray:
        """Predict coefficients for ``atoms``.

        Returns
        -------
        np.ndarray of shape ``(n_atoms, basis_spec.n_coeffs_per_atom)``,
        float64, deterministic, finite.
        """
        if self.ckpt_path is None:
            return self._stub_predict(atoms)
        return self._rholearn_predict(atoms)

    def reconstruct_density(
        self, atoms: ase.Atoms, grid_shape: tuple[int, int, int]
    ) -> np.ndarray:
        """Predict coefficients, then reconstruct the real-space density.

        Equivalent to::

            c = model(atoms)
            reconstruct_grid_from_basis(c, atoms, grid_shape, basis_spec)

        Provided as a convenience for the VASP comparison pipeline,
        which always wants the grid form.
        """
        coeffs = self(atoms)
        return reconstruct_grid_from_basis(coeffs, atoms, grid_shape, self.basis_spec)

    # ------------------------------------------------------------------
    # Implementations
    # ------------------------------------------------------------------
    def _stub_predict(self, atoms: ase.Atoms) -> np.ndarray:
        """Deterministic position-dependent coefficients without rholearn.

        Recipe: seed a NumPy random generator with a hash of the atomic
        positions, atomic numbers, and basis spec. Same atoms in -> same
        coefficients out. Different atom positions -> different coefficients.

        The numbers are small (order 1e-3) so reconstructed densities
        don't blow up the metric ranges in downstream tests.
        """
        n_atoms = len(atoms)
        n_coeffs = self.basis_spec.n_coeffs_per_atom
        positions = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()

        # Build a deterministic seed from the inputs. NumPy's
        # SeedSequence handles arbitrary-length input cleanly.
        seed_bytes = (
            positions.astype(np.float64).tobytes()
            + numbers.astype(np.int64).tobytes()
            + str(self.basis_spec).encode("utf-8")
        )
        seed_int = int.from_bytes(seed_bytes[:16], byteorder="little", signed=False)
        rng = np.random.default_rng(seed_int)
        return rng.standard_normal((n_atoms, n_coeffs), dtype=np.float64) * 1e-3

    def _rholearn_predict(self, atoms: ase.Atoms) -> np.ndarray:
        """Real rholearn forward pass. Lands in PR gamma-prime."""
        raise NotImplementedError(
            "Real rholearn forward pass is deferred to PR gamma-prime. "
            "Construct SALTEDModel with ckpt_path=None for stub mode."
        )
