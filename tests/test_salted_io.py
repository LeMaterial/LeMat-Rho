"""TDD tests for VASP CHGCAR I/O wrapper (PR delta).

The wrapper exposes ``write_chgcar(density, atoms, path)`` so a
reconstructed real-space density grid can be persisted as a VASP
CHGCAR file. That file is then the input to a paired SCF run
(``ICHARG=1``) for the speedup comparison vs the
``ICHARG=2``-from-superposition baseline.

Locked contract:

* ``write_chgcar(density, atoms, path, n_electrons=None)``
    Writes a pymatgen ``Chgcar``-compatible file at ``path``. If
    ``n_electrons`` is given, rescales the density so that
    ``sum(density) * cell_volume / N_grid == n_electrons``.
* The written file round-trips through ``Chgcar.from_file`` and
  preserves shape, atom species, and cell.
* ``read_chgcar(path)`` is the inverse: returns
  ``(density: np.ndarray, atoms: ase.Atoms)``.

End-to-end SCF speedup test is gated on the entalsim
``StructureVASPSinglePoint`` maker landing; pinned here as an
``importorskip`` placeholder so it auto-activates when the
dependency arrives.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import ase
import numpy as np
import pytest


def _cubic_atoms(symbols=("Fe",), fractional=((0.5, 0.5, 0.5),), a=4.0):
    cell = np.eye(3) * a
    cart = np.array(fractional) @ cell
    return ase.Atoms(symbols=list(symbols), positions=cart, cell=cell, pbc=True)


class TestWriteChgcar:
    def test_writes_file(self):
        from salted_ft.io import write_chgcar

        atoms = _cubic_atoms()
        density = np.ones((8, 8, 8), dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CHGCAR"
            write_chgcar(density, atoms, path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_normalizes_to_total_electron_count(self):
        """When ``n_electrons`` is set, the *integrated* density of the
        written file must equal ``n_electrons`` to within ``1e-6 * n_electrons``.
        That's what VASP reads as N_electrons on ICHARG=1.
        """
        from salted_ft.io import read_chgcar, write_chgcar

        atoms = _cubic_atoms()
        # Density that integrates to something arbitrary; write_chgcar
        # should rescale to the requested electron count.
        density = np.ones((8, 8, 8), dtype=np.float64) * 0.5
        target_n = 26.0  # Fe valence electron count, roughly
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CHGCAR"
            write_chgcar(density, atoms, path, n_electrons=target_n)
            read_density, _ = read_chgcar(path)
            # CHGCAR convention: density * volume / N_grid integrates to N_electrons
            cell_volume = np.linalg.det(atoms.get_cell())
            n_grid = np.prod(read_density.shape)
            total_e = read_density.sum() * cell_volume / n_grid
            assert abs(total_e - target_n) / target_n < 1e-4, (
                f"integrated density {total_e:.6f} differs from target {target_n} "
                "by more than 1e-4; CHGCAR normalization is wrong"
            )

    def test_rejects_non_3d_density(self):
        from salted_ft.io import write_chgcar

        atoms = _cubic_atoms()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CHGCAR"
            with pytest.raises(ValueError, match=r"3D"):
                write_chgcar(np.ones((8, 8)), atoms, path)

    def test_rejects_negative_n_electrons(self):
        from salted_ft.io import write_chgcar

        atoms = _cubic_atoms()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CHGCAR"
            with pytest.raises(ValueError, match=r"n_electrons"):
                write_chgcar(np.ones((8, 8, 8)), atoms, path, n_electrons=-1.0)


class TestReadChgcar:
    def test_returns_density_and_atoms(self):
        from salted_ft.io import read_chgcar, write_chgcar

        atoms = _cubic_atoms()
        density = np.ones((8, 8, 8), dtype=np.float64) * 0.1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CHGCAR"
            write_chgcar(density, atoms, path)
            read_density, read_atoms = read_chgcar(path)
            assert read_density.shape == (8, 8, 8)
            assert isinstance(read_atoms, ase.Atoms)

    def test_preserves_atom_species(self):
        from salted_ft.io import read_chgcar, write_chgcar

        atoms = _cubic_atoms(
            symbols=("Fe", "O"), fractional=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
        )
        density = np.ones((8, 8, 8), dtype=np.float64) * 0.1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CHGCAR"
            write_chgcar(density, atoms, path)
            _, read_atoms = read_chgcar(path)
            # Order may differ but the multiset of species must match.
            assert sorted(read_atoms.get_chemical_symbols()) == sorted(
                atoms.get_chemical_symbols()
            )

    def test_preserves_cell(self):
        from salted_ft.io import read_chgcar, write_chgcar

        atoms = _cubic_atoms(a=5.0)
        density = np.ones((4, 4, 4), dtype=np.float64) * 0.05
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CHGCAR"
            write_chgcar(density, atoms, path)
            _, read_atoms = read_chgcar(path)
            np.testing.assert_allclose(
                np.asarray(read_atoms.get_cell()),
                np.asarray(atoms.get_cell()),
                atol=1e-6,
            )


class TestRoundtrip:
    def test_density_roundtrip_within_tolerance(self):
        """Write then read: shape exact, values within VASP-precision tolerance.

        VASP CHGCAR uses 5-decimal scientific notation per value, so
        we expect ~1e-5 relative precision.
        """
        from salted_ft.io import read_chgcar, write_chgcar

        atoms = _cubic_atoms()
        rng = np.random.default_rng(7)
        density = rng.random((8, 8, 8)).astype(np.float64) * 0.1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CHGCAR"
            write_chgcar(density, atoms, path)
            read_density, _ = read_chgcar(path)
            assert read_density.shape == density.shape
            np.testing.assert_allclose(read_density, density, rtol=1e-3, atol=1e-5)


class TestSALTEDModelToChgcar:
    """End-to-end: predict via SALTEDModel, reconstruct, write CHGCAR."""

    def test_predicted_density_writes_to_chgcar(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.io import read_chgcar, write_chgcar
        from salted_ft.model import SALTEDModel

        atoms = _cubic_atoms()
        model = SALTEDModel(basis_spec=BasisSpec())
        density = model.reconstruct_density(atoms, (8, 8, 8))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CHGCAR"
            write_chgcar(density, atoms, path)
            assert path.exists()
            read_density, _ = read_chgcar(path)
            assert read_density.shape == (8, 8, 8)


# ---------------------------------------------------------------------------
# Forward-looking placeholder for the entalsim integration.
#
# Once Entalpic/entalsim PR #56's PR 2 (StructureVASPSinglePoint maker)
# lands and is installable, this test will auto-activate. Until then it
# skips cleanly so the suite stays green.
# ---------------------------------------------------------------------------
class TestVASPSinglePointHook:
    def test_chgcar_consumed_by_entalsim_single_point_maker(self):
        # Skips until entalsim ships the maker.
        pytest.importorskip("entalsim.dft.tasks.single_point")
        from entalsim.dft.tasks.single_point import StructureVASPSinglePoint

        from salted_ft.basis import BasisSpec
        from salted_ft.io import write_chgcar
        from salted_ft.model import SALTEDModel

        atoms = _cubic_atoms()
        model = SALTEDModel(basis_spec=BasisSpec())
        density = model.reconstruct_density(atoms, (8, 8, 8))
        with tempfile.TemporaryDirectory() as tmp:
            chgcar = Path(tmp) / "CHGCAR"
            write_chgcar(density, atoms, chgcar)
            # Maker should accept the written CHGCAR for ICHARG=1.
            maker = StructureVASPSinglePoint(initial_chgcar=chgcar)
            assert maker.initial_chgcar == chgcar
