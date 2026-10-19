"""TDD tests for VASP CHGCAR <-> SALTED basis projection / reconstruction.

These two operations are the DIY bridge layer between VASP plane-wave
densities and the rholearn/SALTED localized-basis world (see the
``plan_salted_graph2mat_basis_choice_may_20_pm.md`` memo for context).

Locked contracts here:

* ``project_chgcar_to_basis(density, atoms, basis_spec)``
    -> ``np.ndarray (n_atoms, n_coeffs_per_atom)`` float64.
    Zero density gives zero coefficients. Linear in the input density.

* ``reconstruct_grid_from_basis(coefficients, atoms, grid_shape, basis_spec)``
    -> ``np.ndarray (Nx, Ny, Nz)`` float64.
    Zero coefficients give a zero grid. Linear in the coefficients.
    A single-atom, l=0, n=0 unit coefficient produces a Gaussian peaked
    at the atom position.

The roundtrip is intentionally NOT pinned to high accuracy in this PR.
A simple orthonormal-approximation projection is enough to land the
contract; a future PR will swap in least-squares solving against the
full basis overlap matrix for tight roundtrip accuracy.
"""

from __future__ import annotations

import ase
import numpy as np


# ---------------------------------------------------------------------------
# Helpers — small synthetic structures so tests stay fast and inspectable.
# ---------------------------------------------------------------------------
def _cubic_atoms(symbols=("Fe",), fractional=((0.0, 0.0, 0.0),), a=4.0):
    """Single-cell ase.Atoms with the requested species/positions in fractional coords."""
    cell = np.eye(3) * a
    cart = np.array(fractional) @ cell
    return ase.Atoms(symbols=list(symbols), positions=cart, cell=cell, pbc=True)


def _zero_grid(shape=(8, 8, 8)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def _random_grid(shape=(8, 8, 8), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# Projection: density grid -> coefficients
# ---------------------------------------------------------------------------
class TestProjectChgcarToBasis:
    def test_output_shape_is_n_atoms_by_n_coeffs(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import project_chgcar_to_basis

        spec = BasisSpec()
        atoms = _cubic_atoms(
            symbols=("Fe", "Fe"), fractional=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
        )
        coeffs = project_chgcar_to_basis(_zero_grid(), atoms, spec)
        assert coeffs.shape == (2, spec.n_coeffs_per_atom)

    def test_zero_density_gives_zero_coefficients(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import project_chgcar_to_basis

        coeffs = project_chgcar_to_basis(_zero_grid(), _cubic_atoms(), BasisSpec())
        np.testing.assert_array_equal(coeffs, 0.0)

    def test_output_dtype_is_float64(self):
        """float64 because we'll feed these to scipy/least-squares downstream."""
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import project_chgcar_to_basis

        coeffs = project_chgcar_to_basis(_random_grid(), _cubic_atoms(), BasisSpec())
        assert coeffs.dtype == np.float64

    def test_linearity_in_density(self):
        """project(alpha * rho) == alpha * project(rho); a basic sanity check
        since both projection and reconstruction must be linear maps.
        """
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import project_chgcar_to_basis

        atoms = _cubic_atoms()
        spec = BasisSpec()
        rho = _random_grid(seed=1)
        c1 = project_chgcar_to_basis(rho, atoms, spec)
        c_scaled = project_chgcar_to_basis(2.5 * rho, atoms, spec)
        np.testing.assert_allclose(c_scaled, 2.5 * c1, rtol=1e-5, atol=1e-8)

    def test_additivity_in_density(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import project_chgcar_to_basis

        atoms = _cubic_atoms()
        spec = BasisSpec()
        rho1 = _random_grid(seed=2)
        rho2 = _random_grid(seed=3)
        c1 = project_chgcar_to_basis(rho1, atoms, spec)
        c2 = project_chgcar_to_basis(rho2, atoms, spec)
        c_sum = project_chgcar_to_basis(rho1 + rho2, atoms, spec)
        np.testing.assert_allclose(c_sum, c1 + c2, rtol=1e-5, atol=1e-8)

    def test_output_is_finite(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import project_chgcar_to_basis

        coeffs = project_chgcar_to_basis(_random_grid(), _cubic_atoms(), BasisSpec())
        assert np.isfinite(coeffs).all()


# ---------------------------------------------------------------------------
# Reconstruction: coefficients -> density grid
# ---------------------------------------------------------------------------
class TestReconstructGridFromBasis:
    def test_output_shape_matches_grid_shape(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import reconstruct_grid_from_basis

        spec = BasisSpec()
        atoms = _cubic_atoms()
        coeffs = np.zeros((1, spec.n_coeffs_per_atom))
        grid = reconstruct_grid_from_basis(coeffs, atoms, (8, 8, 8), spec)
        assert grid.shape == (8, 8, 8)

    def test_zero_coefficients_give_zero_grid(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import reconstruct_grid_from_basis

        spec = BasisSpec()
        atoms = _cubic_atoms()
        coeffs = np.zeros((1, spec.n_coeffs_per_atom))
        grid = reconstruct_grid_from_basis(coeffs, atoms, (8, 8, 8), spec)
        np.testing.assert_array_equal(grid, 0.0)

    def test_output_dtype_is_float64(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import reconstruct_grid_from_basis

        spec = BasisSpec()
        atoms = _cubic_atoms()
        rng = np.random.default_rng(4)
        coeffs = rng.standard_normal((1, spec.n_coeffs_per_atom))
        grid = reconstruct_grid_from_basis(coeffs, atoms, (8, 8, 8), spec)
        assert grid.dtype == np.float64

    def test_linearity_in_coefficients(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import reconstruct_grid_from_basis

        spec = BasisSpec()
        atoms = _cubic_atoms()
        rng = np.random.default_rng(5)
        c = rng.standard_normal((1, spec.n_coeffs_per_atom))
        g1 = reconstruct_grid_from_basis(c, atoms, (8, 8, 8), spec)
        g_scaled = reconstruct_grid_from_basis(3.0 * c, atoms, (8, 8, 8), spec)
        np.testing.assert_allclose(g_scaled, 3.0 * g1, rtol=1e-5, atol=1e-8)

    def test_single_atom_l0_n0_peaks_at_atom_position(self):
        """Unit s-coefficient on the first radial channel: density should peak
        at the atom position (not somewhere else in the cell)."""
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import reconstruct_grid_from_basis

        spec = BasisSpec()
        # Atom at the (0.5, 0.5, 0.5) interior point, away from cell edges.
        atoms = _cubic_atoms(fractional=((0.5, 0.5, 0.5),), a=4.0)
        coeffs = np.zeros((1, spec.n_coeffs_per_atom))
        coeffs[0, 0] = 1.0  # l=0, m=0, n=0 (the most localized s channel)
        grid = reconstruct_grid_from_basis(coeffs, atoms, (16, 16, 16), spec)

        # Peak index in (i, j, k) integer grid should be near the center.
        peak_idx = np.unravel_index(np.argmax(grid), grid.shape)
        center = (8, 8, 8)  # fractional 0.5 on a 16-point grid
        for actual, expected in zip(peak_idx, center, strict=True):
            assert abs(actual - expected) <= 1, (
                f"density peak {peak_idx} is far from atom (expected near {center}); "
                "either the atom-position lookup or the basis evaluation is wrong"
            )

    def test_output_is_finite(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import reconstruct_grid_from_basis

        spec = BasisSpec()
        atoms = _cubic_atoms()
        rng = np.random.default_rng(6)
        coeffs = rng.standard_normal((1, spec.n_coeffs_per_atom))
        grid = reconstruct_grid_from_basis(coeffs, atoms, (8, 8, 8), spec)
        assert np.isfinite(grid).all()


# ---------------------------------------------------------------------------
# Roundtrip: project then reconstruct (and vice versa).
# ---------------------------------------------------------------------------
class TestProjectionReconstructionRoundtrip:
    def test_roundtrip_of_zero_density_is_zero(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import (
            project_chgcar_to_basis,
            reconstruct_grid_from_basis,
        )

        atoms = _cubic_atoms()
        spec = BasisSpec()
        coeffs = project_chgcar_to_basis(_zero_grid(), atoms, spec)
        roundtrip = reconstruct_grid_from_basis(coeffs, atoms, (8, 8, 8), spec)
        np.testing.assert_array_equal(roundtrip, 0.0)

    def test_roundtrip_of_zero_coefficients_is_zero(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.projection import (
            project_chgcar_to_basis,
            reconstruct_grid_from_basis,
        )

        atoms = _cubic_atoms()
        spec = BasisSpec()
        c = np.zeros((1, spec.n_coeffs_per_atom))
        grid = reconstruct_grid_from_basis(c, atoms, (8, 8, 8), spec)
        c_back = project_chgcar_to_basis(grid, atoms, spec)
        np.testing.assert_array_equal(c_back, 0.0)
