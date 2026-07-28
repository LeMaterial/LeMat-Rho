"""VASP density grid <-> atom-centered Gaussian * Y_lm basis coefficients.

The two operations defined here are the DIY bridge between VASP plane-wave
CHGCAR data and the rholearn/SALTED localized-basis world. See the memo
``plan_salted_graph2mat_basis_choice_may_20_pm.md`` (Phase A) for why we
have to build this layer ourselves.

Math
----

The basis expansion is
::

    rho(r) ~= sum_i sum_n sum_{l,m} c_{i,n,l,m} phi_n(|r - r_i|) Y_lm(rhat)

where ``i`` indexes atoms, ``n`` is the radial channel, ``(l, m)`` are the
real spherical harmonic indices, ``phi_n`` is a Gaussian of width
``sigma_n``, and ``Y_lm`` is a real spherical harmonic.

Projection solves a single global least-squares system: we build the
per-structure design matrix of every basis function evaluated at every
grid point and fit all atoms' coefficients simultaneously with
``np.linalg.lstsq``. This accounts for the strong overlap between our
Gaussians (an earlier per-channel orthonormal approximation overcounted
overlapping contributions and produced ~1000% NMAPE).

Reconstruction is the literal sum on the right-hand side.

Both maps are linear in their input (linearity is a pinned test).

PBC
---

Minimum-image convention via the cell inverse. Each grid point sees each
atom at its closest periodic image. Adequate for cells where 2*cutoff
fits inside the smallest cell vector; for very small cells we'd want
full Ewald-style supercell expansion. Not in scope for PR beta.
"""

from __future__ import annotations

import ase
import numpy as np

from salted_ft.basis import BasisSpec


# ---------------------------------------------------------------------------
# Grid-position generation (matches charge3net's `calculate_grid_pos` plus
# `deepdft_ft.data._calculate_grid_pos` so the three pipelines agree on
# where grid point (i, j, k) lives in space).
# ---------------------------------------------------------------------------
def _grid_positions(grid_shape: tuple[int, int, int], cell: np.ndarray) -> np.ndarray:
    """Cartesian coordinates of every grid point.

    Parameters
    ----------
    grid_shape : (Nx, Ny, Nz)
    cell : (3, 3) lattice matrix with rows as vectors

    Returns
    -------
    (Nx * Ny * Nz, 3) Cartesian coordinates, ``[i, j, k]`` order matching
    ``np.ravel`` of an array of that shape.
    """
    # Silence harmless RuntimeWarnings from intermediate matmul reductions.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Nx, Ny, Nz = grid_shape
        fx = np.arange(Nx, dtype=np.float64) / Nx
        fy = np.arange(Ny, dtype=np.float64) / Ny
        fz = np.arange(Nz, dtype=np.float64) / Nz
        fX, fY, fZ = np.meshgrid(fx, fy, fz, indexing="ij")
        frac = np.stack([fX.ravel(), fY.ravel(), fZ.ravel()], axis=-1)
        return frac @ cell  # (n_grid, 3)


# ---------------------------------------------------------------------------
# Real spherical harmonics. We hand-roll real Y_lm for lmax up to 4
# (covers our default lmax=4) because the alternatives are either heavy
# (e3nn/torch in a pure-numpy module) or complex-valued (scipy.special).
# ---------------------------------------------------------------------------
_SQRT_PI = np.sqrt(np.pi)


def _real_sph_harm(rhat: np.ndarray, lmax: int) -> np.ndarray:
    """Real spherical harmonics on unit vectors, l = 0..lmax inclusive.

    Returns an array of shape ``(..., (lmax + 1) ** 2)`` where the last
    axis is ordered ``[Y_00, Y_1{-1}, Y_10, Y_11, Y_2{-2}, ..., Y_l l]``
    (the standard SOAP / SALTED ordering).

    Parameters
    ----------
    rhat : (..., 3) array
        Unit vectors. Zero-length inputs are handled by the caller.
    lmax :
        Maximum angular momentum, inclusive.
    """
    if lmax > 4:
        raise NotImplementedError(
            f"_real_sph_harm only implements l = 0..4 (lmax={lmax} requested). "
            "Extend or swap in e3nn.o3.spherical_harmonics for higher lmax."
        )
    x, y, z = rhat[..., 0], rhat[..., 1], rhat[..., 2]
    n_lm = (lmax + 1) ** 2
    out = np.empty(rhat.shape[:-1] + (n_lm,), dtype=np.float64)

    # l = 0
    out[..., 0] = 0.5 / _SQRT_PI

    if lmax >= 1:
        # l = 1: Y_1{-1} ~ y, Y_10 ~ z, Y_11 ~ x
        c1 = 0.5 * np.sqrt(3.0 / np.pi)
        out[..., 1] = c1 * y
        out[..., 2] = c1 * z
        out[..., 3] = c1 * x

    if lmax >= 2:
        # l = 2
        c2_xy = 0.5 * np.sqrt(15.0 / np.pi)  # Y_2{-2}, Y_21, Y_2{-1} prefactors
        c2_z2 = 0.25 * np.sqrt(5.0 / np.pi)
        c2_x2y2 = 0.25 * np.sqrt(15.0 / np.pi)
        out[..., 4] = c2_xy * x * y  # Y_2{-2}
        out[..., 5] = c2_xy * y * z  # Y_2{-1}
        out[..., 6] = c2_z2 * (3 * z * z - 1)  # Y_20
        out[..., 7] = c2_xy * x * z  # Y_21
        out[..., 8] = c2_x2y2 * (x * x - y * y)  # Y_22

    if lmax >= 3:
        # l = 3
        c3a = 0.25 * np.sqrt(35.0 / (2.0 * np.pi))
        c3b = 0.5 * np.sqrt(105.0 / np.pi)
        c3c = 0.25 * np.sqrt(21.0 / (2.0 * np.pi))
        c3d = 0.25 * np.sqrt(7.0 / np.pi)
        out[..., 9] = c3a * y * (3 * x * x - y * y)  # Y_3{-3}
        out[..., 10] = c3b * x * y * z  # Y_3{-2}
        out[..., 11] = c3c * y * (5 * z * z - 1)  # Y_3{-1}
        out[..., 12] = c3d * z * (5 * z * z - 3)  # Y_30
        out[..., 13] = c3c * x * (5 * z * z - 1)  # Y_31
        out[..., 14] = 0.25 * np.sqrt(105.0 / np.pi) * z * (x * x - y * y)  # Y_32
        out[..., 15] = c3a * x * (x * x - 3 * y * y)  # Y_33

    if lmax >= 4:
        # l = 4
        c4a = 0.75 * np.sqrt(35.0 / np.pi)
        c4b = 0.75 * np.sqrt(35.0 / (2.0 * np.pi))
        c4c = 0.75 * np.sqrt(5.0 / np.pi)
        c4d = 0.75 * np.sqrt(5.0 / (2.0 * np.pi))
        c4e = 3.0 / 16.0 * np.sqrt(1.0 / np.pi)
        out[..., 16] = c4a * x * y * (x * x - y * y)  # Y_4{-4}
        out[..., 17] = c4b * y * z * (3 * x * x - y * y)  # Y_4{-3}
        out[..., 18] = c4c * x * y * (7 * z * z - 1)  # Y_4{-2}
        out[..., 19] = c4d * y * z * (7 * z * z - 3)  # Y_4{-1}
        out[..., 20] = c4e * (35 * z**4 - 30 * z * z + 3)  # Y_40
        out[..., 21] = c4d * x * z * (7 * z * z - 3)  # Y_41
        out[..., 22] = (
            0.375 * np.sqrt(5.0 / np.pi) * (x * x - y * y) * (7 * z * z - 1)
        )  # Y_42
        out[..., 23] = c4b * x * z * (x * x - 3 * y * y)  # Y_43
        out[..., 24] = (
            0.1875
            * np.sqrt(35.0 / np.pi)
            * (x * x * (x * x - 3 * y * y) - y * y * (3 * x * x - y * y))
        )  # Y_44

    return out


# ---------------------------------------------------------------------------
# Per-atom basis-function evaluation at grid points
# ---------------------------------------------------------------------------
def _eval_basis_at_grid(
    atom_position: np.ndarray,
    grid_positions: np.ndarray,
    cell: np.ndarray,
    basis_spec: BasisSpec,
) -> np.ndarray:
    """Evaluate every basis function centered on ``atom_position`` at every
    grid point, using minimum-image convention.

    Returns ``(n_grid, n_coeffs_per_atom)`` array of basis-function values.
    """
    # The masked points outside the cutoff intentionally produce some
    # 0/0 and large-magnitude intermediates whose results we throw away
    # via ``mask``. Silence the harmless RuntimeWarnings to keep test
    # output readable.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv_cell = np.linalg.inv(cell)
        rel = grid_positions - atom_position[None, :]  # (n_grid, 3)
        # Minimum-image: wrap fractional displacement to [-0.5, 0.5]
        frac_disp = rel @ inv_cell
        frac_disp = frac_disp - np.round(frac_disp)
        rel = frac_disp @ cell  # (n_grid, 3) in Cartesian, wrapped

        r = np.linalg.norm(rel, axis=-1)  # (n_grid,)
        mask = r < basis_spec.cutoff
        r_safe = np.where(r > 0, r, 1.0)
        rhat = rel / r_safe[:, None]

    # Real spherical harmonics, (n_grid, (lmax+1)^2)
    ylm = _real_sph_harm(rhat, basis_spec.max_l)

    n_grid = grid_positions.shape[0]
    n_lm = ylm.shape[-1]
    n_radial = basis_spec.n_radial
    out = np.empty((n_grid, n_radial * n_lm), dtype=np.float64)

    for n_idx, sigma in enumerate(basis_spec.sigma):
        radial = np.exp(-0.5 * (r / sigma) ** 2) * mask  # (n_grid,)
        # block layout: [n=0 lm=0..nlm-1, n=1 lm=0..nlm-1, ...]
        out[:, n_idx * n_lm : (n_idx + 1) * n_lm] = radial[:, None] * ylm

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def project_chgcar_to_basis(
    density_grid: np.ndarray,
    atoms: ase.Atoms,
    basis_spec: BasisSpec,
) -> np.ndarray:
    """Project a real-space density grid onto the atom-centered basis.

    Solves one global least-squares system (``np.linalg.lstsq``) over
    the full design matrix of all atoms' basis functions evaluated at
    every grid point, so overlap between basis functions is handled
    exactly rather than via a per-channel orthonormal approximation.

    Parameters
    ----------
    density_grid : (Nx, Ny, Nz) array
        Real-space density on the grid (CHGCAR-like).
    atoms : ase.Atoms
        Periodic structure. Provides positions and cell.
    basis_spec : BasisSpec
        Basis to project onto.

    Returns
    -------
    (n_atoms, n_coeffs_per_atom) float64 array of coefficients.
    """
    grid_shape = density_grid.shape
    cell = np.asarray(atoms.get_cell())
    grid_pos = _grid_positions(grid_shape, cell)  # (n_grid, 3)
    rho_flat = density_grid.astype(np.float64).ravel()  # (n_grid,)

    n_atoms = len(atoms)
    coeffs = np.zeros((n_atoms, basis_spec.n_coeffs_per_atom), dtype=np.float64)
    positions = atoms.get_positions()

    # Build the full per-structure design matrix B_global of shape
    # (n_grid, n_atoms * n_coeffs_per_atom) and solve a single least-
    # squares system for ALL atoms' coefficients simultaneously. This
    # is the correct way to handle the strong overlap between our
    # Gaussian basis functions (sigma ~ cutoff means heavy overlap).
    #
    # The previous orthonormal-approx (numer/denom per channel)
    # produced ~1000% NMAPE on real LeMat-Rho rows because it
    # overcounted contributions from overlapping basis functions
    # (recorded in D1 sanity check, 2026-05-21).
    n_per_atom = basis_spec.n_coeffs_per_atom
    B_global = np.empty((grid_pos.shape[0], n_atoms * n_per_atom), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i, pos in enumerate(positions):
            B_global[:, i * n_per_atom : (i + 1) * n_per_atom] = _eval_basis_at_grid(
                pos, grid_pos, cell, basis_spec
            )
        # lstsq is overdetermined (n_grid > n_atoms * n_per_atom for our
        # 10x10x10 grids), so the solution is the unique minimum-residual
        # least-squares fit.
        c_flat, *_ = np.linalg.lstsq(B_global, rho_flat, rcond=None)
    coeffs = c_flat.reshape(n_atoms, n_per_atom)

    return coeffs


def reconstruct_grid_from_basis(
    coefficients: np.ndarray,
    atoms: ase.Atoms,
    grid_shape: tuple[int, int, int],
    basis_spec: BasisSpec,
) -> np.ndarray:
    """Reconstruct a density grid from per-atom basis coefficients.

    Just evaluates the basis at every grid point and contracts with the
    coefficients. The reverse of ``project_chgcar_to_basis`` in the
    sense that ``reconstruct(project(rho))`` is the best basis-set
    approximation to ``rho``.

    Parameters
    ----------
    coefficients : (n_atoms, n_coeffs_per_atom) array
    atoms : ase.Atoms
    grid_shape : (Nx, Ny, Nz)
    basis_spec : BasisSpec

    Returns
    -------
    (Nx, Ny, Nz) float64 density grid.
    """
    n_atoms = len(atoms)
    if coefficients.shape != (n_atoms, basis_spec.n_coeffs_per_atom):
        raise ValueError(
            f"coefficients shape {coefficients.shape} mismatches "
            f"({n_atoms}, {basis_spec.n_coeffs_per_atom})"
        )

    cell = np.asarray(atoms.get_cell())
    grid_pos = _grid_positions(grid_shape, cell)
    positions = atoms.get_positions()

    rho_flat = np.zeros(grid_pos.shape[0], dtype=np.float64)
    coefficients = coefficients.astype(np.float64)
    # Same harmless matmul warnings from masked-out grid points as in
    # _eval_basis_at_grid; silence them at the caller too.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i, pos in enumerate(positions):
            B = _eval_basis_at_grid(pos, grid_pos, cell, basis_spec)
            rho_flat += B @ coefficients[i]

    return rho_flat.reshape(grid_shape)
