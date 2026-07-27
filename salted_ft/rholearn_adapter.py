"""SALTED -> rholearn data-format adapter.

rholearn's training loop consumes basis-coefficient vectors in
metatensor ``TensorMap`` format, with a specific flat-vector layout
that differs from our internal one:

================== ===================================================
Our layout         atom (outer) -> n (radial) -> lambda -> mu
                   (this is what ``project_chgcar_to_basis`` returns)
rholearn layout    atom (outer) -> lambda -> n (radial) -> mu
                   (see ``rholearn/utils/convert.py::_get_flat_index``)
================== ===================================================

This module provides three things:

1. ``build_lmax_nmax(basis_spec, species)`` -- our uniform BasisSpec
   expanded into rholearn's per-species ``lmax`` / ``nmax`` dicts.
2. ``dense_to_rholearn_flat`` / ``rholearn_flat_to_dense`` -- the
   permutation between the two layouts, ndarray <-> ndarray. Roundtrip
   is exact and pinned by tests.
3. ``dense_to_tensormap`` -- the full path that calls rholearn's
   ``convert.coeff_vector_ndarray_to_tensormap`` to produce a
   ``metatensor.TensorMap``. Lazy-imports rholearn / metatensor.

The permutation is the load-bearing piece. Get it wrong and rholearn
trains on misordered data; the value at index k of the flat vector
no longer corresponds to the (lambda, n, mu) channel rholearn thinks
it does.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from salted_ft.basis import BasisSpec

# Path setup for lazy rholearn import. Same pattern as
# charge3net_ft/model.py and deepdft_ft/runner.py.
_RHOLEARN_ROOT = Path(__file__).resolve().parent.parent.parent / "rholearn"


def _ensure_rholearn_importable() -> None:
    if not _RHOLEARN_ROOT.exists():
        raise RuntimeError(
            f"rholearn repo not found at {_RHOLEARN_ROOT}.\n"
            "Clone it with: git clone https://github.com/lab-cosmo/rholearn "
            f"{_RHOLEARN_ROOT}"
        )
    if str(_RHOLEARN_ROOT) not in sys.path:
        sys.path.insert(0, str(_RHOLEARN_ROOT))


# ---------------------------------------------------------------------------
# Basis spec dict builder
# ---------------------------------------------------------------------------
def build_lmax_nmax(
    basis_spec: BasisSpec, species: Iterable[str]
) -> tuple[dict[str, int], dict[tuple[str, int], int]]:
    """Expand our uniform BasisSpec into rholearn's per-species dicts.

    Returns
    -------
    lmax : ``{species: max_l}`` for every species in ``species``
    nmax : ``{(species, lambda): n_radial}`` for every (species, lambda)
    """
    species = list(species)
    lmax = {s: basis_spec.max_l for s in species}
    nmax = {
        (s, lam): basis_spec.n_radial
        for s in species
        for lam in range(basis_spec.max_l + 1)
    }
    return lmax, nmax


# ---------------------------------------------------------------------------
# Permutation between our layout and rholearn's
# ---------------------------------------------------------------------------
def _our_to_rholearn_permutation(basis_spec: BasisSpec) -> np.ndarray:
    """Return the index permutation ``p`` such that ``rholearn_flat[k] ==
    our_flat[p[k]]`` for a SINGLE atom.

    Our per-atom layout (length ``n_radial * (max_l + 1) ** 2``):
        for n in 0..n_radial:
            for lambda in 0..max_l:
                for mu in -lambda..+lambda:
                    yield (n, lambda, mu)

    rholearn's per-atom layout (same total length):
        for lambda in 0..max_l:
            for n in 0..n_radial:
                for mu in -lambda..+lambda:
                    yield (lambda, n, mu)

    The permutation is independent of the species (uniform basis).
    """
    n_radial = basis_spec.n_radial
    max_l = basis_spec.max_l

    # Source flat index for (n, lambda, mu) in OUR layout:
    #   n * (max_l + 1) ** 2 + lambda * lambda + (mu + lambda)
    # (the second-and-third pieces together index the standard Y_lm slot)
    def our_idx(n: int, lam: int, mu: int) -> int:
        return n * (max_l + 1) ** 2 + lam * lam + (mu + lam)

    # Build the permutation by walking rholearn's order
    perm = np.empty(n_radial * (max_l + 1) ** 2, dtype=np.int64)
    k = 0
    for lam in range(max_l + 1):
        for n in range(n_radial):
            for mu in range(-lam, lam + 1):
                perm[k] = our_idx(n, lam, mu)
                k += 1
    return perm


def dense_to_rholearn_flat(
    coeffs: np.ndarray,
    basis_spec: BasisSpec,
    symbols: Iterable[str],
) -> np.ndarray:
    """Convert our dense ``(n_atoms, n_coeffs_per_atom)`` coefficients to
    rholearn's flat per-structure vector.

    Output length: ``n_atoms * n_coeffs_per_atom``. ``symbols`` is
    accepted for API symmetry with the inverse and species-aware
    extensions; today the permutation is species-independent because
    our BasisSpec is uniform across species.
    """
    n_atoms = coeffs.shape[0]
    assert coeffs.shape == (n_atoms, basis_spec.n_coeffs_per_atom)
    perm = _our_to_rholearn_permutation(basis_spec)
    # ``coeffs[:, perm]`` reorders each atom's row from our layout to rholearn's
    return coeffs[:, perm].ravel().astype(np.float64)


def rholearn_flat_to_dense(
    flat: np.ndarray,
    basis_spec: BasisSpec,
    symbols: Iterable[str],
) -> np.ndarray:
    """Inverse of ``dense_to_rholearn_flat``. Returns the dense
    ``(n_atoms, n_coeffs_per_atom)`` array.
    """
    n_coeffs = basis_spec.n_coeffs_per_atom
    if flat.size % n_coeffs != 0:
        raise ValueError(
            f"flat vector length {flat.size} is not a multiple of "
            f"n_coeffs_per_atom={n_coeffs}; cannot reshape to (n_atoms, n_coeffs)"
        )
    n_atoms = flat.size // n_coeffs
    reshaped = flat.reshape(n_atoms, n_coeffs).astype(np.float64)
    # Inverse permutation: ``inv[perm[k]] = k``.
    perm = _our_to_rholearn_permutation(basis_spec)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    return reshaped[:, inv]


# ---------------------------------------------------------------------------
# Full TensorMap path
# ---------------------------------------------------------------------------
def dense_to_tensormap(
    coeffs: np.ndarray,
    basis_spec: BasisSpec,
    symbols: Iterable[str],
    positions: np.ndarray,
    cell: np.ndarray,
    structure_idx: int = 0,
):
    """Convert dense coefficients to a ``metatensor.TensorMap`` using
    rholearn's converter.

    Lazy-imports rholearn + metatensor so this module is importable
    without those deps installed (the permutation tests above are
    pure numpy).
    """
    _ensure_rholearn_importable()
    import chemfiles
    from rholearn.utils import convert  # type: ignore[import-not-found]

    flat = dense_to_rholearn_flat(coeffs, basis_spec, symbols)
    lmax, nmax = build_lmax_nmax(basis_spec, set(symbols))

    # Build a chemfiles Frame from the structure (rholearn's converter
    # expects one).
    frame = chemfiles.Frame()
    frame.cell = chemfiles.UnitCell(np.asarray(cell, dtype=np.float64))
    for sym, pos in zip(list(symbols), np.asarray(positions), strict=True):
        atom = chemfiles.Atom(sym)
        frame.add_atom(atom, list(pos))

    return convert.coeff_vector_ndarray_to_tensormap(
        frame,
        coeff_vector=flat,
        lmax=lmax,
        nmax=nmax,
        structure_idx=structure_idx,
        tests=0,
    )
