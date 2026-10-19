"""Per-atom coefficient projection for the Graph2Mat arm (PR zeta-beta).

Path A of the Graph2Mat plan: the regression target is the same
per-atom basis-coefficient vector that SALTED predicts (see
``salted_ft.projection.project_chgcar_to_basis``). Graph2Mat then
acts as a different backbone over the same target.

This module exposes:

* ``pack_coeffs_to_point_labels(coeffs, basis_spec, symbols)`` --
   flatten ``(N_atoms, n_coeffs_per_atom)`` into the atom-major
   concatenation Graph2Mat consumes as per-node targets.

* ``unpack_point_labels_to_coeffs(flat, basis_spec, symbols)`` --
   inverse.

* ``make_basis_configuration(positions, cell, symbols, basis_spec)``
   -- wrap a structure into ``graph2mat.BasisConfiguration`` so it
   can be fed to Graph2Mat's data processor without us reaching
   into graph2mat internals from the training driver.

We do not lift the coefficients into a true density-matrix
representation (that was Path B). v1 has no off-site terms.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from salted_ft.basis import BasisSpec


def pack_coeffs_to_point_labels(
    coeffs: np.ndarray,
    basis_spec: BasisSpec,
    symbols: Sequence[str],
) -> np.ndarray:
    """Flatten per-atom coefficients into Graph2Mat per-node labels.

    Parameters
    ----------
    coeffs :
        ``(N_atoms, n_coeffs_per_atom)`` from ``salted_ft``.
    basis_spec :
        Locks ``n_coeffs_per_atom``; used to validate shape.
    symbols :
        Per-atom species symbols. Length must match ``N_atoms``.

    Returns
    -------
    1D array of length ``N_atoms * n_coeffs_per_atom``, atom-major
    (atom 0's block first, then atom 1, ...).
    """
    if coeffs.shape[1] != basis_spec.n_coeffs_per_atom:
        raise ValueError(
            f"coeffs has {coeffs.shape[1]} channels per atom but BasisSpec "
            f"declares {basis_spec.n_coeffs_per_atom}"
        )
    if coeffs.shape[0] != len(symbols):
        raise ValueError(
            f"coeffs has {coeffs.shape[0]} atoms but got {len(symbols)} symbols"
        )
    # ravel keeps the input dtype; explicit C order is the contract we test
    return coeffs.reshape(-1).copy()


def unpack_point_labels_to_coeffs(
    flat: np.ndarray,
    basis_spec: BasisSpec,
    symbols: Sequence[str],
) -> np.ndarray:
    """Inverse of ``pack_coeffs_to_point_labels``."""
    expected = len(symbols) * basis_spec.n_coeffs_per_atom
    if flat.shape[0] != expected:
        raise ValueError(
            f"flat has length {flat.shape[0]} but expected "
            f"{len(symbols)} atoms x {basis_spec.n_coeffs_per_atom} "
            f"channels = {expected}"
        )
    return flat.reshape(len(symbols), basis_spec.n_coeffs_per_atom).copy()


def make_basis_configuration(
    positions: np.ndarray,
    cell: np.ndarray,
    symbols: Sequence[str],
    basis_spec: BasisSpec,
):
    """Bundle one structure into a Graph2Mat ``BasisConfiguration``.

    The basis list is built once per call from the unique species in
    ``symbols`` so the resulting config carries only the species it
    actually contains (a downstream BasisTableWithEdges may union
    these across the dataset).

    Parameters
    ----------
    positions :
        ``(N_atoms, 3)`` Cartesian atomic positions in Angstroms.
    cell :
        ``(3, 3)`` lattice matrix.
    symbols :
        Per-atom species symbols.
    basis_spec :
        Defines the per-species ``PointBasis`` (uniform across
        species in v1).
    """
    # Lazy-import keeps the module importable without graph2mat installed
    # (the test class importorskips, so this only runs when present).
    from graph2mat import BasisConfiguration

    from graph2mat_ft.basis import basis_table_for_species

    table = basis_table_for_species(symbols, basis_spec)
    basis_list = list(table.values())
    symbol_to_idx = {pb.type: i for i, pb in enumerate(basis_list)}
    point_types = np.array([symbol_to_idx[s] for s in symbols], dtype=np.int64)

    return BasisConfiguration(
        point_types=point_types,
        positions=np.asarray(positions, dtype=np.float64),
        basis=basis_list,
        cell=np.asarray(cell, dtype=np.float64),
        pbc=(True, True, True),
    )
