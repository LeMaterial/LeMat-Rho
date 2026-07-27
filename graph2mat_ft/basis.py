"""Adapter from our uniform ``BasisSpec`` to Graph2Mat's ``PointBasis``.

Graph2Mat ships ``PointBasis`` as the per-species basis description.
For each species, ``PointBasis(type, R, basis, basis_convention)``
carries the cutoff, the per-l radial count, and the spherical-
harmonic convention. Our ``salted_ft.basis.BasisSpec`` is
species-uniform in v1, so the adapter just expands the same spec
into one PointBasis per species.

Graph2Mat's expected ``basis`` argument when given a sequence of
ints: the integer at index ``l`` is the number of radial functions
at angular momentum ``l``. So our ``n_radial=4, max_l=4`` maps to
``basis=[4, 4, 4, 4, 4]`` (4 radials at each of l=0..4). The
``basis_size`` Graph2Mat computes from that = sum_l (2l+1) * n_radial
= 100, matching ``BasisSpec.n_coeffs_per_atom``.
"""

from __future__ import annotations

from collections.abc import Iterable

from graph2mat import PointBasis

from salted_ft.basis import BasisSpec


def point_basis_for_species(symbol: str, basis_spec: BasisSpec) -> PointBasis:
    """Build a Graph2Mat ``PointBasis`` for a single species.

    Parameters
    ----------
    symbol :
        Atomic symbol (``"H"``, ``"Fe"``, etc.) -- becomes ``PointBasis.type``.
    basis_spec :
        The same BasisSpec used by salted_ft. cutoff -> ``R``,
        n_radial -> uniform per-l radial count, max_l -> length of basis list.

    Returns
    -------
    PointBasis with ``basis_size == basis_spec.n_coeffs_per_atom``
    and ``basis_convention == 'spherical'``.
    """
    # Per-l radial counts as a list of ints. List index = angular momentum.
    per_l_radials = [basis_spec.n_radial] * (basis_spec.max_l + 1)
    return PointBasis(
        type=symbol,
        R=float(basis_spec.cutoff),
        basis=per_l_radials,
        basis_convention="spherical",
    )


def basis_table_for_species(
    symbols: Iterable[str], basis_spec: BasisSpec
) -> dict[str, PointBasis]:
    """Build a ``{symbol: PointBasis}`` dict for a list of species.

    Duplicates in the input are collapsed. Downstream Graph2Mat data
    processors (``BasisTableWithEdges``, etc.) take this dict to know
    every basis a structure can have.
    """
    unique = list(dict.fromkeys(symbols))  # preserves order, deduplicates
    return {s: point_basis_for_species(s, basis_spec) for s in unique}
