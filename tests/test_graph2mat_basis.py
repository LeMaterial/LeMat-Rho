"""TDD tests for the Graph2Mat-arm basis adapter (PR zeta-alpha).

Wraps our uniform ``salted_ft.basis.BasisSpec`` into Graph2Mat's
``PointBasis`` per-species objects. Graph2Mat expects one
``PointBasis`` per atomic species, each carrying its own basis-size,
cutoff, and basis-convention. Our BasisSpec is species-uniform in v1
so the adapter expands the same spec across every species in a
structure.

Locked contracts:

* ``point_basis_for_species(symbol, basis_spec)`` -> ``PointBasis``
   ``.type == symbol``, ``.R == basis_spec.cutoff``,
   ``.basis_size == basis_spec.n_coeffs_per_atom``,
   ``.basis_convention == 'spherical'``.

* ``basis_table_for_species(symbols, basis_spec)`` -> dict
   ``{symbol: PointBasis}`` so downstream Graph2Mat data processors
   can look up by atomic symbol.

Graph2Mat 0.0.13 PointBasis API:

  PointBasis(
    type: str | int,
    R: float | ndarray,
    basis: str | Sequence[int | (int, int, int)] = (),
    basis_convention: 'cartesian'|'spherical'|'siesta_spherical'|'qe_spherical' = 'spherical',
  )

  When ``basis`` is a sequence of ints, the int at position ``l``
  is the number of radial functions for that angular momentum.
  So ``basis=[4, 4, 4, 4, 4]`` is 4 radials at each of l=0..4.
  ``basis_size`` is the resulting total number of basis functions
  per atom: sum_l (2l + 1) * n_radial[l].
"""

from __future__ import annotations

import pytest


class TestPointBasisForSpecies:
    def test_returns_pointbasis_instance(self):
        pytest.importorskip("graph2mat")
        from graph2mat import PointBasis

        from graph2mat_ft.basis import point_basis_for_species
        from salted_ft.basis import BasisSpec

        pb = point_basis_for_species("Fe", BasisSpec())
        assert isinstance(pb, PointBasis)

    def test_type_field_is_species_symbol(self):
        pytest.importorskip("graph2mat")
        from graph2mat_ft.basis import point_basis_for_species
        from salted_ft.basis import BasisSpec

        pb = point_basis_for_species("Fe", BasisSpec())
        assert pb.type == "Fe"

    def test_R_matches_basis_spec_cutoff(self):
        """Radial cutoff: must equal our BasisSpec.cutoff so the
        neighbor structure inside Graph2Mat matches charge3net_ft /
        deepdft_ft / salted_ft."""
        pytest.importorskip("graph2mat")
        from graph2mat_ft.basis import point_basis_for_species
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        pb = point_basis_for_species("Fe", spec)
        assert float(pb.R) == pytest.approx(spec.cutoff)

    def test_basis_size_matches_n_coeffs_per_atom(self):
        """The per-atom basis function count Graph2Mat sees must equal
        the per-atom coefficient count salted_ft.projection produces.
        Mismatch means our projected coefficients couldn't be loaded
        into a Graph2Mat density-matrix at all.
        """
        pytest.importorskip("graph2mat")
        from graph2mat_ft.basis import point_basis_for_species
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        pb = point_basis_for_species("Fe", spec)
        assert pb.basis_size == spec.n_coeffs_per_atom

    def test_basis_convention_is_spherical(self):
        """Real spherical harmonics. Cartesian would be the wrong basis
        for our projected coefficients (we use real Y_lm in
        salted_ft.projection._real_sph_harm).
        """
        pytest.importorskip("graph2mat")
        from graph2mat_ft.basis import point_basis_for_species
        from salted_ft.basis import BasisSpec

        pb = point_basis_for_species("Fe", BasisSpec())
        assert pb.basis_convention == "spherical"

    def test_basis_has_one_entry_per_l(self):
        """basis is sanitised by Graph2Mat into a tuple of (n_radial, l, parity)
        triples. We expect one triple per l in 0..max_l, each with the same
        n_radial value matching our uniform spec.
        """
        pytest.importorskip("graph2mat")
        from graph2mat_ft.basis import point_basis_for_species
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        pb = point_basis_for_species("Fe", spec)
        # After PointBasis.__post_init__ sanitisation, .basis is
        # tuple[tuple[int, int, int], ...] with one entry per l value.
        assert len(pb.basis) == spec.max_l + 1
        for entry in pb.basis:
            n_radial, lam, _parity = entry
            assert n_radial == spec.n_radial, (
                f"n_radial mismatch at l={lam}: got {n_radial}, want {spec.n_radial}"
            )

    def test_different_species_give_separate_pointbasis(self):
        """Same spec, different species type field. Sanity check that
        adapter doesn't cache or share across species.
        """
        pytest.importorskip("graph2mat")
        from graph2mat_ft.basis import point_basis_for_species
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        pb_h = point_basis_for_species("H", spec)
        pb_fe = point_basis_for_species("Fe", spec)
        assert pb_h.type == "H" and pb_fe.type == "Fe"
        # But size + cutoff are the same since the spec is uniform
        assert pb_h.basis_size == pb_fe.basis_size
        assert float(pb_h.R) == float(pb_fe.R)


class TestBasisTableForSpecies:
    def test_returns_dict_keyed_by_symbol(self):
        pytest.importorskip("graph2mat")
        from graph2mat_ft.basis import basis_table_for_species
        from salted_ft.basis import BasisSpec

        table = basis_table_for_species(("H", "O", "Fe"), BasisSpec())
        assert set(table) == {"H", "O", "Fe"}

    def test_values_are_pointbasis(self):
        pytest.importorskip("graph2mat")
        from graph2mat import PointBasis

        from graph2mat_ft.basis import basis_table_for_species
        from salted_ft.basis import BasisSpec

        table = basis_table_for_species(("H", "Fe"), BasisSpec())
        for v in table.values():
            assert isinstance(v, PointBasis)

    def test_deduplicates_repeated_species(self):
        pytest.importorskip("graph2mat")
        from graph2mat_ft.basis import basis_table_for_species
        from salted_ft.basis import BasisSpec

        # Repeated species in the input list should collapse to one entry.
        table = basis_table_for_species(("Fe", "Fe", "Fe", "O"), BasisSpec())
        assert set(table) == {"Fe", "O"}
