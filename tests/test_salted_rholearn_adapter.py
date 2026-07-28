"""TDD tests for the SALTED -> rholearn data adapter (Phase D3).

rholearn's training loop consumes basis-coefficient vectors in a
specific flat layout (see ``rholearn/utils/convert.py::_get_flat_index``):

    atom (outer) -> o3_lambda -> n (radial, INNER to lambda) -> o3_mu (innermost)

Our ``salted_ft.projection`` layout differs:

    atom (outer) -> n (radial, OUTER to lambda) -> (lambda, mu) packed

The adapter functions tested here move between the two layouts and
produce the ``lmax`` / ``nmax`` dicts rholearn's metatensor converter
needs to know the basis shape.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# rholearn's lmax / nmax dict format (from rholearn/utils/convert.py docstrings)
#
#   lmax = {"H": 1, "C": 2}                          per-species max lambda
#   nmax = {("H", 0): 2, ("H", 1): 3, ("C", 0): 4, ...}  per-species per-lambda n
#
# Our uniform BasisSpec has max_l + n_radial constant across species. The
# adapter expands that into rholearn's per-species dicts so the same basis
# spec works for arbitrary species sets.
# ---------------------------------------------------------------------------


class TestBuildLmaxNmaxDicts:
    """Convert our uniform BasisSpec into rholearn's per-species dicts."""

    def test_lmax_contains_every_species(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.rholearn_adapter import build_lmax_nmax

        lmax, _nmax = build_lmax_nmax(BasisSpec(), species=("H", "O", "Fe"))
        assert set(lmax) == {"H", "O", "Fe"}

    def test_lmax_value_matches_basis_spec(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.rholearn_adapter import build_lmax_nmax

        spec = BasisSpec()
        lmax, _ = build_lmax_nmax(spec, species=("Fe",))
        assert lmax["Fe"] == spec.max_l

    def test_nmax_keyed_by_species_and_lambda(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.rholearn_adapter import build_lmax_nmax

        spec = BasisSpec()
        _, nmax = build_lmax_nmax(spec, species=("H", "Fe"))
        # Both species share the same n_radial at every lambda
        for s in ("H", "Fe"):
            for lam in range(spec.max_l + 1):
                assert nmax[(s, lam)] == spec.n_radial, (
                    f"nmax[({s!r}, {lam})] must be {spec.n_radial}, "
                    f"got {nmax[(s, lam)]}"
                )

    def test_total_per_atom_coefficients_match(self):
        """Sum of ``(2*l + 1) * nmax[(s, l)]`` across l must equal
        ``BasisSpec.n_coeffs_per_atom``. If this drifts the flat vector
        produced by the adapter will be the wrong length.
        """
        from salted_ft.basis import BasisSpec
        from salted_ft.rholearn_adapter import build_lmax_nmax

        spec = BasisSpec()
        lmax, nmax = build_lmax_nmax(spec, species=("Fe",))
        total = sum((2 * lam + 1) * nmax[("Fe", lam)] for lam in range(lmax["Fe"] + 1))
        assert total == spec.n_coeffs_per_atom


# ---------------------------------------------------------------------------
# Reordering: our (atom, n_outer, lm_packed) <-> rholearn (atom, l, n, mu).
# Pure ndarray math, no metatensor required.
# ---------------------------------------------------------------------------


class TestDenseToRholearnFlat:
    """``dense_to_rholearn_flat(coeffs, basis_spec, symbols) -> np.ndarray``."""

    def test_output_length_matches_total_basis(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.rholearn_adapter import dense_to_rholearn_flat

        spec = BasisSpec()
        atoms = ("Fe", "Fe")
        coeffs = np.zeros((2, spec.n_coeffs_per_atom))
        flat = dense_to_rholearn_flat(coeffs, spec, atoms)
        assert flat.shape == (2 * spec.n_coeffs_per_atom,)

    def test_zero_in_gives_zero_out(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.rholearn_adapter import dense_to_rholearn_flat

        spec = BasisSpec()
        flat = dense_to_rholearn_flat(
            np.zeros((1, spec.n_coeffs_per_atom)), spec, ("Fe",)
        )
        np.testing.assert_array_equal(flat, 0.0)

    def test_dtype_preserved(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.rholearn_adapter import dense_to_rholearn_flat

        spec = BasisSpec()
        rng = np.random.default_rng(0)
        coeffs = rng.standard_normal((1, spec.n_coeffs_per_atom)).astype(np.float64)
        flat = dense_to_rholearn_flat(coeffs, spec, ("Fe",))
        assert flat.dtype == np.float64

    def test_concatenates_atoms_in_order(self):
        """Per-atom blocks must appear in input order (atom 0 first, then 1, ...)."""
        from salted_ft.basis import BasisSpec
        from salted_ft.rholearn_adapter import dense_to_rholearn_flat

        spec = BasisSpec()
        # Use distinguishable per-atom values
        coeffs = np.zeros((2, spec.n_coeffs_per_atom))
        coeffs[0, :] = 1.0
        coeffs[1, :] = 2.0
        flat = dense_to_rholearn_flat(coeffs, spec, ("Fe", "Fe"))
        per_atom = spec.n_coeffs_per_atom
        assert np.allclose(flat[:per_atom], 1.0)
        assert np.allclose(flat[per_atom:], 2.0)


class TestRoundtrip:
    """dense -> rholearn-flat -> dense must be exactly the identity."""

    def test_roundtrip_random_single_atom(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.rholearn_adapter import (
            dense_to_rholearn_flat,
            rholearn_flat_to_dense,
        )

        spec = BasisSpec()
        rng = np.random.default_rng(1)
        coeffs = rng.standard_normal((1, spec.n_coeffs_per_atom))
        flat = dense_to_rholearn_flat(coeffs, spec, ("Fe",))
        restored = rholearn_flat_to_dense(flat, spec, ("Fe",))
        np.testing.assert_array_equal(restored, coeffs)

    def test_roundtrip_random_multi_atom(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.rholearn_adapter import (
            dense_to_rholearn_flat,
            rholearn_flat_to_dense,
        )

        spec = BasisSpec()
        rng = np.random.default_rng(2)
        symbols = ("Fe", "O", "Fe", "H")
        coeffs = rng.standard_normal((len(symbols), spec.n_coeffs_per_atom))
        flat = dense_to_rholearn_flat(coeffs, spec, symbols)
        restored = rholearn_flat_to_dense(flat, spec, symbols)
        np.testing.assert_array_equal(restored, coeffs)

    def test_permutation_is_actually_nontrivial(self):
        """The reordering must MOVE values around -- if dense -> flat were
        the identity that would mean we'd silently fed misordered data to
        rholearn. Pinning this catches a future 'simplification' that
        accidentally drops the permutation.
        """
        from salted_ft.basis import BasisSpec
        from salted_ft.rholearn_adapter import dense_to_rholearn_flat

        spec = BasisSpec()
        # Distinguishable per-channel values via arange
        coeffs = np.arange(spec.n_coeffs_per_atom, dtype=np.float64).reshape(
            1, spec.n_coeffs_per_atom
        )
        flat = dense_to_rholearn_flat(coeffs, spec, ("Fe",))
        # rholearn's ordering is atom -> lambda -> n -> mu; ours is
        # atom -> n -> lambda -> mu. So flat[0] is c[atom=0, lambda=0, n=0, mu=0]
        # which in OUR layout is at position [n=0, lm=0] = 0. So flat[0] == 0.
        # But flat[1] is c[atom=0, lambda=1, n=0, mu=-1] which in OUR layout
        # is at [n=0, lm=1] = 1. flat[1] == 1.
        # The DIFFERENT ordering kicks in for flat[3]: rholearn says lambda=1
        # n=1 mu=-1, which in ours is at [n=1, lm=1] = 25, not 3.
        # So flat[3] != coeffs[0, 3] is the load-bearing check.
        assert flat[3] != coeffs[0, 3], (
            "ordering is trivial; the reordering should move values around"
        )


# ---------------------------------------------------------------------------
# Smoke test for the full TensorMap path. Heavier dependency on metatensor
# but the test is short.
# ---------------------------------------------------------------------------


class TestDenseToTensorMap:
    """``dense_to_tensormap`` returns a metatensor TensorMap with the right keys.

    Requires the rholearn sibling repo at ``../rholearn/`` (auto-skips
    when absent). On Adastra (where rholearn IS installed) this test
    activates and exercises the full conversion path.
    """

    def test_tensormap_has_o3_lambda_center_type_keys(self):
        pytest.importorskip("metatensor")
        pytest.importorskip("chemfiles")

        from pathlib import Path

        if not (Path(__file__).resolve().parent.parent.parent / "rholearn").exists():
            pytest.skip("rholearn sibling repo not present; skipping live conversion")

        from salted_ft.basis import BasisSpec
        from salted_ft.rholearn_adapter import dense_to_tensormap

        spec = BasisSpec()
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        cell = np.eye(3) * 4.0
        symbols = ("Fe", "Fe")
        rng = np.random.default_rng(3)
        coeffs = rng.standard_normal((2, spec.n_coeffs_per_atom))
        tmap = dense_to_tensormap(
            coeffs, spec, symbols, positions, cell, structure_idx=0
        )
        # Keys must contain ``o3_lambda`` and ``center_type`` per rholearn's
        # convention (see rholearn/utils/convert.py docstrings).
        names = list(tmap.keys.names)
        assert "o3_lambda" in names
        assert "center_type" in names
