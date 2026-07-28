"""TDD tests for the Graph2Mat coefficient projection (PR zeta-beta).

Path A of the Graph2Mat arm: we keep the same regression target as
SALTED (per-atom basis coefficient vectors from
``salted_ft.projection``) and only ask Graph2Mat for a different
backbone. So the "projection" here is a layout transform, not a
basis change.

Layout we map between:

* dense ``coeffs[N_atoms, n_coeffs_per_atom]`` -- what
   ``salted_ft.project_chgcar_to_basis`` returns
* flat ``point_labels[N_atoms * n_coeffs_per_atom]`` -- atom-major
   concatenation, the shape Graph2Mat's per-node targets take
   when every node has the same uniform basis

Per-atom blocks are kept *contiguous* and *in input order* so the
flat vector lines up with the graph node order Graph2Mat builds
from the structure.

These tests pin the pack/unpack roundtrip and order contract --
they do not exercise Graph2Mat's matrix machinery (we do not have
off-site coefficients in v1).
"""

from __future__ import annotations

import numpy as np
import pytest


class TestPackCoeffsToPointLabels:
    def test_output_shape(self):
        from graph2mat_ft.projection import pack_coeffs_to_point_labels
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        coeffs = np.zeros((3, spec.n_coeffs_per_atom))
        flat = pack_coeffs_to_point_labels(coeffs, spec, ("Fe", "O", "H"))
        assert flat.shape == (3 * spec.n_coeffs_per_atom,)

    def test_dtype_preserved(self):
        from graph2mat_ft.projection import pack_coeffs_to_point_labels
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        rng = np.random.default_rng(0)
        coeffs = rng.standard_normal((2, spec.n_coeffs_per_atom)).astype(np.float64)
        flat = pack_coeffs_to_point_labels(coeffs, spec, ("Fe", "Fe"))
        assert flat.dtype == np.float64

    def test_atoms_concatenated_in_input_order(self):
        """Per-atom blocks must appear contiguously and in the order of
        the symbols argument, so the flat vector aligns with the graph
        node order Graph2Mat builds from the structure."""
        from graph2mat_ft.projection import pack_coeffs_to_point_labels
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        per_atom = spec.n_coeffs_per_atom
        coeffs = np.zeros((2, per_atom))
        coeffs[0, :] = 1.0
        coeffs[1, :] = 2.0
        flat = pack_coeffs_to_point_labels(coeffs, spec, ("Fe", "O"))
        assert np.allclose(flat[:per_atom], 1.0)
        assert np.allclose(flat[per_atom:], 2.0)

    def test_within_atom_order_preserved(self):
        """Within one atom's block, channels must keep their input order
        (no reordering across the channel axis). This is the
        load-bearing contract for matching what the Graph2Mat model
        head learns to emit."""
        from graph2mat_ft.projection import pack_coeffs_to_point_labels
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        per_atom = spec.n_coeffs_per_atom
        coeffs = np.arange(per_atom, dtype=np.float64).reshape(1, per_atom)
        flat = pack_coeffs_to_point_labels(coeffs, spec, ("Fe",))
        np.testing.assert_array_equal(flat, np.arange(per_atom, dtype=np.float64))

    def test_empty_structure_returns_empty(self):
        from graph2mat_ft.projection import pack_coeffs_to_point_labels
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        flat = pack_coeffs_to_point_labels(
            np.zeros((0, spec.n_coeffs_per_atom)), spec, ()
        )
        assert flat.shape == (0,)

    def test_symbol_length_mismatch_raises(self):
        """N_atoms in coeffs must match len(symbols). Catching this at
        the boundary stops a silent off-by-one from polluting the
        training set."""
        from graph2mat_ft.projection import pack_coeffs_to_point_labels
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        coeffs = np.zeros((2, spec.n_coeffs_per_atom))
        with pytest.raises(ValueError):
            pack_coeffs_to_point_labels(coeffs, spec, ("Fe",))

    def test_wrong_channel_width_raises(self):
        """coeffs.shape[1] must equal spec.n_coeffs_per_atom."""
        from graph2mat_ft.projection import pack_coeffs_to_point_labels
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        coeffs = np.zeros((1, spec.n_coeffs_per_atom + 1))
        with pytest.raises(ValueError):
            pack_coeffs_to_point_labels(coeffs, spec, ("Fe",))


class TestUnpackPointLabelsToCoeffs:
    def test_output_shape(self):
        from graph2mat_ft.projection import unpack_point_labels_to_coeffs
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        flat = np.zeros(2 * spec.n_coeffs_per_atom)
        coeffs = unpack_point_labels_to_coeffs(flat, spec, ("Fe", "O"))
        assert coeffs.shape == (2, spec.n_coeffs_per_atom)

    def test_wrong_length_raises(self):
        from graph2mat_ft.projection import unpack_point_labels_to_coeffs
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        bad = np.zeros(2 * spec.n_coeffs_per_atom + 1)
        with pytest.raises(ValueError):
            unpack_point_labels_to_coeffs(bad, spec, ("Fe", "O"))


class TestRoundtrip:
    def test_roundtrip_single_atom(self):
        from graph2mat_ft.projection import (
            pack_coeffs_to_point_labels,
            unpack_point_labels_to_coeffs,
        )
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        rng = np.random.default_rng(1)
        coeffs = rng.standard_normal((1, spec.n_coeffs_per_atom))
        flat = pack_coeffs_to_point_labels(coeffs, spec, ("Fe",))
        restored = unpack_point_labels_to_coeffs(flat, spec, ("Fe",))
        np.testing.assert_array_equal(restored, coeffs)

    def test_roundtrip_multi_atom_mixed_species(self):
        from graph2mat_ft.projection import (
            pack_coeffs_to_point_labels,
            unpack_point_labels_to_coeffs,
        )
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        rng = np.random.default_rng(2)
        symbols = ("Fe", "O", "Fe", "H", "O")
        coeffs = rng.standard_normal((len(symbols), spec.n_coeffs_per_atom))
        flat = pack_coeffs_to_point_labels(coeffs, spec, symbols)
        restored = unpack_point_labels_to_coeffs(flat, spec, symbols)
        np.testing.assert_array_equal(restored, coeffs)


class TestBasisConfiguration:
    """Bundle structure + symbols + (optional) coefficients into a
    Graph2Mat-ready container. Used by the ZETA-GAMMA training
    driver. Lazy-imports graph2mat so test only runs when the dep is
    installed (it is in our pyproject)."""

    def test_returns_basisconfiguration_instance(self):
        pytest.importorskip("graph2mat")
        from graph2mat import BasisConfiguration

        from graph2mat_ft.projection import make_basis_configuration
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        cell = np.eye(3) * 4.0
        symbols = ("Fe", "O")
        cfg = make_basis_configuration(positions, cell, symbols, spec)
        assert isinstance(cfg, BasisConfiguration)

    def test_point_types_indexes_into_basis(self):
        """point_types[i] must point at the PointBasis whose type
        equals symbols[i]. If this drifts Graph2Mat assigns the wrong
        per-species head to each atom."""
        pytest.importorskip("graph2mat")

        from graph2mat_ft.projection import make_basis_configuration
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        cell = np.eye(3) * 4.0
        symbols = ("Fe", "O")
        cfg = make_basis_configuration(positions, cell, symbols, spec)
        # Graph2Mat resolves point_types as indices into the cfg.basis list
        types_via_basis = [cfg.basis[t].type for t in cfg.point_types]
        assert tuple(types_via_basis) == symbols

    def test_positions_and_cell_round_trip(self):
        pytest.importorskip("graph2mat")

        from graph2mat_ft.projection import make_basis_configuration
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        positions = np.array([[0.1, 0.2, 0.3], [1.5, 1.5, 1.5]])
        cell = np.diag([3.0, 4.0, 5.0])
        cfg = make_basis_configuration(positions, cell, ("Fe", "O"), spec)
        np.testing.assert_allclose(cfg.positions, positions)
        np.testing.assert_allclose(cfg.cell, cell)
