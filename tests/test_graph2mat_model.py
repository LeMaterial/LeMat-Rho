"""TDD tests for ``Graph2MatModel`` (PR zeta-gamma).

Mirrors ``salted_ft.model.SALTEDModel``: a single-call wrapper that
takes an ASE Atoms and returns ``(n_atoms, n_coeffs_per_atom)``
coefficients. In stub mode (``ckpt_path=None``) the coefficients
are deterministic and seeded from positions / numbers / basis_spec.
The real Graph2Mat forward pass lands in D6 and is asserted here
to raise NotImplementedError until then -- so the failure mode is
loud rather than silently returning stub output.
"""

from __future__ import annotations

import ase
import numpy as np
import pytest


def _h2_atoms() -> ase.Atoms:
    return ase.Atoms(
        symbols=("H", "H"),
        positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )


def _feo_atoms() -> ase.Atoms:
    return ase.Atoms(
        symbols=("Fe", "O"),
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )


class TestStubMode:
    def test_constructible_without_ckpt(self):
        from graph2mat_ft.model import Graph2MatModel
        from salted_ft.basis import BasisSpec

        m = Graph2MatModel(BasisSpec())
        assert m.ckpt_path is None

    def test_output_shape(self):
        from graph2mat_ft.model import Graph2MatModel
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        m = Graph2MatModel(spec)
        out = m(_h2_atoms())
        assert out.shape == (2, spec.n_coeffs_per_atom)

    def test_output_dtype(self):
        from graph2mat_ft.model import Graph2MatModel
        from salted_ft.basis import BasisSpec

        m = Graph2MatModel(BasisSpec())
        out = m(_h2_atoms())
        assert out.dtype == np.float64

    def test_output_finite(self):
        from graph2mat_ft.model import Graph2MatModel
        from salted_ft.basis import BasisSpec

        m = Graph2MatModel(BasisSpec())
        out = m(_feo_atoms())
        assert np.isfinite(out).all()

    def test_deterministic_same_input(self):
        """Same atoms in -> same coefficients out. Required for the
        downstream evaluation pipeline to be reproducible."""
        from graph2mat_ft.model import Graph2MatModel
        from salted_ft.basis import BasisSpec

        spec = BasisSpec()
        m = Graph2MatModel(spec)
        out1 = m(_h2_atoms())
        out2 = m(_h2_atoms())
        np.testing.assert_array_equal(out1, out2)

    def test_position_dependent(self):
        """Different positions -> different coefficients. Catches the
        bug where the stub accidentally seeds only on species (which
        would make every Fe2O3 polymorph have identical coeffs)."""
        from graph2mat_ft.model import Graph2MatModel
        from salted_ft.basis import BasisSpec

        m = Graph2MatModel(BasisSpec())
        a = _h2_atoms()
        b = _h2_atoms()
        b.positions[1, 0] += 0.1  # nudge the second H
        out_a = m(a)
        out_b = m(b)
        assert not np.array_equal(out_a, out_b)

    def test_species_dependent(self):
        """Different atomic numbers should change the seed even at
        identical positions."""
        from graph2mat_ft.model import Graph2MatModel
        from salted_ft.basis import BasisSpec

        m = Graph2MatModel(BasisSpec())
        a = _h2_atoms()
        b = _h2_atoms()
        b.numbers[1] = 8  # H -> O
        out_a = m(a)
        out_b = m(b)
        assert not np.array_equal(out_a, out_b)

    def test_small_magnitude(self):
        """Stub coefficients should be small (order 1e-3) so the
        reconstructed densities stay in the regime where downstream
        metric tests run without overflow."""
        from graph2mat_ft.model import Graph2MatModel
        from salted_ft.basis import BasisSpec

        m = Graph2MatModel(BasisSpec())
        out = m(_h2_atoms())
        assert np.max(np.abs(out)) < 1.0


class TestRealMode:
    def test_with_ckpt_raises_until_d6(self):
        """Real Graph2Mat forward is deferred to D6. Until then a real
        ckpt path must fail loudly rather than silently fall back to
        the stub (which would corrupt benchmark results)."""
        from graph2mat_ft.model import Graph2MatModel
        from salted_ft.basis import BasisSpec

        m = Graph2MatModel(BasisSpec(), ckpt_path="/tmp/fake.ckpt")
        with pytest.raises(NotImplementedError):
            m(_h2_atoms())


class TestReconstructDensity:
    """Convenience helper: predict + reconstruct on a VASP-like grid."""

    def test_shape_matches_grid(self):
        from graph2mat_ft.model import Graph2MatModel
        from salted_ft.basis import BasisSpec

        m = Graph2MatModel(BasisSpec())
        grid_shape = (8, 8, 8)
        rho = m.reconstruct_density(_h2_atoms(), grid_shape)
        assert rho.shape == grid_shape
