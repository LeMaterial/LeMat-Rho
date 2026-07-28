"""TDD tests for the SALTEDModel wrapper (PR gamma).

The wrapper exposes ``__call__(atoms) -> coefficients`` so SALTED-style
predictions plug into the projection / reconstruction layer from PR beta.

Locked contract:

* ``SALTEDModel(basis_spec, ckpt_path=None)`` — construct. When
  ``ckpt_path`` is None the wrapper produces deterministic
  position-dependent stub coefficients (lets us run tests + the
  reconstruction pipeline without a real rholearn checkpoint).

* ``model(atoms)`` returns ``np.ndarray (n_atoms, n_coeffs_per_atom)``,
  float64, finite, deterministic for fixed inputs.

* ``model.reconstruct_density(atoms, grid_shape)`` returns the density
  grid in the same shape ``reconstruct_grid_from_basis`` would have
  produced from the predicted coefficients. Convenience method for the
  VASP comparison pipeline.

* Metric integration: the predicted density grid feeds into
  ``compute_nmape`` / ``compute_rmse`` / ``compute_nrmse`` from
  ``charge3net_ft.train`` and they return finite scalars. Pinned per the
  brief: "Keep the metric calculations identical to our ChargE3Net pipeline."
"""

from __future__ import annotations

import ase
import numpy as np
import torch


def _cubic_atoms(symbols=("Fe",), fractional=((0.0, 0.0, 0.0),), a=4.0):
    cell = np.eye(3) * a
    cart = np.array(fractional) @ cell
    return ase.Atoms(symbols=list(symbols), positions=cart, cell=cell, pbc=True)


class TestSALTEDModelConstruct:
    def test_constructs_with_basis_spec(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        spec = BasisSpec()
        m = SALTEDModel(basis_spec=spec)
        assert m.basis_spec is spec

    def test_default_ckpt_is_none(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        m = SALTEDModel(basis_spec=BasisSpec())
        assert m.ckpt_path is None


class TestSALTEDModelOutputShape:
    def test_single_atom_output_shape(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        spec = BasisSpec()
        m = SALTEDModel(basis_spec=spec)
        coeffs = m(_cubic_atoms())
        assert coeffs.shape == (1, spec.n_coeffs_per_atom)

    def test_multi_atom_output_shape(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        spec = BasisSpec()
        m = SALTEDModel(basis_spec=spec)
        atoms = _cubic_atoms(
            symbols=("Fe", "O", "Fe"),
            fractional=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
        )
        coeffs = m(atoms)
        assert coeffs.shape == (3, spec.n_coeffs_per_atom)

    def test_output_dtype_is_float64(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        m = SALTEDModel(basis_spec=BasisSpec())
        coeffs = m(_cubic_atoms())
        assert coeffs.dtype == np.float64

    def test_output_is_finite(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        m = SALTEDModel(basis_spec=BasisSpec())
        coeffs = m(_cubic_atoms())
        assert np.isfinite(coeffs).all()


class TestSALTEDModelDeterminism:
    def test_same_input_gives_same_output(self):
        """Reproducibility: critical for CI + regression tests."""
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        m = SALTEDModel(basis_spec=BasisSpec())
        atoms = _cubic_atoms(
            symbols=("Fe", "Fe"), fractional=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
        )
        c1 = m(atoms)
        c2 = m(atoms)
        np.testing.assert_array_equal(c1, c2)

    def test_different_positions_give_different_coefficients(self):
        """A degenerate stub that always returned zeros would pass shape
        + determinism but be useless. Require some position-dependent
        variation so downstream tests have signal to work with.
        """
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        m = SALTEDModel(basis_spec=BasisSpec())
        atoms_a = _cubic_atoms(fractional=((0.0, 0.0, 0.0),))
        atoms_b = _cubic_atoms(fractional=((0.5, 0.5, 0.5),))
        c_a = m(atoms_a)
        c_b = m(atoms_b)
        assert not np.allclose(c_a, c_b), (
            "predicted coefficients must depend on atom positions; the stub "
            "appears to return position-independent constants"
        )

    def test_baseline_ckpt_loads_and_predicts(self, tmp_path):
        """Real-mode path: save a D6 baseline ckpt, instantiate SALTEDModel
        with its path, and verify forward returns the expected shape and
        the prediction differs from stub-mode output (so we know the
        ckpt actually drove the result)."""
        import pytest

        pytest.importorskip("torch")
        import torch

        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel
        from salted_ft.train_baseline import SaltedBaselineModel

        spec = BasisSpec()
        torch.manual_seed(42)
        baseline = SaltedBaselineModel(spec)
        ckpt = tmp_path / "salted_baseline.pt"
        torch.save({"basis_spec": spec, "model": baseline.state_dict()}, ckpt)

        atoms = _cubic_atoms(
            symbols=("Fe", "Fe"), fractional=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
        )

        m_stub = SALTEDModel(spec)
        m_loaded = SALTEDModel(spec, ckpt_path=ckpt)

        out_stub = m_stub(atoms)
        out_loaded = m_loaded(atoms)

        assert out_loaded.shape == (2, spec.n_coeffs_per_atom)
        assert not np.allclose(out_loaded, out_stub), (
            "loaded ckpt produced the same output as the stub seed; "
            "the ckpt path likely is not being exercised"
        )

    def test_bad_ckpt_format_raises_clearly(self, tmp_path):
        import pytest

        pytest.importorskip("torch")
        import torch

        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        ckpt = tmp_path / "bad.pt"
        torch.save({"not_a_baseline": "anything"}, ckpt)
        m = SALTEDModel(BasisSpec(), ckpt_path=ckpt)
        atoms = _cubic_atoms()
        with pytest.raises(RuntimeError, match="baseline format"):
            m(atoms)

    def test_perturbing_non_first_atom_changes_coefficients(self):
        """Regression test for the int.from_bytes(seed_bytes[:16], ...)
        bug: with the old seeding, only atom 0's xyz (the first 24
        bytes) contributed to the seed, so perturbing atom 1+ produced
        identical coefficients. The blake2b hash fixes this.
        """
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        m = SALTEDModel(basis_spec=BasisSpec())
        atoms_a = _cubic_atoms(
            symbols=("Fe", "Fe"), fractional=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
        )
        atoms_b = _cubic_atoms(
            symbols=("Fe", "Fe"), fractional=((0.0, 0.0, 0.0), (0.6, 0.5, 0.5))
        )
        c_a = m(atoms_a)
        c_b = m(atoms_b)
        assert not np.array_equal(c_a, c_b), (
            "perturbing atom 1 must change the coefficient output; "
            "if not, the stub seed only uses atom 0's bytes"
        )


class TestSALTEDModelReconstructDensity:
    def test_reconstruct_density_shape(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        m = SALTEDModel(basis_spec=BasisSpec())
        grid = m.reconstruct_density(_cubic_atoms(), (8, 8, 8))
        assert grid.shape == (8, 8, 8)

    def test_reconstruct_density_matches_explicit_path(self):
        """``model.reconstruct_density(atoms, shape)`` must equal calling
        ``model(atoms)`` then ``reconstruct_grid_from_basis(c, ...)``.
        Convenience method is just sugar.
        """
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel
        from salted_ft.projection import reconstruct_grid_from_basis

        spec = BasisSpec()
        atoms = _cubic_atoms(
            symbols=("Fe", "O"), fractional=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
        )
        m = SALTEDModel(basis_spec=spec)
        c = m(atoms)
        expected = reconstruct_grid_from_basis(c, atoms, (8, 8, 8), spec)
        got = m.reconstruct_density(atoms, (8, 8, 8))
        np.testing.assert_array_equal(got, expected)

    def test_reconstruct_density_dtype_and_finite(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        m = SALTEDModel(basis_spec=BasisSpec())
        grid = m.reconstruct_density(_cubic_atoms(), (8, 8, 8))
        assert grid.dtype == np.float64
        assert np.isfinite(grid).all()


class TestMetricIntegration:
    """Predicted density grid feeds the existing ChargE3Net metric functions."""

    def _to_torch_batch(self, grid: np.ndarray) -> torch.Tensor:
        """Flatten a (Nx, Ny, Nz) grid into a (B=1, N_probes) torch tensor.

        ChargE3Net's compute_nmape signature is (preds, targets, num_probes).
        For full-grid evaluation we use B=1 and num_probes=None.
        """
        return torch.from_numpy(grid.astype(np.float32).reshape(1, -1))

    def test_compute_nmape_returns_finite_scalar(self):
        from charge3net_ft.train import compute_nmape
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        atoms = _cubic_atoms(fractional=((0.5, 0.5, 0.5),))
        m = SALTEDModel(basis_spec=BasisSpec())
        preds = self._to_torch_batch(m.reconstruct_density(atoms, (8, 8, 8)))
        # Synthetic target: same shape, non-zero so the NMAPE denominator is positive
        targets = torch.ones_like(preds)
        nmape = compute_nmape(preds, targets, num_probes=None)
        assert nmape.numel() == 1
        assert torch.isfinite(nmape).all()

    def test_compute_rmse_returns_finite_scalar(self):
        from charge3net_ft.train import compute_rmse
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        atoms = _cubic_atoms(fractional=((0.5, 0.5, 0.5),))
        m = SALTEDModel(basis_spec=BasisSpec())
        preds = self._to_torch_batch(m.reconstruct_density(atoms, (8, 8, 8)))
        targets = torch.ones_like(preds)
        rmse = compute_rmse(preds, targets, num_probes=None)
        assert torch.isfinite(rmse).all()

    def test_compute_nrmse_returns_finite_scalar(self):
        from charge3net_ft.train import compute_nrmse
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        atoms = _cubic_atoms(fractional=((0.5, 0.5, 0.5),))
        m = SALTEDModel(basis_spec=BasisSpec())
        preds = self._to_torch_batch(m.reconstruct_density(atoms, (8, 8, 8)))
        targets = torch.ones_like(preds)
        nrmse = compute_nrmse(preds, targets, num_probes=None)
        assert torch.isfinite(nrmse).all()

    def test_perfect_prediction_gives_zero_nmape(self):
        """Sanity check: NMAPE of a tensor against itself is zero."""
        from charge3net_ft.train import compute_nmape
        from salted_ft.basis import BasisSpec
        from salted_ft.model import SALTEDModel

        atoms = _cubic_atoms(fractional=((0.5, 0.5, 0.5),))
        m = SALTEDModel(basis_spec=BasisSpec())
        preds = self._to_torch_batch(m.reconstruct_density(atoms, (8, 8, 8)))
        # Self-similarity: target identical to prediction => zero error.
        nmape = compute_nmape(preds, preds.clone(), num_probes=None)
        assert nmape.item() == 0.0
