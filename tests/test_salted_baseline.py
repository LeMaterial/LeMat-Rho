"""TDD tests for ``salted_ft.train_baseline`` (D6 path B).

A pragmatic PyTorch baseline that predicts per-atom basis
coefficients (the same target SALTED projects to). Architecture is
a small SchNet-style invariant message-passing net + linear
readout to ``n_coeffs_per_atom`` channels. Loss is MSE on the
ground-truth coefficient vectors from D2.

Tests cover the model contract and the training-loop sanity
check (loss must decrease over a few steps). Real Adastra runs
validate end-to-end NMAPE on the held-out split.
"""

from __future__ import annotations

from pathlib import Path

import ase
import numpy as np
import pandas as pd
import pytest


def _h2_atoms() -> ase.Atoms:
    return ase.Atoms(
        "HH",
        positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )


def _feo_atoms() -> ase.Atoms:
    return ase.Atoms(
        "FeO",
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )


class TestModelForward:
    def test_output_shape(self):
        pytest.importorskip("torch")
        from salted_ft.basis import BasisSpec
        from salted_ft.train_baseline import SaltedBaselineModel

        m = SaltedBaselineModel(BasisSpec())
        out = m(_h2_atoms())
        assert out.shape == (2, BasisSpec().n_coeffs_per_atom)

    def test_output_finite(self):
        pytest.importorskip("torch")
        import torch

        from salted_ft.basis import BasisSpec
        from salted_ft.train_baseline import SaltedBaselineModel

        m = SaltedBaselineModel(BasisSpec())
        out = m(_feo_atoms())
        assert torch.isfinite(out).all()

    def test_output_dtype_is_float32(self):
        pytest.importorskip("torch")
        import torch

        from salted_ft.basis import BasisSpec
        from salted_ft.train_baseline import SaltedBaselineModel

        m = SaltedBaselineModel(BasisSpec())
        out = m(_h2_atoms())
        assert out.dtype == torch.float32

    def test_deterministic_with_same_seed(self):
        """Same model state + same atoms in -> same coefficients out.
        Required for the eval pipeline to be reproducible."""
        pytest.importorskip("torch")
        import torch

        from salted_ft.basis import BasisSpec
        from salted_ft.train_baseline import SaltedBaselineModel

        torch.manual_seed(0)
        m1 = SaltedBaselineModel(BasisSpec())
        torch.manual_seed(0)
        m2 = SaltedBaselineModel(BasisSpec())
        out1 = m1(_h2_atoms())
        out2 = m2(_h2_atoms())
        torch.testing.assert_close(out1, out2)

    def test_different_species_changes_output(self):
        """Species embedding must carry signal. If H and Fe atoms with
        identical positions give identical outputs the embedding is
        ignored."""
        pytest.importorskip("torch")
        import torch

        from salted_ft.basis import BasisSpec
        from salted_ft.train_baseline import SaltedBaselineModel

        torch.manual_seed(0)
        m = SaltedBaselineModel(BasisSpec())
        a_hh = ase.Atoms(
            "HH", positions=[[0, 0, 0], [2, 0, 0]], cell=np.eye(3) * 5.0, pbc=True
        )
        a_he = ase.Atoms(
            "HHe", positions=[[0, 0, 0], [2, 0, 0]], cell=np.eye(3) * 5.0, pbc=True
        )
        out_hh = m(a_hh)
        out_he = m(a_he)
        assert not torch.allclose(out_hh, out_he)


class TestTrainingStep:
    def test_loss_decreases_after_few_steps(self):
        """Sanity: optimiser can drive the loss down on a tiny dataset.
        Catches obvious wiring bugs (no grads flowing, frozen embedding)."""
        pytest.importorskip("torch")
        import torch

        from salted_ft.basis import BasisSpec
        from salted_ft.train_baseline import SaltedBaselineModel

        torch.manual_seed(0)
        spec = BasisSpec()
        model = SaltedBaselineModel(spec)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        atoms = _feo_atoms()
        target = torch.randn(len(atoms), spec.n_coeffs_per_atom) * 0.1

        # Loss before any training
        with torch.no_grad():
            loss_before = torch.nn.functional.mse_loss(model(atoms), target).item()

        for _ in range(20):
            opt.zero_grad()
            pred = model(atoms)
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            opt.step()

        with torch.no_grad():
            loss_after = torch.nn.functional.mse_loss(model(atoms), target).item()
        assert loss_after < loss_before, (
            f"loss did not decrease: before={loss_before:.6f}, after={loss_after:.6f}"
        )


class TestSaveLoad:
    def test_save_load_preserves_predictions(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        from salted_ft.basis import BasisSpec
        from salted_ft.train_baseline import SaltedBaselineModel

        torch.manual_seed(0)
        spec = BasisSpec()
        m = SaltedBaselineModel(spec)
        atoms = _h2_atoms()
        out_before = m(atoms)

        ckpt = tmp_path / "model.pt"
        torch.save({"basis_spec": spec, "model": m.state_dict()}, ckpt)

        m2 = SaltedBaselineModel(spec)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        m2.load_state_dict(state["model"])
        out_after = m2(atoms)
        torch.testing.assert_close(out_before, out_after)


def _toy_dataset_dirs(tmp_path: Path, basis_spec, n_rows: int = 2):
    """Create matched D2 source + projected parquets in two subdirs.

    The training dataset joins them by ``row_index`` per chunk; the
    file basename matches across the two directories so ``chunk_0000``
    in ``source/`` lines up with ``chunk_0000`` in ``coeffs/``.
    """
    src_dir = tmp_path / "charge3net_data"
    coeffs_dir = tmp_path / "salted_projected_coefficients"
    src_dir.mkdir()
    coeffs_dir.mkdir()
    rng = np.random.default_rng(0)

    src_rows = []
    coeffs_rows = []
    for i in range(n_rows):
        n_atoms = 2
        atomic_numbers = [1, 1]
        positions = [[0.0, 0.0, 0.0], [0.74 + 0.01 * i, 0.0, 0.0]]
        cell = (np.eye(3) * 5.0).tolist()
        src_rows.append(
            {
                "row_index": i,
                "material_id": f"mp-{i}",
                "n_atoms": n_atoms,
                "atomic_numbers": atomic_numbers,
                "cartesian_site_positions": [c for row in positions for c in row],
                "lattice_vectors": [c for row in cell for c in row],
                # Tiny grid so this stays cheap; the projected file is what
                # the training loop actually consumes
                "grid_shape": [4, 4, 4],
                "compressed_charge_density": rng.standard_normal(np.prod((4, 4, 4)))
                .astype(np.float32)
                .tobytes(),
            }
        )
        coeffs_rows.append(
            {
                "row_index": i,
                "material_id": f"mp-{i}",
                "n_atoms": n_atoms,
                "atomic_numbers": atomic_numbers,
                "lattice_vectors": cell,
                "n_electrons": 2.0,
                "grid_shape": [4, 4, 4],
                "coefficients": rng.standard_normal(
                    (n_atoms, basis_spec.n_coeffs_per_atom)
                ).tolist(),
                "basis_set_NMAPE": 5.0,
            }
        )
    pd.DataFrame(src_rows).to_parquet(src_dir / "chunk_0000.parquet")
    pd.DataFrame(coeffs_rows).to_parquet(coeffs_dir / "chunk_0000.parquet")
    return src_dir, coeffs_dir


class TestTrainCLI:
    """Higher-level: ``train`` end-to-end on a synthetic 2-row dataset.

    Validates that the full data path (parquet pair -> dataset ->
    training loop -> ckpt) works without crashing. Real ckpts come
    from running ``submit_salted_baseline_adastra.sh``.
    """

    def test_train_writes_ckpt(self, tmp_path):
        pytest.importorskip("torch")

        from salted_ft.basis import BasisSpec
        from salted_ft.train_baseline import train

        spec = BasisSpec()
        src_dir, coeffs_dir = _toy_dataset_dirs(tmp_path, spec, n_rows=2)
        ckpt = tmp_path / "salted_baseline.pt"
        train(
            source_dir=src_dir,
            coeffs_dir=coeffs_dir,
            output_ckpt=ckpt,
            basis_spec=spec,
            n_epochs=1,
            batch_size=1,
            learning_rate=1e-3,
        )
        assert ckpt.exists()
        import torch

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert "model" in state
