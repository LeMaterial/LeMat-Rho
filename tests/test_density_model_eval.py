"""TDD tests for ``scripts/density_model_eval.py`` (D7).

Per-structure density evaluation across the LeMat-Rho arms. This
test exercises the SALTED stub path end-to-end (synthesize a tiny
parquet, run the eval, read back the result) and the structural
contract of the arm dispatcher.

ChargE3Net and DeepDFT grid prediction lands in D7-beta (probe
batching); the eval script must raise NotImplementedError for them
rather than silently fall back to stubs, so a future user does not
get fake metrics on real arms.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import ase
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def eval_module():
    """Import scripts.density_model_eval, adding scripts/ to sys.path."""
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    if "density_model_eval" in sys.modules:
        del sys.modules["density_model_eval"]
    return importlib.import_module("density_model_eval")


def _toy_parquet(tmp_path: Path, n_rows: int = 2) -> Path:
    """Synthesise a tiny LeMat-Rho-shaped parquet for eval tests.

    Layout matches the columns ``salted_ft.project_dataset`` writes
    plus a ``charge_density`` grid and ``grid_shape`` (the eval is
    grid-comparison so we need ground-truth grids)."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_rows):
        grid_shape = (4, 4, 4)
        rows.append(
            {
                "row_index": i,
                "material_id": f"mp-toy-{i}",
                "n_atoms": 2,
                "atomic_numbers": np.array([1, 1], dtype=np.int64),
                "positions": np.array(
                    [[0.0, 0.0, 0.0], [0.74 + 0.01 * i, 0.0, 0.0]], dtype=np.float64
                ).reshape(-1),
                "lattice_vectors": (np.eye(3) * 5.0).reshape(-1),
                "charge_density": rng.standard_normal(np.prod(grid_shape)).astype(
                    np.float64
                ),
                "grid_shape": np.array(grid_shape, dtype=np.int64),
            }
        )
    df = pd.DataFrame(rows)
    out = tmp_path / "toy_test.parquet"
    df.to_parquet(out)
    return out


class TestMetrics:
    def test_nmape_perfect_prediction_is_zero(self, eval_module):
        rho = np.array([1.0, 2.0, 3.0])
        assert eval_module.density_nmape(rho, rho) == pytest.approx(0.0)

    def test_nmape_zero_prediction_against_unit_target(self, eval_module):
        pred = np.zeros(4)
        target = np.ones(4)
        # NMAPE = sum(|0 - 1|) / sum(|1|) * 100 = 4 / 4 * 100 = 100
        assert eval_module.density_nmape(pred, target) == pytest.approx(100.0)

    def test_rmse_perfect_prediction_is_zero(self, eval_module):
        rho = np.array([1.0, 2.0])
        assert eval_module.density_rmse(rho, rho) == pytest.approx(0.0)

    def test_rmse_known(self, eval_module):
        pred = np.array([0.0, 0.0])
        target = np.array([3.0, 4.0])  # MSE = (9+16)/2 = 12.5, RMSE = sqrt(12.5)
        assert eval_module.density_rmse(pred, target) == pytest.approx(np.sqrt(12.5))

    def test_nrmse_perfect_prediction_is_zero(self, eval_module):
        rho = np.array([1.0, 2.0])
        assert eval_module.density_nrmse(rho, rho) == pytest.approx(0.0)

    def test_metrics_handle_3d_grids(self, eval_module):
        """Metrics must work on (Nx, Ny, Nz) arrays, not just flat."""
        rng = np.random.default_rng(1)
        pred = rng.standard_normal((4, 4, 4))
        target = rng.standard_normal((4, 4, 4))
        # Should not error and should be finite
        for fn in (
            eval_module.density_nmape,
            eval_module.density_rmse,
            eval_module.density_nrmse,
        ):
            assert np.isfinite(fn(pred, target))


class TestPredictDensity:
    """Per-arm dispatcher contract."""

    def test_salted_stub_returns_grid_of_correct_shape(self, eval_module):
        from salted_ft.basis import BasisSpec

        atoms = ase.Atoms(
            "HH",
            positions=[[0, 0, 0], [0.74, 0, 0]],
            cell=np.eye(3) * 5.0,
            pbc=True,
        )
        grid_shape = (6, 6, 6)
        rho = eval_module.predict_density(
            "salted", atoms, grid_shape, None, BasisSpec()
        )
        assert rho.shape == grid_shape

    def test_charge3net_with_mock_model_returns_grid(self, eval_module):
        """Charge3Net dispatcher must build the input dict, batch probes,
        and reshape to grid. We mock the network with a callable that
        returns ones at every probe so we can pin the shape contract
        and the reshape order without a real ckpt."""
        pytest.importorskip("torch")
        import torch

        from salted_ft.basis import BasisSpec

        if not (Path(__file__).resolve().parent.parent.parent / "charge3net").exists():
            pytest.skip("charge3net sibling repo not present; integration only")

        atoms = ase.Atoms(
            "HH",
            positions=[[0, 0, 0], [0.74, 0, 0]],
            cell=np.eye(3) * 5.0,
            pbc=True,
        )

        class MockModel:
            calls = 0

            def train(self, mode):
                return self

            def __call__(self, sub_batch):
                MockModel.calls += 1
                n = int(sub_batch["num_probes"].item())
                # Charge3net returns shape [B=1, n_probes]
                return torch.ones((1, n), dtype=torch.float32)

        grid_shape = (6, 6, 6)
        rho = eval_module.predict_density(
            "charge3net",
            atoms,
            grid_shape,
            None,
            BasisSpec(),
            model=MockModel(),
            max_probe_batch=64,
        )
        assert rho.shape == grid_shape
        np.testing.assert_array_equal(rho, np.ones(grid_shape, dtype=np.float32))
        # 6^3 = 216 probes, max_probe_batch=64 -> at least 3 forward calls
        assert MockModel.calls >= 3

    def test_charge3net_max_probe_batch_controls_chunking(self, eval_module):
        """Lowering max_probe_batch must increase the number of forward
        passes proportionally."""
        pytest.importorskip("torch")
        import torch

        from salted_ft.basis import BasisSpec

        if not (Path(__file__).resolve().parent.parent.parent / "charge3net").exists():
            pytest.skip("charge3net sibling repo not present")

        atoms = ase.Atoms(
            "HH",
            positions=[[0, 0, 0], [0.74, 0, 0]],
            cell=np.eye(3) * 5.0,
            pbc=True,
        )

        class CountingMock:
            def __init__(self):
                self.calls = 0

            def train(self, mode):
                return self

            def __call__(self, sub_batch):
                self.calls += 1
                n = int(sub_batch["num_probes"].item())
                return torch.zeros((1, n), dtype=torch.float32)

        m1 = CountingMock()
        eval_module.predict_density(
            "charge3net",
            atoms,
            (8, 8, 8),
            None,
            BasisSpec(),
            model=m1,
            max_probe_batch=512,
        )
        m2 = CountingMock()
        eval_module.predict_density(
            "charge3net",
            atoms,
            (8, 8, 8),
            None,
            BasisSpec(),
            model=m2,
            max_probe_batch=32,
        )
        # Smaller batch -> more sub-batches
        assert m2.calls > m1.calls

    def test_deepdft_with_mock_model_returns_grid(self, eval_module):
        """DeepDFT shares ChargE3Net's input dict format (the latter was
        forked from the former), so the dispatcher should reuse the same
        probe-batching machinery with a DeepDFT-built model. Mock model
        pins the shape contract."""
        pytest.importorskip("torch")
        import torch

        from salted_ft.basis import BasisSpec

        # DeepDFT sibling repo is required because the dispatcher's
        # sys.path side effect goes through deepdft_ft.runner.
        if not (Path(__file__).resolve().parent.parent.parent / "DeepDFT").exists():
            pytest.skip("DeepDFT sibling repo not present; integration only")
        if not (Path(__file__).resolve().parent.parent.parent / "charge3net").exists():
            pytest.skip("charge3net sibling repo not present")

        atoms = ase.Atoms(
            "HH",
            positions=[[0, 0, 0], [0.74, 0, 0]],
            cell=np.eye(3) * 5.0,
            pbc=True,
        )

        class DeepDFTMock:
            def train(self, mode):
                return self

            def __call__(self, sub_batch):
                n = int(sub_batch["num_probes"].item())
                return torch.full((1, n), 0.5, dtype=torch.float32)

        grid_shape = (4, 4, 4)
        rho = eval_module.predict_density(
            "deepdft",
            atoms,
            grid_shape,
            None,
            BasisSpec(),
            model=DeepDFTMock(),
            max_probe_batch=32,
        )
        assert rho.shape == grid_shape
        np.testing.assert_allclose(rho, np.full(grid_shape, 0.5, dtype=np.float32))

    def test_unknown_arm_raises_value_error(self, eval_module):
        from salted_ft.basis import BasisSpec

        atoms = ase.Atoms(
            "HH", positions=[[0, 0, 0], [0.74, 0, 0]], cell=np.eye(3) * 5.0, pbc=True
        )
        with pytest.raises(ValueError, match="unknown"):
            eval_module.predict_density("bogus", atoms, (6, 6, 6), None, BasisSpec())


class TestEvaluateDataset:
    def test_writes_parquet_with_per_row_metrics(self, tmp_path, eval_module):
        from salted_ft.basis import BasisSpec

        in_path = _toy_parquet(tmp_path, n_rows=2)
        out_path = tmp_path / "eval_out.parquet"
        eval_module.evaluate_dataset(
            model_name="salted",
            test_parquet=in_path,
            ckpt=None,
            basis_spec=BasisSpec(),
            output=out_path,
        )
        assert out_path.exists()
        df = pd.read_parquet(out_path)
        assert len(df) == 2
        for col in ("material_id", "nmape", "rmse", "nrmse"):
            assert col in df.columns

    def test_metrics_are_finite(self, tmp_path, eval_module):
        from salted_ft.basis import BasisSpec

        in_path = _toy_parquet(tmp_path, n_rows=2)
        out_path = tmp_path / "eval_out.parquet"
        eval_module.evaluate_dataset(
            model_name="salted",
            test_parquet=in_path,
            ckpt=None,
            basis_spec=BasisSpec(),
            output=out_path,
        )
        df = pd.read_parquet(out_path)
        for col in ("nmape", "rmse", "nrmse"):
            assert np.isfinite(df[col]).all()

    def test_records_model_and_ckpt_in_output(self, tmp_path, eval_module):
        """Output rows must carry the arm name + ckpt path so a downstream
        comparison table can group by model without re-deriving."""
        from salted_ft.basis import BasisSpec

        in_path = _toy_parquet(tmp_path, n_rows=1)
        out_path = tmp_path / "eval_out.parquet"
        eval_module.evaluate_dataset(
            model_name="salted",
            test_parquet=in_path,
            ckpt=None,
            basis_spec=BasisSpec(),
            output=out_path,
        )
        df = pd.read_parquet(out_path)
        assert (df["model"] == "salted").all()
        assert df["ckpt"].iloc[0] in (None, "", "stub")

    def test_limit_caps_n_rows_evaluated(self, tmp_path, eval_module):
        from salted_ft.basis import BasisSpec

        in_path = _toy_parquet(tmp_path, n_rows=5)
        out_path = tmp_path / "eval_out.parquet"
        eval_module.evaluate_dataset(
            model_name="salted",
            test_parquet=in_path,
            ckpt=None,
            basis_spec=BasisSpec(),
            output=out_path,
            limit=2,
        )
        df = pd.read_parquet(out_path)
        assert len(df) == 2
