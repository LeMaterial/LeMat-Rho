"""TDD tests for ``scripts/density_model_comparison_table.py`` (D8).

Takes the per-arm parquet outputs from D7 and aggregates into a
single comparison table (markdown + CSV). Per-row metrics are
summarised per arm: mean +/- std and median, with the number of
structures evaluated.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def comparison_module():
    """Import scripts.density_model_comparison_table."""
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    if "density_model_comparison_table" in sys.modules:
        del sys.modules["density_model_comparison_table"]
    return importlib.import_module("density_model_comparison_table")


def _toy_eval_parquet(
    tmp_path: Path,
    model_name: str,
    nmape_values: list[float],
    rmse_values: list[float],
    nrmse_values: list[float],
) -> Path:
    """Write a D7-shaped eval-output parquet with known metric values."""
    df = pd.DataFrame(
        {
            "model": model_name,
            "ckpt": "stub",
            "material_id": [f"mp-{model_name}-{i}" for i in range(len(nmape_values))],
            "n_atoms": 2,
            "nmape": nmape_values,
            "rmse": rmse_values,
            "nrmse": nrmse_values,
        }
    )
    out = tmp_path / f"eval_{model_name}.parquet"
    df.to_parquet(out)
    return out


class TestAggregate:
    def test_returns_one_row_per_arm(self, tmp_path, comparison_module):
        p1 = _toy_eval_parquet(
            tmp_path, "salted", [10.0, 20.0], [0.1, 0.2], [5.0, 10.0]
        )
        p2 = _toy_eval_parquet(
            tmp_path, "charge3net", [5.0, 7.0], [0.05, 0.07], [2.0, 3.0]
        )
        df = comparison_module.aggregate_per_arm([p1, p2])
        assert set(df["model"]) == {"salted", "charge3net"}

    def test_mean_nmape_matches_input(self, tmp_path, comparison_module):
        p = _toy_eval_parquet(tmp_path, "salted", [10.0, 30.0], [0.1, 0.3], [5.0, 15.0])
        df = comparison_module.aggregate_per_arm([p])
        row = df.iloc[0]
        assert row["nmape_mean"] == pytest.approx(20.0)
        assert row["rmse_mean"] == pytest.approx(0.2)
        assert row["nrmse_mean"] == pytest.approx(10.0)

    def test_std_present(self, tmp_path, comparison_module):
        p = _toy_eval_parquet(tmp_path, "salted", [10.0, 30.0], [0.1, 0.3], [5.0, 15.0])
        df = comparison_module.aggregate_per_arm([p])
        for col in ("nmape_std", "rmse_std", "nrmse_std"):
            assert col in df.columns
            assert np.isfinite(df[col].iloc[0])

    def test_median_present(self, tmp_path, comparison_module):
        p = _toy_eval_parquet(
            tmp_path, "salted", [10.0, 20.0, 30.0], [0.1, 0.2, 0.3], [5.0, 10.0, 15.0]
        )
        df = comparison_module.aggregate_per_arm([p])
        assert df["nmape_median"].iloc[0] == pytest.approx(20.0)

    def test_n_structures_counts_rows(self, tmp_path, comparison_module):
        p = _toy_eval_parquet(tmp_path, "salted", [1.0, 2.0, 3.0], [0.1] * 3, [1.0] * 3)
        df = comparison_module.aggregate_per_arm([p])
        assert df["n_structures"].iloc[0] == 3

    def test_aggregates_multiple_files_per_arm(self, tmp_path, comparison_module):
        """If the same arm is split across two parquets, aggregate
        should treat them as one group. Useful when sharded eval
        runs write per-chunk outputs."""
        (tmp_path / "p1").mkdir()
        (tmp_path / "p2").mkdir()
        p1 = _toy_eval_parquet(tmp_path / "p1", "salted", [10.0], [0.1], [5.0])
        p2 = _toy_eval_parquet(tmp_path / "p2", "salted", [30.0], [0.3], [15.0])
        df = comparison_module.aggregate_per_arm([p1, p2])
        assert len(df) == 1
        assert df.iloc[0]["n_structures"] == 2
        assert df.iloc[0]["nmape_mean"] == pytest.approx(20.0)


class TestRenderMarkdown:
    def test_markdown_contains_arm_names(self, tmp_path, comparison_module):
        p1 = _toy_eval_parquet(
            tmp_path, "salted", [10.0, 20.0], [0.1, 0.2], [5.0, 10.0]
        )
        p2 = _toy_eval_parquet(
            tmp_path, "charge3net", [5.0, 7.0], [0.05, 0.07], [2.0, 3.0]
        )
        df = comparison_module.aggregate_per_arm([p1, p2])
        md = comparison_module.render_markdown_table(df)
        assert "salted" in md
        assert "charge3net" in md

    def test_markdown_has_header_row(self, tmp_path, comparison_module):
        p = _toy_eval_parquet(tmp_path, "salted", [10.0], [0.1], [5.0])
        df = comparison_module.aggregate_per_arm([p])
        md = comparison_module.render_markdown_table(df)
        # GitHub-flavored markdown table separator
        assert "|" in md
        assert "---" in md


class TestWriteOutputs:
    def test_writes_csv(self, tmp_path, comparison_module):
        p = _toy_eval_parquet(tmp_path, "salted", [10.0, 20.0], [0.1, 0.2], [5.0, 10.0])
        out_csv = tmp_path / "out.csv"
        out_md = tmp_path / "out.md"
        comparison_module.build_comparison_table(
            inputs=[p], csv_path=out_csv, markdown_path=out_md
        )
        assert out_csv.exists()
        df = pd.read_csv(out_csv)
        assert "model" in df.columns

    def test_writes_markdown(self, tmp_path, comparison_module):
        p = _toy_eval_parquet(tmp_path, "salted", [10.0, 20.0], [0.1, 0.2], [5.0, 10.0])
        out_csv = tmp_path / "out.csv"
        out_md = tmp_path / "out.md"
        comparison_module.build_comparison_table(
            inputs=[p], csv_path=out_csv, markdown_path=out_md
        )
        assert out_md.exists()
        assert "salted" in out_md.read_text()
