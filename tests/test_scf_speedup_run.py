"""TDD tests for ``scripts/scf_speedup_run.py`` (P4).

The driver loops a held-out test parquet, predicts each row's
density via the chosen ML arm, writes a CHGCAR with the right
electron-count rescaling, and submits a paired baseline + predicted
VASP Flow via ``entalsim.dft.scf_speedup.make_scf_speedup_pair`` +
``entalsim.core.submit.submit_workflow``.

Tests use dependency injection (``make_pair_fn`` and ``submit_fn``)
so they pass locally without entalsim installed. The real CLI
imports entalsim's functions at runtime.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def run_module():
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    if "scf_speedup_run" in sys.modules:
        del sys.modules["scf_speedup_run"]
    return importlib.import_module("scf_speedup_run")


def _toy_parquet(tmp_path: Path, n_rows: int = 2) -> Path:
    """Synthesise a held-out-split-shaped parquet.

    Columns mirror what the held-out split builder will emit:
    material_id, atomic_numbers, positions (flat), lattice_vectors
    (flat 9), grid_shape, n_electrons.
    """
    rows = []
    grid_shape = (4, 4, 4)
    for i in range(n_rows):
        n_atoms = 2
        rows.append(
            {
                "material_id": f"mp-toy-{i}",
                "n_atoms": n_atoms,
                "atomic_numbers": np.array([1, 1], dtype=np.int64),
                "positions": np.array(
                    [[0.0, 0.0, 0.0], [0.74 + 0.01 * i, 0.0, 0.0]],
                    dtype=np.float64,
                ).reshape(-1),
                "lattice_vectors": (np.eye(3) * 5.0).reshape(-1),
                "grid_shape": np.array(grid_shape, dtype=np.int64),
                "n_electrons": 2.0,
            }
        )
    out = tmp_path / "held_out.parquet"
    pd.DataFrame(rows).to_parquet(out)
    return out


def _fake_flow(n_jobs: int = 2):
    return SimpleNamespace(
        jobs=[SimpleNamespace(uuid=f"j{i}") for i in range(n_jobs)],
        name="fake_flow",
    )


def _make_pair_mock(captured: list):
    """Returns a (mock, captured) pair. ``captured`` records each call."""

    def make_pair(structure, predicted_chgcar_dir, metadata):
        captured.append(
            {
                "structure_formula": structure.composition.reduced_formula,
                "predicted_chgcar_dir": str(predicted_chgcar_dir),
                "metadata": dict(metadata),
                "chgcar_exists": (Path(predicted_chgcar_dir) / "CHGCAR").exists(),
            }
        )
        return _fake_flow()

    return make_pair


def _submit_mock(captured: list):
    def submit(flow, project, worker):
        captured.append(
            {"project": project, "worker": worker, "n_jobs": len(flow.jobs)}
        )

    return submit


class TestDriverBasics:
    def test_dry_run_writes_one_chgcar_per_row(self, tmp_path, run_module):
        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=2)
        chgcar_dir = tmp_path / "chgcars"
        make_calls: list = []
        submit_calls: list = []

        records = run_module.run_experiment(
            model_name="salted",
            test_parquet=in_parquet,
            chgcar_dir=chgcar_dir,
            basis_spec=BasisSpec(),
            project="test_project",
            worker="test_worker",
            dry_run=True,
            make_pair_fn=_make_pair_mock(make_calls),
            submit_fn=_submit_mock(submit_calls),
        )
        assert len(records) == 2
        for r in records:
            assert Path(r["chgcar_path"]).exists()
        assert submit_calls == [], "dry_run=True must not submit"

    def test_make_pair_invoked_with_metadata(self, tmp_path, run_module):
        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=2)
        chgcar_dir = tmp_path / "chgcars"
        make_calls: list = []

        run_module.run_experiment(
            model_name="salted",
            test_parquet=in_parquet,
            chgcar_dir=chgcar_dir,
            basis_spec=BasisSpec(),
            project="test_project",
            worker="test_worker",
            dry_run=True,
            make_pair_fn=_make_pair_mock(make_calls),
            submit_fn=_submit_mock([]),
        )
        assert len(make_calls) == 2
        for call in make_calls:
            md = call["metadata"]
            assert md["experiment"] == "scf_speedup"
            assert md["model"] == "salted"
            assert md["material_id"].startswith("mp-toy-")
            assert call["chgcar_exists"], (
                "make_scf_speedup_pair must see a real CHGCAR file at the path "
                "we hand it; otherwise its FileNotFoundError fires on every row"
            )

    def test_limit_caps_rows_processed(self, tmp_path, run_module):
        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=5)
        chgcar_dir = tmp_path / "chgcars"
        make_calls: list = []

        records = run_module.run_experiment(
            model_name="salted",
            test_parquet=in_parquet,
            chgcar_dir=chgcar_dir,
            basis_spec=BasisSpec(),
            project="p",
            worker="w",
            limit=2,
            dry_run=True,
            make_pair_fn=_make_pair_mock(make_calls),
            submit_fn=_submit_mock([]),
        )
        assert len(records) == 2
        assert len(make_calls) == 2


class TestSubmitWiring:
    def test_non_dry_run_calls_submit_per_row(self, tmp_path, run_module):
        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=2)
        chgcar_dir = tmp_path / "chgcars"
        submit_calls: list = []

        run_module.run_experiment(
            model_name="salted",
            test_parquet=in_parquet,
            chgcar_dir=chgcar_dir,
            basis_spec=BasisSpec(),
            project="jz_scf_speedup",
            worker="jean_zay_cpu",
            dry_run=False,
            make_pair_fn=_make_pair_mock([]),
            submit_fn=_submit_mock(submit_calls),
        )
        assert len(submit_calls) == 2
        for call in submit_calls:
            assert call["project"] == "jz_scf_speedup"
            assert call["worker"] == "jean_zay_cpu"
            assert call["n_jobs"] == 2

    def test_records_include_submitted_flag(self, tmp_path, run_module):
        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=1)
        chgcar_dir = tmp_path / "chgcars"

        dry = run_module.run_experiment(
            model_name="salted",
            test_parquet=in_parquet,
            chgcar_dir=chgcar_dir,
            basis_spec=BasisSpec(),
            project="p",
            worker="w",
            dry_run=True,
            make_pair_fn=_make_pair_mock([]),
            submit_fn=_submit_mock([]),
        )
        wet = run_module.run_experiment(
            model_name="salted",
            test_parquet=in_parquet,
            chgcar_dir=tmp_path / "chgcars_wet",
            basis_spec=BasisSpec(),
            project="p",
            worker="w",
            dry_run=False,
            make_pair_fn=_make_pair_mock([]),
            submit_fn=_submit_mock([]),
        )
        assert dry[0]["submitted"] is False
        assert wet[0]["submitted"] is True


class TestArmCheckpointGuard:
    def test_charge3net_without_ckpt_fails_fast(self, tmp_path, run_module):
        """ChargE3Net and DeepDFT without a checkpoint run as random-init
        models. Their predictions would be meaningless, and we would
        silently waste HPC time. The driver must refuse before any
        prediction or submit.
        """
        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=1)
        with pytest.raises(ValueError, match="ckpt"):
            run_module.run_experiment(
                model_name="charge3net",
                test_parquet=in_parquet,
                chgcar_dir=tmp_path / "c",
                basis_spec=BasisSpec(),
                project="p",
                worker="w",
                ckpt=None,
                dry_run=True,
                make_pair_fn=_make_pair_mock([]),
                submit_fn=_submit_mock([]),
            )

    def test_deepdft_without_ckpt_fails_fast(self, tmp_path, run_module):
        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=1)
        with pytest.raises(ValueError, match="ckpt"):
            run_module.run_experiment(
                model_name="deepdft",
                test_parquet=in_parquet,
                chgcar_dir=tmp_path / "c",
                basis_spec=BasisSpec(),
                project="p",
                worker="w",
                ckpt=None,
                dry_run=True,
                make_pair_fn=_make_pair_mock([]),
                submit_fn=_submit_mock([]),
            )

    def test_salted_without_ckpt_uses_stub(self, tmp_path, run_module):
        """SALTED stub mode is the documented fallback. The driver must
        let it through so we can dry-run the pipeline before D6 trained
        weights are available."""
        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=1)
        records = run_module.run_experiment(
            model_name="salted",
            test_parquet=in_parquet,
            chgcar_dir=tmp_path / "c",
            basis_spec=BasisSpec(),
            project="p",
            worker="w",
            ckpt=None,
            dry_run=True,
            make_pair_fn=_make_pair_mock([]),
            submit_fn=_submit_mock([]),
        )
        assert records[0]["ckpt"] == "stub"


class TestChgcarOrganisation:
    def test_per_row_chgcar_dirs_are_unique(self, tmp_path, run_module):
        """make_scf_speedup_pair takes a directory and stages CHGCAR
        from it. Multiple rows must NOT share one directory or the
        last write wins."""
        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=3)
        chgcar_dir = tmp_path / "chgcars"
        make_calls: list = []

        records = run_module.run_experiment(
            model_name="salted",
            test_parquet=in_parquet,
            chgcar_dir=chgcar_dir,
            basis_spec=BasisSpec(),
            project="p",
            worker="w",
            dry_run=True,
            make_pair_fn=_make_pair_mock(make_calls),
            submit_fn=_submit_mock([]),
        )
        seen = {Path(call["predicted_chgcar_dir"]).resolve() for call in make_calls}
        assert len(seen) == len(records) == 3
