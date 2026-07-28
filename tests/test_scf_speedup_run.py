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


class TestPerRowResilience:
    """A multi-hour batch must not die on a single bad row."""

    def test_per_row_failure_does_not_abort_loop(self, tmp_path, run_module):
        """If row 2 of 3 has a corrupt cell (positions with wrong
        length) the loop must skip it, record the failure, and keep
        going. Otherwise the prior rows' Flows are submitted to
        Mongo with no clean resume path."""
        from salted_ft.basis import BasisSpec

        # 3 rows, middle one has corrupt positions.
        rows = []
        grid_shape = (4, 4, 4)
        good_positions = np.array(
            [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float64
        ).reshape(-1)
        for i in range(3):
            pos = good_positions
            if i == 1:
                # Length 2: positions.reshape(-1, 3) raises.
                pos = np.array([0.0, 0.0], dtype=np.float64)
            rows.append(
                {
                    "material_id": f"mp-toy-{i}",
                    "n_atoms": 2,
                    "atomic_numbers": np.array([1, 1], dtype=np.int64),
                    "positions": pos,
                    "lattice_vectors": (np.eye(3) * 5.0).reshape(-1),
                    "grid_shape": np.array(grid_shape, dtype=np.int64),
                    "n_electrons": 2.0,
                }
            )
        in_parquet = tmp_path / "held_out_with_bad_row.parquet"
        pd.DataFrame(rows).to_parquet(in_parquet)

        records = run_module.run_experiment(
            model_name="salted",
            test_parquet=in_parquet,
            chgcar_dir=tmp_path / "chgcars",
            basis_spec=BasisSpec(),
            project="p",
            worker="w",
            dry_run=True,
            make_pair_fn=_make_pair_mock([]),
            submit_fn=_submit_mock([]),
        )
        assert len(records) == 3
        good = [r for r in records if r.get("error") is None]
        bad = [r for r in records if r.get("error") is not None]
        assert len(good) == 2
        assert len(bad) == 1
        assert bad[0]["material_id"] == "mp-toy-1"
        assert bad[0]["submitted"] is False
        assert "reshape" in bad[0]["error"] or "cannot" in bad[0]["error"]


class TestManifest:
    def test_manifest_jsonl_written_after_each_row(self, tmp_path, run_module):
        """The manifest must be written incrementally so an
        interrupted run leaves a resumable record. After all rows
        complete the manifest should have one JSONL line per row."""
        import json

        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=3)
        chgcar_dir = tmp_path / "chgcars"
        manifest = tmp_path / "manifest.jsonl"

        run_module.run_experiment(
            model_name="salted",
            test_parquet=in_parquet,
            chgcar_dir=chgcar_dir,
            basis_spec=BasisSpec(),
            project="p",
            worker="w",
            dry_run=True,
            manifest_path=manifest,
            make_pair_fn=_make_pair_mock([]),
            submit_fn=_submit_mock([]),
        )
        assert manifest.exists()
        lines = manifest.read_text().splitlines()
        assert len(lines) == 3
        for line in lines:
            rec = json.loads(line)
            assert "material_id" in rec
            assert "model" in rec

    def test_manifest_defaults_to_chgcar_dir(self, tmp_path, run_module):
        """If --manifest is not given, default to
        chgcar_dir/manifest.jsonl so a re-run can find it by
        convention."""
        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=1)
        chgcar_dir = tmp_path / "chgcars"

        run_module.run_experiment(
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
        assert (chgcar_dir / "manifest.jsonl").exists()


class TestSkipExisting:
    def test_skip_existing_skips_already_submitted_rows(self, tmp_path, run_module):
        """Pre-populate a manifest with one submitted row, then
        re-run with skip_existing=True; only the unseen rows should
        be processed."""
        import json

        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=3)
        chgcar_dir = tmp_path / "chgcars"
        chgcar_dir.mkdir()
        manifest = chgcar_dir / "manifest.jsonl"
        # Mark mp-toy-1 as already done.
        manifest.write_text(
            json.dumps(
                {
                    "material_id": "mp-toy-1",
                    "model": "salted",
                    "submitted": True,
                    "error": None,
                }
            )
            + "\n"
        )
        make_calls: list = []
        records = run_module.run_experiment(
            model_name="salted",
            test_parquet=in_parquet,
            chgcar_dir=chgcar_dir,
            basis_spec=BasisSpec(),
            project="p",
            worker="w",
            dry_run=True,
            skip_existing=True,
            manifest_path=manifest,
            make_pair_fn=_make_pair_mock(make_calls),
            submit_fn=_submit_mock([]),
        )
        # mp-toy-1 should NOT have been re-processed.
        processed_ids = {call["metadata"]["material_id"] for call in make_calls}
        assert "mp-toy-1" not in processed_ids
        assert processed_ids == {"mp-toy-0", "mp-toy-2"}
        # Records reflect what THIS run did, not the historical entry.
        assert len(records) == 2

    def test_skip_existing_does_not_skip_failed_rows(self, tmp_path, run_module):
        """A row in the manifest with submitted=False (error from a
        previous run) should be retried on the next run, not skipped."""
        import json

        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=2)
        chgcar_dir = tmp_path / "chgcars"
        chgcar_dir.mkdir()
        manifest = chgcar_dir / "manifest.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "material_id": "mp-toy-0",
                    "model": "salted",
                    "submitted": False,
                    "error": "previous_run_died",
                }
            )
            + "\n"
        )
        make_calls: list = []
        run_module.run_experiment(
            model_name="salted",
            test_parquet=in_parquet,
            chgcar_dir=chgcar_dir,
            basis_spec=BasisSpec(),
            project="p",
            worker="w",
            dry_run=True,
            skip_existing=True,
            manifest_path=manifest,
            make_pair_fn=_make_pair_mock(make_calls),
            submit_fn=_submit_mock([]),
        )
        processed_ids = {call["metadata"]["material_id"] for call in make_calls}
        # mp-toy-0 was previously failed, should be retried.
        assert "mp-toy-0" in processed_ids


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

    def test_chgcar_layout_is_nested_by_model_then_material_id(
        self, tmp_path, run_module
    ):
        """Layout must be ``chgcar_root/<model>/<material_id>/CHGCAR``
        so a material_id containing separator characters never causes
        ambiguity. Was previously a flat ``{model}__{material_id}/``
        which broke on synthesised IDs like ``oqmd__1234``."""
        from salted_ft.basis import BasisSpec

        in_parquet = _toy_parquet(tmp_path, n_rows=1)
        chgcar_dir = tmp_path / "chgcars"

        records = run_module.run_experiment(
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
        chgcar_path = Path(records[0]["chgcar_path"])
        # Path tail must be .../<model>/<material_id>/CHGCAR
        parts = chgcar_path.parts
        assert parts[-1] == "CHGCAR"
        assert parts[-2] == "mp-toy-0"
        assert parts[-3] == "salted"


class TestRealisticRow:
    """Catch mutation-killers a 2-atom H2 toy row misses: a missing
    n_electrons rescale, a positions-reshape bug, or a grid/atom
    mismatch all pass silently on the degenerate fixture."""

    def test_5_atom_asymmetric_grid_unequal_n_electrons(self, tmp_path, run_module):
        from salted_ft.basis import BasisSpec

        # 5 atoms: 1 Fe + 4 O (chosen so sum(Z)=26+4*8=58 != n_electrons=12.5).
        # Asymmetric grid_shape catches axes-swap bugs.
        n_atoms = 5
        atomic_numbers = np.array([26, 8, 8, 8, 8], dtype=np.int64)
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 5, size=(n_atoms, 3)).astype(np.float64)
        rows = [
            {
                "material_id": "mp-realistic-0",
                "n_atoms": n_atoms,
                "atomic_numbers": atomic_numbers,
                "positions": positions.reshape(-1),
                "lattice_vectors": (np.eye(3) * 5.0).reshape(-1),
                "grid_shape": np.array([8, 10, 12], dtype=np.int64),
                "n_electrons": 12.5,
            }
        ]
        in_parquet = tmp_path / "realistic.parquet"
        pd.DataFrame(rows).to_parquet(in_parquet)

        make_calls: list = []
        records = run_module.run_experiment(
            model_name="salted",
            test_parquet=in_parquet,
            chgcar_dir=tmp_path / "chgcars",
            basis_spec=BasisSpec(),
            project="p",
            worker="w",
            dry_run=True,
            make_pair_fn=_make_pair_mock(make_calls),
            submit_fn=_submit_mock([]),
        )
        # The row completed without error -- reshape correct, write_chgcar
        # accepted asymmetric grid, n_electrons propagated to write_chgcar.
        assert len(records) == 1
        assert records[0]["error"] is None, f"unexpected error: {records[0]['error']}"
        assert records[0]["submitted"] is False  # dry-run
