"""TDD tests for the DeepDFT validation-pass host-RAM OOM fix.

Job 5004725 (--mem=64000M) was OOM-killed during the step-0 validation
pass. The val collate asked for 5000 probes per material, and upstream
``probes_to_graph`` inserts every probe as a dummy atom into a periodic
neighborlist, so probe-probe pairs grow quadratically with the probe
count (5000 probes was ~75M pairs for the median LeMat-Rho cell, far
beyond 64 GB transient). Probes were also drawn WITH replacement from
grids of only 1000 points (5x duplicates, zero statistical value).

These tests pin the fix contract:

- ``deepdft_ft.data.sample_probe_indices`` never returns more indices
  than the configured cap (checked on a 15x15x15 = 3375-point grid,
  the upgraded LeMat-Rho grid size),
- sampling is without replacement whenever the grid has at least as
  many points as the cap,
- the per-worker parquet table cache in ``deepdft_ft.data`` is bounded
  (LRU), mirroring ``charge3net_ft.data``,
- ``submit_deepdft_adastra.sh`` passes the new --val-probes and
  --val-max-samples flags and points at the 15cube dataset.

Kept free of any DeepDFT-repo import: ``deepdft_ft.runner`` needs the
upstream DeepDFT clone on sys.path, but the sampling helper and the
dataset adapter live in ``deepdft_ft.data``, which needs only the
``../charge3net`` sibling (for the shared parquet helpers). The whole
module skips when that sibling is absent.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

try:
    import charge3net_ft.data  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"charge3net sibling repo unavailable: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Probe index sampling (used by the validation collate in deepdft_ft.runner)
# ---------------------------------------------------------------------------
class TestSampleProbeIndices:
    def test_probe_count_never_exceeds_cap_on_15cube_grid(self):
        """A 15x15x15 grid has 3375 points; the cap (1000) must still hold."""
        from deepdft_ft.data import sample_probe_indices

        indices = sample_probe_indices(15**3, 1000)
        assert len(indices) == 1000, (
            f"expected exactly 1000 probe indices; got {len(indices)}"
        )

    def test_without_replacement_when_grid_at_least_cap(self):
        """n_probes <= grid points implies no duplicate probes."""
        from deepdft_ft.data import sample_probe_indices

        indices = sample_probe_indices(15**3, 1000)
        assert len(np.unique(indices)) == len(indices), (
            "probe indices must be sampled without replacement when the grid "
            "has at least n_probes points"
        )

    def test_grid_equal_to_cap_uses_every_point_exactly_once(self):
        """1000-point grid + 1000-probe cap: the old code drew 5x duplicates."""
        from deepdft_ft.data import sample_probe_indices

        indices = sample_probe_indices(1000, 1000)
        assert sorted(indices.tolist()) == list(range(1000))

    def test_indices_stay_within_grid(self):
        from deepdft_ft.data import sample_probe_indices

        indices = sample_probe_indices(3375, 1000)
        assert indices.min() >= 0
        assert indices.max() < 3375

    def test_small_grid_falls_back_to_exact_count(self):
        """Grids smaller than the cap still yield exactly n_probes indices.

        Upstream padding/eval assumes a uniform probe count per sample, so
        the fallback samples with replacement rather than shrinking.
        """
        from deepdft_ft.data import sample_probe_indices

        indices = sample_probe_indices(10, 32)
        assert len(indices) == 32
        assert indices.min() >= 0
        assert indices.max() < 10

    def test_seeded_rng_is_deterministic(self):
        from deepdft_ft.data import sample_probe_indices

        a = sample_probe_indices(3375, 1000, rng=np.random.default_rng(7))
        b = sample_probe_indices(3375, 1000, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Bounded per-worker parquet table cache
# ---------------------------------------------------------------------------
def _write_synthetic_chunk(path: Path, n_valid: int = 1) -> None:
    """Same schema as the real LeMat-Rho chunks (see test_deepdft_data.py)."""
    grid = json.dumps(np.ones((10, 10, 10), dtype=np.float32).tolist())
    table = pa.table(
        {
            "compressed_charge_density": pa.array([grid] * n_valid, type=pa.string()),
            "species_at_sites": pa.array([["Fe"]] * n_valid),
            "cartesian_site_positions": pa.array([[[0.0, 0.0, 0.0]]] * n_valid),
            "lattice_vectors": pa.array(
                [[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]] * n_valid
            ),
        }
    )
    pq.write_table(table, path)


class TestBoundedTableCache:
    def test_cache_never_exceeds_cap(self, tmp_path):
        """Touching more chunks than the cap must evict, not grow unbounded.

        The unbounded dict cache held one decompressed pyarrow table per
        chunk file forever (same failure mode that OOM-killed the
        charge3net jobs 4971293/4971343 before its LRU port).
        """
        import deepdft_ft.data as dd

        cap = dd._DEEPDFT_TABLE_CACHE_MAX_CHUNKS
        n_chunks = cap + 2
        for i in range(n_chunks):
            _write_synthetic_chunk(tmp_path / f"chunk_{i:03d}.parquet")

        dd._DEEPDFT_TABLE_CACHE.clear()
        ds = dd.LeMatRhoDeepDFTDataset(parquet_dir=tmp_path)
        # One row per chunk: touches every file once.
        for i in range(n_chunks):
            ds[i]
        assert len(dd._DEEPDFT_TABLE_CACHE) <= cap, (
            f"table cache grew to {len(dd._DEEPDFT_TABLE_CACHE)} entries; cap is {cap}"
        )

    def test_cache_hit_returns_same_row(self, tmp_path):
        """Eviction must not corrupt reads: re-reading a row round-trips."""
        import deepdft_ft.data as dd

        n_chunks = dd._DEEPDFT_TABLE_CACHE_MAX_CHUNKS + 2
        for i in range(n_chunks):
            _write_synthetic_chunk(tmp_path / f"chunk_{i:03d}.parquet")

        dd._DEEPDFT_TABLE_CACHE.clear()
        ds = dd.LeMatRhoDeepDFTDataset(parquet_dir=tmp_path)
        first = ds[0]["density"]
        for i in range(n_chunks):  # cycle far enough to evict chunk 0
            ds[i]
        again = ds[0]["density"]  # re-read forces a cache miss + reload
        np.testing.assert_array_equal(first, again)


# ---------------------------------------------------------------------------
# Submit script wiring
# ---------------------------------------------------------------------------
SUBMIT_SCRIPT = Path(__file__).resolve().parent.parent / "submit_deepdft_adastra.sh"


def _run_submit(tmp_path: Path, env_extra: dict | None = None):
    """Run submit_deepdft_adastra.sh under bash with LEMATRHO_DRY_RUN=1."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available in test environment")
    env = {
        **os.environ,
        "LEMATRHO_DRY_RUN": "1",
        # Avoid touching the user's real Adastra setup.
        "LEMATRHO_ADASTRA_SETUP": str(tmp_path),
        **(env_extra or {}),
    }
    return subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class TestSubmitScriptValWiring:
    def test_dry_run_passes_val_probe_and_subsample_flags(self, tmp_path):
        result = _run_submit(tmp_path)
        assert result.returncode == 0, (
            f"dry-run exited {result.returncode}; stderr={result.stderr}"
        )
        assert "--val-probes 1000" in result.stdout, (
            f"run line must cap val probes at 1000; stdout={result.stdout}"
        )
        assert "--val-max-samples 200" in result.stdout, (
            f"run line must subsample the val split to 200; stdout={result.stdout}"
        )

    def test_dataset_points_at_15cube_dir(self, tmp_path):
        result = _run_submit(tmp_path)
        assert "charge3net_data_15cube" in result.stdout, (
            f"DATA_DIR must be the 15cube dataset; stdout={result.stdout}"
        )

    def test_venv_is_fresh(self):
        text = SUBMIT_SCRIPT.read_text()
        assert "venv311_fresh/bin/activate" in text, (
            "script must activate venv311_fresh"
        )
        assert "venv311/bin/activate" not in text, (
            "stale venv311 activation still present"
        )
