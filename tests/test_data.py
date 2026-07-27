"""
Unit tests for charge3net_ft data utilities.

Uses synthetic in-memory data — no real Parquet files, no charge3net dep.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# ---------------------------------------------------------------------------
# We test only the pure utility functions that don't import charge3net.
# Import them by reaching into the module after patching the sys.path block.
# ---------------------------------------------------------------------------
def _import_data_utils():
    """Import _parse_grid_json and _row_to_atoms_and_density without triggering
    the charge3net RuntimeError (which fires if the sibling repo is absent)."""
    import importlib
    import sys
    from unittest.mock import patch

    # Stub out the charge3net modules so the import succeeds without the repo
    fake_modules = [
        "src",
        "src.charge3net",
        "src.charge3net.data",
        "src.charge3net.data.collate",
        "src.charge3net.data.graph_construction",
        "src.utils",
        "src.utils.data",
    ]
    stubs = {}
    for mod in fake_modules:
        stubs[mod] = type(sys)("mod")
    stubs["src.charge3net.data.collate"].collate_list_of_dicts = lambda *a, **kw: None
    stubs["src.charge3net.data.graph_construction"].KdTreeGraphConstructor = object
    stubs["src.utils.data"].calculate_grid_pos = lambda *a, **kw: None

    # Also patch the existence check so it doesn't raise
    with (
        patch.dict(sys.modules, stubs),
        patch("pathlib.Path.exists", return_value=True),
    ):
        import importlib

        # Force reimport with stubs in place
        if "charge3net_ft.data" in sys.modules:
            del sys.modules["charge3net_ft.data"]
        mod = importlib.import_module("charge3net_ft.data")
    return mod


class TestParseGridJson:
    def test_roundtrip_3d(self):
        grid = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        json_str = json.dumps(grid)
        from charge3net_ft.data import _parse_grid_json

        result = _parse_grid_json(json_str)
        assert result.shape == (2, 2, 2)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, np.array(grid, dtype=np.float32))

    def test_10x10x10(self):
        from charge3net_ft.data import _parse_grid_json

        grid = np.random.rand(10, 10, 10).tolist()
        result = _parse_grid_json(json.dumps(grid))
        assert result.shape == (10, 10, 10)


class TestRowToAtomsAndDensity:
    def _make_row(self):
        return {
            "species_at_sites": ["Fe", "O"],
            "cartesian_site_positions": [[0.0, 0.0, 0.0], [1.4, 1.4, 1.4]],
            "lattice_vectors": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            "compressed_charge_density": json.dumps(np.ones((10, 10, 10)).tolist()),
        }

    def test_atoms_species(self):
        import ase

        from charge3net_ft.data import _row_to_atoms_and_density

        row = self._make_row()
        atoms, _density, _origin = _row_to_atoms_and_density(row)
        assert isinstance(atoms, ase.Atoms)
        assert list(atoms.get_chemical_symbols()) == ["Fe", "O"]

    def test_pbc(self):
        from charge3net_ft.data import _row_to_atoms_and_density

        atoms, _, _ = _row_to_atoms_and_density(self._make_row())
        assert all(atoms.pbc)

    def test_density_shape(self):
        from charge3net_ft.data import _row_to_atoms_and_density

        _, density, _ = _row_to_atoms_and_density(self._make_row())
        assert density.shape == (10, 10, 10)

    def test_origin_is_zero(self):
        from charge3net_ft.data import _row_to_atoms_and_density

        _, _, origin = _row_to_atoms_and_density(self._make_row())
        np.testing.assert_array_equal(origin, [0.0, 0.0, 0.0])

    def test_unknown_species_raises(self):
        from charge3net_ft.data import _row_to_atoms_and_density

        row = self._make_row()
        row["species_at_sites"] = ["Xx"]  # invalid symbol
        with pytest.raises(KeyError):
            _row_to_atoms_and_density(row)


class TestBuildParquetIndex:
    def _write_chunk(self, path: Path, n_valid: int, n_null: int):
        """Write a synthetic chunk_*.parquet file."""
        valid = [json.dumps(np.ones((10, 10, 10)).tolist())] * n_valid
        null = [None] * n_null
        table = pa.table(
            {
                "compressed_charge_density": pa.array(valid + null, type=pa.string()),
                "species_at_sites": pa.array([["Fe"]] * (n_valid + n_null)),
                "cartesian_site_positions": pa.array(
                    [[[0.0, 0.0, 0.0]]] * (n_valid + n_null)
                ),
                "lattice_vectors": pa.array(
                    [[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]]
                    * (n_valid + n_null)
                ),
            }
        )
        pq.write_table(table, path)

    def test_counts_valid_rows(self):
        from charge3net_ft.data import _build_parquet_index

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_chunk(d / "chunk_000.parquet", n_valid=5, n_null=2)
            self._write_chunk(d / "chunk_001.parquet", n_valid=3, n_null=1)
            file_paths, index = _build_parquet_index(d)
            assert len(index) == 8  # 5 + 3 valid
            assert len(file_paths) == 2

    def test_index_entries_reference_correct_file(self):
        from charge3net_ft.data import _build_parquet_index

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_chunk(d / "chunk_000.parquet", n_valid=3, n_null=0)
            self._write_chunk(d / "chunk_001.parquet", n_valid=2, n_null=0)
            _, index = _build_parquet_index(d)
            file_indices = [fi for fi, _ in index]
            assert file_indices[:3] == [0, 0, 0]
            assert file_indices[3:] == [1, 1]

    def test_raises_on_empty_dir(self):
        from charge3net_ft.data import _build_parquet_index

        with tempfile.TemporaryDirectory() as tmp, pytest.raises(FileNotFoundError):
            _build_parquet_index(Path(tmp))

    def test_ignores_extra_columns(self):
        """Newer LeMat-Rho dataset versions add Bader-analysis columns (e.g.
        bader_charges, bader_volumes) alongside the four required columns.
        _build_parquet_index and _row_to_atoms_and_density should ignore the
        extras transparently: data.py:46 declares an explicit _COLUMNS allowlist
        and pq.read_table is called with columns=_COLUMNS.
        """
        from charge3net_ft.data import _build_parquet_index, _row_to_atoms_and_density

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            n = 3
            grid = json.dumps(np.ones((10, 10, 10)).tolist())
            table = pa.table(
                {
                    # required columns
                    "compressed_charge_density": pa.array([grid] * n, type=pa.string()),
                    "species_at_sites": pa.array([["Fe"]] * n),
                    "cartesian_site_positions": pa.array([[[0.0, 0.0, 0.0]]] * n),
                    "lattice_vectors": pa.array(
                        [[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]] * n
                    ),
                    # extras analogous to what Entalpic/lemat-rho-v1 added in 2026:
                    "bader_charges": pa.array([[0.42]] * n),
                    "bader_volumes": pa.array([[11.7]] * n),
                    "material_id": pa.array([f"mat_{i}" for i in range(n)]),
                }
            )
            pq.write_table(table, d / "chunk_000.parquet")

            # build_parquet_index should still find all 3 valid rows
            file_paths, index = _build_parquet_index(d)
            assert len(index) == n
            assert len(file_paths) == 1

            # _row_to_atoms_and_density should produce a usable atoms+density
            # even when the row dict contains the extras (it indexes the
            # required keys directly, so the extras are dead weight).
            row = {
                "species_at_sites": ["Fe"],
                "cartesian_site_positions": [[0.0, 0.0, 0.0]],
                "lattice_vectors": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                "compressed_charge_density": grid,
                "bader_charges": [0.42],
                "bader_volumes": [11.7],
                "material_id": "mat_0",
            }
            atoms, density, origin = _row_to_atoms_and_density(row)
            assert len(atoms) == 1
            assert density.shape == (10, 10, 10)
            np.testing.assert_array_equal(origin, np.zeros(3))


# ---------------------------------------------------------------------------
# LRU eviction for the per-worker parquet table cache.
#
# Why this is here (regression test for the OOM that killed jobs 4971293 and
# 4971343): without eviction, each DataLoader worker accumulates every chunk
# it has ever read. With 8 workers per rank x 4 DDP ranks = 32 workers, and
# ~2 GB of pyarrow-decompressed table per chunk, the cache alone can grow to
# ~140 GB on a long run. The OOM hit at MaxRSS=35 GB per rank x 4 = 140 GB,
# above our 125 GB --mem budget.
#
# The fix: cap the cache. A small LRU bounded by `_TABLE_CACHE_MAX_CHUNKS`
# evicts the least-recently-used chunk before adding a new one.
# ---------------------------------------------------------------------------


class TestTableCacheLRU:
    """LeMatRhoDataset's _TABLE_CACHE must evict to stay below a bounded size."""

    def _write_n_chunks(self, d: Path, n: int):
        for i in range(n):
            _write_one_row_chunk(d / f"chunk_{i:03d}.parquet")

    def test_cache_size_is_bounded(self):
        """After reading from many chunks, the cache must not contain all of them."""
        from charge3net_ft import data as data_mod

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            n_chunks = 10
            self._write_n_chunks(d, n_chunks)

            # Force a small cap so the test is fast and unambiguous.
            original_max = getattr(data_mod, "_TABLE_CACHE_MAX_CHUNKS", None)
            data_mod._TABLE_CACHE_MAX_CHUNKS = 3
            data_mod._TABLE_CACHE.clear()
            try:
                ds = data_mod.LeMatRhoDataset(parquet_dir=d, num_probes=None)
                for i in range(len(ds)):
                    _ = ds._read_row(i)
                assert len(data_mod._TABLE_CACHE) <= 3, (
                    "cache grew beyond _TABLE_CACHE_MAX_CHUNKS=3; "
                    f"actual size {len(data_mod._TABLE_CACHE)}"
                )
            finally:
                if original_max is not None:
                    data_mod._TABLE_CACHE_MAX_CHUNKS = original_max
                data_mod._TABLE_CACHE.clear()

    def test_cache_evicts_least_recently_used(self):
        """When the cache is full, the next miss should drop the LRU entry."""
        from charge3net_ft import data as data_mod

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_n_chunks(d, 5)
            data_mod._TABLE_CACHE_MAX_CHUNKS = 2
            data_mod._TABLE_CACHE.clear()
            try:
                ds = data_mod.LeMatRhoDataset(parquet_dir=d, num_probes=None)
                # Touch chunks 0, 1 -> cache holds {0, 1}
                ds._read_row(0)
                ds._read_row(1)
                assert set(data_mod._TABLE_CACHE.keys()) == {0, 1}
                # Touch chunk 2 -> the LRU (0) should evict, cache holds {1, 2}
                ds._read_row(2)
                assert set(data_mod._TABLE_CACHE.keys()) == {1, 2}, (
                    f"expected LRU eviction of chunk 0, got cache keys "
                    f"{set(data_mod._TABLE_CACHE.keys())}"
                )
                # Re-access 1 -> bumps 1 to most-recent; cache still {1, 2}
                ds._read_row(1)
                # Touch 3 -> 2 is now LRU, evict 2, cache holds {1, 3}
                ds._read_row(3)
                assert set(data_mod._TABLE_CACHE.keys()) == {1, 3}, (
                    f"expected LRU eviction of chunk 2 after re-access of 1; "
                    f"got cache keys {set(data_mod._TABLE_CACHE.keys())}"
                )
            finally:
                data_mod._TABLE_CACHE.clear()

    def test_cache_max_default_is_reasonable(self):
        """The default cap must be > 0 and small enough that 8 workers x cap
        worth of cached chunks fits well below per-rank memory budgets.

        With ~2 GB per chunk and ~8 workers per rank, a default of 5 caps
        the per-rank cache at ~80 GB worst case (only chunks the worker
        actually saw count; in practice well under). We pick 5 to leave
        plenty of margin under a 32-GB-per-rank shared-mode allocation.
        """
        from charge3net_ft import data as data_mod

        assert hasattr(data_mod, "_TABLE_CACHE_MAX_CHUNKS"), (
            "_TABLE_CACHE_MAX_CHUNKS must be defined for the LRU to work"
        )
        assert 1 <= data_mod._TABLE_CACHE_MAX_CHUNKS <= 20, (
            f"_TABLE_CACHE_MAX_CHUNKS={data_mod._TABLE_CACHE_MAX_CHUNKS} is "
            "outside the sensible range [1, 20]; very small evicts too "
            "aggressively for shuffled access, very large defeats the cap"
        )


def _write_one_row_chunk(path: Path):
    """Helper: one valid row per chunk; used by the LRU eviction tests."""
    table = pa.table(
        {
            "compressed_charge_density": pa.array(
                [json.dumps(np.ones((10, 10, 10)).tolist())], type=pa.string()
            ),
            "species_at_sites": pa.array([["Fe"]]),
            "cartesian_site_positions": pa.array([[[0.0, 0.0, 0.0]]]),
            "lattice_vectors": pa.array(
                [[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]]
            ),
        }
    )
    pq.write_table(table, path)
