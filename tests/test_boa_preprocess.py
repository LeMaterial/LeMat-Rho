"""Tests for the boa_ft parquet -> LMDB preprocessor.

Covers the pure row-to-graph conversion, the datasplit writer, and a small
end-to-end run that round-trips through BOA's ``LmdbDataset``. The pure-helper
tests (max-z filter, datasplit writer, bounded table cache) run anywhere; the
graph and LMDB round-trip tests require the sibling ``boa`` clone (for
``scdp``) and ``charge3net`` (for the shared parquet decoder), so they skip
outside the boa_ft environment described in boa_ft/README.md.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# A 4x4x4 grid keeps the probe count small (64) while still being a real 3D box.
_GRID_N = 4
_CELL = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]


def _write_synthetic_chunk(path: Path, n_valid: int = 3) -> None:
    """Write a chunk_*.parquet with the LeMat-Rho schema (H2 in a cubic box)."""
    grid = json.dumps(np.ones((_GRID_N, _GRID_N, _GRID_N), dtype=np.float32).tolist())
    table = pa.table(
        {
            "compressed_charge_density": pa.array([grid] * n_valid, type=pa.string()),
            "species_at_sites": pa.array([["H", "H"]] * n_valid),
            "cartesian_site_positions": pa.array(
                [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]]] * n_valid
            ),
            "lattice_vectors": pa.array([_CELL] * n_valid),
            # extras the preprocessor must ignore
            "material_id": pa.array([f"mat_{i}" for i in range(n_valid)]),
        }
    )
    pq.write_table(table, path)


class TestRowToAtomicData:
    """A parquet row becomes an scdp AtomicData with the expected fields."""

    def _one_graph(self, tmp: Path):
        pytest.importorskip("scdp")
        from scdp.scripts.preprocess import get_atomic_number_table_from_zs

        from boa_ft.preprocess import row_to_atomic_data
        from charge3net_ft.data import _build_parquet_index

        _write_synthetic_chunk(tmp / "chunk_000.parquet", n_valid=1)
        file_paths, _ = _build_parquet_index(tmp)
        table = pq.read_table(file_paths[0])
        row = {c: table.column(c)[0].as_py() for c in table.column_names}
        z_table = get_atomic_number_table_from_zs(np.arange(100).tolist())
        return row_to_atomic_data(
            row, "mat_0", z_table, atom_cutoff=4.0, max_neighbors=None
        )

    def test_atoms_and_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self._one_graph(Path(tmp))
            assert data.n_atom == 2
            assert data.atom_types.tolist() == [1, 1]
            np.testing.assert_allclose(data.cell[0].numpy(), np.array(_CELL), atol=1e-5)

    def test_no_virtual_nodes(self):
        """vnode_method='none' means the graph carries only real atoms."""
        with tempfile.TemporaryDirectory() as tmp:
            data = self._one_graph(Path(tmp))
            assert data.n_vnode == 0
            assert int(data.num_nodes) == 2

    def test_probe_grid_matches_density(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self._one_graph(Path(tmp))
            n_points = _GRID_N**3
            assert data.chg_labels.shape[0] == n_points
            assert data.probe_coords.shape == (n_points, 3)

    def test_grid_origin_is_cell_corner(self):
        """Fractional (0, 0, 0) probe maps to Cartesian origin."""
        with tempfile.TemporaryDirectory() as tmp:
            data = self._one_graph(Path(tmp))
            np.testing.assert_allclose(
                data.probe_coords[0].numpy(), np.zeros(3), atol=1e-5
            )


class TestRowExceedsMaxZ:
    """Rows with elements beyond the basis coverage are flagged for skipping."""

    def test_actinide_row_exceeds_def2_svp_ceiling(self):
        from boa_ft.preprocess import row_exceeds_max_z

        assert row_exceeds_max_z({"species_at_sites": ["U", "O", "O"]}, 86)

    def test_light_row_passes(self):
        from boa_ft.preprocess import row_exceeds_max_z

        assert not row_exceeds_max_z({"species_at_sites": ["H", "Rn"]}, 86)


class TestWriteDatasplits:
    """Splits are disjoint, cover the whole range, and match charge3net sizes."""

    def test_split_sizes_and_disjoint(self):
        from boa_ft.preprocess import write_datasplits

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            splits = write_datasplits(100, out, val_frac=0.05, test_frac=0.05, seed=42)
            assert len(splits["train"]) == 90
            assert len(splits["validation"]) == 5
            assert len(splits["test"]) == 5
            all_idx = splits["train"] + splits["validation"] + splits["test"]
            assert sorted(all_idx) == list(range(100))
            assert (out / "datasplits.json").exists()

    def test_split_is_seed_deterministic(self):
        from boa_ft.preprocess import write_datasplits

        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            a = write_datasplits(50, Path(t1), 0.1, 0.1, seed=42)
            b = write_datasplits(50, Path(t2), 0.1, 0.1, seed=42)
            assert a == b


class TestReadRowCached:
    """The preprocess table cache is a bounded LRU, not an unbounded dict.

    Mirrors the 5-chunk LRU in ``deepdft_ft.data`` (the unbounded variant
    held every decompressed pyarrow table for the whole run).
    """

    def test_cache_never_exceeds_cap(self):
        import collections

        from boa_ft.preprocess import _TABLE_CACHE_MAX_CHUNKS, read_row_cached

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            n_chunks = _TABLE_CACHE_MAX_CHUNKS + 2
            file_paths = []
            for i in range(n_chunks):
                p = tmp / f"chunk_{i:03d}.parquet"
                _write_synthetic_chunk(p, n_valid=1)
                file_paths.append(p)
            cache: collections.OrderedDict = collections.OrderedDict()
            for fi in range(n_chunks):
                read_row_cached(file_paths, fi, 0, cache)
            assert len(cache) <= _TABLE_CACHE_MAX_CHUNKS

    def test_reread_after_eviction_roundtrips(self):
        import collections

        from boa_ft.preprocess import _TABLE_CACHE_MAX_CHUNKS, read_row_cached

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            n_chunks = _TABLE_CACHE_MAX_CHUNKS + 2
            file_paths = []
            for i in range(n_chunks):
                p = tmp / f"chunk_{i:03d}.parquet"
                _write_synthetic_chunk(p, n_valid=1)
                file_paths.append(p)
            cache: collections.OrderedDict = collections.OrderedDict()
            first = read_row_cached(file_paths, 0, 0, cache)
            for fi in range(n_chunks):  # cycle far enough to evict chunk 0
                read_row_cached(file_paths, fi, 0, cache)
            again = read_row_cached(file_paths, 0, 0, cache)
            assert again == first


class TestPreprocessEndToEnd:
    """A full run writes shards + metadata that LmdbDataset can read back."""

    def test_roundtrip_through_lmdb_dataset(self):
        import argparse

        pytest.importorskip("scdp")
        pytest.importorskip("boa")
        from boa.data.dataset import LmdbDataset

        from boa_ft.preprocess import main

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            parquet_dir = tmp / "parquet"
            parquet_dir.mkdir()
            _write_synthetic_chunk(parquet_dir / "chunk_000.parquet", n_valid=5)
            _write_synthetic_chunk(parquet_dir / "chunk_001.parquet", n_valid=3)
            out_dir = tmp / "lematrho"

            args = argparse.Namespace(
                parquet_dir=str(parquet_dir),
                out_dir=str(out_dir),
                num_shards=2,
                limit=None,
                atom_cutoff=4.0,
                max_neighbors=None,
                map_size_gb=1,
                val_frac=0.2,
                test_frac=0.2,
                seed=42,
            )
            main(args)

            # atomic_numbers.json reflects the only element present.
            atomic_numbers = json.loads((out_dir / "atomic_numbers.json").read_text())
            assert atomic_numbers == [1]

            # datasplits cover every written sample.
            splits = json.loads((out_dir / "datasplits.json").read_text())
            n = sum(len(splits[k]) for k in ("train", "validation", "test"))
            assert n == 8

            # LmdbDataset sees all 8 samples across the two shards.
            ds = LmdbDataset(out_dir / "data")
            assert len(ds) == 8
            sample = ds[0]
            assert sample.n_atom == 2
