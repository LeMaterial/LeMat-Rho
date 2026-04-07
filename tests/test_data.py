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
        "src", "src.charge3net", "src.charge3net.data",
        "src.charge3net.data.collate", "src.charge3net.data.graph_construction",
        "src.utils", "src.utils.data",
    ]
    stubs = {}
    for mod in fake_modules:
        stubs[mod] = type(sys)("mod")
    stubs["src.charge3net.data.collate"].collate_list_of_dicts = lambda *a, **kw: None
    stubs["src.charge3net.data.graph_construction"].KdTreeGraphConstructor = object
    stubs["src.utils.data"].calculate_grid_pos = lambda *a, **kw: None

    with patch.dict(sys.modules, stubs):
        # Also patch the existence check so it doesn't raise
        with patch("pathlib.Path.exists", return_value=True):
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
        atoms, density, origin = _row_to_atoms_and_density(row)
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
        table = pa.table({
            "compressed_charge_density": pa.array(valid + null, type=pa.string()),
            "species_at_sites": pa.array([["Fe"]] * (n_valid + n_null)),
            "cartesian_site_positions": pa.array([[[0.0, 0.0, 0.0]]] * (n_valid + n_null)),
            "lattice_vectors": pa.array([[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]] * (n_valid + n_null)),
        })
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
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                _build_parquet_index(Path(tmp))
