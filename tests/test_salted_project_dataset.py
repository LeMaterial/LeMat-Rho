"""TDD tests for the Phase D2 dataset-projection module.

Locks the contract for ``salted_ft.project_dataset.project_chunk``,
which reads a LeMat-Rho-format parquet chunk, runs
``project_chgcar_to_basis`` row by row, and writes a parallel parquet
chunk of projected coefficients.

Output schema per row::

    {
        "row_index":       int (matches the original chunk row index),
        "material_id":     str (carried through if present, else "" ),
        "n_atoms":         int,
        "atomic_numbers":  list[int],
        "lattice_vectors": list[list[float]],   # 3x3
        "n_electrons":     float (integrated density * cell_volume / n_grid),
        "grid_shape":      list[int],            # [Nx, Ny, Nz]
        "coefficients":    list[list[float]],    # (n_atoms, n_coeffs_per_atom)
        "basis_set_NMAPE": float (basis-ceiling NMAPE for this row),
    }

The basis_set_NMAPE column is the per-row reconstruction error from
roundtripping; we keep it so downstream sanity-checks can know each
sample's basis ceiling.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _write_synthetic_chunk(path: Path, n_rows: int = 3) -> None:
    """Write a LeMat-Rho-format chunk for use by the projection script."""
    rng = np.random.default_rng(42)
    grids = [
        json.dumps(rng.random((10, 10, 10), dtype=np.float64).tolist())
        for _ in range(n_rows)
    ]
    table = pa.table(
        {
            "compressed_charge_density": pa.array(grids, type=pa.string()),
            "species_at_sites": pa.array([["Fe"]] * n_rows),
            "cartesian_site_positions": pa.array([[[2.0, 2.0, 2.0]]] * n_rows),
            "lattice_vectors": pa.array(
                [[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]] * n_rows
            ),
            # extras: confirm they get ignored
            "bader_charges": pa.array([[0.4]] * n_rows),
            "material_id": pa.array([f"mat_{i:03d}" for i in range(n_rows)]),
        }
    )
    pq.write_table(table, path)


class TestProjectChunkContract:
    """``project_chunk(in_path, out_path, basis_spec)`` -> None.

    Reads ``in_path`` (LeMat-Rho format), projects each row, writes
    ``out_path`` in the schema documented at the top of this file.
    """

    def test_output_file_written(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.project_dataset import project_chunk

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "in.parquet", n_rows=2)
            out = d / "out.parquet"
            project_chunk(d / "in.parquet", out, BasisSpec())
            assert out.exists()
            assert out.stat().st_size > 0

    def test_row_count_matches_valid_input(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.project_dataset import project_chunk

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "in.parquet", n_rows=3)
            out = d / "out.parquet"
            project_chunk(d / "in.parquet", out, BasisSpec())
            t = pq.read_table(out)
            assert len(t) == 3

    def test_required_columns_present(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.project_dataset import project_chunk

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "in.parquet", n_rows=2)
            out = d / "out.parquet"
            project_chunk(d / "in.parquet", out, BasisSpec())
            t = pq.read_table(out)
            required = {
                "row_index",
                "material_id",
                "n_atoms",
                "atomic_numbers",
                "lattice_vectors",
                "n_electrons",
                "grid_shape",
                "coefficients",
                "basis_set_NMAPE",
            }
            missing = required - set(t.column_names)
            assert not missing, f"missing required columns: {missing}"

    def test_coefficient_shape_per_row(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.project_dataset import project_chunk

        spec = BasisSpec()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "in.parquet", n_rows=2)
            out = d / "out.parquet"
            project_chunk(d / "in.parquet", out, spec)
            t = pq.read_table(out).to_pydict()
            for c, n_atoms in zip(t["coefficients"], t["n_atoms"], strict=True):
                # Each row has its own coefficient block; first dim is n_atoms,
                # second is n_coeffs_per_atom.
                arr = np.asarray(c)
                assert arr.shape == (n_atoms, spec.n_coeffs_per_atom), (
                    f"row coefficient shape mismatch: got {arr.shape}, "
                    f"expected ({n_atoms}, {spec.n_coeffs_per_atom})"
                )

    def test_basis_set_NMAPE_is_finite_and_nonnegative(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.project_dataset import project_chunk

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "in.parquet", n_rows=3)
            out = d / "out.parquet"
            project_chunk(d / "in.parquet", out, BasisSpec())
            t = pq.read_table(out).to_pydict()
            for x in t["basis_set_NMAPE"]:
                assert np.isfinite(x)
                assert x >= 0.0

    def test_material_id_preserved(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.project_dataset import project_chunk

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "in.parquet", n_rows=3)
            out = d / "out.parquet"
            project_chunk(d / "in.parquet", out, BasisSpec())
            t = pq.read_table(out).to_pydict()
            assert t["material_id"] == ["mat_000", "mat_001", "mat_002"]

    def test_handles_null_charge_density_rows(self):
        """Real LeMat-Rho chunks have some rows with NULL density (failed
        DFT extraction). Those should be skipped, not crash the projection.
        """
        from salted_ft.basis import BasisSpec
        from salted_ft.project_dataset import project_chunk

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            grids = [
                json.dumps(np.ones((10, 10, 10)).tolist()),
                None,  # null density - should be skipped
                json.dumps(np.ones((10, 10, 10)).tolist()),
            ]
            table = pa.table(
                {
                    "compressed_charge_density": pa.array(grids, type=pa.string()),
                    "species_at_sites": pa.array([["Fe"]] * 3),
                    "cartesian_site_positions": pa.array([[[2.0, 2.0, 2.0]]] * 3),
                    "lattice_vectors": pa.array(
                        [[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]] * 3
                    ),
                    "material_id": pa.array(["a", "b", "c"]),
                }
            )
            pq.write_table(table, d / "in.parquet")
            out = d / "out.parquet"
            project_chunk(d / "in.parquet", out, BasisSpec())
            t = pq.read_table(out).to_pydict()
            assert len(t["row_index"]) == 2
            assert t["row_index"] == [0, 2]


class TestProjectDirectory:
    """Driver that runs project_chunk over every chunk_*.parquet in a dir."""

    def test_processes_all_chunks(self):
        from salted_ft.basis import BasisSpec
        from salted_ft.project_dataset import project_directory

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            in_d = d / "in"
            in_d.mkdir()
            out_d = d / "out"
            for i in range(3):
                _write_synthetic_chunk(in_d / f"chunk_{i:06d}.parquet", n_rows=2)
            project_directory(in_d, out_d, BasisSpec())
            outputs = sorted(out_d.glob("chunk_*.parquet"))
            assert len(outputs) == 3
            for out in outputs:
                assert pq.read_table(out).num_rows == 2

    def test_skips_existing_outputs(self):
        """Idempotent: a re-run does not re-project chunks that already exist.

        Lets us resume a partially-completed projection job after an
        interruption without paying the LSQR cost again.
        """
        from salted_ft.basis import BasisSpec
        from salted_ft.project_dataset import project_directory

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            in_d = d / "in"
            in_d.mkdir()
            out_d = d / "out"
            _write_synthetic_chunk(in_d / "chunk_000000.parquet", n_rows=2)
            # First run
            project_directory(in_d, out_d, BasisSpec())
            first_mtime = (out_d / "chunk_000000.parquet").stat().st_mtime
            # Second run should be a no-op
            project_directory(in_d, out_d, BasisSpec())
            second_mtime = (out_d / "chunk_000000.parquet").stat().st_mtime
            assert first_mtime == second_mtime
