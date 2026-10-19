"""TDD tests for the LeMat-Rho → DeepDFT data adapter.

DeepDFT (peterbjorgensen/DeepDFT) consumes a per-sample dict of the form::

    {
        "density":       np.ndarray (Nx, Ny, Nz),
        "atoms":         ase.Atoms,
        "origin":        np.ndarray (3,),
        "grid_position": np.ndarray (Nx, Ny, Nz, 3),
        "metadata":      dict,                    # must contain "filename"
    }

Our adapter ``LeMatRhoDeepDFTDataset`` reuses the existing
``_row_to_atoms_and_density`` and ``_build_parquet_index`` helpers in
``charge3net_ft.data`` (so the input pipeline is shared between models) and
returns DeepDFT's dict shape directly. No tar/CHGCAR conversion needed.

``charge3net_ft.data`` needs the ``../charge3net`` sibling clone (its
module-level sys.path block raises RuntimeError without it), so the whole
module skips when the sibling is absent.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import ase
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

try:
    import charge3net_ft.data  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"charge3net sibling repo unavailable: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers — write a synthetic chunk_*.parquet with the same schema the real
# LeMat-Rho data has, plus the Bader columns it gained in v1.
# ---------------------------------------------------------------------------
def _write_synthetic_chunk(path: Path, n_valid: int = 3) -> None:
    grid = json.dumps(np.ones((10, 10, 10), dtype=np.float32).tolist())
    table = pa.table(
        {
            "compressed_charge_density": pa.array([grid] * n_valid, type=pa.string()),
            "species_at_sites": pa.array([["Fe"]] * n_valid),
            "cartesian_site_positions": pa.array([[[0.0, 0.0, 0.0]]] * n_valid),
            "lattice_vectors": pa.array(
                [[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]] * n_valid
            ),
            # extras DeepDFT must ignore
            "bader_charges": pa.array([[0.42]] * n_valid),
            "material_id": pa.array([f"mat_{i}" for i in range(n_valid)]),
        }
    )
    pq.write_table(table, path)


class TestLeMatRhoDeepDFTDataset:
    """Adapter __getitem__ returns DeepDFT's exact dict contract."""

    def test_length_matches_valid_rows(self):
        from deepdft_ft.data import LeMatRhoDeepDFTDataset

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "chunk_000.parquet", n_valid=5)
            _write_synthetic_chunk(d / "chunk_001.parquet", n_valid=3)
            ds = LeMatRhoDeepDFTDataset(parquet_dir=d)
            assert len(ds) == 8

    def test_item_has_all_required_keys(self):
        """DeepDFT's collate_fn reads density, atoms, origin, grid_position, metadata."""
        from deepdft_ft.data import LeMatRhoDeepDFTDataset

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "chunk_000.parquet", n_valid=1)
            ds = LeMatRhoDeepDFTDataset(parquet_dir=d)
            sample = ds[0]
            for key in ("density", "atoms", "origin", "grid_position", "metadata"):
                assert key in sample, (
                    f"DeepDFT expects key {key!r}; got {list(sample.keys())}"
                )

    def test_item_density_is_3d_numpy(self):
        from deepdft_ft.data import LeMatRhoDeepDFTDataset

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "chunk_000.parquet", n_valid=1)
            sample = LeMatRhoDeepDFTDataset(parquet_dir=d)[0]
            assert isinstance(sample["density"], np.ndarray)
            assert sample["density"].shape == (10, 10, 10), (
                f"expected (10, 10, 10) density; got {sample['density'].shape}"
            )

    def test_item_atoms_is_ase_atoms_with_pbc(self):
        """Periodic boundary conditions matter for any solid-state density."""
        from deepdft_ft.data import LeMatRhoDeepDFTDataset

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "chunk_000.parquet", n_valid=1)
            sample = LeMatRhoDeepDFTDataset(parquet_dir=d)[0]
            assert isinstance(sample["atoms"], ase.Atoms)
            assert all(sample["atoms"].pbc), (
                "LeMat-Rho cells are periodic; atoms.pbc must be (True, True, True)"
            )

    def test_item_origin_is_3vec_zeros(self):
        """LeMat-Rho stores grids at fractional (0, 0, 0); the adapter mirrors that."""
        from deepdft_ft.data import LeMatRhoDeepDFTDataset

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "chunk_000.parquet", n_valid=1)
            sample = LeMatRhoDeepDFTDataset(parquet_dir=d)[0]
            assert isinstance(sample["origin"], np.ndarray)
            np.testing.assert_array_equal(sample["origin"], np.zeros(3))

    def test_item_grid_position_shape_matches_density(self):
        """grid_position is (Nx, Ny, Nz, 3) Cartesian probe coordinates."""
        from deepdft_ft.data import LeMatRhoDeepDFTDataset

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "chunk_000.parquet", n_valid=1)
            sample = LeMatRhoDeepDFTDataset(parquet_dir=d)[0]
            assert sample["grid_position"].shape == (10, 10, 10, 3), (
                f"grid_position must be (Nx, Ny, Nz, 3); got {sample['grid_position'].shape}"
            )

    def test_grid_position_origin_is_zero(self):
        """grid_position[0, 0, 0] must be the cell origin (0, 0, 0)."""
        from deepdft_ft.data import LeMatRhoDeepDFTDataset

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "chunk_000.parquet", n_valid=1)
            sample = LeMatRhoDeepDFTDataset(parquet_dir=d)[0]
            np.testing.assert_allclose(sample["grid_position"][0, 0, 0], np.zeros(3))

    def test_grid_position_uses_cell_matrix(self):
        """grid_position[1, 0, 0] should be one step along the a vector.

        For our synthetic 10×10×10 grid with a 4-Å cubic cell:
          frac coord at index (1, 0, 0) = (1/10, 0, 0)
          Cartesian   = frac @ cell    = (4/10, 0, 0) = (0.4, 0, 0)
        """
        from deepdft_ft.data import LeMatRhoDeepDFTDataset

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "chunk_000.parquet", n_valid=1)
            sample = LeMatRhoDeepDFTDataset(parquet_dir=d)[0]
            np.testing.assert_allclose(
                sample["grid_position"][1, 0, 0], [0.4, 0.0, 0.0], atol=1e-5
            )

    def test_item_metadata_has_filename(self):
        """DeepDFT logs reference filename — must be a stable string per sample."""
        from deepdft_ft.data import LeMatRhoDeepDFTDataset

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "chunk_000.parquet", n_valid=2)
            ds = LeMatRhoDeepDFTDataset(parquet_dir=d)
            for i in range(len(ds)):
                meta = ds[i]["metadata"]
                assert "filename" in meta, f"metadata missing 'filename'; got {meta}"
                assert isinstance(meta["filename"], str)
            # Filenames should differ across samples so DeepDFT logs don't collide.
            assert ds[0]["metadata"]["filename"] != ds[1]["metadata"]["filename"]

    def test_ignores_extra_columns(self):
        """Bader / material_id columns added to LeMat-Rho v1 are dead weight here.

        Same regression we already pinned for charge3net_ft.data; mirroring it
        on the DeepDFT path keeps the two adapters honest in lockstep.
        """
        from deepdft_ft.data import LeMatRhoDeepDFTDataset

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_synthetic_chunk(d / "chunk_000.parquet", n_valid=1)
            sample = LeMatRhoDeepDFTDataset(parquet_dir=d)[0]
            # The synthetic chunk includes bader_charges + material_id columns.
            # The adapter should successfully ingest the row regardless.
            assert sample["density"].shape == (10, 10, 10)


class TestGenerateDatasplits:
    """The runner's train/val split must be restart-stable.

    submit_deepdft_adastra.sh resumes with --load_model and no --split_file,
    so the split is regenerated on every restart. An unseeded permutation
    then leaks previous val rows into train (and DDP ranks disagree); two
    computations with identical args must produce identical splits.
    """

    def test_same_args_give_same_split(self):
        from deepdft_ft.data import generate_datasplits

        assert generate_datasplits(100) == generate_datasplits(100)

    def test_split_is_disjoint_and_complete(self):
        from deepdft_ft.data import generate_datasplits

        splits = generate_datasplits(100)
        assert len(splits["validation"]) == 5  # ceil(100 * 0.05), as upstream
        assert sorted(splits["train"] + splits["validation"]) == list(range(100))

    def test_different_seed_changes_split(self):
        from deepdft_ft.data import generate_datasplits

        assert generate_datasplits(100, seed=0) != generate_datasplits(100, seed=1)


class TestRaisesOnEmptyDir:
    def test_no_chunks_in_dir_raises(self):
        from deepdft_ft.data import LeMatRhoDeepDFTDataset

        with tempfile.TemporaryDirectory() as tmp, pytest.raises(FileNotFoundError):
            LeMatRhoDeepDFTDataset(parquet_dir=Path(tmp))
