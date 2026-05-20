"""LeMat-Rho → DeepDFT data adapter.

DeepDFT's ``runner.py`` expects a ``torch.utils.data.Dataset`` that yields
per-sample dicts of the form::

    {
        "density":       np.ndarray (Nx, Ny, Nz),
        "atoms":         ase.Atoms,
        "origin":        np.ndarray (3,),
        "grid_position": np.ndarray (Nx, Ny, Nz, 3),
        "metadata":      {"filename": str, ...},
    }

That dict is fed into DeepDFT's ``CollateFuncRandomSample`` which samples
random probe points, builds the atom/probe graph via asap3, and pads the
batch. The only thing we provide is a path from a directory of LeMat-Rho
parquet chunks to that dict shape.

The parquet schema, the index building, and the row → (atoms, density, origin)
conversion live in ``charge3net_ft.data`` and are reused verbatim. Keeping a
single source of truth for the input pipeline means a future Bader/extra-column
addition only needs one regression test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from charge3net_ft.data import (
    _COLUMNS,
    _build_parquet_index,
    _row_to_atoms_and_density,
)


# Per-worker cache, separate from charge3net_ft's so the two pipelines don't
# step on each other when running side by side in the same process.
_DEEPDFT_TABLE_CACHE: dict = {}


def _calculate_grid_pos(density: np.ndarray, origin: np.ndarray, cell) -> np.ndarray:
    """Cartesian probe positions for an (Nx, Ny, Nz) density grid.

    Same formula DeepDFT uses internally (see DeepDFT/dataset.py:_calculate_grid_pos).
    Kept here so we don't need DeepDFT importable at test time.

    Parameters
    ----------
    density : np.ndarray of shape (Nx, Ny, Nz)
        Used only for its shape.
    origin : np.ndarray of shape (3,)
        Cell-frame origin in Cartesian coordinates.
    cell : ASE Cell or 3x3 array
        Lattice vectors as rows.

    Returns
    -------
    grid_pos : np.ndarray of shape (Nx, Ny, Nz, 3)
        Cartesian coordinates of every grid point.
    """
    ngridpts = np.array(density.shape)
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / density.shape[0],
        np.arange(ngridpts[1]) / density.shape[1],
        np.arange(ngridpts[2]) / density.shape[2],
        indexing="ij",
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, np.asarray(cell))
    grid_pos = grid_pos + origin
    return grid_pos


class LeMatRhoDeepDFTDataset(Dataset):
    """Iterate LeMat-Rho parquet chunks as DeepDFT-shaped sample dicts.

    Parameters
    ----------
    parquet_dir : str or Path
        Directory containing ``chunk_*.parquet`` files.
    _shared_index : tuple, optional
        Internal: pre-built (file_paths, index) tuple shared between
        train/val splits to avoid scanning files twice.
    """

    def __init__(
        self,
        parquet_dir: str | Path | None = None,
        _shared_index: Optional[tuple] = None,
    ):
        if _shared_index is not None:
            self._file_paths, self._index = _shared_index
        else:
            if parquet_dir is None:
                raise ValueError("Must provide parquet_dir or _shared_index")
            self._file_paths, self._index = _build_parquet_index(Path(parquet_dir))

    def __len__(self) -> int:
        return len(self._index)

    def _read_row(self, idx: int) -> dict:
        """Lazy per-worker chunk caching, mirrors charge3net_ft.data.

        Cache is keyed by the absolute parquet path (not the integer ``fi``)
        so multiple ``LeMatRhoDeepDFTDataset`` instances pointing at different
        directories don't collide on ``fi=0``.
        """
        fi, ri = self._index[idx]
        key = str(self._file_paths[fi].resolve())
        if key not in _DEEPDFT_TABLE_CACHE:
            _DEEPDFT_TABLE_CACHE[key] = pq.read_table(
                self._file_paths[fi], columns=_COLUMNS
            )
        table = _DEEPDFT_TABLE_CACHE[key]
        return {col: table.column(col)[ri].as_py() for col in _COLUMNS}

    def __getitem__(self, idx: int) -> dict:
        row = self._read_row(idx)
        atoms, density, origin = _row_to_atoms_and_density(row)
        grid_pos = _calculate_grid_pos(density, origin, atoms.get_cell())

        # Index-derived filename so DeepDFT logs stay distinguishable across
        # samples. Format mirrors the tar member names DeepDFT normally sees.
        fi, ri = self._index[idx]
        chunk_stem = Path(self._file_paths[fi]).stem  # e.g. "chunk_000017"
        filename = f"{chunk_stem}_row{ri:06d}.parquet"

        return {
            "density": density,
            "atoms": atoms,
            "origin": origin,
            "grid_position": grid_pos,
            "metadata": {"filename": filename},
        }
