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

import collections
import math
from pathlib import Path

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
#
# Bounded LRU, ported from charge3net_ft.data: the unbounded dict version of
# this cache held one decompressed pyarrow table (~2 GB with the inflated
# compressed_charge_density strings) per chunk file forever, the same failure
# mode that OOM-killed charge3net jobs 4971293/4971343. Cap of 5 chunks keeps
# each worker's cache around 10 GB worst case. OrderedDict gives O(1) LRU.
_DEEPDFT_TABLE_CACHE_MAX_CHUNKS = 5
_DEEPDFT_TABLE_CACHE: collections.OrderedDict[str, object] = collections.OrderedDict()


def sample_probe_indices(
    n_grid_points: int,
    n_probes: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample flat grid indices for density probes, capped at ``n_probes``.

    Upstream DeepDFT samples probes WITH replacement
    (``np.random.randint``), which on a 1000-point LeMat-Rho grid with
    5000 probes yields 5x duplicates for zero statistical value while
    ``probes_to_graph`` cost grows quadratically in the probe count
    (the host-RAM OOM in job 5004725). This helper samples without
    replacement whenever the grid has at least ``n_probes`` points, and
    always returns exactly ``n_probes`` indices (upstream padding/eval
    assumes a uniform probe count per sample, so grids smaller than the
    cap fall back to with-replacement sampling rather than shrinking).

    Kept here, DeepDFT-import-free, so it stays testable without the
    upstream clone on sys.path (same rationale as ``_calculate_grid_pos``).

    Parameters
    ----------
    n_grid_points : int
        Total number of grid points (``prod(grid_shape)``).
    n_probes : int
        Number of probe indices to draw.
    rng : np.random.Generator, optional
        Source of randomness; a fresh default generator when omitted.

    Returns
    -------
    indices : np.ndarray of shape (n_probes,)
        Flat indices into the grid, unique when
        ``n_grid_points >= n_probes``.

    .. code-block:: python

        flat = sample_probe_indices(15**3, 1000)
        probe_choice = np.unravel_index(flat, grid_pos.shape[0:3])
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_grid_points >= n_probes:
        return rng.choice(n_grid_points, size=n_probes, replace=False)
    return rng.integers(n_grid_points, size=n_probes)


def generate_datasplits(datalen: int, seed: int = 0) -> dict[str, list[int]]:
    """Generate the runner's 95/5 train/validation split of ``datalen`` rows.

    Upstream DeepDFT drew ``np.random.permutation`` unseeded, so a job that
    restarts with ``--load_model`` and no ``--split_file`` (exactly what
    submit_deepdft_adastra.sh does) regenerated a DIFFERENT split and leaked
    previous validation rows into train; DDP ranks could disagree the same
    way. Seeding makes the split a pure function of (datalen, seed).

    Kept here, DeepDFT-import-free, so it stays testable without the
    upstream clone on sys.path (same rationale as ``sample_probe_indices``).

    Parameters
    ----------
    datalen : int
        Total number of samples in the dataset.
    seed : int
        Split RNG seed (the runner's ``--split-seed``, default 0).

    Returns
    -------
    dict
        ``{"train": [...], "validation": [...]}`` index lists; validation
        holds ``ceil(0.05 * datalen)`` rows, matching upstream.

    .. code-block:: python

        splits = generate_datasplits(len(dataset), seed=args.split_seed)
    """
    num_validation = math.ceil(datalen * 0.05)
    indices = np.random.default_rng(seed).permutation(datalen)
    return {
        "train": indices[num_validation:].tolist(),
        "validation": indices[:num_validation].tolist(),
    }


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
        _shared_index: tuple | None = None,
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
        directories don't collide on ``fi=0``. Capped at
        ``_DEEPDFT_TABLE_CACHE_MAX_CHUNKS`` entries; on a miss past capacity
        the least recently used chunk is evicted.
        """
        fi, ri = self._index[idx]
        key = str(self._file_paths[fi].resolve())
        if key in _DEEPDFT_TABLE_CACHE:
            # Refresh recency on hit so hot chunks survive eviction.
            _DEEPDFT_TABLE_CACHE.move_to_end(key)
        else:
            if len(_DEEPDFT_TABLE_CACHE) >= _DEEPDFT_TABLE_CACHE_MAX_CHUNKS:
                _DEEPDFT_TABLE_CACHE.popitem(last=False)
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
