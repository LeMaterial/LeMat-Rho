"""
Dataset and DataLoader for LeMatRho charge density data.

Reads Parquet files produced by lematerial-fetcher, converts to charge3net's
expected input format (padded dict batches), and builds PBC-aware atom-atom
and atom-probe graphs using charge3net's KdTreeGraphConstructor.

Uses lazy loading: only an index of valid rows is stored in memory (~2 MB).
Parquet rows are read on-the-fly in __getitem__. Each DataLoader worker caches
opened tables per chunk file so each file is read from disk only once per worker.
"""

import collections
import json
import sys
from functools import partial
from pathlib import Path

import ase
import ase.data
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# charge3net imports — add the cloned repo to sys.path so that its internal
# `from src.charge3net.data...` imports resolve correctly.
# Expects: <parent of LeMat-Rho>/charge3net/ (cloned from AIforGreatGood/charge3net)
# ---------------------------------------------------------------------------
_CHARGE3NET_ROOT = Path(__file__).resolve().parent.parent.parent / "charge3net"
if not _CHARGE3NET_ROOT.exists():
    raise RuntimeError(
        f"charge3net repo not found at {_CHARGE3NET_ROOT}.\n"
        "Clone it with: git clone https://github.com/AIforGreatGood/charge3net "
        f"{_CHARGE3NET_ROOT}"
    )
if str(_CHARGE3NET_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHARGE3NET_ROOT))

from src.charge3net.data.collate import collate_list_of_dicts
from src.charge3net.data.graph_construction import KdTreeGraphConstructor
from src.utils.data import calculate_grid_pos

# Columns we actually need from Parquet
_COLUMNS = [
    "species_at_sites",
    "cartesian_site_positions",
    "lattice_vectors",
    "compressed_charge_density",
]

# ---------------------------------------------------------------------------
# Element symbol -> atomic number lookup
# ---------------------------------------------------------------------------
_SYMBOL_TO_Z = {s: z for z, s in enumerate(ase.data.chemical_symbols)}

# Process-local LRU table cache: keyed by file index, populated on first access.
# Each DataLoader worker has its own cache (workers fork the parent), so each
# chunk file is read from disk at most once per worker per cache cycle.
#
# Bounded LRU because the previous unbounded version OOM-killed jobs 4971293
# and 4971343 at MaxRSS=35 GB/rank. Per-chunk decompressed pyarrow tables
# weigh ~2 GB (the compressed_charge_density JSON strings inflate 6x from
# disk). With 8 workers x 4 DDP ranks = 32 workers, an unbounded cache grew
# to ~140 GB total in 6 h.
#
# Cap of 5 chunks per worker keeps each worker's cache around 10 GB worst
# case, well under any per-rank memory budget. OrderedDict gives O(1) LRU.
_TABLE_CACHE_MAX_CHUNKS = 5
_TABLE_CACHE: "collections.OrderedDict[int, object]" = collections.OrderedDict()


def _parse_grid_json(json_str: str) -> np.ndarray:
    """Parse a JSON-serialised 3D grid string into a float32 numpy array."""
    grid = json.loads(json_str)
    return np.array(grid, dtype=np.float32)


def _row_to_atoms_and_density(row: dict) -> tuple:
    """
    Convert a single Parquet row dict into an ase.Atoms object, a 3D density
    grid, and the grid origin.

    Returns
    -------
    atoms : ase.Atoms
        The periodic structure.
    density : np.ndarray
        Shape (Nx, Ny, Nz) charge density grid.
    origin : np.ndarray
        Grid origin, [0, 0, 0] (charge3net convention).
    """
    species = row["species_at_sites"]
    atomic_numbers = [_SYMBOL_TO_Z[s] for s in species]
    positions = np.array(row["cartesian_site_positions"], dtype=np.float64)
    cell = np.array(row["lattice_vectors"], dtype=np.float64)

    atoms = ase.Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=True,
    )

    density = _parse_grid_json(row["compressed_charge_density"])
    origin = np.zeros(3)

    return atoms, density, origin


def _build_parquet_index(parquet_dir: Path) -> tuple:
    """
    Scan all Parquet chunks and build a lightweight index of valid rows
    (those with non-null compressed_charge_density).

    Returns
    -------
    file_paths : list[Path]
        Sorted list of Parquet file paths.
    index : list[tuple[int, int]]
        Each entry is (file_index, row_index_within_file).
    """
    file_paths = sorted(parquet_dir.glob("chunk_*.parquet"))
    if not file_paths:
        raise FileNotFoundError(f"No chunk_*.parquet files in {parquet_dir}")

    index = []
    n_total = 0
    for fi, fp in enumerate(file_paths):
        # Read only the charge density column to check for nulls.
        # Use .is_valid (Arrow scalar) rather than .as_py() is not None to
        # avoid creating Python objects for every row.
        col = pq.read_table(fp, columns=["compressed_charge_density"]).column(
            "compressed_charge_density"
        )
        n_rows = len(col)
        n_total += n_rows
        for ri in range(n_rows):
            if col[ri].is_valid:
                index.append((fi, ri))

    n_valid = len(index)
    print(
        f"LeMatRhoDataset: {n_valid}/{n_total} valid rows indexed from {len(file_paths)} files"
    )
    return file_paths, index


class LeMatRhoDataset(Dataset):
    """
    PyTorch Dataset that lazily reads LeMatRho Parquet chunks and yields
    charge3net-compatible graph dicts.

    Only a lightweight index (~2 MB for 65k rows) is stored in memory.
    Each __getitem__ call retrieves a single row. Chunk files are cached
    per-worker so each file is loaded from disk only once per worker process.

    Parameters
    ----------
    parquet_dir : str or Path
        Directory containing chunk_*.parquet files.
    cutoff : float
        Cutoff radius for neighbor finding (Angstrom). Must match model.
    num_probes : int or None
        If set, randomly subsample this many probe points per sample
        (used during training to save memory). If None, use all grid points.
    _shared_index : tuple or None
        Internal: pre-computed (file_paths, index) to share between
        train/val datasets without scanning files twice.
    """

    def __init__(
        self,
        parquet_dir: str | None = None,
        cutoff: float = 4.0,
        num_probes: int | None = None,
        _shared_index: tuple | None = None,
    ):
        self.cutoff = cutoff
        self.num_probes = num_probes

        self.graph_constructor = KdTreeGraphConstructor(
            cutoff=cutoff,
            num_probes=num_probes,
            disable_pbc=False,
        )

        if _shared_index is not None:
            self._file_paths, self._index = _shared_index
        else:
            self._file_paths, self._index = _build_parquet_index(Path(parquet_dir))

    def __len__(self):
        return len(self._index)

    def _read_row(self, idx: int) -> dict:
        """
        Read a single row from disk via its index entry.

        Uses a process-local LRU cache (_TABLE_CACHE) so each chunk file is
        loaded from disk at most once per worker per cache cycle. Cache is
        capped at _TABLE_CACHE_MAX_CHUNKS entries; on a miss past capacity
        the least-recently-used chunk is evicted. Re-access of a present
        entry promotes it to most-recent so the running shuffled-access
        pattern from RandomSampler doesn't constantly thrash.
        """
        fi, ri = self._index[idx]
        if fi in _TABLE_CACHE:
            # Hit: bump to most-recent and return.
            _TABLE_CACHE.move_to_end(fi)
        else:
            # Miss: evict LRU if at capacity, then read.
            if len(_TABLE_CACHE) >= _TABLE_CACHE_MAX_CHUNKS:
                _TABLE_CACHE.popitem(last=False)
            _TABLE_CACHE[fi] = pq.read_table(self._file_paths[fi], columns=_COLUMNS)
        table = _TABLE_CACHE[fi]
        row = {}
        for col in _COLUMNS:
            row[col] = table.column(col)[ri].as_py()
        return row

    def __getitem__(self, idx: int) -> dict:
        row = self._read_row(idx)
        atoms, density, origin = _row_to_atoms_and_density(row)

        # Generate grid positions — same function charge3net uses internally.
        # For a (Nx, Ny, Nz) density grid, this creates fractional coordinates
        # [0/Nx, 1/Nx, ..., (Nx-1)/Nx] in each dimension, then maps to
        # Cartesian via the cell matrix.
        grid_pos = calculate_grid_pos(density, origin, atoms.get_cell())

        # Build the graph dict using charge3net's KdTreeGraphConstructor.
        # This handles:
        #   - Random probe subsampling (if num_probes is set)
        #   - Dynamic supercell expansion for PBC
        #   - KD-tree based atom->probe neighbor finding
        #   - ASE-based atom->atom neighbor finding
        graph_dict = self.graph_constructor(density, atoms, grid_pos)

        return graph_dict


def build_dataloaders(
    parquet_dir: str,
    cutoff: float = 4.0,
    train_probes: int = 200,
    val_probes: int = 1000,
    batch_size: int = 4,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = False,
    distributed: bool = False,
) -> tuple:
    """
    Build train, validation, and test DataLoaders.

    Scans Parquet files once, then creates three datasets (with different
    probe sampling) that share the same lightweight index.

    The split is deterministic given (seed, val_frac, test_frac). These
    values must not change between runs if reported metrics are to remain
    comparable.

    Parameters
    ----------
    parquet_dir : str
        Path to directory with chunk_*.parquet files.
    cutoff : float
        Neighbor cutoff (Angstrom).
    train_probes : int
        Number of randomly sampled probes per sample during training.
    val_probes : int
        Number of probes per sample during validation and test.
    batch_size : int
        Batch size.
    val_frac : float
        Fraction of data to use for validation.
    test_frac : float
        Fraction of data to hold out as a test set (never used for model
        selection — only evaluated once at the end of training).
    num_workers : int
        DataLoader workers.
    seed : int
        Random seed for split reproducibility.
    pin_memory : bool
        Pin memory for GPU transfer.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader, DataLoader, DataLoader
    """
    # Build the index once, share between all three datasets
    shared_index = _build_parquet_index(Path(parquet_dir))

    train_dataset = LeMatRhoDataset(
        cutoff=cutoff, num_probes=train_probes, _shared_index=shared_index
    )
    val_dataset = LeMatRhoDataset(
        cutoff=cutoff, num_probes=val_probes, _shared_index=shared_index
    )
    test_dataset = LeMatRhoDataset(
        cutoff=cutoff, num_probes=val_probes, _shared_index=shared_index
    )

    # Split indices (same for all three datasets, different probe sampling)
    n = len(train_dataset)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices, test_indices = random_split(
        range(n), [n_train, n_val, n_test], generator=generator
    )

    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices.indices)

    collate_fn = partial(collate_list_of_dicts, pin_memory=pin_memory)

    # DDP path: shard the training set across ranks via DistributedSampler.
    # Val/test stay non-distributed (each rank evaluates the whole set; only
    # rank 0 reports). This wastes V+T compute but keeps eval simple and
    # rank-agnostic. The data is tiny (5%+5% of 65k) so it's fine.
    train_sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(
            train_subset,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        # shuffle and sampler are mutually exclusive in DataLoader.
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
