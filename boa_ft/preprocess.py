"""LeMat-Rho parquet -> sharded LMDB of scdp ``AtomicData`` for BOA.

BOA trains from an LMDB directory of pickled ``scdp.data.data.AtomicData``
graphs plus a ``datasplits.json`` (see ``boa.data.dataset.LmdbDataset`` and
``boa.data.datamodule.ProbeDataModule``). This script builds both from the
LeMat-Rho parquet chunks, reusing the exact schema and row decoder that the
charge3net and deepdft arms use (imported, not copied, so the input pipeline
stays a single source of truth).

Each valid row becomes an ``AtomicData`` via ``build_graph_with_vnodes`` with
``disable_pbc=False`` (periodic neighbor graph) and ``vnode_method="none"``
(BOA's function-centric path assigns coefficients to real atom pairs, so no
virtual nodes). Rows whose ``compressed_charge_density`` is null are already
dropped by the shared index builder; any row that still fails to convert is
skipped and counted.

.. code-block:: bash

    python -m boa_ft.preprocess \\
        --parquet-dir /path/to/lemat_rho_chunks \\
        --out-dir $BOA_DATA/lematrho \\
        --num-shards 16
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
import torch
from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS
from tqdm import tqdm

# scdp (sibling boa clone, editable-installed; see boa_ft/README.md) and the
# shared charge3net_ft parquet pipeline (needs the ../charge3net sibling) are
# imported lazily inside the functions that need them, so the pure helpers
# (row_exceeds_max_z, write_datasplits) import cleanly without either sibling.
if TYPE_CHECKING:
    from scdp.data.data import AtomicData


def row_to_atomic_data(
    row: dict,
    metadata: str,
    z_table,
    atom_cutoff: float,
    max_neighbors: int | None,
) -> AtomicData:
    """Convert one parquet row dict into an scdp ``AtomicData`` graph.

    Parameters
    ----------
    row : dict
        Row with the LeMat-Rho columns (species, positions, lattice, density).
    metadata : str
        Human-readable id stored on the graph (used for logging).
    z_table : scdp.data.data.AtomicNumberTable
        Atomic-number-to-index table spanning all elements.
    atom_cutoff : float
        Cutoff radius (Angstrom) for the periodic atom-atom neighbor graph.
    max_neighbors : int or None
        Optional cap on neighbors per atom.

    Returns
    -------
    scdp.data.data.AtomicData
        Graph carrying atoms, cell, full probe grid, and flattened density.
    """
    from scdp.data.data import AtomicData

    from charge3net_ft.data import _row_to_atoms_and_density

    atoms, density, _origin = _row_to_atoms_and_density(row)

    atom_types = torch.from_numpy(atoms.numbers).long()
    atom_coords = torch.from_numpy(atoms.positions).float()
    cell = torch.from_numpy(np.asarray(atoms.cell.array)).float()
    chg_density = torch.from_numpy(np.ascontiguousarray(density)).float()

    # origin=None: LeMat-Rho grids start at fractional (0, 0, 0), so the grid
    # positions computed inside build_graph_with_vnodes need no offset.
    return AtomicData.build_graph_with_vnodes(
        atom_types,
        atom_coords,
        cell,
        chg_density,
        None,
        metadata=metadata,
        z_table=z_table,
        atom_cutoff=atom_cutoff,
        disable_pbc=False,
        vnode_method="none",
        device="cpu",
        max_neighbors=max_neighbors,
        struct=None,
    )


# Same bound as the per-worker LRU table caches in deepdft_ft.data and
# charge3net_ft.data: each cached entry is a fully decompressed pyarrow
# table, so an unbounded cache eventually holds the whole dataset in RAM.
_TABLE_CACHE_MAX_CHUNKS = 5


def read_row_cached(
    file_paths: list[Path],
    fi: int,
    ri: int,
    cache: collections.OrderedDict,
    max_chunks: int = _TABLE_CACHE_MAX_CHUNKS,
) -> dict:
    """Read one parquet row, keeping at most ``max_chunks`` tables cached (LRU).

    Parameters
    ----------
    file_paths : list of Path
        Parquet chunk files, indexed by ``fi``.
    fi, ri : int
        File index and row index within that file.
    cache : collections.OrderedDict
        Caller-owned LRU cache mapping file index to pyarrow table.
    max_chunks : int
        Cache capacity; least recently used tables are evicted beyond it.

    Returns
    -------
    dict
        Column-name-to-value mapping for the requested row.
    """
    if fi in cache:
        cache.move_to_end(fi)
    else:
        cache[fi] = pq.read_table(file_paths[fi])
        while len(cache) > max_chunks:
            cache.popitem(last=False)
    table = cache[fi]
    return {col: table.column(col)[ri].as_py() for col in table.column_names}


def row_exceeds_max_z(row: dict, max_z: int) -> bool:
    """Return True when a row contains an element heavier than ``max_z``.

    Used to drop materials the GTO basis cannot represent (def2-svp covers
    H through Rn, Z <= 86, so actinide-bearing rows must be excluded).

    Parameters
    ----------
    row : dict
        Row with the LeMat-Rho columns (needs ``species_at_sites``).
    max_z : int
        Highest allowed atomic number.

    Returns
    -------
    bool
        True if any site's element has Z above ``max_z``.

    .. code-block:: python

        row_exceeds_max_z({"species_at_sites": ["U", "O", "O"]}, 86)  # True
    """
    return any(ASE_ATOMIC_NUMBERS[s] > max_z for s in row["species_at_sites"])


def write_datasplits(
    n_samples: int,
    out_dir: Path,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict:
    """Write ``datasplits.json`` with the charge3net split convention.

    Uses the same ``torch.Generator`` seed and (val, test) fractions as
    ``charge3net_ft.data.build_dataloaders`` so splits stay comparable across
    arms. Indices are sorted global dataset indices (matching how
    ``LmdbDataset`` concatenates shards in sorted filename order).

    Parameters
    ----------
    n_samples : int
        Total number of samples written to the LMDB.
    out_dir : Path
        Dataset directory where ``datasplits.json`` is written.
    val_frac, test_frac : float
        Validation and test fractions.
    seed : int
        Split RNG seed.

    Returns
    -------
    dict
        The split dict with keys ``train``, ``validation``, ``test``.
    """
    n_val = int(n_samples * val_frac)
    n_test = int(n_samples * test_frac)
    n_train = n_samples - n_val - n_test
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = torch.utils.data.random_split(
        range(n_samples), [n_train, n_val, n_test], generator=generator
    )
    splits = {
        "train": sorted(int(i) for i in train_idx.indices),
        "validation": sorted(int(i) for i in val_idx.indices),
        "test": sorted(int(i) for i in test_idx.indices),
    }
    with open(out_dir / "datasplits.json", "w") as f:
        json.dump(splits, f)
    return splits


def main(args: argparse.Namespace) -> None:
    """Run the parquet -> LMDB conversion and write splits + basis metadata."""
    from scdp.scripts.preprocess import get_atomic_number_table_from_zs

    from charge3net_ft.data import _build_parquet_index

    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    file_paths, index = _build_parquet_index(Path(args.parquet_dir))
    if args.limit is not None:
        index = index[: args.limit]

    # Atomic-number table spanning all elements (matches scdp's own preprocess).
    z_table = get_atomic_number_table_from_zs(np.arange(100).tolist())

    # Contiguous blocks keep the LMDB global-index order equal to processing
    # order, so datasplits.json indices line up with LmdbDataset.
    num_shards = min(args.num_shards, len(index))
    shard_blocks = np.array_split(np.arange(len(index)), num_shards)
    map_size = args.map_size_gb * (1024**3)

    import lmdb  # local import: lmdb is only needed when actually writing.

    n_written = 0
    n_failed = 0
    n_heavy = 0
    # Per-sample element sets, indexed by global write order. BOA's
    # construct_orbitals aligns the basis positionally against the *training*
    # split's unique elements, so the basis element list must be exactly the
    # union over the train split (see write below), not the whole dataset.
    sample_elements: list[set[int]] = []

    # Bounded per-chunk LRU table cache (see read_row_cached): rows are
    # processed in contiguous blocks, so a small cache still gives one
    # read per file without holding every decompressed table in RAM.
    table_cache: collections.OrderedDict = collections.OrderedDict()

    for shard_id, block in enumerate(shard_blocks):
        if len(block) == 0:
            continue
        shard_path = str(data_dir / f"data.{shard_id:04d}.lmdb")
        db = lmdb.open(
            shard_path,
            map_size=map_size,
            subdir=False,
            meminit=False,
            map_async=True,
        )
        local_idx = 0
        for global_pos in tqdm(block, desc=f"shard {shard_id}", position=0):
            fi, ri = index[int(global_pos)]
            chunk_stem = file_paths[fi].stem
            metadata = f"{chunk_stem}_row{ri:06d}"
            row = read_row_cached(file_paths, fi, ri, table_cache)
            if args.max_z is not None and row_exceeds_max_z(row, args.max_z):
                n_heavy += 1
                continue
            try:
                data = row_to_atomic_data(
                    row, metadata, z_table, args.atom_cutoff, args.max_neighbors
                )
            except Exception as exc:  # noqa: BLE001 - skip and count bad rows.
                n_failed += 1
                print(f"skip {metadata}: {type(exc).__name__}: {exc}")
                continue
            sample_elements.append({int(z) for z in data.atom_types.tolist() if z > 0})
            txn = db.begin(write=True)
            txn.put(f"{local_idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()
            local_idx += 1
            n_written += 1
        # LmdbDataset reads this "length" key to size the shard.
        txn = db.begin(write=True)
        txn.put("length".encode("ascii"), pickle.dumps(local_idx, protocol=-1))
        txn.commit()
        db.sync()
        db.close()

    splits = write_datasplits(
        n_written, out_dir, args.val_frac, args.test_frac, args.seed
    )

    # Basis element list = union of elements over the train split, matching the
    # metadata BOA computes over the same split. Elements that appear only in
    # val/test are excluded here; BOA has no orbital for an unseen element, so
    # for real runs the train split must cover every element (a 90% split of a
    # large dataset does).
    train_elements: set[int] = set()
    for i in splits["train"]:
        train_elements.update(sample_elements[i])
    atomic_numbers = sorted(train_elements)
    with open(out_dir / "atomic_numbers.json", "w") as f:
        json.dump(atomic_numbers, f)

    print(
        f"wrote {n_written} samples to {data_dir} across {num_shards} shard(s); "
        f"skipped {n_failed} unconvertible row(s) and {n_heavy} row(s) above max-z"
    )
    print(f"atomic_numbers ({len(atomic_numbers)}): {atomic_numbers}")
    print(
        f"pass these to training via data.basis_info.atomic_numbers='{atomic_numbers}'"
    )


def get_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet-dir",
        required=True,
        help="Directory of LeMat-Rho chunk_*.parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Dataset directory to create (holds data/ LMDB shards + datasplits.json).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=16,
        help="Number of LMDB shards to split the data into.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N valid rows (for smoke tests).",
    )
    parser.add_argument(
        "--atom-cutoff",
        type=float,
        default=4.0,
        help="Cutoff radius (Angstrom) for the periodic atom-atom graph.",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=None,
        help="Optional cap on neighbors per atom in the stored graph.",
    )
    parser.add_argument(
        "--max-z",
        type=int,
        default=None,
        help=(
            "Skip rows containing any element with Z above this "
            "(86 keeps def2-svp basis coverage; actinides have no basis)."
        ),
    )
    parser.add_argument(
        "--map-size-gb",
        type=int,
        default=64,
        help="LMDB max map size in GiB (virtual; file grows sparsely).",
    )
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
