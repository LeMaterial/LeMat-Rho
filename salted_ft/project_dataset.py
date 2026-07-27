"""Phase D2: project the LeMat-Rho parquet dataset onto the SALTED basis.

One-time job. Reads every ``chunk_*.parquet`` produced by
lematerial-fetcher (rows of densities + structures), runs
``project_chgcar_to_basis`` row by row, writes a parallel
``chunk_*.parquet`` of basis coefficients that downstream training
loops (rholearn, Graph2Mat, etc.) consume.

Output schema per row::

    row_index        int            position in the source chunk
    material_id      str            carried from source if present, else ""
    n_atoms          int
    atomic_numbers   list[int]      ASE atomic numbers, length n_atoms
    lattice_vectors  list[list]     3x3 cell matrix in Angstrom
    n_electrons      float          integrated density * cell_volume / n_grid
    grid_shape       list[int]      [Nx, Ny, Nz]
    coefficients     list[list]     (n_atoms, n_coeffs_per_atom)
    basis_set_NMAPE  float          per-row reconstruction error (%)

CLI::

    uv run python -m salted_ft.project_dataset \\
        --input-dir  $SETUP/charge3net_data \\
        --output-dir $SETUP/salted_projected_coefficients
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from charge3net_ft.data import _COLUMNS, _row_to_atoms_and_density
from salted_ft.basis import BasisSpec
from salted_ft.projection import (
    project_chgcar_to_basis,
    reconstruct_grid_from_basis,
)


def _row_nmape(true: np.ndarray, pred: np.ndarray) -> float:
    """Integral-normalised mean absolute percentage error (%) for one row."""
    return float(100.0 * np.sum(np.abs(true - pred)) / (np.sum(np.abs(true)) + 1e-12))


def project_chunk(
    in_path: str | Path,
    out_path: str | Path,
    basis_spec: BasisSpec,
) -> None:
    """Project every valid row in ``in_path`` and write ``out_path``."""
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = list(_COLUMNS)
    # material_id is optional; include it if present so downstream can match
    # to the source LeMat-Rho row.
    schema = pq.read_schema(in_path)
    has_material_id = "material_id" in schema.names
    if has_material_id:
        columns.append("material_id")

    table = pq.read_table(in_path, columns=columns)
    n_rows = len(table)

    out_rows: list[dict] = []
    for ri in range(n_rows):
        chgd = table.column("compressed_charge_density")[ri]
        if not chgd.is_valid:
            continue  # skip null density (failed DFT extraction in source)

        row = {col: table.column(col)[ri].as_py() for col in _COLUMNS}
        atoms, density, _origin = _row_to_atoms_and_density(row)

        coeffs = project_chgcar_to_basis(density, atoms, basis_spec)
        reconstructed = reconstruct_grid_from_basis(
            coeffs, atoms, density.shape, basis_spec
        )
        nmape = _row_nmape(density, reconstructed)

        cell = np.asarray(atoms.get_cell(), dtype=np.float64)
        cell_volume = float(np.abs(np.linalg.det(cell)))
        n_grid = int(np.prod(density.shape))
        n_electrons = float(density.sum() * cell_volume / n_grid)

        out_rows.append(
            {
                "row_index": ri,
                "material_id": (
                    table.column("material_id")[ri].as_py() if has_material_id else ""
                ),
                "n_atoms": len(atoms),
                "atomic_numbers": atoms.get_atomic_numbers().tolist(),
                "lattice_vectors": cell.tolist(),
                "n_electrons": n_electrons,
                "grid_shape": list(density.shape),
                "coefficients": coeffs.tolist(),
                "basis_set_NMAPE": nmape,
            }
        )

    out_table = pa.Table.from_pylist(out_rows)
    pq.write_table(out_table, out_path)


def project_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    basis_spec: BasisSpec,
) -> None:
    """Run :func:`project_chunk` over every ``chunk_*.parquet`` in ``input_dir``.

    Idempotent: a chunk whose output already exists is left untouched
    so partially-completed runs can resume cheaply.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(input_dir.glob("chunk_*.parquet"))
    if not inputs:
        raise FileNotFoundError(f"no chunk_*.parquet files under {input_dir}")

    for in_path in inputs:
        out_path = output_dir / in_path.name
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        project_chunk(in_path, out_path, basis_spec)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project the LeMat-Rho parquet dataset onto the SALTED basis."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--basis-spec",
        type=str,
        default=None,
        help="JSON-encoded BasisSpec overrides. If omitted, defaults are used.",
    )
    args = parser.parse_args(argv)

    if args.basis_spec:
        overrides = json.loads(args.basis_spec)
        # sigma must be tuple-ified to satisfy BasisSpec's frozen dataclass
        if "sigma" in overrides:
            overrides["sigma"] = tuple(overrides["sigma"])
        spec = BasisSpec(**overrides)
    else:
        spec = BasisSpec()
    print(
        f"BasisSpec: lmax={spec.max_l}, n_radial={spec.n_radial}, "
        f"sigma={spec.sigma}, cutoff={spec.cutoff}, "
        f"n_coeffs_per_atom={spec.n_coeffs_per_atom}"
    )

    project_directory(args.input_dir, args.output_dir, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
