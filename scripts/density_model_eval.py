"""Single-model density evaluation across the LeMat-Rho arms (D7).

Per-structure evaluator: load a model arm, predict the real-space
density on a regular grid for each test row, and write per-structure
NMAPE / RMSE / NRMSE against the ground-truth density into a
parquet file. Driven from the CLI; importable for D8 (the
comparison-table builder) which calls ``evaluate_dataset`` directly.

Arm coverage
------------

* ``salted`` -- fully wired. Stub mode (no ckpt) is supported via
   ``SALTEDModel(basis_spec, ckpt_path=None)``; real mode lands when
   D6 (SALTED training driver) produces a checkpoint.
* ``charge3net`` -- grid prediction (probe batching over Nx*Ny*Nz
   grid coordinates) lands in D7-beta. Raises NotImplementedError
   here so a future user does not silently get stub metrics from a
   real-arm name.
* ``deepdft`` -- same as ``charge3net``.

The Graph2Mat arm is parked (see graph2mat_ft/__init__.py); not
exposed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ase
import numpy as np
import pandas as pd

from salted_ft.basis import BasisSpec


def density_nmape(pred: np.ndarray, target: np.ndarray) -> float:
    """Integral-normalised MAPE: sum(|target - pred|) / sum(|target|) * 100."""
    return float(np.abs(pred - target).sum() / (np.abs(target).sum() + 1e-10) * 100.0)


def density_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error across all grid points."""
    return float(np.sqrt(((pred - target) ** 2).mean()))


def density_nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    """RMSE / mean(|target|) * 100. Comparable across electron counts."""
    return float(
        np.sqrt(((pred - target) ** 2).mean()) / (np.abs(target).mean() + 1e-10) * 100.0
    )


def predict_density(
    model_name: str,
    atoms: ase.Atoms,
    grid_shape: tuple[int, int, int],
    ckpt: str | Path | None,
    basis_spec: BasisSpec,
) -> np.ndarray:
    """Dispatch to the per-arm grid prediction path."""
    if model_name == "salted":
        # Lazy import: the deepdft / charge3net branches do not need
        # rholearn or sibling repos available.
        from salted_ft.model import SALTEDModel

        m = SALTEDModel(basis_spec, ckpt_path=ckpt)
        return m.reconstruct_density(atoms, grid_shape)
    if model_name in ("charge3net", "deepdft"):
        raise NotImplementedError(
            f"{model_name} grid prediction lands in D7-beta "
            "(probe batching over the Nx*Ny*Nz grid). "
            "Construct a probe coordinate list from the cell + grid_shape "
            "and batch through the model's forward pass."
        )
    raise ValueError(f"unknown model arm: {model_name!r}")


def _row_to_atoms(row: pd.Series) -> ase.Atoms:
    """Reconstruct an ase.Atoms from a LeMat-Rho-shaped parquet row."""
    positions = np.asarray(row["positions"]).reshape(-1, 3)
    cell = np.asarray(row["lattice_vectors"]).reshape(3, 3)
    numbers = np.asarray(row["atomic_numbers"])
    return ase.Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)


def _row_target_grid(row: pd.Series) -> tuple[np.ndarray, tuple[int, int, int]]:
    grid_shape = tuple(int(x) for x in row["grid_shape"])
    target = np.asarray(row["charge_density"]).reshape(grid_shape)
    return target, grid_shape


def evaluate_dataset(
    model_name: str,
    test_parquet: str | Path,
    ckpt: str | Path | None,
    basis_spec: BasisSpec,
    output: str | Path,
    limit: int | None = None,
) -> Path:
    """Loop over rows in ``test_parquet`` and write per-row metrics."""
    df_in = pd.read_parquet(test_parquet)
    if limit is not None:
        df_in = df_in.head(limit)

    rows = []
    ckpt_label = str(ckpt) if ckpt is not None else "stub"
    for _, row in df_in.iterrows():
        atoms = _row_to_atoms(row)
        target, grid_shape = _row_target_grid(row)
        pred = predict_density(model_name, atoms, grid_shape, ckpt, basis_spec)
        rows.append(
            {
                "model": model_name,
                "ckpt": ckpt_label,
                "material_id": row.get("material_id"),
                "n_atoms": int(row.get("n_atoms", len(atoms))),
                "nmape": density_nmape(pred, target),
                "rmse": density_rmse(pred, target),
                "nrmse": density_nrmse(pred, target),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = Path(output)
    out_df.to_parquet(out_path)
    return out_path


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-structure density-prediction eval for LeMat-Rho arms."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("salted", "charge3net", "deepdft"),
        help="Which arm to evaluate.",
    )
    parser.add_argument(
        "--test-parquet",
        required=True,
        type=Path,
        help="Path to test split parquet (LeMat-Rho row layout).",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output parquet path."
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Model checkpoint. Omit for stub mode (where supported).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows (smoke-test).",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    out_path = evaluate_dataset(
        model_name=args.model,
        test_parquet=args.test_parquet,
        ckpt=args.ckpt,
        basis_spec=BasisSpec(),
        output=args.output,
        limit=args.limit,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
