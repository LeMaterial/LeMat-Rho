"""SCF-speedup experiment driver (P4).

For each row in a held-out test parquet, the driver:

1. Reconstructs the ``ase.Atoms`` + grid_shape + n_electrons.
2. Predicts the density via the chosen ML arm
   (``scripts.density_model_eval.predict_density`` already supports
   ``salted``, ``charge3net``, and ``deepdft``).
3. Writes a CHGCAR with VASP's electron-count rescaling so
   ``ICHARG=1`` reads a self-consistent total.
4. Builds a paired baseline + predicted Flow via
   ``entalsim.dft.scf_speedup.make_scf_speedup_pair`` and submits it
   to MongoDB via ``entalsim.core.submit.submit_workflow``.

The two entalsim callables are dependency-injectable so the driver
unit-tests pass locally without entalsim installed; the CLI imports
them at runtime.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Callable

import ase
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

from salted_ft.basis import BasisSpec
from salted_ft.io import write_chgcar

# scripts/ is not a package; reach the sibling module via sys.path
# (same pattern the test fixture uses).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
_density_eval = importlib.import_module("density_model_eval")
predict_density = _density_eval.predict_density


_ARMS_REQUIRING_CKPT = ("charge3net", "deepdft")


def _row_to_atoms(row: pd.Series) -> ase.Atoms:
    positions = np.asarray(row["positions"]).reshape(-1, 3)
    cell = np.asarray(row["lattice_vectors"]).reshape(3, 3)
    numbers = np.asarray(row["atomic_numbers"])
    return ase.Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)


def _row_grid_shape(row: pd.Series) -> tuple[int, int, int]:
    return tuple(int(x) for x in row["grid_shape"])


def run_experiment(
    model_name: str,
    test_parquet: str | Path,
    chgcar_dir: str | Path,
    basis_spec: BasisSpec,
    project: str,
    worker: str,
    ckpt: str | Path | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    make_pair_fn: Callable[..., Any] | None = None,
    submit_fn: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Loop the test parquet and submit one paired Flow per row.

    Returns one record per processed row with the CHGCAR path, the
    flow's job count, and a ``submitted`` flag — useful for tests
    and for end-of-run sanity logging.
    """
    if model_name in _ARMS_REQUIRING_CKPT and ckpt is None:
        raise ValueError(
            f"--ckpt is required for arm {model_name!r}; running without "
            "weights produces random-init predictions and wastes HPC time. "
            "Stub mode is supported only for 'salted'."
        )

    # Lazy-import entalsim callables when the caller did not inject
    # mocks. Keeps the test suite passable without entalsim installed.
    if make_pair_fn is None:
        from entalsim.dft.scf_speedup import make_scf_speedup_pair as make_pair_fn
    if submit_fn is None:
        from entalsim.core.submit import submit_workflow as submit_fn

    df_in = pd.read_parquet(test_parquet)
    if limit is not None:
        df_in = df_in.head(limit)

    chgcar_root = Path(chgcar_dir)
    chgcar_root.mkdir(parents=True, exist_ok=True)
    ckpt_label = str(ckpt) if ckpt is not None else "stub"

    records: list[dict[str, Any]] = []
    for _, row in df_in.iterrows():
        material_id = str(row["material_id"])
        atoms = _row_to_atoms(row)
        grid_shape = _row_grid_shape(row)
        n_electrons = float(row["n_electrons"])

        # Predict density (ML forward pass).
        density = predict_density(model_name, atoms, grid_shape, ckpt, basis_spec)

        # One directory per (model, material_id) so make_scf_speedup_pair's
        # prev_dir mechanism stages the right file (it copies CHGCAR from
        # the directory). Different rows must not share one directory.
        row_dir = chgcar_root / f"{model_name}__{material_id}"
        row_dir.mkdir(parents=True, exist_ok=True)
        chgcar_path = row_dir / "CHGCAR"
        write_chgcar(density, atoms, chgcar_path, n_electrons=n_electrons)

        # Build the paired Flow. We pass a pymatgen Structure because
        # entalsim's atomate2 Makers consume that.
        structure = AseAtomsAdaptor.get_structure(atoms)
        metadata = {
            "experiment": "scf_speedup",
            "material_id": material_id,
            "model": model_name,
            "ckpt": ckpt_label,
        }
        flow = make_pair_fn(structure, row_dir, metadata)

        if not dry_run:
            submit_fn(flow, project=project, worker=worker)

        records.append(
            {
                "material_id": material_id,
                "model": model_name,
                "ckpt": ckpt_label,
                "chgcar_path": str(chgcar_path),
                "n_jobs": len(flow.jobs),
                "submitted": not dry_run,
            }
        )

    return records


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SCF-speedup experiment driver: predict CHGCAR, "
        "submit paired r2SCAN single-point Flow per structure."
    )
    p.add_argument(
        "--model",
        required=True,
        choices=("salted", "charge3net", "deepdft"),
        help="Which ML arm to evaluate.",
    )
    p.add_argument(
        "--test-parquet",
        required=True,
        type=Path,
        help="Held-out test split parquet (P-ID or P-OOD).",
    )
    p.add_argument(
        "--chgcar-dir",
        required=True,
        type=Path,
        help="Directory for predicted CHGCAR files; per-row subdirs created.",
    )
    p.add_argument(
        "--project",
        required=True,
        help="jobflow_remote project name (matches a jfremote YAML).",
    )
    p.add_argument(
        "--worker",
        required=True,
        help="jobflow_remote worker name from the project YAML.",
    )
    p.add_argument("--ckpt", type=Path, default=None, help="Model checkpoint path.")
    p.add_argument(
        "--limit", type=int, default=None, help="Process only the first N rows."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write CHGCARs and build Flows but do not submit_workflow.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_cli().parse_args(argv)
    records = run_experiment(
        model_name=args.model,
        test_parquet=args.test_parquet,
        chgcar_dir=args.chgcar_dir,
        basis_spec=BasisSpec(),
        project=args.project,
        worker=args.worker,
        ckpt=args.ckpt,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    submitted = sum(1 for r in records if r["submitted"])
    print(
        f"Processed {len(records)} rows for arm={args.model}; "
        f"submitted={submitted}, dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
