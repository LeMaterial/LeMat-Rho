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
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ase
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm.auto import tqdm

from salted_ft.basis import BasisSpec
from salted_ft.io import write_chgcar

logger = logging.getLogger(__name__)

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


def _load_submitted_ids(manifest_path: Path, model_name: str) -> set[str]:
    """Read a JSONL manifest and return material_ids previously submitted.

    Failed rows (``submitted=False``) are intentionally NOT counted so
    the next run retries them.
    """
    if not manifest_path.exists():
        return set()
    submitted: set[str] = set()
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed manifest line: %s", line[:80])
            continue
        if rec.get("model") == model_name and rec.get("submitted") is True:
            submitted.add(str(rec["material_id"]))
    return submitted


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
    manifest_path: str | Path | None = None,
    skip_existing: bool = False,
    make_pair_fn: Callable[..., Any] | None = None,
    submit_fn: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Loop the test parquet and submit one paired Flow per row.

    The driver is resilient to per-row failures: a bad row records
    an ``error`` entry and the loop continues. Results stream to a
    JSONL manifest after each row so an interrupted run leaves a
    resumable record. ``skip_existing=True`` skips rows whose
    ``material_id`` is already marked ``submitted=True`` in the
    manifest for this ``model_name`` (failed rows are retried).
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

    chgcar_root = Path(chgcar_dir)
    chgcar_root.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = chgcar_root / "manifest.jsonl"
    else:
        manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    already_done = (
        _load_submitted_ids(manifest_path, model_name) if skip_existing else set()
    )
    if already_done:
        logger.info(
            "Skipping %d rows already submitted (manifest=%s)",
            len(already_done),
            manifest_path,
        )

    df_in = pd.read_parquet(test_parquet)
    if limit is not None:
        df_in = df_in.head(limit)

    ckpt_label = str(ckpt) if ckpt is not None else "stub"
    records: list[dict[str, Any]] = []

    for _, row in tqdm(
        df_in.iterrows(),
        total=len(df_in),
        desc=f"scf_speedup({model_name})",
    ):
        material_id = str(row["material_id"])
        if material_id in already_done:
            logger.info("Skipping %s (already submitted)", material_id)
            continue

        record: dict[str, Any] = {
            "material_id": material_id,
            "model": model_name,
            "ckpt": ckpt_label,
            "submitted": False,
            "error": None,
        }
        try:
            atoms = _row_to_atoms(row)
            grid_shape = _row_grid_shape(row)
            n_electrons = float(row["n_electrons"])

            density = predict_density(model_name, atoms, grid_shape, ckpt, basis_spec)

            # One directory per (model, material_id) so make_scf_speedup_pair's
            # prev_dir mechanism stages the right file. Nested layout
            # (chgcar_root/<model>/<material_id>/CHGCAR) avoids ambiguity
            # for material_ids that contain separator characters.
            row_dir = chgcar_root / model_name / material_id
            row_dir.mkdir(parents=True, exist_ok=True)
            chgcar_path = row_dir / "CHGCAR"
            write_chgcar(density, atoms, chgcar_path, n_electrons=n_electrons)

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

            record.update(
                {
                    "chgcar_path": str(chgcar_path),
                    "n_jobs": len(flow.jobs),
                    "submitted": not dry_run,
                }
            )
            logger.info(
                "%s arm=%s n_jobs=%d submitted=%s",
                material_id,
                model_name,
                record["n_jobs"],
                record["submitted"],
            )
        except Exception as exc:
            # Catch broadly: any per-row exception (corrupt parquet, ML
            # OOM, mongo timeout) must not kill the rest of the batch.
            record["error"] = repr(exc)
            logger.exception(
                "Row failed material_id=%s arm=%s",
                material_id,
                model_name,
            )
        finally:
            # Stream to manifest after every row so an interrupted
            # run leaves a resumable record. Open in append mode so
            # parallel runs (different arms, different parquets) can
            # share a manifest if pointed at the same path.
            with manifest_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            records.append(record)

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
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSONL manifest path (default: <chgcar-dir>/manifest.jsonl). "
        "Streamed after each row so interrupted runs are resumable.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows whose material_id is already submitted=True in the "
        "manifest for this model. Failed rows are always retried.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
        manifest_path=args.manifest,
        skip_existing=args.skip_existing,
    )
    submitted = sum(1 for r in records if r["submitted"])
    failed = sum(1 for r in records if r.get("error"))
    logger.info(
        "Processed %d rows for arm=%s; submitted=%d, failed=%d, dry_run=%s",
        len(records),
        args.model,
        submitted,
        failed,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
