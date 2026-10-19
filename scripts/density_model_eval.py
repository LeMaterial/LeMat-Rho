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
* ``charge3net`` -- fully wired via ``_charge3net_predict_grid``
   (full-grid graph built with charge3net's KdTreeGraphConstructor,
   probes batched over the Nx*Ny*Nz grid coordinates).
* ``deepdft`` -- fully wired via ``_deepdft_predict_grid`` (reuses the
   same graph construction; needs the DeepDFT sibling clone on
   sys.path via ``deepdft_ft.runner``).

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
    model: object | None = None,
    max_probe_batch: int = 2500,
) -> np.ndarray:
    """Dispatch to the per-arm grid prediction path.

    Parameters
    ----------
    model :
        Optional pre-loaded model. If provided, ``ckpt`` is ignored.
        Lets tests inject a mock without going through real ckpt loading.
    max_probe_batch :
        ChargE3Net / DeepDFT probe-batching size. Lower if the device
        runs out of memory on big grids.
    """
    if model_name == "salted":
        # Lazy import: the deepdft / charge3net branches do not need
        # rholearn or sibling repos available.
        from salted_ft.model import SALTEDModel

        m = model if model is not None else SALTEDModel(basis_spec, ckpt_path=ckpt)
        return m.reconstruct_density(atoms, grid_shape)
    if model_name == "charge3net":
        return _charge3net_predict_grid(
            model=model,
            ckpt=ckpt,
            atoms=atoms,
            grid_shape=grid_shape,
            max_probe_batch=max_probe_batch,
        )
    if model_name == "deepdft":
        return _deepdft_predict_grid(
            model=model,
            ckpt=ckpt,
            atoms=atoms,
            grid_shape=grid_shape,
            max_probe_batch=max_probe_batch,
        )
    raise ValueError(f"unknown model arm: {model_name!r}")


def _charge3net_predict_grid(
    model: object | None,
    ckpt: str | Path | None,
    atoms: ase.Atoms,
    grid_shape: tuple[int, int, int],
    max_probe_batch: int,
) -> np.ndarray:
    """ChargE3Net grid prediction via probe-batched forward.

    Builds the full-grid graph using charge3net's own
    ``KdTreeGraphConstructor`` so atom and probe edges match what
    the model saw during training, batches probes through
    ``split_batch``, and reshapes to ``(Nx, Ny, Nz)``.

    Loading paths
    -------------
    * ``model`` provided: use it directly. The path tests rely on
      to mock the network without a real ckpt.
    * Else, ``ChargE3NetWrapper(ckpt_path=ckpt)`` is constructed.
      Requires the charge3net sibling repo present at
      ``../charge3net/`` (resolved by ``charge3net_ft.model``).
    """
    import torch

    # Import charge3net_ft.model unconditionally for the sys.path side
    # effect (it adds ../charge3net to sys.path so the src.* helpers
    # below resolve). When the caller supplies a model directly we still
    # need charge3net's data utilities to build the graph.
    import charge3net_ft.model as _c3n_wrapper_module  # noqa: F401

    if model is None:
        from charge3net_ft.model import ChargE3NetWrapper

        model = ChargE3NetWrapper(ckpt_path=ckpt)

    from src.charge3net.data.collate import collate_list_of_dicts
    from src.charge3net.data.graph_construction import KdTreeGraphConstructor
    from src.utils.data import calculate_grid_pos
    from src.utils.predictions import split_batch

    grid_shape_arr = np.asarray(grid_shape, dtype=np.int64)
    dummy_density = np.zeros(tuple(grid_shape_arr), dtype=np.float32)
    origin = np.zeros(3, dtype=np.float64)
    grid_pos = calculate_grid_pos(dummy_density, origin, atoms.get_cell())

    constructor = KdTreeGraphConstructor(cutoff=4.0, num_probes=None, disable_pbc=False)
    graph_dict = constructor(dummy_density, atoms, grid_pos)
    batched = collate_list_of_dicts([graph_dict], pin_memory=False)

    if hasattr(model, "train"):
        model.train(False)

    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for sub_batch in split_batch(batched, max_probe_batch):
            out = model(sub_batch)
            preds.append(out.detach().cpu().squeeze(0))

    rho_flat = torch.cat(preds, dim=0).numpy()
    return rho_flat.reshape(tuple(grid_shape_arr))


def _deepdft_predict_grid(
    model: object | None,
    ckpt: str | Path | None,
    atoms: ase.Atoms,
    grid_shape: tuple[int, int, int],
    max_probe_batch: int,
    num_interactions: int = 3,
    node_size: int = 128,
    cutoff: float = 4.0,
    use_painn: bool = True,
) -> np.ndarray:
    """DeepDFT grid prediction via probe-batched forward.

    DeepDFT is the upstream code that ChargE3Net forked, so the
    forward input dict shape is identical: same probe_xyz /
    probe_edges / num_probes / etc. We reuse charge3net's data
    utilities (already imported by ``_charge3net_predict_grid``)
    to build the graph. The arm-specific bits are:

    * sys.path side effect from ``deepdft_ft.runner`` (adds
      ``../DeepDFT`` and stubs ``asap3`` if it is missing).
    * model construction via ``densitymodel.PainnDensityModel`` or
      ``densitymodel.DensityModel`` (SchNet variant).
    * defaults match ``submit_deepdft_adastra.sh``:
      num_interactions=3, node_size=128, cutoff=4.0, PaiNN.

    Loading paths
    -------------
    * ``model`` provided: use it directly (tests inject mocks here).
    * Else, build the model and ``torch.load`` the ckpt.
    """
    import torch

    # sys.path side effect + asap3 stub, must happen before importing
    # densitymodel even when caller supplied the model.
    import deepdft_ft.runner as _deepdft_runner_module  # noqa: F401

    if model is None:
        import densitymodel

        if use_painn:
            model = densitymodel.PainnDensityModel(num_interactions, node_size, cutoff)
        else:
            model = densitymodel.DensityModel(num_interactions, node_size, cutoff)
        if ckpt is not None:
            state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            # DeepDFT's ckpts wrap the state dict in a "model" key
            state_dict = state.get("model", state)
            model.load_state_dict(state_dict)

    # Reuse the charge3net data layer (DeepDFT input dict is the same).
    from src.charge3net.data.collate import collate_list_of_dicts
    from src.charge3net.data.graph_construction import KdTreeGraphConstructor
    from src.utils.data import calculate_grid_pos
    from src.utils.predictions import split_batch

    import charge3net_ft.model as _c3n_wrapper_module  # noqa: F401

    grid_shape_arr = np.asarray(grid_shape, dtype=np.int64)
    dummy_density = np.zeros(tuple(grid_shape_arr), dtype=np.float32)
    origin = np.zeros(3, dtype=np.float64)
    grid_pos = calculate_grid_pos(dummy_density, origin, atoms.get_cell())

    constructor = KdTreeGraphConstructor(
        cutoff=cutoff, num_probes=None, disable_pbc=False
    )
    graph_dict = constructor(dummy_density, atoms, grid_pos)
    batched = collate_list_of_dicts([graph_dict], pin_memory=False)

    if hasattr(model, "train"):
        model.train(False)

    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for sub_batch in split_batch(batched, max_probe_batch):
            out = model(sub_batch)
            preds.append(out.detach().cpu().squeeze(0))

    rho_flat = torch.cat(preds, dim=0).numpy()
    return rho_flat.reshape(tuple(grid_shape_arr))


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
