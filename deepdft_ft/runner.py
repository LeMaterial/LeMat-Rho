"""DeepDFT training runner — vendored from peterbjorgensen/DeepDFT@main.

Vendored rather than monkey-patched because the DDP integration touches
many points throughout `main()` (dataset construction, model wrap,
sampler, checkpoint save, logging gates). Keeping the patched copy here
makes the delta auditable and the code testable.

Diff vs upstream:
- Adds DDP setup via `_setup_ddp`/`_is_main` helpers (mirrors the pattern
  used in `charge3net_ft/train.py`). DDP activates iff `WORLD_SIZE>1`.
- Detects parquet directories and uses `LeMatRhoDeepDFTDataset` instead
  of `dataset.DensityData`. Other arg formats are passed through to
  upstream unchanged so the runner still works on the original tar/dir
  datasets.
- `RandomSampler` swapped for `DistributedSampler` when DDP active.
- Model wrapped in `DistributedDataParallel`; checkpoint save/load unwraps
  via `_unwrap`.
- Logging + checkpoint writes gated on rank 0.
- Validation uses `ValCollateFuncRandomSample` (capped probe count via
  `--val-probes`, sampled without replacement) over a deterministic
  seeded subsample of the val split (`--val-max-samples`). Upstream's
  5000-probe with-replacement val collate OOM-killed job 5004725.
- Generated train/val splits are seeded (`--split-seed`, default 0) via
  `generate_datasplits` instead of upstream's unseeded
  `np.random.permutation`, so restarts with `--load_model` and all DDP
  ranks reproduce the same split without a `--split_file`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import sys
import timeit
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler

torch.set_num_threads(1)  # Try to avoid thread overload on cluster

# ---------------------------------------------------------------------------
# Make the DeepDFT sibling repo importable. Expected layout (mirrors
# how charge3net is set up):
#   <repo-root>/         <-- LeMat-Rho
#   <repo-root>/../DeepDFT/   <-- AIforGreatGood/DeepDFT clone
# ---------------------------------------------------------------------------
_DEEPDFT_ROOT = Path(__file__).resolve().parent.parent.parent / "DeepDFT"
if not _DEEPDFT_ROOT.exists():
    raise RuntimeError(
        f"DeepDFT repo not found at {_DEEPDFT_ROOT}.\n"
        "Clone it with: git clone https://github.com/peterbjorgensen/DeepDFT "
        f"{_DEEPDFT_ROOT}"
    )
if str(_DEEPDFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEEPDFT_ROOT))

# ---------------------------------------------------------------------------
# Stub `asap3` if it isn't available. Building asap3 from source requires
# Python.h which isn't installed on Adastra (and getting it would need
# admin). Upstream DeepDFT supports an ASE-based fallback via
# `AseNeigborListWrapper`; we expose the same interface from `asap3.FullNeighborList`
# so the upstream `import asap3 ; asap3.FullNeighborList(...)` calls work.
# ---------------------------------------------------------------------------
try:
    import asap3  # noqa: F401
except ImportError:
    import types

    import ase.neighborlist
    import numpy as np

    _asap3_stub = types.ModuleType("asap3")

    class _AseFullNeighborList:
        """Drop-in `asap3.FullNeighborList` replacement using ASE primitives.

        Behaviourally equivalent for DeepDFT's use case: ``get_neighbors(i, cutoff)``
        returns ``(indices, rel_positions, dist2)`` arrays. Much slower than real
        asap3 but works without C++ headers.
        """

        def __init__(self, cutoff, atoms):
            self._cutoff = cutoff
            self._positions = atoms.get_positions()
            self._cell = np.asarray(atoms.get_cell())
            nl = ase.neighborlist.NewPrimitiveNeighborList(
                cutoff, skin=0.0, self_interaction=False, bothways=True
            )
            nl.build(atoms.get_pbc(), atoms.get_cell(), atoms.get_positions())
            self._nl = nl

        def get_neighbors(self, i, cutoff):
            assert cutoff == self._cutoff, (
                "cutoff must match the one used at FullNeighborList init"
            )
            indices, offsets = self._nl.get_neighbors(i)
            rel_positions = (
                self._positions[indices] + offsets @ self._cell - self._positions[i]
            )
            dist2 = (rel_positions**2).sum(axis=1)
            return indices, rel_positions, dist2

    _asap3_stub.FullNeighborList = _AseFullNeighborList
    sys.modules["asap3"] = _asap3_stub

import dataset
import densitymodel

from deepdft_ft.data import (
    LeMatRhoDeepDFTDataset,
    generate_datasplits,
    sample_probe_indices,
)


# ---------------------------------------------------------------------------
# Distributed-training helpers (same pattern as charge3net_ft/train.py).
# ---------------------------------------------------------------------------
def _is_ddp() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_ddp() -> tuple[int, int, int]:
    """Returns (rank, local_rank, world_size). No-op when WORLD_SIZE=1."""
    if not _is_ddp():
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # nccl routes through RCCL on AMD ROCm builds.
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _is_main(rank: int) -> bool:
    return rank == 0


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DistributedDataParallel for state_dict access."""
    return model.module if hasattr(model, "module") else model


def _is_parquet_dir(path: str | Path) -> bool:
    """LeMat-Rho parquet dirs contain ``chunk_*.parquet``; tar/cube paths don't."""
    p = Path(path)
    return p.is_dir() and any(p.glob("chunk_*.parquet"))


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train graph convolution network", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Load model parameters from previous run",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Atomic interaction cutoff distance [Å]",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Train/test/validation split file json",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed for the generated train/val split (used when no "
        "--split_file is given). Fixed default so restarts with "
        "--load_model and all DDP ranks reproduce the same split",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=3,
        help="Number of interaction layers used",
    )
    parser.add_argument(
        "--node_size", type=int, default=64, help="Size of hidden node states"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/model_output",
        help="Path to output directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/qm9.db",
        help="Path to ASE database",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=int(1e6),
        help="Maximum number of optimisation steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Set which device to use for training e.g. 'cuda' or 'cpu'",
    )

    parser.add_argument(
        "--use_painn_model",
        action="store_true",
        help="Enable equivariant message passing model (PaiNN)",
    )

    parser.add_argument(
        "--ignore_pbc",
        action="store_true",
        help="If flag is given, disable periodic boundary conditions (force to False) in atoms data",
    )

    parser.add_argument(
        "--force_pbc",
        action="store_true",
        help="If flag is given, force periodic boundary conditions to True in atoms data",
    )

    parser.add_argument(
        "--val-probes",
        type=int,
        default=1000,
        help="Probes per material in validation. probes_to_graph cost grows "
        "quadratically in this number; 5000 OOM-killed the 64 GB job 5004725",
    )

    parser.add_argument(
        "--val-max-samples",
        type=int,
        default=200,
        help="Deterministic seeded subsample size of the validation split "
        "(upstream val sets were ~100 materials; ours is ~3.3k)",
    )

    return parser.parse_args(arg_list)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ValCollateFuncRandomSample(dataset.CollateFuncRandomSample):
    """Validation collate with capped, without-replacement probe sampling.

    Upstream ``atoms_and_probe_sample_to_graph_dict`` draws probes WITH
    replacement (``np.random.randint``) and ``probes_to_graph`` inserts
    every probe as a dummy atom into a periodic bothways neighborlist,
    so probe-probe pairs grow quadratically in the probe count. On the
    tiny LeMat-Rho cells (volume p50 89 A^3) 5000 probes meant ~75M
    pairs per median material, which OOM-killed the 64 GB jobs
    5003891/5004725 during the step-0 val pass. This subclass keeps the
    upstream graph construction but samples probe indices via
    ``sample_probe_indices`` (without replacement when the grid has at
    least ``num_probes`` points).

    .. code-block:: python

        collate = ValCollateFuncRandomSample(cutoff=4.0, num_probes=1000)
        batch = collate([sample_dict_a, sample_dict_b])
    """

    def __call__(self, input_dicts: list) -> dict:
        graphs = []
        for i in input_dicts:
            if self.set_pbc is not None:
                atoms = i["atoms"].copy()
                atoms.set_pbc(self.set_pbc)
            else:
                atoms = i["atoms"]
            graphs.append(self._sample_to_graph_dict(i, atoms))
        return dataset.collate_list_of_dicts(graphs, pin_memory=self.pin_memory)

    def _sample_to_graph_dict(self, sample: dict, atoms) -> dict:
        """Upstream ``atoms_and_probe_sample_to_graph_dict`` with the probe
        selection swapped for capped, without-replacement sampling."""
        grid_pos = sample["grid_position"]
        density = sample["density"]
        flat_choice = sample_probe_indices(
            int(np.prod(grid_pos.shape[0:3])), self.num_probes
        )
        probe_choice = np.unravel_index(flat_choice, grid_pos.shape[0:3])
        probe_pos = grid_pos[probe_choice]
        probe_target = density[probe_choice]

        atom_edges, atom_edges_displacement, neighborlist, inv_cell_T = (
            dataset.atoms_to_graph(atoms, self.cutoff)
        )
        probe_edges, probe_edges_displacement = dataset.probes_to_graph(
            atoms,
            probe_pos,
            self.cutoff,
            neighborlist=neighborlist,
            inv_cell_T=inv_cell_T,
        )

        default_type = torch.get_default_dtype()
        if not probe_edges:
            probe_edges = [np.zeros((0, 2), dtype=int)]
            probe_edges_displacement = [np.zeros((0, 3), dtype=int)]
        res = {
            "nodes": torch.tensor(atoms.get_atomic_numbers()),
            "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
            "atom_edges_displacement": torch.tensor(
                np.concatenate(atom_edges_displacement, axis=0), dtype=default_type
            ),
            "probe_edges": torch.tensor(np.concatenate(probe_edges, axis=0)),
            "probe_edges_displacement": torch.tensor(
                np.concatenate(probe_edges_displacement, axis=0), dtype=default_type
            ),
            "probe_target": torch.tensor(probe_target, dtype=default_type),
        }
        res["num_nodes"] = torch.tensor(res["nodes"].shape[0])
        res["num_atom_edges"] = torch.tensor(res["atom_edges"].shape[0])
        res["num_probe_edges"] = torch.tensor(res["probe_edges"].shape[0])
        res["num_probes"] = torch.tensor(res["probe_target"].shape[0])
        res["probe_xyz"] = torch.tensor(probe_pos, dtype=default_type)
        res["atom_xyz"] = torch.tensor(atoms.get_positions(), dtype=default_type)
        res["cell"] = torch.tensor(np.array(atoms.get_cell()), dtype=default_type)
        return res


def split_data(dataset, args):
    # Load or generate splits
    if args.split_file:
        with open(args.split_file, "r") as fp:
            splits = json.load(fp)
    else:
        # Seeded and restart-stable: resumes with --load_model and no
        # --split_file regenerate this split, and upstream's unseeded
        # np.random.permutation leaked previous val rows into train.
        splits = generate_datasplits(len(dataset), args.split_seed)

        # Save split file
        with open(os.path.join(args.output_dir, "datasplits.json"), "w") as f:
            json.dump(splits, f)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)
    return datasplits


def eval_model(model, dataloader, device):
    with torch.no_grad():
        running_ae = torch.tensor(0.0, device=device)
        running_se = torch.tensor(0.0, device=device)
        running_count = torch.tensor(0.0, device=device)
        for batch in dataloader:
            device_batch = {
                k: v.to(device=device, non_blocking=True) for k, v in batch.items()
            }
            outputs = model(device_batch)
            targets = device_batch["probe_target"]

            running_ae += torch.sum(torch.abs(targets - outputs))
            running_se += torch.sum(torch.square(targets - outputs))
            running_count += torch.sum(device_batch["num_probes"])

        mae = (running_ae / running_count).item()
        rmse = (torch.sqrt(running_se / running_count)).item()

    return mae, rmse


def get_normalization(dataset, per_atom=True):
    try:
        num_targets = len(dataset.transformer.targets)
    except AttributeError:
        num_targets = 1
    x_sum = torch.zeros(num_targets)
    x_2 = torch.zeros(num_targets)
    num_objects = 0
    for sample in dataset:
        x = sample["targets"]
        if per_atom:
            x = x / sample["num_nodes"]
        x_sum += x
        x_2 += x**2.0
        num_objects += 1
    # Var(X) = E[X^2] - E[X]^2
    x_mean = x_sum / num_objects
    x_var = x_2 / num_objects - x_mean**2.0

    return x_mean, torch.sqrt(x_var)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = get_arguments()

    # DDP setup (no-op when WORLD_SIZE=1). Must precede device + dataset
    # construction; each rank pins itself to its own GCD via local_rank.
    rank, local_rank, _world_size = _setup_ddp()
    is_main = _is_main(rank)

    # Override device for DDP runs.
    if _is_ddp():
        args.device = f"cuda:{local_rank}"

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    # Save command line args
    with open(os.path.join(args.output_dir, "commandline_args.txt"), "w") as f:
        f.write("\n".join(sys.argv[1:]))
    # Save parsed command line arguments
    with open(os.path.join(args.output_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # Setup dataset and loader. If args.dataset points at a directory of
    # LeMat-Rho chunk_*.parquet files, use our adapter; otherwise fall
    # through to upstream's tar/cube/dir loader unchanged.
    if _is_parquet_dir(args.dataset):
        if is_main:
            logging.info("loading LeMat-Rho parquet dir %s", args.dataset)
        densitydata = LeMatRhoDeepDFTDataset(parquet_dir=args.dataset)
    else:
        if args.dataset.endswith(".txt"):
            # Text file contains list of datafiles
            with open(args.dataset, "r") as datasetfiles:
                filelist = [
                    os.path.join(os.path.dirname(args.dataset), line.strip("\n"))
                    for line in datasetfiles
                ]
        else:
            filelist = [args.dataset]
        if is_main:
            logging.info("loading data %s", args.dataset)
        densitydata = torch.utils.data.ConcatDataset(
            [dataset.DensityData(path) for path in filelist]
        )

    # Split data into train and validation sets
    datasplits = split_data(densitydata, args)
    # Pool_size and num_workers downsized from the upstream 20*4 = 80.
    # (An earlier comment here blamed 200-300^3 cells for the 64 GB OOMs;
    # LeMat-Rho grids are actually tiny, 10-15 points per axis. The real
    # host-RAM cost is probe-count quadratic neighborlist growth in
    # probes_to_graph, measured on jobs 5003891/5004725; see
    # ValCollateFuncRandomSample. The small pool is still kept: it bounds
    # how many decoded parquet rows sit in RAM per worker.)
    datasplits["train"] = dataset.RotatingPoolData(datasplits["train"], 5)

    if args.ignore_pbc and args.force_pbc:
        raise ValueError(
            "ignore_pbc and force_pbc are mutually exclusive and can't both be set at the same time"
        )
    elif args.ignore_pbc:
        set_pbc = False
    elif args.force_pbc:
        set_pbc = True
    else:
        set_pbc = None

    # Setup loaders. With DDP, the train sampler shards data across ranks
    # so each rank sees a disjoint subset per epoch. Val stays
    # non-distributed and only rank 0 actually uses it.
    if _is_ddp():
        train_sampler = DistributedSampler(
            datasplits["train"], shuffle=True, drop_last=True
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(datasplits["train"])
    train_loader = torch.utils.data.DataLoader(
        datasplits["train"],
        2,
        # See RotatingPoolData(...5) above; num_workers compounds the RAM
        # footprint of the rotating pool (pool_size * num_workers decoded
        # rows resident per worker). The dominant transient cost per batch
        # is the probe neighborlist, quadratic in the probe count (1000
        # train probes -> ~3M pairs / ~6.6 GB peak on the median material,
        # measured on jobs 5003891/5004725), so keep both knobs small.
        num_workers=2,
        sampler=train_sampler,
        collate_fn=dataset.CollateFuncRandomSample(
            args.cutoff, 1000, pin_memory=False, set_pbc_to=set_pbc
        ),
    )
    # Deterministic seeded subsample of the val split. The 5% split is
    # ~3.3k materials while upstream val sets were ~100; a full pass with
    # the python-loop collate takes hours per log interval. This subsample
    # is only restart-stable because the parent split is too (seeded via
    # --split-seed in split_data); together they keep best_val_mae
    # comparable across restarts.
    val_split = datasplits["validation"]
    if args.val_max_samples is not None and len(val_split) > args.val_max_samples:
        val_rng = np.random.default_rng(0)
        val_keep = val_rng.choice(
            len(val_split), size=args.val_max_samples, replace=False
        )
        val_split = torch.utils.data.Subset(val_split, sorted(val_keep.tolist()))
    val_loader = torch.utils.data.DataLoader(
        val_split,
        2,
        # ValCollateFuncRandomSample caps probes at args.val_probes (default
        # 1000, NOT prod(grid_shape): at the 15^3 = 3375-point grids the
        # quadratic neighborlist cost would already flirt with the 64 GB
        # ceiling) and samples without replacement. The upstream 5000-probe
        # with-replacement collate OOM-killed job 5004725 at step 0.
        collate_fn=ValCollateFuncRandomSample(
            args.cutoff, args.val_probes, pin_memory=False, set_pbc_to=set_pbc
        ),
        num_workers=0,
    )
    # Upstream materialised the full val_loader into a list at startup for
    # speed ("Preloading validation batch"); with our larger val split we
    # keep it streaming instead (eager preloading OOM-killed job 4971720).

    # Initialise model
    device = torch.device(args.device)
    if args.use_painn_model:
        net = densitymodel.PainnDensityModel(
            args.num_interactions, args.node_size, args.cutoff
        )
    else:
        net = densitymodel.DensityModel(
            args.num_interactions, args.node_size, args.cutoff
        )
    if is_main:
        logging.debug("model has %d parameters", count_parameters(net))
    net = net.to(device)
    if _is_ddp():
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[local_rank], output_device=local_rank
        )

    # Setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()
    scheduler_fn = lambda step: 0.96 ** (step / 100000)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)

    log_interval = 5000
    running_loss = torch.tensor(0.0, device=device)
    running_loss_count = torch.tensor(0, device=device)
    best_val_mae = np.inf
    step = 0
    # Restore checkpoint
    if args.load_model:
        state_dict = torch.load(args.load_model, map_location=device)
        _unwrap(net).load_state_dict(state_dict["model"])
        step = state_dict["step"]
        best_val_mae = state_dict["best_val_mae"]
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])

    if is_main:
        logging.info("start training")

    data_timer = AverageMeter("data_timer")
    transfer_timer = AverageMeter("transfer_timer")
    train_timer = AverageMeter("train_timer")
    eval_timer = AverageMeter("eval_time")

    endtime = timeit.default_timer()
    for _ in itertools.count():
        for batch_host in train_loader:
            data_timer.update(timeit.default_timer() - endtime)
            tstart = timeit.default_timer()
            # Transfer to 'device'
            batch = {
                k: v.to(device=device, non_blocking=True)
                for (k, v) in batch_host.items()
            }
            transfer_timer.update(timeit.default_timer() - tstart)

            tstart = timeit.default_timer()
            # Reset gradient
            optimizer.zero_grad()

            # Forward, backward and optimize
            outputs = net(batch)
            loss = criterion(outputs, batch["probe_target"])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_loss += (
                    loss
                    * batch["probe_target"].shape[0]
                    * batch["probe_target"].shape[1]
                )
                running_loss_count += torch.sum(batch["num_probes"])

            train_timer.update(timeit.default_timer() - tstart)

            # print(step, loss_value)
            # Validate and save model
            if (step % log_interval == 0) or ((step + 1) == args.max_steps):
                tstart = timeit.default_timer()
                with torch.no_grad():
                    train_loss = (running_loss / running_loss_count).item()
                    running_loss = running_loss_count = 0

                val_mae, val_rmse = eval_model(net, val_loader, device)

                if is_main:
                    logging.info(
                        "step=%d, val_mae=%g, val_rmse=%g, sqrt(train_loss)=%g",
                        step,
                        val_mae,
                        val_rmse,
                        math.sqrt(train_loss),
                    )

                # Save checkpoint (rank 0 only). _unwrap so the state_dict
                # is interchangeable between single-GPU and DDP runs.
                if is_main and val_mae < best_val_mae:
                    best_val_mae = val_mae
                    torch.save(
                        {
                            "model": _unwrap(net).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step,
                            "best_val_mae": best_val_mae,
                        },
                        os.path.join(args.output_dir, "best_model.pth"),
                    )

                eval_timer.update(timeit.default_timer() - tstart)
                logging.debug(
                    "%s %s %s %s"
                    % (data_timer, transfer_timer, train_timer, eval_timer)
                )
            step += 1

            scheduler.step()

            if step >= args.max_steps:
                if is_main:
                    logging.info("Max steps reached, exiting")
                if _is_ddp():
                    torch.distributed.destroy_process_group()
                sys.exit(0)

            endtime = timeit.default_timer()


if __name__ == "__main__":
    main()
