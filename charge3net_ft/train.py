"""
Training script for fine-tuning ChargE3Net on LeMatRho charge density data.

Usage:
    python -m charge3net_ft.train \
        --parquet-dir /path/to/lematrho_full_10x10x10 \
        --ckpt-path /path/to/charge3net/models/charge3net_mp.pt \
        --epochs 50

    # Or set LEMATRHO_DATA_DIR env var and omit --parquet-dir.

To do a quick smoke test (1 batch, no checkpoint):
    python -m charge3net_ft.train \
        --parquet-dir /path/to/lematrho_full_10x10x10 \
        --smoke-test
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# charge3net imports (for the LR scheduler)
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

from src.charge3net.models.scheduler import PowerDecayScheduler

from .data import build_dataloaders
from .model import ChargE3NetWrapper


# ---------------------------------------------------------------------------
# Distributed training helpers
# ---------------------------------------------------------------------------
def _is_ddp() -> bool:
    """True if SLURM/torchrun has set up multi-process training."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_ddp() -> tuple[int, int, int]:
    """Initialize the process group and return (rank, local_rank, world_size).

    No-op (returns 0, 0, 1) if we're not in a distributed environment.

    The submit script is expected to export the standard torch env vars from
    SLURM:
        WORLD_SIZE  = $SLURM_NTASKS
        RANK        = $SLURM_PROCID
        LOCAL_RANK  = $SLURM_LOCALID
        MASTER_ADDR = $(scontrol show hostname $SLURM_NODELIST | head -1)
        MASTER_PORT = some unused port (e.g. 29500)
    """
    if not _is_ddp():
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # nccl works on AMD ROCm because PyTorch routes it through RCCL.
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _is_main(rank: int) -> bool:
    """True on rank 0; used to gate prints, wandb, and checkpoint saves."""
    return rank == 0


def _probe_mask(targets: torch.Tensor, num_probes: torch.Tensor) -> torch.Tensor:
    """Boolean mask [B, max_probes], True for real probe points (not padding)."""
    return (
        torch.arange(targets.shape[1], device=targets.device)[None]
        < num_probes[:, None]
    )


def compute_nmape(
    preds: torch.Tensor, targets: torch.Tensor, num_probes: torch.Tensor = None
) -> torch.Tensor:
    """
    Integral-Normalized Mean Absolute Percentage Error (%).

    NMAPE = sum(|target - pred|) / sum(|target|) * 100

    Computed per-sample in the batch, then averaged.
    This is charge3net's primary validation metric.

    Parameters
    ----------
    num_probes : torch.Tensor, optional
        Shape [B]. If provided, masks out zero-padding before computing
        metrics (required when samples have variable probe counts).
    """
    if num_probes is not None:
        mask = _probe_mask(targets, num_probes)
        diff = (torch.abs(targets - preds) * mask).sum(dim=1)
        denom = (torch.abs(targets) * mask).sum(dim=1) + 1e-10
    else:
        diff = torch.abs(targets - preds).sum(dim=1)
        denom = torch.abs(targets).sum(dim=1) + 1e-10
    return (diff / denom * 100.0).mean()


def compute_rmse(
    preds: torch.Tensor, targets: torch.Tensor, num_probes: torch.Tensor = None
) -> torch.Tensor:
    """Root Mean Squared Error (e/Å³), averaged over the batch."""
    if num_probes is not None:
        mask = _probe_mask(targets, num_probes)
        n = mask.sum(dim=1).float()
        mse = ((targets - preds) ** 2 * mask).sum(dim=1) / (n + 1e-10)
    else:
        mse = ((targets - preds) ** 2).mean(dim=1)
    return mse.sqrt().mean()


def compute_nrmse(
    preds: torch.Tensor, targets: torch.Tensor, num_probes: torch.Tensor = None
) -> torch.Tensor:
    """Normalized RMSE (%) — RMSE / mean(|target|) * 100, per-sample then averaged."""
    if num_probes is not None:
        mask = _probe_mask(targets, num_probes)
        n = mask.sum(dim=1).float()
        mse = ((targets - preds) ** 2 * mask).sum(dim=1) / (n + 1e-10)
        mean_abs = (torch.abs(targets) * mask).sum(dim=1) / (n + 1e-10)
    else:
        mse = ((targets - preds) ** 2).mean(dim=1)
        mean_abs = torch.abs(targets).mean(dim=1)
    rmse = mse.sqrt()
    return (rmse / (mean_abs + 1e-10) * 100.0).mean()


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    global_step,
    log_every=50,
    use_wandb=False,
):
    """Run one training epoch, return (average loss, updated global_step)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for i, batch in enumerate(train_loader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        preds = model(batch)
        loss = F.l1_loss(preds, batch["probe_target"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        global_step += 1

        if (i + 1) % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  step {i + 1}: loss={loss.item():.6f}  lr={lr:.2e}")
            if use_wandb:
                wandb.log({"train/loss_step": loss.item(), "lr": lr}, step=global_step)

    return total_loss / max(n_batches, 1), global_step


@torch.no_grad()
def validate(model, loader, device):
    """Run evaluation, return average L1, NMAPE, RMSE, and NRMSE."""
    model.eval()
    total_loss = 0.0
    total_nmape = 0.0
    total_rmse = 0.0
    total_nrmse = 0.0
    n_batches = 0

    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        preds = model(batch)
        targets = batch["probe_target"]
        num_probes = batch.get("num_probes")

        total_loss += F.l1_loss(preds, targets).item()
        total_nmape += compute_nmape(preds, targets, num_probes).item()
        total_rmse += compute_rmse(preds, targets, num_probes).item()
        total_nrmse += compute_nrmse(preds, targets, num_probes).item()
        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "L1": total_loss / denom,
        "NMAPE": total_nmape / denom,
        "RMSE": total_rmse / denom,
        "NRMSE": total_nrmse / denom,
    }


def _unwrap(model):
    """Return the underlying ChargE3NetWrapper regardless of DDP wrapping.

    DistributedDataParallel wraps the user model in a ``.module`` attribute;
    state_dict() and load_state_dict() should always target the inner model
    so checkpoints are interchangeable between single-GPU and DDP runs.
    """
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model, optimizer, scheduler, epoch, best_nmape, global_step, path):
    """Save training checkpoint (rank 0 should be the only caller in DDP)."""
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_nmape": best_nmape,
            "global_step": global_step,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Load training checkpoint, return (start_epoch, best_nmape, global_step)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    _unwrap(model).model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    best_nmape = ckpt["best_nmape"]
    global_step = ckpt.get("global_step", 0)
    lr = optimizer.param_groups[0]["lr"]
    print(
        f"Resumed from checkpoint: epoch {start_epoch}, "
        f"best_nmape={best_nmape:.2f}%, step={global_step}, lr={lr:.2e}"
    )
    return start_epoch, best_nmape, global_step


def main():
    load_dotenv()  # load .env (WANDB_API_KEY, etc.)

    parser = argparse.ArgumentParser(description="Fine-tune ChargE3Net on LeMatRho")
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default=os.environ.get("LEMATRHO_DATA_DIR"),
        help=(
            "Directory with chunk_*.parquet files. "
            "Defaults to $LEMATRHO_DATA_DIR env var."
        ),
    )
    parser.add_argument(
        "--ckpt-path", type=str, default=None, help="Pre-trained checkpoint (.pt)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./checkpoints", help="Save directory"
    )
    parser.add_argument("--cutoff", type=float, default=4.0, help="Neighbor cutoff (A)")
    parser.add_argument(
        "--train-probes", type=int, default=200, help="Probes per sample (train)"
    )
    parser.add_argument(
        "--val-probes", type=int, default=1000, help="Probes per sample (val/test)"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.05,
        help="Validation fraction. Do not change after first run.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.05,
        help="Test fraction (held out, evaluated once at end). "
        "Do not change after first run.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run 1 forward pass and exit"
    )
    parser.add_argument(
        "--overfit-single-batch",
        action="store_true",
        help="Overfit on a single batch to verify the model can learn",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (cpu, cuda, mps). Auto-detect if not set.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to training checkpoint (latest.pt) to resume from",
    )
    parser.add_argument("--wandb-project", type=str, default="lemat-rho-charge3net")
    parser.add_argument("--wandb-entity", type=str, default="dtts")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode (use 'offline' on air-gapped clusters)",
    )
    args = parser.parse_args()

    if args.parquet_dir is None:
        parser.error(
            "--parquet-dir is required (or set the LEMATRHO_DATA_DIR environment variable)"
        )

    # Seed everything for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # DDP setup (no-op when WORLD_SIZE=1). Must happen before device
    # selection because each rank pins itself to its own GPU via local_rank.
    rank, local_rank, world_size = _setup_ddp()
    is_main = _is_main(rank)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif _is_ddp():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if is_main:
        print(f"Using device: {device}; world_size={world_size}")

    # W&B (rank 0 only). Soft-fail: if init times out (e.g. compute node
    # can't reach api.wandb.ai through the cluster proxy), degrade to
    # disabled mode and keep training. Used to be fatal — caused the
    # 1h47m job 4969727 timeout-then-crash on Adastra.
    use_wandb = (not args.no_wandb and not args.smoke_test) and is_main
    if use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                settings=wandb.Settings(init_timeout=300),
                mode=args.wandb_mode,
            )
        except Exception as e:  # noqa: BLE001 — really do want broad here
            print(
                f"WARNING: wandb.init failed ({type(e).__name__}: {e}); "
                "continuing with wandb disabled. Training output is still "
                "saved to checkpoints + stdout."
            )
            use_wandb = False

    # Data
    if is_main:
        print("Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        parquet_dir=args.parquet_dir,
        cutoff=args.cutoff,
        train_probes=args.train_probes,
        val_probes=args.val_probes,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        num_workers=args.num_workers,
        seed=args.seed,
        distributed=_is_ddp(),
    )
    if is_main:
        print(
            f"Train: {len(train_loader.dataset)} samples, "
            f"Val: {len(val_loader.dataset)} samples, "
            f"Test: {len(test_loader.dataset)} samples"
        )

    # Model. Loaded on every rank (each gets its own copy of the weights);
    # DDP will sync gradients across ranks at backward.
    if is_main:
        print("Initializing ChargE3Net...")
    model = ChargE3NetWrapper(ckpt_path=args.ckpt_path, cutoff=args.cutoff)
    model = model.to(device)
    if _is_ddp():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    n_params = sum(
        p.numel() for p in (model.module if _is_ddp() else model).parameters()
    )
    if is_main:
        print(f"Model parameters: {n_params:,}")

    # Smoke test: just run one forward pass
    if args.smoke_test:
        print("\n--- Smoke test ---")
        model.eval()
        batch = next(iter(train_loader))
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        print(f"Batch keys: {list(batch.keys())}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        with torch.no_grad():
            preds = model(batch)
        print(f"Predictions shape: {preds.shape}")
        loss = F.l1_loss(preds, batch["probe_target"])
        print(f"L1 loss: {loss.item():.6f}")
        print("Smoke test passed.")
        return

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -----------------------------------------------------------------------
    # Single-batch overfit test
    # -----------------------------------------------------------------------
    if args.overfit_single_batch:
        print("\n--- Single-batch overfit test ---")
        print(f"lr={args.lr}  epochs={args.epochs}  probes={args.train_probes}")

        # Fetch exactly one batch and pin it to device
        fixed_batch = next(iter(train_loader))
        fixed_batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in fixed_batch.items()
        }
        n_atoms = fixed_batch["num_nodes"].tolist()
        n_probes = fixed_batch["num_probes"].tolist()
        print(f"Batch: {len(n_atoms)} samples, atoms={n_atoms}, probes={n_probes}")

        model.train()
        for epoch in range(1, args.epochs + 1):
            preds = model(fixed_batch)
            targets = fixed_batch["probe_target"]
            num_probes = fixed_batch.get("num_probes")
            loss = F.l1_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            nmape = compute_nmape(preds, targets, num_probes)
            rmse = compute_rmse(preds, targets, num_probes)
            nrmse = compute_nrmse(preds, targets, num_probes)
            print(
                f"Epoch {epoch:>4d}/{args.epochs}  L1={loss.item():.6f}  "
                f"NMAPE={nmape.item():.2f}%  RMSE={rmse.item():.4f}  NRMSE={nrmse.item():.2f}%"
            )
            if use_wandb:
                wandb.log(
                    {
                        "overfit/L1": loss.item(),
                        "overfit/NMAPE": nmape.item(),
                        "overfit/RMSE": rmse.item(),
                        "overfit/NRMSE": nrmse.item(),
                        "epoch": epoch,
                    }
                )

        print("\nOverfit test complete.")
        if use_wandb:
            wandb.finish()
        return

    # -----------------------------------------------------------------------
    # Normal training loop
    # -----------------------------------------------------------------------
    # charge3net's power decay scheduler: lr = alpha^(step/beta)
    # For fine-tuning we use a gentler decay (higher beta = slower decay)
    scheduler = PowerDecayScheduler(optimizer, alpha=0.96, beta=5e4)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_nmape = float("inf")

    global_step = 0
    start_epoch = 0

    if args.resume_from:
        start_epoch, best_nmape, global_step = load_checkpoint(
            args.resume_from,
            model,
            optimizer,
            scheduler,
            device,
        )

    if is_main:
        print(f"\nStarting training from epoch {start_epoch + 1} to {args.epochs}...")
    for epoch in range(start_epoch, args.epochs):
        # DDP requires set_epoch on the sampler each epoch for proper shuffling.
        if _is_ddp() and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            global_step,
            log_every=args.log_every,
            use_wandb=use_wandb,
        )
        val = validate(model, val_loader, device)
        elapsed = time.time() - t0

        if is_main:
            print(
                f"Epoch {epoch + 1}/{args.epochs}  "
                f"train_L1={train_loss:.6f}  "
                f"val_L1={val['L1']:.6f}  "
                f"val_NMAPE={val['NMAPE']:.2f}%  "
                f"val_RMSE={val['RMSE']:.4f}  "
                f"val_NRMSE={val['NRMSE']:.2f}%  "
                f"time={elapsed:.0f}s"
            )

        if use_wandb:
            wandb.log(
                {
                    "train/L1": train_loss,
                    "val/L1": val["L1"],
                    "val/NMAPE": val["NMAPE"],
                    "val/RMSE": val["RMSE"],
                    "val/NRMSE": val["NRMSE"],
                    "epoch": epoch + 1,
                },
                step=global_step,
            )

        # Save best checkpoint (selected on val NMAPE). Only rank 0 writes.
        if is_main and val["NMAPE"] < best_nmape:
            best_nmape = val["NMAPE"]
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_nmape,
                global_step,
                save_dir / "best.pt",
            )
            print(f"  -> New best val NMAPE: {best_nmape:.2f}%")

        # Save latest checkpoint every epoch (for SLURM resumption).
        if is_main:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_nmape,
                global_step,
                save_dir / "latest.pt",
            )

        # Keep ranks in lockstep so a slow saver doesn't get lapped.
        if _is_ddp():
            torch.distributed.barrier()

    # -----------------------------------------------------------------------
    # Test set evaluation — run once at the end using the best checkpoint.
    # These numbers are the uncontaminated held-out performance estimate.
    # -----------------------------------------------------------------------
    print("\nLoading best checkpoint for test evaluation...")
    best_ckpt_path = save_dir / "best.pt"
    if best_ckpt_path.exists():
        load_checkpoint(best_ckpt_path, model, optimizer, scheduler, device)
    else:
        print("  Warning: best.pt not found, using current model weights.")

    test = validate(model, test_loader, device)
    print(
        f"\nTest set results (best.pt, held-out):\n"
        f"  L1={test['L1']:.6f}  NMAPE={test['NMAPE']:.2f}%  "
        f"RMSE={test['RMSE']:.4f}  NRMSE={test['NRMSE']:.2f}%"
    )
    if use_wandb:
        wandb.log(
            {
                "test/L1": test["L1"],
                "test/NMAPE": test["NMAPE"],
                "test/RMSE": test["RMSE"],
                "test/NRMSE": test["NRMSE"],
            }
        )

    if is_main:
        print(f"\nTraining complete. Best val NMAPE: {best_nmape:.2f}%")
        print(f"Checkpoints saved to {save_dir}")
    if use_wandb:
        wandb.finish()
    if _is_ddp():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
