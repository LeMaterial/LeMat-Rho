"""
Training script for fine-tuning ChargE3Net on LeMatRho charge density data.

Usage:
    uv run python train.py \
        --parquet-dir /path/to/lematrho_full_10x10x10 \
        --ckpt-path models/charge3net_mp.pt \
        --epochs 50

To do a quick smoke test (1 batch, no checkpoint):
    uv run python train.py \
        --parquet-dir /path/to/lematrho_full_10x10x10 \
        --smoke-test
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# charge3net imports (for the LR scheduler)
# ---------------------------------------------------------------------------
_CHARGE3NET_ROOT = Path(__file__).resolve().parent.parent / "charge3net"
if str(_CHARGE3NET_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHARGE3NET_ROOT))

from src.charge3net.models.scheduler import PowerDecayScheduler  # noqa: E402

from data import build_dataloaders  # noqa: E402
from model import ChargE3NetWrapper  # noqa: E402


def compute_nmape(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Integral-Normalized Mean Absolute Percentage Error (%).

    NMAPE = sum(|target - pred|) / sum(|target|) * 100

    Computed per-sample in the batch, then averaged.
    This is charge3net's primary validation metric.
    """
    # preds, targets: [B, num_probes] (padded)
    diff = torch.abs(targets - preds)
    # Sum over probes dimension
    nmape = diff.sum(dim=1) / (torch.abs(targets).sum(dim=1) + 1e-10) * 100.0
    return nmape.mean()


def train_one_epoch(model, train_loader, optimizer, scheduler, device, log_every=50):
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for i, batch in enumerate(train_loader):
        # Move tensors to device
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

        if (i + 1) % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  step {i+1}: loss={loss.item():.6f}  lr={lr:.2e}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, val_loader, device):
    """Run validation, return average L1 loss and NMAPE."""
    model.eval()
    total_loss = 0.0
    total_nmape = 0.0
    n_batches = 0

    for batch in val_loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        preds = model(batch)
        loss = F.l1_loss(preds, batch["probe_target"])
        nmape = compute_nmape(preds, batch["probe_target"])

        total_loss += loss.item()
        total_nmape += nmape.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_nmape = total_nmape / max(n_batches, 1)
    return avg_loss, avg_nmape


def save_checkpoint(model, optimizer, scheduler, epoch, best_nmape, path):
    """Save training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model": model.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_nmape": best_nmape,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ChargE3Net on LeMatRho")
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default="/Users/dts/Documents/entalpic/lematerial-fetcher/lematrho_full_10x10x10",
        help="Directory with chunk_*.parquet files",
    )
    parser.add_argument("--ckpt-path", type=str, default=None, help="Pre-trained checkpoint")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--cutoff", type=float, default=4.0, help="Neighbor cutoff (A)")
    parser.add_argument("--train-probes", type=int, default=200, help="Probes per sample (train)")
    parser.add_argument("--val-probes", type=int, default=1000, help="Probes per sample (val)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--val-frac", type=float, default=0.05, help="Validation fraction")
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data
    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        parquet_dir=args.parquet_dir,
        cutoff=args.cutoff,
        train_probes=args.train_probes,
        val_probes=args.val_probes,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Model
    print("Initializing ChargE3Net...")
    model = ChargE3NetWrapper(ckpt_path=args.ckpt_path, cutoff=args.cutoff)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Smoke test: just run one forward pass
    if args.smoke_test:
        print("\n--- Smoke test ---")
        model.eval()
        batch = next(iter(train_loader))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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
            loss = F.l1_loss(preds, fixed_batch["probe_target"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            nmape = compute_nmape(preds, fixed_batch["probe_target"])
            print(f"Epoch {epoch:>4d}/{args.epochs}  L1={loss.item():.6f}  NMAPE={nmape.item():.2f}%")

        print("\nOverfit test complete.")
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

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, log_every=args.log_every
        )
        val_loss, val_nmape = validate(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_L1={train_loss:.6f}  "
            f"val_L1={val_loss:.6f}  "
            f"val_NMAPE={val_nmape:.2f}%  "
            f"time={elapsed:.0f}s"
        )

        # Save best checkpoint
        if val_nmape < best_nmape:
            best_nmape = val_nmape
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_nmape, save_dir / "best.pt"
            )
            print(f"  -> New best NMAPE: {best_nmape:.2f}%")

        # Save latest checkpoint every epoch
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_nmape, save_dir / "latest.pt"
        )

    print(f"\nTraining complete. Best NMAPE: {best_nmape:.2f}%")
    print(f"Checkpoints saved to {save_dir}")


if __name__ == "__main__":
    main()
