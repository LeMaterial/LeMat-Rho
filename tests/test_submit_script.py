"""TDD tests for the parameterized Adastra submit script.

The script `submit_charge3net_adastra.sh` is now configurable via two env
vars:

  LEMATRHO_TRAINING_MODE   "pretrained" (default) or "from_scratch"
  LEMATRHO_DRY_RUN         "1" prints the resolved train command and exits

These tests pin the contract.

They don't depend on Adastra. The script is sourced under bash with
LEMATRHO_DRY_RUN=1 so the venv activate / rocm-smi / srun calls are
skipped and the train invocation is printed instead of executed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SUBMIT_SCRIPT = Path(__file__).resolve().parent.parent / "submit_charge3net_adastra.sh"


def _run(env_extra: dict) -> subprocess.CompletedProcess:
    """Run the submit script under bash with LEMATRHO_DRY_RUN=1."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available in test environment")
    env = {
        **os.environ,
        "LEMATRHO_DRY_RUN": "1",
        # Avoid touching the user's real Adastra setup or W&B credentials.
        "LEMATRHO_ADASTRA_SETUP": "/tmp/fake_setup_for_tests",
        # SLURM env vars that the script would normally inherit.
        "SLURM_NTASKS": "4",
        "SLURM_NODELIST": "g0001",
        "SLURM_JOB_ACCOUNT": "c1816212_mi250",
        **env_extra,
    }
    return subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_dry_run_mode_prints_train_command():
    """LEMATRHO_DRY_RUN=1 must print the resolved train command and exit 0."""
    result = _run({})
    assert result.returncode == 0, (
        f"dry-run exited {result.returncode}; stderr={result.stderr}"
    )
    assert "charge3net_ft.train" in result.stdout, (
        f"dry-run output missing the train invocation; stdout={result.stdout}"
    )


def test_default_mode_is_pretrained():
    """Unset LEMATRHO_TRAINING_MODE -> pretrained MP checkpoint path is used."""
    result = _run({})
    assert result.returncode == 0
    out = result.stdout
    assert "--ckpt-path" in out, (
        f"default (pretrained) run must pass --ckpt-path; stdout={out}"
    )
    assert "charge3net_mp.pt" in out, (
        f"default run must point --ckpt-path at the MP checkpoint; stdout={out}"
    )


def test_pretrained_mode_uses_default_save_dir():
    """Pretrained mode writes to charge3net_checkpoints/ (no fromscratch suffix)."""
    result = _run({"LEMATRHO_TRAINING_MODE": "pretrained"})
    assert result.returncode == 0
    assert (
        "charge3net_checkpoints " in (result.stdout + " ")
        or "charge3net_checkpoints\n" in result.stdout
        or "/charge3net_checkpoints" in result.stdout
    )
    assert "charge3net_checkpoints_fromscratch" not in result.stdout, (
        f"pretrained mode must NOT use the fromscratch save dir; stdout={result.stdout}"
    )


def test_from_scratch_mode_drops_ckpt_path():
    """LEMATRHO_TRAINING_MODE=from_scratch -> no --ckpt-path flag at all.

    Without --ckpt-path, ChargE3NetWrapper.__init__ initializes weights
    fresh (no MP transfer). This is the comparison arm for the
    pretrained vs from-scratch experiment.
    """
    result = _run({"LEMATRHO_TRAINING_MODE": "from_scratch"})
    assert result.returncode == 0, (
        f"from_scratch run exited {result.returncode}; stderr={result.stderr}"
    )
    out = result.stdout
    assert "--ckpt-path" not in out, (
        f"from_scratch must not pass --ckpt-path; stdout={out}"
    )
    # also confirm charge3net_mp.pt isn't referenced anywhere in the
    # resolved command (defense against accidental partial passing)
    assert "charge3net_mp.pt" not in out, (
        f"from_scratch must not reference the MP checkpoint; stdout={out}"
    )


def test_from_scratch_mode_uses_separate_save_dir():
    """From-scratch run writes to a different dir so checkpoints don't collide
    with the pretrained run.
    """
    result = _run({"LEMATRHO_TRAINING_MODE": "from_scratch"})
    assert result.returncode == 0
    out = result.stdout
    assert "charge3net_checkpoints_fromscratch" in out, (
        f"from_scratch must write to charge3net_checkpoints_fromscratch/; stdout={out}"
    )


def test_from_scratch_mode_uses_distinct_wandb_name():
    """W&B run name differs between the two modes so the dashboard tells them apart."""
    # WANDB_NAME is what wandb reads at init time when no --name is passed.
    pretrained = _run({"LEMATRHO_TRAINING_MODE": "pretrained"}).stdout
    fromscratch = _run({"LEMATRHO_TRAINING_MODE": "from_scratch"}).stdout
    # Both must mention WANDB_NAME or set it somehow.
    assert "WANDB_NAME" in pretrained or "wandb-run-name" in pretrained, (
        f"pretrained mode must set the wandb run name; stdout={pretrained}"
    )
    assert "WANDB_NAME" in fromscratch or "wandb-run-name" in fromscratch, (
        f"from_scratch mode must set the wandb run name; stdout={fromscratch}"
    )

    # And they must differ.
    # Extract WANDB_NAME value from each (simple regex-free parsing).
    def _wandb_name(blob: str) -> str:
        for line in blob.splitlines():
            if "WANDB_NAME=" in line:
                return line.split("WANDB_NAME=", 1)[1].split()[0].strip("'\"")
        return ""

    p_name = _wandb_name(pretrained)
    f_name = _wandb_name(fromscratch)
    assert p_name and f_name and p_name != f_name, (
        f"WANDB_NAME must differ between modes; pretrained={p_name!r}, fromscratch={f_name!r}"
    )


def test_invalid_mode_exits_with_clear_error():
    """An unrecognized mode must fail fast with a helpful message."""
    result = _run({"LEMATRHO_TRAINING_MODE": "garbage"})
    assert result.returncode != 0, (
        f"invalid mode must exit non-zero; stdout={result.stdout} stderr={result.stderr}"
    )
    combined = (result.stdout + " " + result.stderr).lower()
    assert "garbage" in combined or "training_mode" in combined or "mode" in combined, (
        f"error message should mention the bad value or the env var; "
        f"stdout={result.stdout} stderr={result.stderr}"
    )


def test_batch_size_and_val_probes_match_paper():
    """Regression test: per-GPU batch=16, val_probes=1000 match the upstream paper."""
    result = _run({})
    assert "--batch-size 16" in result.stdout, (
        f"per-GPU batch must be 16 (paper); stdout={result.stdout}"
    )
    assert "--val-probes 1000" in result.stdout, (
        f"val_probes must be 1000 (paper); stdout={result.stdout}"
    )


def test_wandb_mode_is_offline():
    """W&B must default to offline; api.wandb.ai is unreachable from
    Adastra compute nodes (caused job 4969727 to crash after 1h47m).
    """
    result = _run({})
    assert "--wandb-mode offline" in result.stdout, (
        f"wandb-mode must default to offline; stdout={result.stdout}"
    )
