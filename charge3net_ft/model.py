"""
ChargE3Net model wrapper for fine-tuning on LeMatRho data.

Thin wrapper around the official E3DensityModel that handles:
1. Instantiation with the correct MP hyperparameters.
2. Loading pre-trained checkpoint weights (both legacy PL and new format).
3. Forward pass returning predicted density at probe points.
"""

import sys
from pathlib import Path

import torch
from torch import nn

# ---------------------------------------------------------------------------
# charge3net imports
# Expects: <parent of LeMat-Rho>/charge3net/ (cloned from AIforGreatGood/charge3net)
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

from src.charge3net.models.e3 import E3DensityModel

# ---------------------------------------------------------------------------
# Default hyperparameters matching the Materials Project checkpoint
# (from configs/charge3net/model/e3_density.yaml)
# ---------------------------------------------------------------------------
MP_MODEL_DEFAULTS = {
    "num_interactions": 3,
    "num_neighbors": 20,
    "mul": 500,
    "lmax": 4,
    "cutoff": 4.0,
    "basis": "gaussian",
    "num_basis": 20,
}


class ChargE3NetWrapper(nn.Module):
    """
    Wrapper around the official E3DensityModel.

    Instantiates the model with hyperparameters matching the pre-trained
    Materials Project checkpoint, and provides utilities for loading weights
    and running the forward pass.

    Parameters
    ----------
    ckpt_path : str or None
        Path to a pre-trained checkpoint (.pt file). If provided, weights
        are loaded immediately.
    model_kwargs : dict
        Override any of the default MP hyperparameters.
    """

    def __init__(self, ckpt_path: str | None = None, **model_kwargs):
        super().__init__()

        # Merge user overrides with MP defaults
        params = {**MP_MODEL_DEFAULTS, **model_kwargs}
        self.model = E3DensityModel(**params)
        self.cutoff = params["cutoff"]

        if ckpt_path is not None:
            self.load_pretrained(ckpt_path)

    def load_pretrained(self, ckpt_path: str):
        """
        Load pre-trained weights from a checkpoint file.

        Handles three formats:
        1. Legacy PyTorch Lightning: state_dict keys prefixed with "network."
        2. New charge3net format: checkpoint["model"] contains the state_dict.
        3. Raw state_dict: the file IS the state_dict.

        Note: weights_only=False is required for the legacy PL format which
        stores non-tensor objects. The checkpoint is our own internal file
        (AIforGreatGood/charge3net repo), not untrusted user input.
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict):
            if "pytorch-lightning_version" in checkpoint:
                # Legacy PL format: keys like "network.atom_model.interactions.0...."
                state_dict = {
                    k.replace("network.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                }
                print(f"Loaded legacy PL checkpoint from {ckpt_path}")
            elif "model" in checkpoint:
                # New charge3net trainer format
                state_dict = checkpoint["model"]
                print(f"Loaded charge3net checkpoint from {ckpt_path}")
            elif "state_dict" in checkpoint:
                # Generic PL format without version key
                state_dict = {
                    k.replace("network.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                }
                print(f"Loaded state_dict checkpoint from {ckpt_path}")
            else:
                # Assume the dict IS the state_dict
                state_dict = checkpoint
                print(f"Loaded raw state_dict from {ckpt_path}")
        else:
            raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

    def forward(self, input_dict: dict) -> torch.Tensor:
        """
        Run the E3DensityModel forward pass.

        Parameters
        ----------
        input_dict : dict
            Batch dict with keys matching charge3net's expected format:
            nodes, atom_xyz, cell, atom_edges, atom_edges_displacement,
            num_nodes, num_atom_edges, probe_xyz, probe_edges,
            probe_edges_displacement, num_probes, num_probe_edges.

        Returns
        -------
        torch.Tensor
            Predicted density at probe points, shape [B, max_probes].
        """
        return self.model(input_dict)
