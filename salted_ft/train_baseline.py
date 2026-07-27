"""SALTED arm: PyTorch baseline coefficient-prediction model + train loop (D6).

Path B of the D6 plan: skip the rholearn integration, train a small
SchNet-style invariant message-passing network directly on the D2
projected coefficients with MSE loss. Produces a checkpoint that
``scripts/density_model_eval.py`` can load and exercise via the
SALTED arm path.

Architecture is deliberately minimal:

* Per-atom species embedding (Z -> ``hidden_dim`` vector).
* Gaussian RBF distance featurisation over neighbours within the
   ``BasisSpec.cutoff``.
* Two SchNet-style continuous-filter convolution layers.
* Per-atom readout MLP -> ``n_coeffs_per_atom`` channels.

Notes
-----

* The output is *invariant* under rotation. The l>0 channels of the
   SALTED basis are equivariant by construction, so this baseline
   will be systematically wrong on those channels. It still gives a
   reasonable scalar density once reconstructed, and is a starting
   point for the comparison table. Upgrade to e3nn/MACE for proper
   equivariance.
* The dataset reads two parquet directories: D2 source (atom
   positions) and D2 projected coefficients (training targets),
   joined on ``row_index`` per matching chunk filename.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import ase
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from ase.neighborlist import primitive_neighbor_list
from torch import nn

from salted_ft.basis import BasisSpec


class GaussianRBF(nn.Module):
    """Gaussian radial basis expansion of distances."""

    def __init__(self, n_basis: int = 16, cutoff: float = 4.0, sigma: float = 0.4):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_basis))
        self.sigma = sigma

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -0.5 * ((d[:, None] - self.centers[None, :]) / self.sigma) ** 2
        )


class CfConv(nn.Module):
    """SchNet-style continuous filter convolution."""

    def __init__(self, hidden_dim: int, n_basis: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pre = nn.Linear(hidden_dim, hidden_dim)
        self.post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x + self.post(self.pre(x) * 0)
        src, dst = edge_index
        msg = self.pre(x)[src] * self.filter_net(edge_rbf)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        return x + self.post(agg)


class SaltedBaselineModel(nn.Module):
    """SchNet-style invariant message-passing network for per-atom coefficients."""

    def __init__(
        self,
        basis_spec: BasisSpec,
        hidden_dim: int = 64,
        n_basis: int = 16,
        n_layers: int = 2,
        max_z: int = 120,
    ):
        super().__init__()
        self.basis_spec = basis_spec
        self.cutoff = float(basis_spec.cutoff)
        self.z_embed = nn.Embedding(max_z, hidden_dim)
        self.rbf = GaussianRBF(n_basis=n_basis, cutoff=self.cutoff)
        self.layers = nn.ModuleList(
            [CfConv(hidden_dim, n_basis) for _ in range(n_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, basis_spec.n_coeffs_per_atom),
        )

    def forward(self, atoms: ase.Atoms) -> torch.Tensor:
        device = self.z_embed.weight.device
        z = torch.from_numpy(atoms.get_atomic_numbers().astype(np.int64)).to(device)
        positions = atoms.get_positions().astype(np.float64)
        cell = np.asarray(atoms.get_cell(), dtype=np.float64)
        pbc = atoms.get_pbc()

        # ASE PBC-aware neighbour list within the cutoff.
        # 'ijD' -> source idx, dest idx, displacement vector
        i, j, D = primitive_neighbor_list("ijD", pbc, cell, positions, self.cutoff)
        if len(i) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_rbf = torch.zeros((0, self.rbf.centers.numel()), device=device)
        else:
            edge_index = torch.tensor(np.stack([i, j]), dtype=torch.long, device=device)
            dist = torch.tensor(
                np.linalg.norm(D, axis=1), dtype=torch.float32, device=device
            )
            edge_rbf = self.rbf(dist)

        x = self.z_embed(z)
        for layer in self.layers:
            x = layer(x, edge_index, edge_rbf)
        return self.readout(x)


class SaltedTrainingDataset:
    """Join D2 source (positions) + projected coefficients (targets) by row_index."""

    def __init__(
        self,
        source_dir: str | Path,
        coeffs_dir: str | Path,
    ):
        source_dir = Path(source_dir)
        coeffs_dir = Path(coeffs_dir)

        src_files = {p.name: p for p in source_dir.glob("chunk_*.parquet")}
        coeffs_files = {p.name: p for p in coeffs_dir.glob("chunk_*.parquet")}
        common = sorted(set(src_files) & set(coeffs_files))
        if not common:
            raise RuntimeError(
                f"No matching chunk_*.parquet in {source_dir} and {coeffs_dir}"
            )

        self._index: list[tuple[str, int]] = []
        for name in common:
            n = pq.ParquetFile(coeffs_files[name]).metadata.num_rows
            for ri in range(n):
                self._index.append((name, ri))
        self._src_files = src_files
        self._coeffs_files = coeffs_files
        # Per-chunk cache so each parquet is read at most once per worker.
        self._src_cache: dict[str, dict] = {}
        self._coeffs_cache: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _load(self, name: str) -> tuple[dict, dict]:
        if name not in self._src_cache:
            self._src_cache[name] = pq.read_table(self._src_files[name]).to_pydict()
        if name not in self._coeffs_cache:
            self._coeffs_cache[name] = pq.read_table(
                self._coeffs_files[name]
            ).to_pydict()
        return self._src_cache[name], self._coeffs_cache[name]

    def __getitem__(self, idx: int) -> tuple[ase.Atoms, torch.Tensor]:
        name, ri = self._index[idx]
        src, coeffs = self._load(name)
        # Match by row_index in case projected rows are a subset (D2 skips
        # rows with null charge density).
        src_row_indices = src["row_index"]
        try:
            src_ri = src_row_indices.index(coeffs["row_index"][ri])
        except ValueError as err:
            raise RuntimeError(
                f"Row {ri} of {name} (row_index="
                f"{coeffs['row_index'][ri]}) has no source counterpart"
            ) from err

        n_atoms = int(coeffs["n_atoms"][ri])
        positions = np.asarray(src["cartesian_site_positions"][src_ri]).reshape(-1, 3)
        cell = np.asarray(src["lattice_vectors"][src_ri]).reshape(3, 3)
        Z = np.asarray(coeffs["atomic_numbers"][ri])
        target = np.asarray(coeffs["coefficients"][ri]).reshape(n_atoms, -1)
        atoms = ase.Atoms(numbers=Z, positions=positions, cell=cell, pbc=True)
        return atoms, torch.from_numpy(target.astype(np.float32))


def train(
    source_dir: str | Path,
    coeffs_dir: str | Path,
    output_ckpt: str | Path,
    basis_spec: BasisSpec,
    n_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    log_every: int = 50,
) -> None:
    """Standard PyTorch training loop with gradient accumulation per batch."""
    dataset = SaltedTrainingDataset(source_dir, coeffs_dir)
    model = SaltedBaselineModel(basis_spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    step = 0
    for epoch in range(n_epochs):
        order = np.random.permutation(len(dataset))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            opt.zero_grad()
            losses = []
            for i in batch_idx:
                atoms, target = dataset[int(i)]
                target = target.to(device)
                pred = model(atoms)
                loss = F.mse_loss(pred, target)
                (loss / len(batch_idx)).backward()
                losses.append(loss.item())
            opt.step()
            step += 1
            if step % log_every == 0:
                mean = float(np.mean(losses))
                print(f"epoch {epoch} step {step} mse {mean:.6f}")

    torch.save(
        {"basis_spec": basis_spec, "model": model.state_dict()},
        Path(output_ckpt),
    )


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the SALTED baseline model.")
    p.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="D2 input parquet dir (cartesian_site_positions live here).",
    )
    p.add_argument(
        "--coeffs-dir",
        type=Path,
        required=True,
        help="D2 projected coefficients parquet dir.",
    )
    p.add_argument(
        "--output-ckpt",
        type=Path,
        required=True,
        help="Path for the trained checkpoint .pt file.",
    )
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = _build_cli().parse_args(argv)
    train(
        source_dir=args.source_dir,
        coeffs_dir=args.coeffs_dir,
        output_ckpt=args.output_ckpt,
        basis_spec=BasisSpec(),
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
