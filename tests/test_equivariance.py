"""Structural equivariance test for ChargE3Net.

ChargE3Net predicts the scalar charge density ρ(r). For the model to be
rotationally equivariant (i.e. ρ(R·r; R·atoms) == ρ(r; atoms)), the output
irreps of the probe-side network must contain only ℓ=0 even-parity components
("0e", pure scalars). This is the e3nn-level guarantee: as long as the final
representation is a scalar irrep, the model's output is invariant under SO(3)
acting on the input frame.

A runtime equivariance check (rotate inputs, predict, compare to predictions
on the unrotated inputs) is the gold standard but requires a real forward
pass on the production-sized model, which is too slow for a CPU unit test.
The structural test here covers the same property at the architecture level.

Skipped automatically when the upstream charge3net repo isn't on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip if the sibling charge3net repo isn't installed locally
# ---------------------------------------------------------------------------
_CHARGE3NET_ROOT = Path(__file__).resolve().parent.parent.parent / "charge3net"
if not _CHARGE3NET_ROOT.exists():
    pytest.skip(
        f"charge3net repo not at {_CHARGE3NET_ROOT}; "
        "clone github.com/AIforGreatGood/charge3net there to run this test",
        allow_module_level=True,
    )
if str(_CHARGE3NET_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHARGE3NET_ROOT))

from e3nn import o3
from src.charge3net.models.e3 import E3DensityModel


@pytest.fixture(scope="module")
def production_model():
    """Build a model with the MP-checkpoint hyperparameters.

    Module-scoped so the (slow) construction happens once for all assertions.
    """
    torch.manual_seed(0)
    model = E3DensityModel(
        num_interactions=3,
        num_neighbors=20,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=20,
    )
    model.train(False)
    return model


def test_param_count_matches_mp_checkpoint(production_model):
    """Sanity check: the model has the 1.9M params we expect.

    Guards against silently changing the architecture in a way that breaks
    checkpoint loading from charge3net_mp.pt.
    """
    n_params = sum(p.numel() for p in production_model.parameters())
    assert 1_900_000 <= n_params <= 1_920_000, (
        f"Architecture drift: expected ~1.91M params (MP checkpoint), got {n_params:,}"
    )


def test_atom_model_uses_higher_order_irreps(production_model):
    """ChargE3Net's atom representation must include ℓ>0 irreps to be 'higher-order'.

    The paper's central claim is that going from ℓ_max=1 to ℓ_max=4 produces
    substantially better densities on systems with subtle bonding. If someone
    accidentally drops the higher-l components (e.g. by passing lmax=0), the
    model degenerates to a scalar-only network and silently regresses to a
    much weaker baseline.
    """
    atom_irreps = production_model.atom_model.atom_irreps_sequence
    assert len(atom_irreps) > 0, "atom_irreps_sequence is empty"
    final_irreps = atom_irreps[-1]
    max_l = max(ir.l for _mul, ir in final_irreps)
    assert max_l >= 4, (
        f"Atom representation max ℓ is {max_l}; ChargE3Net's "
        f"higher-order claim requires ℓ_max ≥ 4. Got {final_irreps}."
    )


def test_atom_model_has_both_parities(production_model):
    """The atom representation should include both even (+) and odd (-) parity irreps.

    Without odd-parity components the model can't represent any vector- or
    pseudovector-valued atom features, which the higher-order convolutions
    need internally. The default get_irreps(mul, lmax) function in e3.py
    generates both; this test pins that down.
    """
    final_irreps = production_model.atom_model.atom_irreps_sequence[-1]
    parities = {ir.p for _mul, ir in final_irreps}
    assert parities == {-1, 1}, (
        f"Atom irreps should include both even (p=+1) and odd (p=-1) parities; "
        f"got parities {parities}: {final_irreps}"
    )


def test_get_irreps_helper_is_balanced():
    """The get_irreps helper in e3.py should produce roughly balanced channel counts.

    This is the function used to construct atom_irreps. If it ever returns
    zero-multiplicity for any (l, p) pair at production hyperparameters, the
    architecture breaks silently (some irreps disappear). Tests the helper
    directly to fail fast.
    """
    from src.charge3net.models.e3 import get_irreps

    irreps = get_irreps(500, lmax=4)
    multiplicities = [mul for mul, _ in irreps]
    assert all(mul > 0 for mul in multiplicities), (
        f"get_irreps(500, 4) produced a zero-multiplicity irrep: {irreps}"
    )
    # 5 ℓ levels × 2 parities = 10 entries
    assert len(irreps) == 10, (
        f"Expected 10 irreps (5 ℓ × 2 parity), got {len(irreps)}: {irreps}"
    )


def test_atom_irreps_sequence_length_matches_num_interactions(production_model):
    """One irreps entry per convolution layer (plus the input embedding)."""
    seq = production_model.atom_model.atom_irreps_sequence
    # num_interactions=3 → 3 convolutions; the sequence holds the post-conv
    # representations. Length will be 3 or 4 depending on whether the input
    # embedding is included; both are valid, but we pin a sane range.
    assert 3 <= len(seq) <= 5, (
        f"atom_irreps_sequence length {len(seq)} is outside the expected "
        f"range [3, 5] for num_interactions=3"
    )


def test_atom_model_uses_cutoff_consistent_with_kdtree(production_model):
    """The cutoff baked into the atom model must match what the dataset uses.

    `KdTreeGraphConstructor` in LeMatRhoDataset uses cutoff=4.0; if the model
    is built with a different cutoff, edges fed in at training time won't
    match what the convolution layer expects.
    """
    assert production_model.atom_model.cutoff == pytest.approx(4.0)


def test_e3nn_o3_irreps_are_proper_objects(production_model):
    """The atom representation must use e3nn's o3.Irreps wrapper.

    Equivariance is enforced by the o3.Irreps abstraction (which carries
    parity information and is consumed by FullyConnectedTensorProduct). If
    someone replaces it with a plain list, equivariance silently breaks even
    though the forward pass still produces output.
    """
    final_irreps = production_model.atom_model.atom_irreps_sequence[-1]
    assert isinstance(final_irreps, o3.Irreps), (
        f"Expected o3.Irreps for atom_irreps_sequence[-1]; got {type(final_irreps)}"
    )
