"""
Unit tests for charge3net_ft metric functions.

These tests use synthetic tensors with known ground-truth values and do not
require real data, GPU, or the charge3net dependency.
"""

import pytest
import torch

# Import directly so tests work without the charge3net sibling repo installed
import sys

# Patch the charge3net path check so we can import train.py without the repo
sys.modules.setdefault("src", type(sys)("src"))

# We import the metric functions directly by exec'ing only the relevant parts
# of train.py to avoid triggering the charge3net sys.path block at module level.
from charge3net_ft.train import compute_nmape, compute_nrmse, compute_rmse  # noqa: E402


class TestComputeNmape:
    def test_perfect_prediction(self):
        t = torch.tensor([[1.0, 2.0, 3.0]])
        assert compute_nmape(t, t).item() == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self):
        # preds all zero, targets = [2, 2] → NMAPE = 4/(4) * 100 = 100%
        preds = torch.zeros(1, 2)
        targets = torch.tensor([[2.0, 2.0]])
        assert compute_nmape(preds, targets).item() == pytest.approx(100.0, rel=1e-4)

    def test_batch_average(self):
        # Sample 1: 0% error; Sample 2: 100% error → average 50%
        preds = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        targets = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        assert compute_nmape(preds, targets).item() == pytest.approx(50.0, rel=1e-4)

    def test_with_mask_ignores_padding(self):
        # Probe 3 is padding (zero); should give same result as without it
        preds = torch.tensor([[0.0, 0.0, 0.0]])
        targets = torch.tensor([[2.0, 2.0, 0.0]])
        num_probes = torch.tensor([2])
        masked = compute_nmape(preds, targets, num_probes).item()
        assert masked == pytest.approx(100.0, rel=1e-4)

    def test_mask_vs_no_mask_differ_when_padding_nonzero(self):
        # Without mask, zero-padding in targets dilutes the denominator
        preds = torch.tensor([[0.0, 0.0, 0.0]])
        targets = torch.tensor([[2.0, 2.0, 100.0]])  # row 3 is "padding"
        num_probes = torch.tensor([2])
        masked = compute_nmape(preds, targets, num_probes).item()
        unmasked = compute_nmape(preds, targets).item()
        assert masked != pytest.approx(unmasked, rel=1e-2)


class TestComputeRmse:
    def test_perfect_prediction(self):
        t = torch.tensor([[1.0, 2.0]])
        assert compute_rmse(t, t).item() == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self):
        # preds = [0], targets = [3] → MSE = 9 → RMSE = 3
        preds = torch.zeros(1, 1)
        targets = torch.tensor([[3.0]])
        assert compute_rmse(preds, targets).item() == pytest.approx(3.0, rel=1e-4)

    def test_with_mask(self):
        # Same as above but with a padding element appended
        preds = torch.zeros(1, 2)
        targets = torch.tensor([[3.0, 999.0]])
        num_probes = torch.tensor([1])
        assert compute_rmse(preds, targets, num_probes).item() == pytest.approx(3.0, rel=1e-4)


class TestComputeNrmse:
    def test_perfect_prediction(self):
        t = torch.tensor([[2.0, 4.0]])
        assert compute_nrmse(t, t).item() == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self):
        # preds = [0, 0], targets = [2, 2] → RMSE = 2, mean(|target|) = 2 → NRMSE = 100%
        preds = torch.zeros(1, 2)
        targets = torch.tensor([[2.0, 2.0]])
        assert compute_nrmse(preds, targets).item() == pytest.approx(100.0, rel=1e-4)

    def test_with_mask(self):
        preds = torch.zeros(1, 3)
        targets = torch.tensor([[2.0, 2.0, 999.0]])
        num_probes = torch.tensor([2])
        assert compute_nrmse(preds, targets, num_probes).item() == pytest.approx(100.0, rel=1e-4)
