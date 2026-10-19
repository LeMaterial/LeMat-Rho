"""
Unit tests for ChargE3NetWrapper checkpoint loading.

Tests all three checkpoint format branches using mock state dicts,
without requiring the real charge3net repo or a GPU.
"""

import sys
import tempfile
from unittest.mock import patch

import pytest
import torch
from torch import nn


def _make_mock_e3density():
    """Return a tiny 2-param nn.Module standing in for E3DensityModel."""

    class _Tiny(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(2))

        def forward(self, x):
            return x

    return _Tiny


def _import_wrapper():
    """Import ChargE3NetWrapper with charge3net stubbed out."""
    fake = {
        "src": type(sys)("src"),
        "src.charge3net": type(sys)("src.charge3net"),
        "src.charge3net.models": type(sys)("src.charge3net.models"),
        "src.charge3net.models.e3": type(sys)("src.charge3net.models.e3"),
    }
    MockModel = _make_mock_e3density()
    fake["src.charge3net.models.e3"].E3DensityModel = MockModel

    with (
        patch.dict(sys.modules, fake),
        patch("pathlib.Path.exists", return_value=True),
    ):
        if "charge3net_ft.model" in sys.modules:
            del sys.modules["charge3net_ft.model"]
        import importlib

        mod = importlib.import_module("charge3net_ft.model")
    return mod.ChargE3NetWrapper, MockModel


class TestLoadPretrained:
    def _save_and_load(self, checkpoint, wrapper_cls):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        torch.save(checkpoint, path)
        w = wrapper_cls()
        w.load_pretrained(path)
        return w

    def test_new_charge3net_format(self):
        WrapperCls, MockModel = _import_wrapper()
        state_dict = MockModel().state_dict()
        ckpt = {"model": state_dict}
        w = self._save_and_load(ckpt, WrapperCls)
        # Should load without errors
        assert w is not None

    def test_legacy_pl_format(self):
        WrapperCls, MockModel = _import_wrapper()
        state_dict = {f"network.{k}": v for k, v in MockModel().state_dict().items()}
        ckpt = {
            "pytorch-lightning_version": "1.9.0",
            "state_dict": state_dict,
        }
        w = self._save_and_load(ckpt, WrapperCls)
        assert w is not None

    def test_generic_state_dict_format(self):
        WrapperCls, MockModel = _import_wrapper()
        state_dict = {f"network.{k}": v for k, v in MockModel().state_dict().items()}
        ckpt = {"state_dict": state_dict}
        w = self._save_and_load(ckpt, WrapperCls)
        assert w is not None

    def test_raw_state_dict_format(self):
        WrapperCls, MockModel = _import_wrapper()
        ckpt = MockModel().state_dict()  # the dict IS the state_dict
        w = self._save_and_load(ckpt, WrapperCls)
        assert w is not None

    def test_missing_file_raises(self):
        WrapperCls, _ = _import_wrapper()
        w = WrapperCls()
        with pytest.raises(FileNotFoundError):
            w.load_pretrained("/nonexistent/path/checkpoint.pt")

    def test_unexpected_format_raises(self):
        WrapperCls, _ = _import_wrapper()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        torch.save("not_a_dict", path)
        w = WrapperCls()
        with pytest.raises(TypeError):
            w.load_pretrained(path)
