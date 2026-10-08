import numpy as np
import pytest
import torch

from solweig_gpu import device as dev

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS not available")


def test_forced_device(monkeypatch):
    monkeypatch.setenv("SOLWEIG_DEVICE", "cpu")
    assert dev.get_device().type == "cpu"


def test_auto_device_prefers_gpu(monkeypatch):
    monkeypatch.delenv("SOLWEIG_DEVICE", raising=False)
    expected = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    assert dev.get_device().type == expected


def test_empty_cache_calls_cuda_on_cuda_machines(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    dev.empty_cache()
    assert calls == ["cuda"]


def test_as_tensor_keeps_float64_on_cpu():
    t = dev.as_tensor(np.zeros(3, dtype=np.float64), device="cpu")
    assert t.dtype == torch.float64


@mps
def test_as_tensor_casts_float64_on_mps():
    t = dev.as_tensor(np.zeros(3, dtype=np.float64), device="mps")
    assert t.dtype == torch.float32 and t.device.type == "mps"


def _synthetic_scene(device):
    """A 120 x 120 m block with one building and one tree, 1 m pixels, ground at 5 m."""
    a = torch.full((120, 120), 5.0, device=device)
    a[40:60, 40:70] = 25.0
    trees = torch.zeros_like(a)
    trees[85:95, 30:40] = 12.0
    veg = trees + a
    veg[veg == a] = 0
    veg2 = trees * 0.25 + a
    veg2[veg2 == a] = 0
    bush = torch.zeros_like(a)
    amax = torch.maximum(a.max(), veg.max())
    return amax, a, veg, veg2, bush


@mps
@pytest.mark.parametrize("azimuth,altitude", [(0.0, 30.0), (135.0, 45.0), (200.0, 58.0), (290.0, 10.0)])
def test_shadow_same_on_mps_and_cpu(monkeypatch, azimuth, altitude):
    from solweig_gpu.shadow import shadow

    results = {}
    for name in ("cpu", "mps"):
        monkeypatch.setenv("SOLWEIG_DEVICE", name)
        out = shadow(*_synthetic_scene(torch.device(name)), azimuth, altitude, 1.0)
        results[name] = [x.cpu() for x in out]
    for c, m in zip(results["cpu"], results["mps"]):
        assert torch.equal(c, m)
    sh = results["mps"][0]
    assert sh.min() == 0 and sh.max() == 1  # the building really casts a shadow
