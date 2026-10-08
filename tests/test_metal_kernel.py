"""The fused Metal shadow kernel must give exactly the PyTorch results on MPS."""

import numpy as np
import pytest
import torch

from solweig_gpu.device import get_device
from solweig_gpu.metal_kernels import metal_enabled

# The model code puts its own arrays on get_device(), so run only when that is MPS.
mps = pytest.mark.skipif(get_device().type != "mps", reason="needs Apple MPS")


def _scene(n=96, seed=1):
    rng = np.random.default_rng(seed)
    dsm = np.full((n, n), 2.0, dtype=np.float32) + rng.random((n, n), dtype=np.float32) * 0.3
    for _ in range(12):
        x, y = rng.integers(0, n - 12, 2)
        w, h = rng.integers(4, 12, 2)
        dsm[x:x + w, y:y + h] += rng.uniform(5, 30)
    trees = np.zeros_like(dsm)
    for _ in range(20):
        x, y = rng.integers(2, n - 6, 2)
        r = rng.integers(1, 4)
        trees[x - r:x + r, y - r:y + r] = rng.uniform(3, 15)
    trees[dsm > 5] = 0
    veg = trees + dsm
    veg[veg == dsm] = 0
    veg2 = 0.25 * trees + dsm
    veg2[veg2 == dsm] = 0
    bush = np.logical_not(veg2 * veg) * veg
    amax = max(dsm.max(), trees.max())
    dev = torch.device("mps")
    t = [torch.tensor(x, dtype=torch.float32, device=dev) for x in (dsm, veg, veg2, bush)]
    return float(amax), t


def _both(monkeypatch, fn):
    monkeypatch.setenv("SOLWEIG_METAL_KERNEL", "0")
    ref = [o.cpu().numpy() for o in fn()]
    monkeypatch.setenv("SOLWEIG_METAL_KERNEL", "1")
    assert metal_enabled(torch.zeros(1, device="mps"))
    new = [o.cpu().numpy() for o in fn()]
    return ref, new


@mps
@pytest.mark.parametrize("azimuth,altitude", [(0, 6), (23.2, 6), (45, 18), (100, 30), (135, 42),
                                              (180, 54), (225, 10), (270, 66), (315, 78), (0, 90)])
def test_metal_shadow_matches_pytorch(monkeypatch, azimuth, altitude):
    from solweig_gpu.shadow import shadow

    amax, (a, veg, veg2, bush) = _scene()
    ref, new = _both(monkeypatch, lambda: shadow(amax, a, veg, veg2, bush, azimuth, altitude, 1.0))
    for r, n in zip(ref, new):
        np.testing.assert_array_equal(n, r)


@mps
@pytest.mark.parametrize("azimuth,altitude", [(60, 5), (135, 35), (200, 58), (290, 12)])
def test_metal_wallheight_matches_pytorch(monkeypatch, azimuth, altitude):
    from solweig_gpu.solweig import shadowingfunction_wallheight_23

    amax, (a, veg, veg2, bush) = _scene(seed=2)
    walls = torch.clamp(a - torch.roll(a, 1, 0), min=0)
    aspect = torch.remainder(torch.arange(a.numel(), device=a.device, dtype=torch.float32), 6.28).reshape(a.shape)
    ref, new = _both(monkeypatch, lambda: shadowingfunction_wallheight_23(
        a, veg, veg2, azimuth, altitude, 1.0, amax, bush, walls, aspect))
    for r, n in zip(ref, new):
        np.testing.assert_array_equal(n, r)


@mps
@pytest.mark.parametrize("scale,landcover", [(1.0, 0), (2.0, 1)])
def test_metal_gvf_matches_pytorch(monkeypatch, scale, landcover):
    from solweig_gpu.solweig import gvf_2018a, shadowingfunction_wallheight_23

    amax, (a, veg, veg2, bush) = _scene(seed=3)
    dev = a.device
    walls = torch.clamp(a - torch.roll(a, 1, 0), min=0)
    aspect = torch.remainder(torch.arange(a.numel(), device=dev, dtype=torch.float32) * 7.0, 360.0).reshape(a.shape)
    vegsh, sh, _, _, wallsun, _, _, _ = shadowingfunction_wallheight_23(
        a, veg, veg2, 140.0, 30.0, scale, amax, bush, walls, aspect * torch.pi / 180)
    shadow = sh - (1 - vegsh) * (1 - 0.03)
    buildings = (a < 5).float()
    lc = torch.where(a < 2.1, 3.0, 1.0).to(dev)
    gen = torch.Generator().manual_seed(0)
    Tg = (torch.rand(a.shape, generator=gen) * 12).to(dev)
    t = lambda v: torch.tensor(v, device=dev)

    def run():
        return gvf_2018a(wallsun.clone(), walls, buildings, scale, shadow, t(1.0), t(22.0), aspect, Tg.clone(),
                         t(3.5), t(25.0), torch.full_like(a, 0.95), t(0.9), torch.full_like(a, 0.15), 5.67e-8, t(0.2),
                         a.shape[0], a.shape[1], t(20.0), lc, landcover)

    ref, new = _both(monkeypatch, run)
    for r, n in zip(ref, new):
        np.testing.assert_array_equal(n, r)


@mps
def test_metal_svf_matches_pytorch(monkeypatch):
    from solweig_gpu.shadow import svf_calculator

    amax, (a, veg, veg2, bush) = _scene(n=64, seed=4)
    ref, new = _both(monkeypatch, lambda: svf_calculator(2, torch.tensor(amax), a, veg, veg2, bush, 1.0))
    for r, n in zip(ref, new):
        np.testing.assert_array_equal(n, r)
