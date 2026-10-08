"""Sky view factor regression tests against values from UMEP's numpy SOLWEIG code."""

import numpy as np
import torch

from solweig_gpu.device import get_device
from solweig_gpu.shadow import create_patches, svf_calculator

# UMEP svfForProcessing153 on a flat, open 60 x 60 surface (centre pixel).
# N, S and W are below 1 in UMEP itself because of the patch discretisation.
UMEP_FLAT = {"svf": 1.0, "svfE": 1.0, "svfN": 0.9762, "svfS": 0.9475, "svfW": 0.9237}


def test_every_patch_gets_its_own_azimuth():
    _, _, _, skyvaultaltint, aziinterval, _, azistart = create_patches(2)
    assert int(sum(aziinterval)) == 153
    # in float32, int(360 / (360 / 30)) is 29, which used to drop patches
    step32 = torch.tensor([360 / p for p in aziinterval.tolist()], dtype=torch.float32)
    assert any(int(360 / s) != int(n) for s, n in zip(step32, aziinterval))


def test_flat_surface_matches_umep():
    device = get_device()
    a = torch.full((60, 60), 20.0, device=device)
    zeros = torch.zeros_like(a)
    out = svf_calculator(2, torch.tensor(20.0), a, zeros, zeros.clone(), zeros.clone(), 1.0)
    names = ["svf", "svfaveg", "svfE", "svfEaveg", "svfEveg", "svfN", "svfNaveg", "svfNveg",
             "svfS", "svfSaveg", "svfSveg", "svfveg", "svfW", "svfWaveg", "svfWveg"]
    got = dict(zip(names, out))
    for name, expected in UMEP_FLAT.items():
        assert np.isclose(float(got[name][30, 30]), expected, atol=5e-4), name
