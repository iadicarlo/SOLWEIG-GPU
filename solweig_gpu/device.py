"""Device selection shared by all GPU code.

Order of preference: CUDA, then Apple Metal (MPS), then CPU. Set the
environment variable SOLWEIG_DEVICE to "cuda", "mps" or "cpu" to force one.

MPS has no float64 support, so on MPS every floating point tensor must be
float32. ``as_tensor`` takes care of that for arrays coming from numpy.
"""

import os

import numpy as np
import torch


def get_device():
    forced = os.environ.get("SOLWEIG_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def as_tensor(data, device=None, dtype=None):
    """torch.tensor() that never hands float64 to the MPS backend."""
    device = get_device() if device is None else torch.device(device)
    if dtype is None and device.type == "mps":
        if isinstance(data, np.ndarray) and data.dtype == np.float64:
            data = data.astype(np.float32)
        elif isinstance(data, float):
            dtype = torch.float32
    return torch.tensor(data, device=device, dtype=dtype)
