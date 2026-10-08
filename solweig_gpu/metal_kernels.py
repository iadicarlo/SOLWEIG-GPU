"""Fused Metal kernels for the shadow ray march on Apple GPUs (MPS).

The PyTorch shadow routines shift whole 2D arrays once per ray step and make
about twenty full memory passes per step. Here one GPU thread handles one
pixel: it walks its own ray towards the sun, reads the three height grids at
the source cell of every step and keeps its running state in registers. No
intermediate arrays are written, so the work is no longer bound by memory
bandwidth.

The result is bit for bit the same as the PyTorch code:

* The step offsets (dx, dy) and heights (dz, dzprev) are computed here on the
  host with exactly the same Python/numpy double precision arithmetic as in
  shadow() and shadowingfunction_wallheight_23(), then rounded to float32 the
  way PyTorch rounds a Python scalar in ``tensor - dz``.
* Cells whose source lies outside the grid see 0, like the zeroed temp arrays.
* max() propagates NaN like torch.max, and the shader is compiled in safe math
  mode so no comparison is rewritten.

The Metal path is used only for float32 tensors on an MPS device. Set
SOLWEIG_METAL_KERNEL=0 to fall back to the PyTorch code.
"""

import math
import os
import warnings

import numpy as np
import torch

_SOURCE = r"""
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

// torch.max / torch.maximum: NaN wins.
inline float nanmax(float m, float x) {
    return isnan(m) ? m : ((isnan(x) || x > m) ? x : m);
}

// params: sizex, sizey, nsteps
template <bool VOLUMES>
inline void ray(device const float4* h, device const float* bush,
                constant int2* dxy, constant float2* dzs, constant int* params,
                device float* sh_out, device float* vegsh_out, device float* vbsh_out,
                device float* f_out, device float* shvoveg_out, uint idx)
{
    const int sizex = params[0];
    const int sizey = params[1];
    const int nsteps = params[2];
    if ((int)idx >= sizex * sizey) return;
    const int x = (int)idx / sizey;
    const int y = (int)idx - x * sizey;
    const float ap = h[idx].x;
    float f = ap;
    float shvoveg = h[idx].y;
    bool sh = false;
    float vegsh = (bush[idx] > 1.0f) ? 1.0f : 0.0f;
    bool anyveg = false;
    for (int s = 0; s < nsteps; ++s) {
        const int2 d = dxy[s];
        const float2 z = dzs[s];          // (dz, dzprev)
        const int sx = x + d.x;
        const int sy = y + d.y;
        float t = 0.0f, tv = 0.0f, tv2 = 0.0f, lv = 0.0f, lv2 = 0.0f;
        if (sx >= 0 && sx < sizex && sy >= 0 && sy < sizey) {
            const float4 c = h[sx * sizey + sy];   // (a, vegdem, vegdem2, -)
            t = c.x - z.x;
            tv = c.y - z.x;
            tv2 = c.z - z.x;
            lv = c.y - z.y;
            lv2 = c.z - z.y;
        }
        f = nanmax(f, t);
        if (VOLUMES) shvoveg = nanmax(shvoveg, tv);
        sh = f > ap;
        const int cnt = (int)(tv > ap) + (int)(tv2 > ap) + (int)(lv > ap) + (int)(lv2 > ap);
        const float vegsh2 = (cnt > 0 && cnt < 4) ? 1.0f : 0.0f;
        vegsh = nanmax(vegsh, vegsh2);
        if (vegsh * (sh ? 1.0f : 0.0f) > 0.0f) vegsh = 0.0f;
        anyveg = anyveg || (vegsh > 0.0f);
    }
    const float vb = anyveg ? 1.0f : 0.0f;
    sh_out[idx] = 1.0f - (sh ? 1.0f : 0.0f);
    vegsh_out[idx] = 1.0f - vegsh;
    vbsh_out[idx] = 1.0f - (vb - vegsh);
    if (VOLUMES) {
        f_out[idx] = f;
        shvoveg_out[idx] = shvoveg;
    }
}

kernel void shadow_ray(device const float4* h [[buffer(0)]],
                       device const float* bush [[buffer(1)]],
                       constant int2* dxy [[buffer(2)]],
                       constant float2* dzs [[buffer(3)]],
                       constant int* params [[buffer(4)]],
                       device float* sh_out [[buffer(5)]],
                       device float* vegsh_out [[buffer(6)]],
                       device float* vbsh_out [[buffer(7)]],
                       uint idx [[thread_position_in_grid]])
{
    ray<false>(h, bush, dxy, dzs, params, sh_out, vegsh_out, vbsh_out,
               sh_out, sh_out, idx);
}

kernel void shadow_ray_volumes(device const float4* h [[buffer(0)]],
                               device const float* bush [[buffer(1)]],
                               constant int2* dxy [[buffer(2)]],
                               constant float2* dzs [[buffer(3)]],
                               constant int* params [[buffer(4)]],
                               device float* sh_out [[buffer(5)]],
                               device float* vegsh_out [[buffer(6)]],
                               device float* vbsh_out [[buffer(7)]],
                               device float* f_out [[buffer(8)]],
                               device float* shvoveg_out [[buffer(9)]],
                               uint idx [[thread_position_in_grid]])
{
    ray<true>(h, bush, dxy, dzs, params, sh_out, vegsh_out, vbsh_out,
              f_out, shvoveg_out, idx);
}
// Ground view factor sweep of sunonsurface_2018a. The PyTorch temp arrays are
// never cleared, so a cell whose source leaves the grid keeps the last value.
// params: sizex, sizey, nsteps, snapshot step; out holds 16 planes.
kernel void sunonsurface_sweep(device const float4* g [[buffer(0)]],
                               device const float2* g2 [[buffer(1)]],
                               device const float* lwall [[buffer(2)]],
                               constant int2* dxy [[buffer(3)]],
                               constant int* params [[buffer(4)]],
                               constant float& albedo_b [[buffer(5)]],
                               device float* out [[buffer(6)]],
                               uint idx [[thread_position_in_grid]])
{
    const int sizex = params[0];
    const int sizey = params[1];
    const int nsteps = params[2];
    const int snap = params[3];
    const int npix = sizex * sizey;
    if ((int)idx >= npix) return;
    const int x = (int)idx / sizey;
    const int y = (int)idx - x * sizey;
    float f = g[idx].x;
    const float lw = lwall[idx];
    float tb = 0.0f, ts = 0.0f, tl = 0.0f, tas = 0.0f, tan_ = 0.0f, tws = 0.0f;
    float wsh = 0.0f, wlupsh = 0.0f, walbsh = 0.0f, walbnosh = 0.0f;
    float wlwall = 0.0f, walbwall = 0.0f, wwall = 0.0f, walbwallnosh = 0.0f;
    float bub = 0.0f, bubwall = 0.0f;
    float snapv[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int s = 0; s < nsteps; ++s) {
        const int sx = x + dxy[s].x;
        const int sy = y + dxy[s].y;
        if (sx >= 0 && sx < sizex && sy >= 0 && sy < sizey) {
            const int j = sx * sizey + sy;
            const float4 c = g[j];         // buildings, shadow, Lup, albshadow
            const float2 c2 = g2[j];       // alb, sunwall
            tb = c.x; ts = c.y; tl = c.z; tas = c.w; tan_ = c2.x; tws = c2.y;
        }
        f = isnan(f) ? f : ((isnan(tb) || tb < f) ? tb : f);
        wsh += ts * f;
        wlupsh += tl * f;
        walbsh += tas * f;
        walbnosh += tan_ * f;
        const float tempb = tws * f;
        const float tempbwall = f * -1.0f + 1.0f;
        bub = ((tempb + bub) > 0.0f) ? 1.0f : 0.0f;
        bubwall = ((tempbwall + bubwall) > 0.0f) ? 1.0f : 0.0f;
        wlwall += bub * lw;
        walbwall += bub * albedo_b;
        wwall += bub;
        walbwallnosh += bubwall * albedo_b;
        if (s == snap) {
            snapv[0] = wsh; snapv[1] = wlupsh; snapv[2] = walbsh; snapv[3] = walbnosh;
            snapv[4] = wlwall; snapv[5] = walbwall; snapv[6] = wwall; snapv[7] = walbwallnosh;
        }
    }
    out[0 * npix + idx] = wsh;
    out[1 * npix + idx] = wlupsh;
    out[2 * npix + idx] = walbsh;
    out[3 * npix + idx] = walbnosh;
    out[4 * npix + idx] = wlwall;
    out[5 * npix + idx] = walbwall;
    out[6 * npix + idx] = wwall;
    out[7 * npix + idx] = walbwallnosh;
    for (int k = 0; k < 8; ++k) out[(8 + k) * npix + idx] = snapv[k];
}
// One sky patch of svf_calculator: the same float32 sums, in the same order,
// as its loop over the annulus rings k. acc holds 15 planes:
// svf, svfE, svfS, svfW, svfN, svfveg, svfEveg, svfSveg, svfWveg, svfNveg,
// svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg.
// params: npix, nk, direction bits (E=1, S=2, W=4, N=8), patch index, npatch
kernel void svf_accumulate(device const float* sh [[buffer(0)]],
                           device const float* vegsh [[buffer(1)]],
                           device const float* vbsh [[buffer(2)]],
                           device float* acc [[buffer(3)]],
                           device float* shmat [[buffer(4)]],
                           device float* vegshmat [[buffer(5)]],
                           device float* vbshmat [[buffer(6)]],
                           constant float2* w [[buffer(7)]],
                           constant int* params [[buffer(8)]],
                           uint idx [[thread_position_in_grid]])
{
    const int npix = params[0];
    const int nk = params[1];
    const int dirs = params[2];
    const int patch = params[3];
    const int npatch = params[4];
    if ((int)idx >= npix) return;
    const float s = sh[idx];
    const float v = vegsh[idx];
    const float b = vbsh[idx];
    float a[15];
    for (int j = 0; j < 15; ++j) a[j] = acc[j * npix + idx];
    for (int k = 0; k < nk; ++k) {
        const float wf = w[k].x;   // annulus weight for the full ring
        const float wa = w[k].y;   // and for the half ring
        a[0] = a[0] + wf * s;
        const float ws = wa * s;
        if (dirs & 1) a[1] = a[1] + ws;
        if (dirs & 2) a[2] = a[2] + ws;
        if (dirs & 4) a[3] = a[3] + ws;
        if (dirs & 8) a[4] = a[4] + ws;
        a[5] = a[5] + wf * v;
        a[10] = a[10] + wf * b;
        for (int d = 0; d < 4; ++d) {
            if (dirs & (1 << d)) {
                a[6 + d] = a[6 + d] + wa * v;
                a[11 + d] = a[11 + d] + wa * b;
            }
        }
    }
    for (int j = 0; j < 15; ++j) acc[j * npix + idx] = a[j];
    const ulong m = (ulong)idx * (ulong)npatch + (ulong)patch;   // can pass 2^31 on big tiles
    shmat[m] = s;
    vegshmat[m] = v;
    vbshmat[m] = b;
}
"""

_LIB = None
_LIB_FAILED = False


def _library():
    global _LIB, _LIB_FAILED
    if _LIB is None and not _LIB_FAILED:
        try:
            _LIB = torch.mps.compile_shader(_SOURCE)
        except Exception as exc:  # pragma: no cover - depends on the platform
            _LIB_FAILED = True
            warnings.warn(f"Metal shadow kernel unavailable, using PyTorch code: {exc}")
    return _LIB


def metal_enabled(*tensors):
    """True when the fused Metal kernel should handle these tensors."""
    if os.environ.get("SOLWEIG_METAL_KERNEL", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    if not hasattr(torch, "mps") or not hasattr(torch.mps, "compile_shader"):
        return False
    for t in tensors:
        if not isinstance(t, torch.Tensor) or t.device.type != "mps" or t.dtype != torch.float32:
            return False
    return _library() is not None


def ray_steps(azimuth, altitude, scale, amaxvalue, sizex, sizey):
    """Offsets and heights of every ray step, as the PyTorch loop makes them.

    ``azimuth`` and ``altitude`` are in degrees; ``amaxvalue`` is the height
    range the loop runs over (already reduced by the lowest surface).
    Returns int32 array (n, 2) with (dx, dy) and float32 array (n, 2) with
    (dz, dzprev) for each step.
    """
    # Same statements, in the same order, as shadow() in shadow.py.
    degrees = math.pi / 180.
    azimuth = float(azimuth) * degrees
    altitude = float(altitude) * degrees
    pibyfour = math.pi / 4.
    threetimespibyfour = 3. * pibyfour
    fivetimespibyfour = 5. * pibyfour
    seventimespibyfour = 7. * pibyfour
    sinazimuth = float(np.sin(azimuth))
    cosazimuth = float(np.cos(azimuth))
    tanazimuth = float(np.tan(azimuth))
    signsinazimuth = math.copysign(1., sinazimuth) if sinazimuth != 0 else 0.
    signcosazimuth = math.copysign(1., cosazimuth) if cosazimuth != 0 else 0.
    dssin = abs(1. / sinazimuth) if sinazimuth != 0 else math.inf
    dscos = abs(1. / cosazimuth) if cosazimuth != 0 else math.inf
    tanaltitudebyscale = float(np.tan(altitude)) / scale
    index = 0
    dx = dy = dz = 0.
    dzprev = 0.
    dxy = []
    dzs = []
    while (amaxvalue >= dzprev and abs(dx) < sizex and abs(dy) < sizey):
        if (pibyfour <= azimuth < threetimespibyfour or fivetimespibyfour <= azimuth < seventimespibyfour):
            dy = signsinazimuth * index
            dx = -1. * signcosazimuth * abs(round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * abs(round(index * tanazimuth))
            dx = -1. * signcosazimuth * index
            ds = dscos
        dz = ds * index * tanaltitudebyscale
        # the PyTorch slices are only well defined while |dx| <= sizex
        if abs(dx) > sizex or abs(dy) > sizey:
            raise RuntimeError("ray step jumped past the grid edge")
        dxy.append((int(dx), int(dy)))
        dzs.append((dz, dzprev))
        dzprev = dz
        index += 1
    return np.asarray(dxy, dtype=np.int32).reshape(-1, 2), np.asarray(dzs, dtype=np.float32).reshape(-1, 2)


def pack_heights(a, vegdem, vegdem2):
    """(rows, cols, 4) float32 tensor with a, vegdem, vegdem2 in one 16 byte cell."""
    h = torch.empty(a.shape + (4,), device=a.device, dtype=torch.float32)
    h[..., 0] = a
    h[..., 1] = vegdem
    h[..., 2] = vegdem2
    h[..., 3] = 0.
    return h


def shadow_metal(amaxvalue, a, vegdem, vegdem2, bush, azimuth, altitude, scale,
                 volumes=False, packed=None):
    """Fused version of the ray march shared by shadow() and wallheight_23.

    Returns (sh, vegsh, vbshvegsh) already in their final 1 = sunlit form,
    plus (f, shvoveg) when ``volumes`` is True.
    """
    lib = _library()
    sizex, sizey = a.shape
    device = a.device
    rng = max(float(amaxvalue), float(a.max()), float(vegdem.max())) - float(a.min())
    dxy, dzs = ray_steps(azimuth, altitude, scale, rng, sizex, sizey)
    n = dxy.shape[0]
    h = packed if packed is not None else pack_heights(a, vegdem, vegdem2)
    bush = bush.contiguous()
    dxy_t = torch.from_numpy(dxy).to(device)
    dzs_t = torch.from_numpy(dzs).to(device)
    params = torch.tensor([sizex, sizey, n], dtype=torch.int32, device=device)
    sh = torch.empty((sizex, sizey), device=device, dtype=torch.float32)
    vegsh = torch.empty_like(sh)
    vbshvegsh = torch.empty_like(sh)
    threads = sizex * sizey
    if volumes:
        f = torch.empty_like(sh)
        shvoveg = torch.empty_like(sh)
        lib.shadow_ray_volumes(h, bush, dxy_t, dzs_t, params, sh, vegsh, vbshvegsh, f, shvoveg,
                               threads=threads)
        return sh, vegsh, vbshvegsh, f, shvoveg
    lib.shadow_ray(h, bush, dxy_t, dzs_t, params, sh, vegsh, vbshvegsh, threads=threads)
    return sh, vegsh, vbshvegsh


def gvf_offsets(azimuth, nsteps, sizex, sizey):
    """(dx, dy) per step of sunonsurface_2018a, with the same double precision code."""
    azimuth = float(azimuth) * (math.pi / 180)
    pibyfour = math.pi / 4
    threetimespibyfour = 3 * pibyfour
    fivetimespibyfour = 5 * pibyfour
    seventimespibyfour = 7 * pibyfour
    sinazimuth = float(np.sin(azimuth))
    cosazimuth = float(np.cos(azimuth))
    tanazimuth = float(np.tan(azimuth))
    signsinazimuth = math.copysign(1., sinazimuth) if sinazimuth != 0 else 0.
    signcosazimuth = math.copysign(1., cosazimuth) if cosazimuth != 0 else 0.
    dxy = []
    for index in range(nsteps):
        if (pibyfour <= azimuth and azimuth < threetimespibyfour) or (fivetimespibyfour <= azimuth and azimuth < seventimespibyfour):
            dy = signsinazimuth * index
            dx = -1 * signcosazimuth * abs(round(index / tanazimuth))
        else:
            dy = signsinazimuth * abs(round(index * tanazimuth))
            dx = -1 * signcosazimuth * index
        if abs(dx) > sizex or abs(dy) > sizey:
            raise RuntimeError("ray step jumped past the grid edge")
        dxy.append((int(dx), int(dy)))
    return np.asarray(dxy, dtype=np.int32).reshape(-1, 2)


def sunonsurface_sweep(azimuth, nsteps, snap, buildings, shadow, Lup, albshadow, alb, sunwall, Lwall, albedo_b):
    """Weight sums of sunonsurface_2018a after the last step and after step ``snap``.

    Returns a (16, rows, cols) tensor: weightsumsh, weightsumLupsh, weightsumalbsh,
    weightsumalbnosh, weightsumLwall, weightsumalbwall, weightsumwall,
    weightsumalbwallnosh, then the same eight at step ``snap``.
    """
    lib = _library()
    sizex, sizey = buildings.shape
    device = buildings.device
    shape = (sizex, sizey)
    g = torch.stack([torch.broadcast_to(t, shape) for t in (buildings, shadow, Lup, albshadow)], dim=-1).contiguous()
    g2 = torch.stack([torch.broadcast_to(t, shape) for t in (alb, sunwall)], dim=-1).contiguous()
    lwall = torch.broadcast_to(Lwall, shape).contiguous()
    dxy_t = torch.from_numpy(gvf_offsets(azimuth, nsteps, sizex, sizey)).to(device)
    params = torch.tensor([sizex, sizey, nsteps, snap], dtype=torch.int32, device=device)
    out = torch.empty((16, sizex, sizey), device=device, dtype=torch.float32)
    lib.sunonsurface_sweep(g, g2, lwall, dxy_t, params, float(albedo_b), out, threads=sizex * sizey)
    return out


def svf_accumulate(acc, shmat, vegshmat, vbshvegshmat, sh, vegsh, vbshvegsh, weights, azimuth, index):
    """Add one sky patch to the 15 SVF sums and store its shadows in the patch matrices.

    ``weights`` is a float32 array (nk, 2) with the annulus weights for the
    full and the half ring, as annulus_weight() returns them on this device.
    """
    lib = _library()
    rows, cols = sh.shape
    azimuth = float(azimuth)
    dirs = ((1 if 0 <= azimuth < 180 else 0) | (2 if 90 <= azimuth < 270 else 0)
            | (4 if 180 <= azimuth < 360 else 0) | (8 if (azimuth >= 270 or azimuth < 90) else 0))
    params = torch.tensor([rows * cols, weights.shape[0], dirs, index, shmat.shape[2]],
                          dtype=torch.int32, device=sh.device)
    w = torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float32)).to(sh.device)
    lib.svf_accumulate(sh, vegsh, vbshvegsh, acc, shmat, vegshmat, vbshvegshmat, w, params,
                       threads=rows * cols)
