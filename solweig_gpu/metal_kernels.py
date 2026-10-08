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

// Hourly sky patch loops. Each thread takes one pixel and runs through the
// patches in order, so every sum gets its terms in the same order as the
// PyTorch loop. A term like scalar * mask * scalar * mask equals mask * P,
// where P is the product of the scalars computed by PyTorch with the masks
// set to one: with 0/1 masks and finite P this gives the same bits. A sum
// that starts at +0 can never become -0, so terms with a zero mask can be
// skipped.
//
// The three shadow matrices hold only 0 and 1, so they are read as one byte
// per pixel and patch: bit 0 shmat, bit 1 vegshmat, bit 2 vbshvegshmat.
// diffsh is a function of shmat and vegshmat and comes from a 4 entry table
// indexed by bits 0 and 1.

// Sunlit and shaded test of shaded_or_sunlit for one pixel and patch.
inline void sun_or_shade(float hsvf, float yi, float palt, float r2d, thread float& sp, thread float& shp) {
    const float deg = atan(hsvf + yi) * r2d;
    sp = (deg < palt) ? 1.0f : 0.0f;
    shp = (deg > palt) ? 1.0f : 0.0f;
}

// Pack the three shadow matrices into codes; flags a value other than 0 or 1.
// Grid stride loop: n can pass 2^32 on big tiles.
kernel void pack_patch_codes(device const float* shmat [[buffer(0)]],
                             device const float* vegmat [[buffer(1)]],
                             device const float* vbmat [[buffer(2)]],
                             device uchar* codes [[buffer(3)]],
                             device atomic_int* bad [[buffer(4)]],
                             constant long* n [[buffer(5)]],
                             uint idx [[thread_position_in_grid]],
                             uint nthreads [[threads_per_grid]])
{
    for (long i = idx; i < n[0]; i += nthreads) {
        const float s = shmat[i], v = vegmat[i], b = vbmat[i];
        if ((s != 0.0f && s != 1.0f) || (v != 0.0f && v != 1.0f) || (b != 0.0f && b != 1.0f))
            atomic_store_explicit(bad, 1, memory_order_relaxed);
        codes[i] = (uchar)((s == 1.0f ? 1 : 0) | (v == 1.0f ? 2 : 0) | (b == 1.0f ? 4 : 0));
    }
}

// diffsh as a 4 entry table: any entry fills its slot, then every entry is checked.
kernel void diffsh_lut_fill(device const float* diffsh [[buffer(0)]],
                            device const uchar* codes [[buffer(1)]],
                            device atomic_uint* lut [[buffer(2)]],
                            constant long* n [[buffer(3)]],
                            uint idx [[thread_position_in_grid]],
                            uint nthreads [[threads_per_grid]])
{
    for (long i = idx; i < n[0]; i += nthreads)
        atomic_store_explicit(&lut[codes[i] & 3], as_type<uint>(diffsh[i]), memory_order_relaxed);
}

kernel void diffsh_lut_check(device const float* diffsh [[buffer(0)]],
                             device const uchar* codes [[buffer(1)]],
                             device const uint* lut [[buffer(2)]],
                             device atomic_int* bad [[buffer(3)]],
                             constant long* n [[buffer(4)]],
                             uint idx [[thread_position_in_grid]],
                             uint nthreads [[threads_per_grid]])
{
    for (long i = idx; i < n[0]; i += nthreads)
        if (as_type<uint>(diffsh[i]) != lut[codes[i] & 3])
            atomic_store_explicit(bad, 1, memory_order_relaxed);
}

// First patch loop of define_patch_characteristics.
// tab (npatch x 28): 0 Lsky_down, 1 Lsky_side, 2 veg side, 3 veg down,
// 4-7 sky E S W N, 8-11 veg E S W N, 12 sun side, 13 shade side, 14 sun down,
// 15 shade down, 16-19 sun E S W N, 20-23 shade E S W N, 24 yi, 25 patch altitude,
// 26 rad2deg. flags: bits 0-3 directions E S W N, bit 4 sunlit wall branch.
// acc planes: 0 Ldown_sky, 1 Lside_sky, 2 Lside_veg, 3 Ldown_veg, 4-7 Least
// Lsouth Lwest Lnorth, 8 Lside_sun, 9 Lside_sh, 10 Ldown_sun, 11 Ldown_sh,
// 12 Lside_ref, 13 Ldown_ref.
kernel void lw_patches(device const uchar* codes [[buffer(0)]],
                       device const float* hsvf [[buffer(1)]],
                       constant float* tab [[buffer(2)]],
                       constant int* flags [[buffer(3)]],
                       constant int* params [[buffer(4)]],
                       device float* acc [[buffer(5)]],
                       uint idx [[thread_position_in_grid]])
{
    const int npix = params[0];
    const int np = params[1];
    if ((int)idx >= npix) return;
    float a[12];
    for (int j = 0; j < 12; ++j) a[j] = acc[j * npix + idx];
    const float h = hsvf[idx];
    const ulong base = (ulong)idx * (ulong)np;
    for (int i = 0; i < np; ++i) {
        constant float* T = tab + i * 28;
        const int fl = flags[i];
        const int c = codes[base + i];
        const bool s = c & 1, v = c & 2, b = c & 4;
        if (s && v) {                       // temp_sky
            a[0] = a[0] + T[0];
            a[1] = a[1] + T[1];
            for (int d = 0; d < 4; ++d)
                if (fl & (1 << d)) a[4 + d] = a[4 + d] + T[4 + d];
        }
        if (!v || !b) {                     // temp_vegsh
            a[2] = a[2] + T[2];
            a[3] = a[3] + T[3];
            for (int d = 0; d < 4; ++d)
                if (fl & (1 << d)) a[4 + d] = a[4 + d] + T[8 + d];
        }
        if (!s && b) {                      // temp_sh
            if (fl & 16) {
                float sp, shp;
                sun_or_shade(h, T[24], T[25], T[26], sp, shp);
                if (sp != 0.0f) {
                    a[8] = a[8] + T[12];
                    a[10] = a[10] + T[14];
                }
                if (shp != 0.0f) {
                    a[9] = a[9] + T[13];
                    a[11] = a[11] + T[15];
                }
                for (int d = 0; d < 4; ++d) {
                    if (fl & (1 << d)) {
                        if (sp != 0.0f) a[4 + d] = a[4 + d] + T[16 + d];
                        if (shp != 0.0f) a[4 + d] = a[4 + d] + T[20 + d];
                    }
                }
            } else {
                a[9] = a[9] + T[13];
                a[11] = a[11] + T[15];
                for (int d = 0; d < 4; ++d)
                    if (fl & (1 << d)) a[4 + d] = a[4 + d] + T[20 + d];
            }
        }
    }
    for (int j = 0; j < 12; ++j) acc[j * npix + idx] = a[j];
}

// Second patch loop of define_patch_characteristics (reflected longwave).
// tab (npatch x 8): steradian, cos(alt), sin(alt), cos to E S W N.
kernel void lw_reflected(device const uchar* codes [[buffer(0)]],
                         device const float* refl [[buffer(1)]],
                         constant float* tab [[buffer(2)]],
                         constant int* flags [[buffer(3)]],
                         constant int* params [[buffer(4)]],
                         device float* acc [[buffer(5)]],
                         uint idx [[thread_position_in_grid]])
{
    const int npix = params[0];
    const int np = params[1];
    if ((int)idx >= npix) return;
    float dirs[4];
    for (int d = 0; d < 4; ++d) dirs[d] = acc[(4 + d) * npix + idx];
    float side = acc[12 * npix + idx];
    float down = acc[13 * npix + idx];
    const float r = refl[idx];
    const ulong base = (ulong)idx * (ulong)np;
    for (int i = 0; i < np; ++i) {
        // no skipping here: the reflected flux is per pixel and may be NaN
        const float m = ((codes[base + i] & 7) != 7) ? 1.0f : 0.0f;
        constant float* T = tab + i * 8;
        const int fl = flags[i];
        const float rs = r * T[0];
        const float rsc = (rs * T[1]) * m;
        side = side + rsc;
        down = down + (rs * T[2]) * m;
        for (int d = 0; d < 4; ++d)
            if (fl & (1 << d)) dirs[d] = dirs[d] + rsc * T[3 + d];
    }
    for (int d = 0; d < 4; ++d) acc[(4 + d) * npix + idx] = dirs[d];
    acc[12 * npix + idx] = side;
    acc[13 * npix + idx] = down;
}

// Anisotropic patch loop of Kside_veg_v2022a, cylinder (params[2] = 1) or box.
// tab (npatch x 32): 0 lumChi, 1 steradian, 2 anglIncC, 3-6 anglInc E S W N,
// 7 veg, 8 sun, 9 shade, 10 shade (no sun branch), 11-14 veg E S W N,
// 15-18 sun E S W N, 19-22 shade E S W N, 23-26 shade E S W N (no sun branch),
// 27 yi, 28 patch altitude, 29 rad2deg.
// flags: bits 0-3 diffuse directions, bits 4-7 reflected directions, bit 8 sun branch.
// acc planes: 0 KsideD, 1 Kref_veg, 2 Kref_sun, 3 Kref_sh, 4-7 diffRad E S W N,
// 8-11 Kref_veg, 12-15 Kref_sun, 16-19 Kref_sh (E S W N).
kernel void kside_patches(device const uchar* codes [[buffer(0)]],
                          constant float* lut [[buffer(1)]],
                          device const float* hsvf [[buffer(2)]],
                          constant float* tab [[buffer(3)]],
                          constant int* flags [[buffer(4)]],
                          constant int* params [[buffer(5)]],
                          device float* acc [[buffer(6)]],
                          uint idx [[thread_position_in_grid]])
{
    const int npix = params[0];
    const int np = params[1];
    const int cyl = params[2];
    if ((int)idx >= npix) return;
    const float h = hsvf[idx];
    const ulong base = (ulong)idx * (ulong)np;
    if (cyl) {
        float sd = acc[idx], veg = acc[npix + idx], sun = acc[2 * npix + idx], sh = acc[3 * npix + idx];
        for (int i = 0; i < np; ++i) {
            constant float* T = tab + i * 32;
            const int c = codes[base + i];
            sd = sd + ((lut[c & 3] * T[0]) * T[2]) * T[1];
            if (!(c & 2) || !(c & 4)) veg = veg + T[7];
            if (!(c & 1) && (c & 4)) {
                float sp, shp;
                sun_or_shade(h, T[27], T[28], T[29], sp, shp);
                if (sp != 0.0f) sun = sun + T[8];
                if (shp != 0.0f) sh = sh + T[9];
            }
        }
        acc[idx] = sd; acc[npix + idx] = veg; acc[2 * npix + idx] = sun; acc[3 * npix + idx] = sh;
        return;
    }
    float a[20];
    for (int j = 0; j < 20; ++j) a[j] = acc[j * npix + idx];
    for (int i = 0; i < np; ++i) {
        constant float* T = tab + i * 32;
        const int fl = flags[i];
        const int c = codes[base + i];
        const float D = lut[c & 3];
        for (int d = 0; d < 4; ++d)
            if (fl & (1 << d)) a[4 + d] = a[4 + d] + ((D * T[0]) * T[3 + d]) * T[1];
        if (!(c & 2) || !(c & 4)) {
            a[1] = a[1] + T[7];
            for (int d = 0; d < 4; ++d)
                if (fl & (16 << d)) a[8 + d] = a[8 + d] + T[11 + d];
        }
        if (!(c & 1) && (c & 4)) {
            if (fl & 256) {
                float sp, shp;
                sun_or_shade(h, T[27], T[28], T[29], sp, shp);
                if (sp != 0.0f) a[2] = a[2] + T[8];
                if (shp != 0.0f) a[3] = a[3] + T[9];
                for (int d = 0; d < 4; ++d) {
                    if (fl & (16 << d)) {
                        if (sp != 0.0f) a[12 + d] = a[12 + d] + T[15 + d];
                        if (shp != 0.0f) a[16 + d] = a[16 + d] + T[19 + d];
                    }
                }
            } else {
                a[3] = a[3] + T[10];
                for (int d = 0; d < 4; ++d)
                    if (fl & (16 << d)) a[16 + d] = a[16 + d] + T[23 + d];
            }
        }
    }
    for (int j = 0; j < 20; ++j) acc[j * npix + idx] = a[j];
}

// out += sum over patches of diffsh * w[i], in patch order (aniLum).
kernel void patch_weighted_sum(device const uchar* codes [[buffer(0)]],
                               constant float* lut [[buffer(1)]],
                               constant float* w [[buffer(2)]],
                               constant int* params [[buffer(3)]],
                               device float* out [[buffer(4)]],
                               uint idx [[thread_position_in_grid]])
{
    const int npix = params[0];
    const int np = params[1];
    if ((int)idx >= npix) return;
    float a = out[idx];
    const ulong base = (ulong)idx * (ulong)np;
    for (int i = 0; i < np; ++i) a = a + lut[codes[base + i] & 3] * w[i];
    out[idx] = a;
}

// Tail of sunonsurface_2018a plus the sums of gvf_2018a, for one azimuth.
// w: the 16 planes of sunonsurface_sweep. keep, lupg (the ground Lup term),
// buildings, alb, shadow: per pixel. sc: first + 1, first, second + 1, second.
// acc planes: gvfLup, gvfalb, gvfalbnosh, gvfSum, gvfLup E S W N,
// gvfalb E S W N, gvfalbnosh E S W N. params: npix, direction bits.
kernel void gvf_tail(device const float* w [[buffer(0)]],
                     device const float* keep [[buffer(1)]],
                     device const float* lupg [[buffer(2)]],
                     device const float* buildings [[buffer(3)]],
                     device const float* alb [[buffer(4)]],
                     device const float* shadow [[buffer(5)]],
                     constant float* sc [[buffer(6)]],
                     constant int* params [[buffer(7)]],
                     device float* acc [[buffer(8)]],
                     uint idx [[thread_position_in_grid]])
{
    const int npix = params[0];
    const int dirs = params[1];
    if ((int)idx >= npix) return;
    const float F1 = sc[0], F = sc[1], S1 = sc[2], S = sc[3];
    float W[16];
    for (int j = 0; j < 16; ++j) W[j] = w[j * npix + idx];
    // W: 0 sh, 1 Lupsh, 2 albsh, 3 albnosh, 4 Lwall, 5 albwall, 6 wall, 7 albwallnosh, 8-15 the same at "first"
    const float m1 = (W[14] > 0.0f) ? 1.0f : 0.0f, n1 = (W[14] > 0.0f) ? 0.0f : 1.0f;
    const float m2 = (W[6] > 0.0f) ? 1.0f : 0.0f, n2 = (W[6] > 0.0f) ? 0.0f : 1.0f;
    const float i1 = (W[15] > 0.0f) ? 1.0f : 0.0f, j1 = (W[15] > 0.0f) ? 0.0f : 1.0f;
    const float i2 = (W[7] > 0.0f) ? 1.0f : 0.0f, j2 = (W[7] > 0.0f) ? 0.0f : 1.0f;
    const bool k = keep[idx] == 1.0f;
    const float wall2 = k ? 0.0f : W[6];
    const float lwall2 = k ? 0.0f : W[4];
    const float albwall2 = k ? 0.0f : W[5];

    float gvf2 = ((wall2 + W[0]) / S1) * m2 + (W[0] / S) * n2;
    if (gvf2 > 1.0f) gvf2 = 1.0f;
    const float gvfLup1 = ((W[12] + W[9]) / F1) * m1 + (W[9] / F) * n1;
    const float gvfLup2 = ((lwall2 + W[1]) / S1) * m2 + (W[1] / S) * n2;
    const float gvfalb1 = ((W[13] + W[10]) / F1) * m1 + (W[10] / F) * n1;
    const float gvfalb2 = ((albwall2 + W[2]) / S1) * m2 + (W[2] / S) * n2;
    const float gvfalbnosh1 = ((W[15] + W[11]) / F1) * i1 + (W[11] / F) * j1;
    const float gvfalbnosh2 = ((W[7] + W[3]) / S) * i2 + (W[3] / S) * j2;

    const float b = buildings[idx];
    const float bm = b * -1.0f + 1.0f;
    float gvfLup = (gvfLup1 * 0.5f + gvfLup2 * 0.4f) / 0.9f;
    gvfLup = gvfLup + lupg[idx] * bm;
    float gvfalb = (gvfalb1 * 0.5f + gvfalb2 * 0.4f) / 0.9f;
    gvfalb = gvfalb + (alb[idx] * bm) * shadow[idx];
    float gvfalbnosh = (gvfalbnosh1 * 0.5f + gvfalbnosh2 * 0.4f) / 0.9f;
    gvfalbnosh = gvfalbnosh * b + alb[idx] * bm;

    acc[idx] = acc[idx] + gvfLup;
    acc[npix + idx] = acc[npix + idx] + gvfalb;
    acc[2 * npix + idx] = acc[2 * npix + idx] + gvfalbnosh;
    acc[3 * npix + idx] = acc[3 * npix + idx] + gvf2;
    for (int d = 0; d < 4; ++d) {
        if (dirs & (1 << d)) {
            acc[(4 + d) * npix + idx] = acc[(4 + d) * npix + idx] + gvfLup;
            acc[(8 + d) * npix + idx] = acc[(8 + d) * npix + idx] + gvfalb;
            acc[(12 + d) * npix + idx] = acc[(12 + d) * npix + idx] + gvfalbnosh;
        }
    }
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


def _tables(device, tab, flags):
    if isinstance(tab, torch.Tensor):
        tab = tab.to(device=device, dtype=torch.float32).contiguous()
    else:
        tab = torch.as_tensor(np.ascontiguousarray(tab, dtype=np.float32)).to(device)
    flags = torch.as_tensor(np.ascontiguousarray(flags, dtype=np.int32)).to(device)
    return tab, flags


def patch_codes(shmat, vegshmat, vbshvegshmat):
    """One byte per pixel and patch holding the three 0/1 shadow matrices.

    Built once and kept on shmat. Returns None when the matrices are not
    (rows, cols, npatch) float32 on MPS or hold values other than 0 and 1.
    """
    mats = (shmat, vegshmat, vbshvegshmat)
    if not (metal_enabled(*mats) and all(m.dim() == 3 and m.is_contiguous() and m.shape == shmat.shape
                                         for m in mats)):
        return None
    key = tuple(id(m) for m in mats)
    cached = getattr(shmat, "_solweig_patch_codes", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    codes = torch.empty(shmat.shape, dtype=torch.uint8, device=shmat.device)
    bad = torch.zeros(1, dtype=torch.int32, device=shmat.device)
    n = shmat.numel()
    n_t = torch.tensor([n], dtype=torch.int64, device=shmat.device)
    _library().pack_patch_codes(shmat, vegshmat, vbshvegshmat, codes, bad, n_t, threads=min(n, 1 << 24))
    if int(bad.item()) != 0:
        codes = None
    shmat._solweig_patch_codes = (key, codes)
    return codes


def diffsh_table(diffsh, codes):
    """The 4 values of diffsh indexed by codes & 3, checked against every entry.

    Kept on diffsh. Returns None if diffsh is not a function of shmat and vegshmat.
    """
    if codes is None or not metal_enabled(diffsh) or diffsh.shape != codes.shape:
        return None
    cached = getattr(diffsh, "_solweig_lut", None)
    if cached is not None and cached[0] is codes:
        return cached[1]
    if not diffsh.is_contiguous():
        return None
    n = diffsh.numel()
    n_t = torch.tensor([n], dtype=torch.int64, device=diffsh.device)
    bits = torch.zeros(4, dtype=torch.int32, device=diffsh.device)
    bad = torch.zeros(1, dtype=torch.int32, device=diffsh.device)
    _library().diffsh_lut_fill(diffsh, codes, bits, n_t, threads=min(n, 1 << 24))
    _library().diffsh_lut_check(diffsh, codes, bits, bad, n_t, threads=min(n, 1 << 24))
    lut = bits.view(torch.float32) if int(bad.item()) == 0 else None
    diffsh._solweig_lut = (codes, lut)
    return lut


def _check_finite(tab):
    # skipping masked terms is only exact when every factor is finite
    return bool(torch.isfinite(tab).all())


def lw_patches(codes, hsvf, tab, flags, acc):
    """First patch loop of define_patch_characteristics; adds into acc (14, rows, cols)."""
    rows, cols, npatch = codes.shape
    tab, flags = _tables(codes.device, tab, flags)
    params = torch.tensor([rows * cols, npatch], dtype=torch.int32, device=codes.device)
    _library().lw_patches(codes, hsvf.contiguous(), tab, flags, params, acc, threads=rows * cols)


def lw_reflected(codes, refl, tab, flags, acc):
    """Reflected longwave loop of define_patch_characteristics; adds into acc."""
    rows, cols, npatch = codes.shape
    tab, flags = _tables(codes.device, tab, flags)
    params = torch.tensor([rows * cols, npatch], dtype=torch.int32, device=codes.device)
    _library().lw_reflected(codes, refl.contiguous(), tab, flags, params, acc, threads=rows * cols)


def kside_patches(codes, lut, hsvf, tab, flags, cyl, acc):
    """Anisotropic patch loop of Kside_veg_v2022a; adds into acc (20, rows, cols)."""
    rows, cols, npatch = codes.shape
    tab, flags = _tables(codes.device, tab, flags)
    params = torch.tensor([rows * cols, npatch, 1 if cyl else 0], dtype=torch.int32, device=codes.device)
    _library().kside_patches(codes, lut, hsvf.contiguous(), tab, flags, params, acc, threads=rows * cols)


def patch_weighted_sum(codes, lut, weights, out):
    """out += sum_i diffsh[:, :, i] * weights[i], added in patch order."""
    rows, cols, npatch = codes.shape
    w = weights.to(device=codes.device, dtype=torch.float32).contiguous()
    params = torch.tensor([rows * cols, npatch], dtype=torch.int32, device=codes.device)
    _library().patch_weighted_sum(codes, lut, w, params, out, threads=rows * cols)


def gvf_tail(w, keep, lupg, buildings, alb, shadow, first, second, dirs, acc):
    """Tail of sunonsurface_2018a for one azimuth, added into the 16 gvf_2018a sums."""
    _, rows, cols = w.shape
    shape = (rows, cols)
    planes = [torch.broadcast_to(t, shape).to(torch.float32).contiguous() for t in (keep, lupg, buildings, alb, shadow)]
    sc = torch.tensor([first + 1, first, second + 1, second], dtype=torch.float32, device=w.device)
    params = torch.tensor([rows * cols, dirs], dtype=torch.int32, device=w.device)
    _library().gvf_tail(w, *planes, sc, params, acc, threads=rows * cols)
