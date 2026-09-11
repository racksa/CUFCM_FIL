# `rbf_2d_precomputed` filament table — export format & reconstruction spec

This document specifies the binary file exported by
`reconstruct_precomputed.py --export-binary` (e.g. `knut1/theta_table.bin`),
and the reconstruction algorithm needed to consume it — enough to port to
C++/CUDA with no dependency on the Python fitting pipeline.

## What this table is

`beat_fit.py` (with `theta_fit_basis == 'rbf_2d_precomputed'`) fits a
non-separable 2D radial-basis-function model of the filament's tangent
angle `theta(s, psi)` — `s` = normalized arc length in `[0,1]`, `psi` =
beat phase in `[0, 2*pi)` — then evaluates that model once, offline, on a
dense `200 x 201` `(s, psi)` grid. This file is that grid: a fixed-size,
portable lookup table, independent of the original training-set size.
Reconstructing the filament from it needs no kernel evaluation, no
knowledge of the original digitized data, and no Python — just table
lookup, bilinear interpolation, and a numerical integration.

Position is reconstructed by integrating the unit tangent vector
`(cos(theta), sin(theta))` along `s`, starting from the origin: `s=0` is
always `(x,y) = (0,0)` with `theta(0, psi) = 0` for every `psi` (a
structural property of the fit — see `theta_basis_comparison.tex`). The
filament is unit length by construction. Mapping this into your
simulation's own lab frame (scale, rotation, translation) is a separate
step, outside the scope of this table (see `calibration.py`'s
`FIL_LEN`/`R`/`BASE` for how the Python pipeline does it, if useful as a
reference).

## Binary file layout

All values little-endian. Produced by `export_binary()` in
`reconstruct_precomputed.py`.

| Offset (bytes) | Type            | Field       | Notes |
|----------------|-----------------|-------------|-------|
| 0              | `char[8]`       | `magic`     | `"RBF2DPC\0"` |
| 8              | `int32`         | `version`   | `1` |
| 12             | `int32`         | `n_s`       | number of `s` grid points (200) |
| 16             | `int32`         | `n_psi`     | number of `psi` grid points, **including** the wraparound duplicate at `psi=2*pi` (201) |
| 20             | `int32`         | `reserved`  | `0`; pads header to 24 bytes (8-byte aligned) |
| 24             | `float64[n_s]`         | `s_grid`          | arc length, ascending, `s_grid[0]=0`, `s_grid[n_s-1]=1` |
| 24+8·n_s       | `float64[n_psi]`       | `psi_grid_padded` | phase, ascending, `psi_grid_padded[0]=0`, `psi_grid_padded[n_psi-1]=2*pi` |
| 24+8·(n_s+n_psi) | `float64[n_s*n_psi]` | `theta_grid`      | **row-major**: `theta_grid[i*n_psi + j]` = theta at `(s_grid[i], psi_grid_padded[j])` |

Total file size = `24 + 8*(n_s + n_psi + n_s*n_psi)` bytes (324,832 bytes /
~317 KiB for the current `knut1` export, `n_s=200`, `n_psi=201`).

The wraparound duplicate (`psi_grid_padded[n_psi-1] = 2*pi`, with
`theta_grid[:, n_psi-1] == theta_grid[:, 0]`) means a plain, non-periodic
bracket search over `psi_grid_padded` is always correct for any
`psi mod 2*pi` — no special-casing needed at the phase boundary.

### Minimal C++ reader

```cpp
#include <cstdint>
#include <cstdio>
#include <vector>

struct ThetaTable {
    int32_t n_s, n_psi;
    std::vector<double> s_grid;          // n_s
    std::vector<double> psi_grid_padded; // n_psi
    std::vector<double> theta_grid;      // n_s * n_psi, row-major
};

ThetaTable load_theta_table(const char* path) {
    FILE* f = std::fopen(path, "rb");
    char magic[8];
    int32_t version, n_s, n_psi, reserved;
    std::fread(magic, 1, 8, f);
    std::fread(&version, 4, 1, f);
    std::fread(&n_s, 4, 1, f);
    std::fread(&n_psi, 4, 1, f);
    std::fread(&reserved, 4, 1, f);
    // caller should check: memcmp(magic, "RBF2DPC\0", 8) == 0 && version == 1

    ThetaTable t;
    t.n_s = n_s; t.n_psi = n_psi;
    t.s_grid.resize(n_s);
    t.psi_grid_padded.resize(n_psi);
    t.theta_grid.resize((size_t)n_s * n_psi);
    std::fread(t.s_grid.data(), 8, n_s, f);
    std::fread(t.psi_grid_padded.data(), 8, n_psi, f);
    std::fread(t.theta_grid.data(), 8, (size_t)n_s * n_psi, f);
    std::fclose(f);
    return t;
}
```

## Reconstruction algorithm

Two functions, matching `reconstruct_precomputed.py`'s `theta_of_s` and
`evaluate_shape` exactly (same math, restructured for a single-point C++
call instead of Python's vectorized-over-the-whole-grid style — bilinear
interpolation is order-independent, so interpolating s-then-psi here is
mathematically identical to the Python reference's psi-then-s).

### `theta_of_s(s, psi)` — O(1), 4-corner bilinear lookup

```cpp
#include <algorithm>
#include <cmath>

double theta_of_s(double s_query, double psi, const ThetaTable& t) {
    // Wrap psi into [0, 2*pi)
    double psi_w = std::fmod(psi, 2.0 * M_PI);
    if (psi_w < 0.0) psi_w += 2.0 * M_PI;

    // Bracket psi: find j such that psi_grid_padded[j] <= psi_w <= psi_grid_padded[j+1]
    int j = (int)(std::upper_bound(t.psi_grid_padded.begin(), t.psi_grid_padded.end(), psi_w)
                   - t.psi_grid_padded.begin()) - 1;
    j = std::clamp(j, 0, t.n_psi - 2);
    double t_psi = (psi_w - t.psi_grid_padded[j]) / (t.psi_grid_padded[j+1] - t.psi_grid_padded[j]);

    // Bracket s: find i such that s_grid[i] <= s_query <= s_grid[i+1]
    int i = (int)(std::upper_bound(t.s_grid.begin(), t.s_grid.end(), s_query)
                   - t.s_grid.begin()) - 1;
    i = std::clamp(i, 0, t.n_s - 2);
    double t_s = (s_query - t.s_grid[i]) / (t.s_grid[i+1] - t.s_grid[i]);

    // 4-corner bilinear interpolation
    double v00 = t.theta_grid[(size_t)i * t.n_psi + j];
    double v10 = t.theta_grid[(size_t)(i+1) * t.n_psi + j];
    double v01 = t.theta_grid[(size_t)i * t.n_psi + (j+1)];
    double v11 = t.theta_grid[(size_t)(i+1) * t.n_psi + (j+1)];

    double v0 = v00 * (1.0 - t_s) + v10 * t_s;   // interpolate in s at psi_lo
    double v1 = v01 * (1.0 - t_s) + v11 * t_s;   // interpolate in s at psi_hi
    return v0 * (1.0 - t_psi) + v1 * t_psi;      // interpolate in psi
}
```

### `evaluate_shape(s_query[], n_query, psi)` — position via integration

```cpp
void evaluate_shape(const double* s_query, int n_query, double psi,
                     const ThetaTable& t, int n_fine,
                     double* x_out, double* y_out) {
    double s_max = 1e-12;
    for (int k = 0; k < n_query; ++k) s_max = std::max(s_max, s_query[k]);

    std::vector<double> s_fine(n_fine), x_fine(n_fine), y_fine(n_fine);
    for (int k = 0; k < n_fine; ++k)
        s_fine[k] = s_max * k / (n_fine - 1);

    x_fine[0] = 0.0; y_fine[0] = 0.0;
    double theta_prev = theta_of_s(s_fine[0], psi, t);
    for (int k = 1; k < n_fine; ++k) {
        double theta_k = theta_of_s(s_fine[k], psi, t);
        double ds = s_fine[k] - s_fine[k-1];
        x_fine[k] = x_fine[k-1] + 0.5 * (std::cos(theta_prev) + std::cos(theta_k)) * ds;
        y_fine[k] = y_fine[k-1] + 0.5 * (std::sin(theta_prev) + std::sin(theta_k)) * ds;
        theta_prev = theta_k;
    }

    // Linear interpolation of (x_fine, y_fine) back onto the caller's s_query[]
    for (int q = 0; q < n_query; ++q) {
        int k = (int)(std::upper_bound(s_fine.begin(), s_fine.end(), s_query[q])
                       - s_fine.begin()) - 1;
        k = std::clamp(k, 0, n_fine - 2);
        double u = (s_query[q] - s_fine[k]) / (s_fine[k+1] - s_fine[k]);
        x_out[q] = x_fine[k] * (1.0 - u) + x_fine[k+1] * u;
        y_out[q] = y_fine[k] * (1.0 - u) + y_fine[k+1] * u;
    }
}
```

`n_fine = 500` matches `beat_fit.py`'s/`reconstruct_precomputed.py`'s own
default and is what the reference values below were generated with.

## CUDA notes

- **`theta_of_s`'s bilinear lookup is exactly what CUDA texture memory does
  in hardware.** Bind `theta_grid` as a `cudaArray`/`cudaTextureObject_t`
  with `filterMode = cudaFilterModeLinear`, and a single `tex2D()` fetch
  replaces the whole bracket-search-plus-lerp block above — likely faster
  than the hand-rolled version and far less kernel code. (`psi`'s
  wraparound still needs handling before the fetch, e.g. via
  `cudaAddressModeWrap` with the grid normalized to `[0,1)`, or by
  wrapping `psi` manually as above and relying on the padded duplicate
  column.)
- **The integration in `evaluate_shape` is the remaining non-`O(1)` cost**
  (`n_fine` texture fetches + trapezoid accumulation per reconstruction).
  If this is called every timestep for thousands of filaments, consider
  precomputing `x_grid`/`y_grid` alongside `theta_grid` in Python (same
  `(s, psi)` grid, integrated once per training `psi` column offline)
  and exporting a second/third table — that would make *position*
  reconstruction `O(1)` too, with no integration at runtime at all. Not
  implemented here since it wasn't asked for; flagged as the natural next
  step if the integration cost turns out to matter at your simulation's
  scale.

## Validation

Reference values below, generated from `knut1/theta_table.bin` via
`reconstruct_precomputed.py` (`n_fine=500`). Use these to check a new
port for silent indexing/wraparound bugs before trusting it:

| s | psi | theta(s,psi) | x | y |
|---|-----|---------------|---|---|
| 0.0 | 0.0                | 0.0                 | 0.0                | 0.0 |
| 0.5 | 0.0                | -0.3212432764       | 0.4924102551       | -0.0652575295 |
| 1.0 | 0.0                | -0.2835846464       | 0.9544603020       | -0.2545014456 |
| 0.0 | 1.5707963268 (π/2) | 0.0                 | 0.0                | 0.0 |
| 0.5 | 1.5707963268 (π/2) | 0.5559205037        | 0.3925792804       | 0.3046620222 |
| 1.0 | 1.5707963268 (π/2) | 0.0791411643        | 0.8578588552       | 0.4669299291 |
| 0.5 | 3.1415926536 (π)   | 1.7548851777        | 0.2502277979       | 0.1549829432 |
| 0.5 | 4.7123889804 (3π/2)| -0.7389605759       | 0.1344404396       | -0.4618223602 |

(`s=0` always gives `theta=0`, `x=0`, `y=0` for every `psi` — the
structural base-point guarantee described above; a quick sanity check
that doesn't even require the table's actual fitted content.)

## Regenerating / re-exporting

If `beat_fit.py` is re-run on `knut1` (or any dataset) with
`theta_fit_basis == 'rbf_2d_precomputed'`, re-export with:

```bash
python3 reconstruct_precomputed.py <dataset_dir> --export-binary [--output PATH]
```

This always reflects whatever is currently in that dataset's
`coefficients.json` — re-run it after any re-fit.
