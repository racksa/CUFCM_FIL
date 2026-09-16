#pragma once
// MLS precomputed table lookup and shape reconstruction.
// Header-only, CPU-only, C++11-compatible.
// Compile benchmark driver with -DMLS_BENCHMARK_ENABLED.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MLS_N_FINE
#define MLS_N_FINE 500
#endif

#ifndef MLS_TABLE_PATH
#define MLS_TABLE_PATH "input/mls_table.bin"
#endif

struct MlsTable {
    int32_t n_s  = 0;
    int32_t n_psi = 0;
    std::vector<double> s_grid;           // [n_s]
    std::vector<double> psi_grid_padded;  // [n_psi], includes 2*pi wraparound duplicate
    std::vector<double> theta_grid;       // [n_s * n_psi], row-major: [i*n_psi + j]
};

inline MlsTable load_mls_table(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f)
        throw std::runtime_error(std::string("load_mls_table: cannot open ") + path);

    char magic[8];
    int32_t version, n_s, n_psi, reserved;
    std::fread(magic, 1, 8, f);
    std::fread(&version,  4, 1, f);
    std::fread(&n_s,      4, 1, f);
    std::fread(&n_psi,    4, 1, f);
    std::fread(&reserved, 4, 1, f);

    if (std::memcmp(magic, "MLSPC\0\0\0", 8) != 0 || version != 1) {
        std::fclose(f);
        throw std::runtime_error("load_mls_table: bad magic or version");
    }

    MlsTable t;
    t.n_s = n_s; t.n_psi = n_psi;
    t.s_grid.resize(n_s);
    t.psi_grid_padded.resize(n_psi);
    t.theta_grid.resize((size_t)n_s * n_psi);
    std::fread(t.s_grid.data(),          8, n_s,               f);
    std::fread(t.psi_grid_padded.data(), 8, n_psi,             f);
    std::fread(t.theta_grid.data(),      8, (size_t)n_s*n_psi, f);
    std::fclose(f);
    return t;
}

// O(log n) bilinear lookup of theta(s, psi).
inline double theta_of_s(double s_query, double psi, const MlsTable& t) {
    const double TWO_PI = 6.283185307179586;
    double psi_w = std::fmod(psi, TWO_PI);
    if (psi_w < 0.0) psi_w += TWO_PI;

    int j = (int)(std::upper_bound(t.psi_grid_padded.begin(), t.psi_grid_padded.end(), psi_w)
                  - t.psi_grid_padded.begin()) - 1;
    j = std::max(0, std::min(j, t.n_psi - 2));
    double t_psi = (psi_w - t.psi_grid_padded[j]) / (t.psi_grid_padded[j+1] - t.psi_grid_padded[j]);

    int i = (int)(std::upper_bound(t.s_grid.begin(), t.s_grid.end(), s_query)
                  - t.s_grid.begin()) - 1;
    i = std::max(0, std::min(i, t.n_s - 2));
    double t_s = (s_query - t.s_grid[i]) / (t.s_grid[i+1] - t.s_grid[i]);

    double v00 = t.theta_grid[(size_t) i    * t.n_psi +  j   ];
    double v10 = t.theta_grid[(size_t)(i+1) * t.n_psi +  j   ];
    double v01 = t.theta_grid[(size_t) i    * t.n_psi + (j+1)];
    double v11 = t.theta_grid[(size_t)(i+1) * t.n_psi + (j+1)];

    return (v00*(1.0-t_s) + v10*t_s)*(1.0-t_psi)
         + (v01*(1.0-t_s) + v11*t_s)*(    t_psi );
}

// Exact bilinear-interpolated d(theta)/d(psi).
inline double theta_dpsi_of_s(double s_query, double psi, const MlsTable& t) {
    const double TWO_PI = 6.283185307179586;
    double psi_w = std::fmod(psi, TWO_PI);
    if (psi_w < 0.0) psi_w += TWO_PI;

    int j = (int)(std::upper_bound(t.psi_grid_padded.begin(), t.psi_grid_padded.end(), psi_w)
                  - t.psi_grid_padded.begin()) - 1;
    j = std::max(0, std::min(j, t.n_psi - 2));
    double dpsi = t.psi_grid_padded[j+1] - t.psi_grid_padded[j];

    int i = (int)(std::upper_bound(t.s_grid.begin(), t.s_grid.end(), s_query)
                  - t.s_grid.begin()) - 1;
    i = std::max(0, std::min(i, t.n_s - 2));
    double t_s = (s_query - t.s_grid[i]) / (t.s_grid[i+1] - t.s_grid[i]);

    double v00 = t.theta_grid[(size_t) i    * t.n_psi +  j   ];
    double v10 = t.theta_grid[(size_t)(i+1) * t.n_psi +  j   ];
    double v01 = t.theta_grid[(size_t) i    * t.n_psi + (j+1)];
    double v11 = t.theta_grid[(size_t)(i+1) * t.n_psi + (j+1)];

    // Exact partial derivative of the bilinear patch in psi
    return ((1.0 - t_s)*(v01 - v00) + t_s*(v11 - v10)) / dpsi;
}

// Position reconstruction: x_out[q] = int_0^{s_query[q]} cos(theta) ds', etc.
// Uses trapezoid rule on a fine grid of n_fine points up to max(s_query).
inline void evaluate_shape(const double* s_query, int n_query, double psi,
                            const MlsTable& t, int n_fine,
                            double* x_out, double* y_out) {
    double s_max = 1e-12;
    for (int k = 0; k < n_query; ++k)
        if (s_query[k] > s_max) s_max = s_query[k];

    std::vector<double> s_fine(n_fine), x_fine(n_fine), y_fine(n_fine);
    for (int k = 0; k < n_fine; ++k)
        s_fine[k] = s_max * k / (n_fine - 1);

    x_fine[0] = 0.0; y_fine[0] = 0.0;
    double theta_prev = theta_of_s(s_fine[0], psi, t);
    for (int k = 1; k < n_fine; ++k) {
        double theta_k = theta_of_s(s_fine[k], psi, t);
        double ds = s_fine[k] - s_fine[k-1];
        x_fine[k] = x_fine[k-1] + 0.5*(std::cos(theta_prev) + std::cos(theta_k))*ds;
        y_fine[k] = y_fine[k-1] + 0.5*(std::sin(theta_prev) + std::sin(theta_k))*ds;
        theta_prev = theta_k;
    }

    for (int q = 0; q < n_query; ++q) {
        int k = (int)(std::upper_bound(s_fine.begin(), s_fine.end(), s_query[q])
                      - s_fine.begin()) - 1;
        k = std::max(0, std::min(k, n_fine - 2));
        double u = (s_query[q] - s_fine[k]) / (s_fine[k+1] - s_fine[k]);
        x_out[q] = x_fine[k]*(1.0-u) + x_fine[k+1]*u;
        y_out[q] = y_fine[k]*(1.0-u) + y_fine[k+1]*u;
    }
}

// Velocity direction: d(x,y)/d(psi).
// Integrates (-sin(theta)*dtheta/dpsi, cos(theta)*dtheta/dpsi) over s.
inline void evaluate_shape_vel_dir(const double* s_query, int n_query, double psi,
                                    const MlsTable& t, int n_fine,
                                    double* dx_out, double* dy_out) {
    double s_max = 1e-12;
    for (int k = 0; k < n_query; ++k)
        if (s_query[k] > s_max) s_max = s_query[k];

    std::vector<double> s_fine(n_fine), dx_fine(n_fine), dy_fine(n_fine);
    for (int k = 0; k < n_fine; ++k)
        s_fine[k] = s_max * k / (n_fine - 1);

    dx_fine[0] = 0.0; dy_fine[0] = 0.0;
    double theta_prev = theta_of_s(    s_fine[0], psi, t);
    double dtdp_prev  = theta_dpsi_of_s(s_fine[0], psi, t);
    for (int k = 1; k < n_fine; ++k) {
        double theta_k = theta_of_s(    s_fine[k], psi, t);
        double dtdp_k  = theta_dpsi_of_s(s_fine[k], psi, t);
        double ds = s_fine[k] - s_fine[k-1];
        double fx0 = -std::sin(theta_prev)*dtdp_prev,  fx1 = -std::sin(theta_k)*dtdp_k;
        double fy0 =  std::cos(theta_prev)*dtdp_prev,  fy1 =  std::cos(theta_k)*dtdp_k;
        dx_fine[k] = dx_fine[k-1] + 0.5*(fx0 + fx1)*ds;
        dy_fine[k] = dy_fine[k-1] + 0.5*(fy0 + fy1)*ds;
        theta_prev = theta_k;
        dtdp_prev  = dtdp_k;
    }

    for (int q = 0; q < n_query; ++q) {
        int k = (int)(std::upper_bound(s_fine.begin(), s_fine.end(), s_query[q])
                      - s_fine.begin()) - 1;
        k = std::max(0, std::min(k, n_fine - 2));
        double u = (s_query[q] - s_fine[k]) / (s_fine[k+1] - s_fine[k]);
        dx_out[q] = dx_fine[k]*(1.0-u) + dx_fine[k+1]*u;
        dy_out[q] = dy_fine[k]*(1.0-u) + dy_fine[k+1]*u;
    }
}

// =============================================================================
// Benchmark (compiled only when -DMLS_BENCHMARK_ENABLED)
// =============================================================================
#ifdef MLS_BENCHMARK_ENABLED

#include <chrono>

struct MlsBenchResult {
    double theta_lookup_ns;    // ns per theta_of_s call
    double evaluate_shape_us;  // us per evaluate_shape call (n_query = nseg)
    double evaluate_veldir_us; // us per evaluate_shape_vel_dir call
    double full_update_us;     // us per combined shape + vel_dir call
    int nseg, n_fine, n_reps;
};

inline MlsBenchResult mls_benchmark(const MlsTable& t,
                                     int nseg   = 20,
                                     int n_fine = MLS_N_FINE,
                                     int n_reps = 1000) {
    using clock = std::chrono::high_resolution_clock;
    using ns_t  = std::chrono::nanoseconds;

    std::vector<double> sq(nseg), xo(nseg), yo(nseg), dxo(nseg), dyo(nseg);
    for (int n = 0; n < nseg; ++n) sq[n] = double(n) / double(nseg - 1);

    MlsBenchResult r = {};
    r.nseg = nseg; r.n_fine = n_fine; r.n_reps = n_reps;

    // theta_of_s: 100 * n_reps calls
    {
        volatile double sink = 0.0;
        auto t0 = clock::now();
        for (int i = 0; i < n_reps * 100; ++i)
            sink += theta_of_s(sq[i % nseg], 1.0 + i*0.0001, t);
        auto t1 = clock::now();
        (void)sink;
        r.theta_lookup_ns = (double)std::chrono::duration_cast<ns_t>(t1-t0).count()
                            / (n_reps * 100);
    }

    // evaluate_shape
    {
        volatile double sink = 0.0;
        auto t0 = clock::now();
        for (int i = 0; i < n_reps; ++i) {
            evaluate_shape(sq.data(), nseg, 1.0 + i*0.001, t, n_fine, xo.data(), yo.data());
            sink += xo[0];
        }
        auto t1 = clock::now();
        (void)sink;
        r.evaluate_shape_us = (double)std::chrono::duration_cast<ns_t>(t1-t0).count()
                              / n_reps / 1000.0;
    }

    // evaluate_shape_vel_dir
    {
        volatile double sink = 0.0;
        auto t0 = clock::now();
        for (int i = 0; i < n_reps; ++i) {
            evaluate_shape_vel_dir(sq.data(), nseg, 1.0 + i*0.001, t, n_fine, dxo.data(), dyo.data());
            sink += dxo[0];
        }
        auto t1 = clock::now();
        (void)sink;
        r.evaluate_veldir_us = (double)std::chrono::duration_cast<ns_t>(t1-t0).count()
                               / n_reps / 1000.0;
    }

    // full update (shape + vel_dir together)
    {
        volatile double sink = 0.0;
        auto t0 = clock::now();
        for (int i = 0; i < n_reps; ++i) {
            evaluate_shape(sq.data(), nseg, 1.0 + i*0.001, t, n_fine, xo.data(), yo.data());
            evaluate_shape_vel_dir(sq.data(), nseg, 1.0 + i*0.001, t, n_fine, dxo.data(), dyo.data());
            sink += xo[0] + dxo[0];
        }
        auto t1 = clock::now();
        (void)sink;
        r.full_update_us = (double)std::chrono::duration_cast<ns_t>(t1-t0).count()
                          / n_reps / 1000.0;
    }

    return r;
}

inline void mls_benchmark_print(const MlsBenchResult& r) {
    std::printf("=== MLS precomputed benchmark ===\n");
    std::printf("  nseg=%d  n_fine=%d  n_reps=%d\n", r.nseg, r.n_fine, r.n_reps);
    std::printf("  theta_of_s:            %6.1f ns/call\n",  r.theta_lookup_ns);
    std::printf("  evaluate_shape:        %6.1f us/call  (%4.1f ns/seg)\n",
                r.evaluate_shape_us,  r.evaluate_shape_us  * 1000.0 / r.nseg);
    std::printf("  evaluate_shape_veldir: %6.1f us/call  (%4.1f ns/seg)\n",
                r.evaluate_veldir_us, r.evaluate_veldir_us * 1000.0 / r.nseg);
    std::printf("  full update:           %6.1f us/call\n",  r.full_update_us);
}

#endif // MLS_BENCHMARK_ENABLED
