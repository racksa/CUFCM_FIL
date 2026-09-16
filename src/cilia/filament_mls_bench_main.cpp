// filament_mls_bench_main.cpp — standalone validation + benchmark for filament_mls.hpp
//
// Build (from repo root):
//   g++ -O2 -std=c++14 -DMLS_BENCHMARK_ENABLED \
//       -I. -Isrc/general -Isrc/flow_field -Isrc/cilia \
//       src/cilia/filament_mls_bench_main.cpp -o bin/mls_bench
//
// Run:
//   ./bin/mls_bench [table_path [nseg [n_fine [n_reps]]]]
// Defaults: table_path=input/mls_table.bin, nseg=20, n_fine=500, n_reps=1000

#define MLS_BENCHMARK_ENABLED
#include "filament_mls.hpp"

#include <cmath>
#include <cstdlib>

// Reference values from MLS_PRECOMPUTED_SPEC.md (knut1/mls_table.bin, n_fine=500)
struct ValCase { double s, psi, theta_ref, x_ref, y_ref; };

static const ValCase VAL_CASES[] = {
    {0.0, 0.0,                0.0,           0.0,            0.0          },
    {0.5, 0.0,               -0.3212432764,  0.4924102551,  -0.0652575295 },
    {1.0, 0.0,               -0.2835846464,  0.9544603020,  -0.2545014456 },
    {0.0, 1.5707963268,       0.0,           0.0,            0.0          },
    {0.5, 1.5707963268,       0.5559205037,  0.3925792804,   0.3046620222 },
    {1.0, 1.5707963268,       0.0791411643,  0.8578588552,   0.4669299291 },
    {0.5, 3.1415926536,       1.7548851777,  0.2502277979,   0.1549829432 },
    {0.5, 4.7123889804,      -0.7389605759,  0.1344404396,  -0.4618223602 },
};

static bool validate(const MlsTable& t, int n_fine,
                     double tol_theta = 1e-6, double tol_pos = 1e-5) {
    bool ok = true;
    for (size_t ci = 0; ci < sizeof(VAL_CASES)/sizeof(VAL_CASES[0]); ++ci) {
        const ValCase& c = VAL_CASES[ci];
        double th     = theta_of_s(c.s, c.psi, t);
        double xo, yo;
        evaluate_shape(&c.s, 1, c.psi, t, n_fine, &xo, &yo);

        double eth = std::abs(th  - c.theta_ref);
        double ex  = std::abs(xo  - c.x_ref);
        double ey  = std::abs(yo  - c.y_ref);
        bool pass  = (eth <= tol_theta) && (ex <= tol_pos) && (ey <= tol_pos);
        if (!pass) ok = false;

        std::printf("  s=%3.1f psi=%6.4f  theta=%+.10f (ref %+.10f err=%.1e %s)"
                    "  x=%+.10f (err=%.1e %s)  y=%+.10f (err=%.1e %s)\n",
                    c.s, c.psi, th, c.theta_ref, eth, eth<=tol_theta?"OK":"FAIL",
                    xo, ex, ex<=tol_pos?"OK":"FAIL",
                    yo, ey, ey<=tol_pos?"OK":"FAIL");
    }
    return ok;
}

int main(int argc, char** argv) {
    const char* path  = (argc > 1) ? argv[1] : MLS_TABLE_PATH;
    int nseg   = (argc > 2) ? std::atoi(argv[2]) : 20;
    int n_fine = (argc > 3) ? std::atoi(argv[3]) : MLS_N_FINE;
    int n_reps = (argc > 4) ? std::atoi(argv[4]) : 1000;

    std::printf("Loading: %s\n", path);
    MlsTable t;
    try {
        t = load_mls_table(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    std::printf("  n_s=%d  n_psi=%d\n\n", t.n_s, t.n_psi);

    std::printf("--- Validation (n_fine=%d, tol_theta=1e-6, tol_pos=1e-5) ---\n", n_fine);
    bool valid = validate(t, n_fine);
    std::printf("Validation: %s\n\n", valid ? "PASSED" : "FAILED");

    std::printf("--- Benchmark ---\n");
    MlsBenchResult r = mls_benchmark(t, nseg, n_fine, n_reps);
    mls_benchmark_print(r);

    return valid ? 0 : 1;
}
