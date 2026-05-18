import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def equal_area_seeding(N, R=0.5, samples_per_iter=1000, max_iters=1000,
                       seed=None, snapshot_iters=None, init='hemisphere',
                       reset_count_every=None):
    """
    Python port of equal_area_seeding() from seeding.cu.
    MacQueen's online k-means on a sphere of radius R.

    Parameters
    ----------
    snapshot_iters : list of int, optional
        Iterations at which to save a copy of X and count for visualisation.

    Returns
    -------
    pos       : (N, 3) centroid positions
    polar_dir : (N, 3) southward tangent
    azi_dir   : (N, 3) eastward tangent
    normal    : (N, 3) outward normal
    snapshots : list of (iteration, X_copy, count_copy)
    conv_log  : list of (iteration, min/max ratio)
    """
    if N == 0:
        empty = np.empty((0, 3))
        return empty, empty, empty, empty, [], []

    if snapshot_iters is None:
        snapshot_iters = set()
    else:
        snapshot_iters = set(snapshot_iters)

    rng = np.random.default_rng(seed)

    def random_points(n):
        v = rng.standard_normal((n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v * R

    def project(x):
        return x / np.linalg.norm(x) * R

    # --- Initialisation ---
    if init == 'random':
        X = random_points(N)
    elif init == 'hemisphere':
        # All points on the z > 0 hemisphere (extreme case)
        v = rng.standard_normal((N, 3))
        v[:, 2] = np.abs(v[:, 2])          # force z > 0
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        X = v * R
    elif init == 'pole':
        # All points within a small cap around the north pole (most extreme)
        v = rng.standard_normal((N, 3)) * [0.05, 0.05, 1.0]
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        X = v * R
    else:
        raise ValueError(f"Unknown init='{init}'. Choose 'random', 'hemisphere', or 'pole'.")

    count     = np.ones(N, dtype=np.int64)
    old_count = np.ones(N, dtype=np.int64)

    snapshots = []
    conv_log  = []

    # Save iteration 0 (Saff-Kuijlaars initial positions)
    if 0 in snapshot_iters:
        snapshots.append((0, X.copy(), count.copy()))

    for iteration in range(max_iters):
        samples = random_points(samples_per_iter)

        _, nn_ids = KDTree(X).query(samples)

        for i, idx in enumerate(nn_ids):
            c      = count[idx]
            X[idx] = (c * X[idx] + samples[i]) / (c + 1)
            count[idx] += 1
            X[idx] = project(X[idx])

        iter_num = iteration + 1

        if iter_num in snapshot_iters:
            snapshots.append((iter_num, X.copy(), count.copy()))

        print(f"\rIteration {iter_num}", end='', flush=True)

        if reset_count_every is not None and iter_num % reset_count_every == 0:
            count[:] = 1

        if iter_num % 100 == 0:
            delta      = count - old_count
            min_change = int(delta.min())
            max_change = int(delta.max())
            old_count  = count.copy()
            ratio = min_change / max_change if max_change > 0 else 0.0
            conv_log.append((iter_num, ratio))
            if ratio > 0.99:
                print(f"Converged at iteration {iter_num}")
                # Save final snapshot if not already captured
                if iter_num not in snapshot_iters:
                    snapshots.append((iter_num, X.copy(), count.copy()))
                break

    # --- Local frames ---
    theta = np.arctan2(np.sqrt(X[:, 0]**2 + X[:, 1]**2), X[:, 2])
    phi   = np.arctan2(X[:, 1], X[:, 0])
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi),   np.sin(phi)

    polar_dir = np.stack([ ct*cp,  ct*sp, -st         ], axis=1)
    azi_dir   = np.stack([-sp,     cp,    np.zeros(N) ], axis=1)
    normal    = np.cross(polar_dir, azi_dir)

    return X, polar_dir, azi_dir, normal, snapshots, conv_log


def plot_sphere_snapshot(ax, X, count, title, R=0.5):
    """Plot centroids on a sphere."""
    # Wireframe sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0,   np.pi, 20)
    sx = R * np.outer(np.cos(u), np.sin(v))
    sy = R * np.outer(np.sin(u), np.sin(v))
    sz = R * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(sx, sy, sz, color='lightgrey', alpha=0.2, linewidth=0.4)

    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                    color='tab:blue', s=18, zorder=5)
    ax.set_title(title, fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    return sc


if __name__ == '__main__':
    N    = 100
    R    = 0.5
    init = 'pole'   # 'random' | 'hemisphere' | 'pole'
    snaps = [0, 1, 5, 10, 100, 500, 1000, 2000, 5000]

    pos, pd, ad, nr, snapshots, conv_log = equal_area_seeding(
        N, R=R, samples_per_iter=1000, max_iters=5000,
        seed=42, snapshot_iters=snaps, init=init,
        reset_count_every=100
    )

    # --- Figure 1: snapshots ---
    n_snaps = len(snapshots)
    ncols   = 3
    nrows   = (n_snaps + ncols - 1) // ncols
    fig1    = plt.figure(figsize=(4*ncols, 4*nrows))

    for k, (it, Xs, cs) in enumerate(snapshots):
        ax = fig1.add_subplot(nrows, ncols, k+1, projection='3d')
        plot_sphere_snapshot(ax, Xs, cs, f'Iteration {it}', R=R)

    fig1.subplots_adjust(hspace=0.05, wspace=0.05)
    fig1.suptitle(f'MacQueen k-means convergence  (N={N}, init={init!r})', fontsize=11)

    # --- Figure 2: convergence metric ---
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    if conv_log:
        iters, ratios = zip(*conv_log)
        ax2.plot(iters, ratios, marker='.', markersize=4)
        ax2.axhline(0.99, color='red', linestyle='--', linewidth=0.8, label='Convergence threshold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel(r'$\Delta_\mathrm{min}\,/\,\Delta_\mathrm{max}$')
        ax2.set_title('Convergence metric (min/max count change per 100 iterations)')
        ax2.legend()
        ax2.grid(True, alpha=0.4)
    fig2.tight_layout()

    plt.show()