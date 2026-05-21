import numpy as np
from scipy.spatial import KDTree

with open("input/fourier_modes/sphere639.seed") as f:
    vals = np.fromstring(f.read(), sep=' ')

pts = vals.reshape(-1, 3)
N   = len(pts)
R   = np.mean(np.linalg.norm(pts, axis=1))
R = 49.4*7.5

tree = KDTree(pts)
dists, _ = tree.query(pts, k=2)
nn_dists = dists[:, 1]

# Circular-patch theory: each point owns a spherical cap of area 4πR²/N
alpha    = np.arccos(1.0 - 2.0 / N)   # angular cap radius
d_theory = 2.0 * R * np.sin(alpha)    # chord distance between adjacent centres

print(f"N                             = {N}")
print(f"R (mean)                      = {R:.6f}")
print(f"Mean NN distance              = {nn_dists.mean():.6f}")
print(f"Min  NN distance              = {nn_dists.min():.6f}")
print(f"Max  NN distance              = {nn_dists.max():.6f}")
print(f"Std  NN distance              = {nn_dists.std():.6f}")
print(f"Theoretical NN (circular cap) = {d_theory:.6f}")