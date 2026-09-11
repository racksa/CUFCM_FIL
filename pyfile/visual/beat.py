import numpy as np
import struct
from scipy.integrate import cumulative_trapezoid

# Fourier coeffs for the shape
Ay = np.array([[-3.3547e-01, 4.0369e-01, 1.0362e-01], \
            [4.0318e-01, -1.5553e+00, 7.3455e-01], \
            [-9.9513e-02, 3.2829e-02, -1.2106e-01], \
            [8.1046e-02, -3.0982e-01, 1.4568e-01]])

Ax = np.array([[9.7204e-01, -2.8315e-01, 4.9243e-02], \
            [-1.8466e-02, -1.2926e-01, 2.6981e-01], \
            [1.6209e-01, -3.4983e-01, 1.9082e-01], \
            [1.0259e-02, 3.5907e-02, -6.8736e-02]])

By = np.array([[0, 0, 0], \
            [2.9136e-01, 1.0721e+00, -1.0433e+00], \
            [6.1554e-03, 3.2521e-01, -2.8315e-01], \
            [-6.0528e-02, 2.3185e-01, -2.0108e-01]])

Bx = np.array([[0, 0, 0], \
            [1.9697e-01, -5.1193e-01, 3.4778e-01], \
            [-5.1295e-02, 4.3396e-01, -3.3547e-01], \
            [1.2311e-02, 1.4157e-01, -1.1695e-01]])

def lung_cilia_shape(s, phase):
    pos = np.zeros(3)
#     svec = np.array([s, s**2, s**3])
#     fourier_dim = np.shape(Ax)[0]
    degrees = np.shape(Ax)[1]
    fourier_dim = np.shape(Ax)[0]
    svec = np.array([s**(i+1) for i in range(degrees)])
    cosvec = np.array([ np.cos(n*phase) for n in range(fourier_dim)])
    sinvec = np.array([ np.sin(n*phase) for n in range(fourier_dim)])
    cosvec[0] *= 0.5

    x = (cosvec@Ax + sinvec@Bx)@svec
    y = (cosvec@Ay + sinvec@By)@svec
    z = np.zeros(np.shape(x))

    return x, y, z


Ay_volvox = np.array([[-0.01359477, -0.2933222 ,  1.11787568, -0.5797987 ],
       [ 0.17304993, -2.07743633,  1.96507494, -0.48418322],
       [-0.13139014,  1.67275679, -3.30501852,  1.75899013],
       [-0.16029509,  0.86911573, -1.43711565,  0.70427363]])
Ax_volvox = np.array([[ 1.00173453e+00, -1.93437790e-01, -2.09032251e-01, 1.09657224e-01],
       [ 1.77973257e-03, -2.19512900e-01,  1.18549520e+00, -8.75601570e-01],
       [ 1.82160747e-01, -1.19973029e+00,  1.85563800e+00, -7.38088036e-01],
       [-1.18146940e-01,  1.11157299e+00, -2.22516778e+00, 1.24886035e+00]])
By_volvox = np.array([[-1.65423231e-14,  2.22044605e-16, -4.60742555e-15, -1.27675648e-15],
       [-7.16106264e-02,  2.20709826e+00, -4.29445659e+00, 2.11206054e+00],
       [-2.81241335e-01,  1.40574837e+00, -1.48026680e+00, 3.96486282e-01],
       [ 2.78457057e-02, -3.71722651e-01,  1.03114541e+00, -7.11831064e-01]])
Bx_volvox = np.array([[ 4.24660307e-15,  5.89719246e-15, -5.20417043e-15, -3.55271368e-15],
       [ 9.13270401e-02, -5.35979006e-01,  1.37873237e+00, -5.83970315e-01],
       [-9.93255062e-02,  1.16560579e+00, -2.64682150e+00, 1.58062759e+00],
       [-1.57456039e-01,  9.40433425e-01, -1.39914889e+00, 5.95880263e-01]])

def volvox_cilia_shape(s, phase):
    pos = np.zeros(3)
    # svec = np.array([s, s**2, s**3])
    degrees = np.shape(Ax_volvox)[1]
    fourier_dim = np.shape(Ax_volvox)[0]
    svec = np.array([s**(i+1) for i in range(degrees)])
    
    cosvec = np.array([ np.cos(n*phase) for n in range(fourier_dim)])
    sinvec = np.array([ np.sin(n*phase) for n in range(fourier_dim)])

    x = (cosvec@Ax_volvox + sinvec@Bx_volvox)@svec
    y = (cosvec@Ay_volvox + sinvec@By_volvox)@svec
    z = np.zeros(np.shape(x))

    return x, y, z


# ---------------------------------------------------------------------------
# RBF 2D precomputed beat (SHAPE_SEQUENCE=8, theta_table.bin)
# ---------------------------------------------------------------------------

_rbf_theta_cache = {}  # path → (s_grid, psi_grid_padded, theta_grid)

def load_rbf_theta_table(path="input/theta_table.bin"):
    """Load theta_table.bin once; subsequent calls with the same path return the cached result."""
    if path in _rbf_theta_cache:
        return _rbf_theta_cache[path]
    with open(path, 'rb') as f:
        magic = f.read(8)
        assert magic == b'RBF2DPC\x00', f"Bad magic bytes in {path}: {magic!r}"
        version, n_s, n_psi, _reserved = struct.unpack('<4i', f.read(16))
        s_grid     = np.frombuffer(f.read(8 * n_s),          dtype=np.float64).copy()
        psi_grid   = np.frombuffer(f.read(8 * n_psi),        dtype=np.float64).copy()
        theta_grid = np.frombuffer(f.read(8 * n_s * n_psi),  dtype=np.float64).reshape(n_s, n_psi).copy()
    # Add wraparound column so bilinear interpolation works across the psi=2π boundary
    psi_grid_padded = np.append(psi_grid, psi_grid[0] + 2.0 * np.pi)
    theta_grid      = np.column_stack([theta_grid, theta_grid[:, 0]])
    table = (s_grid, psi_grid_padded, theta_grid, n_s, n_psi)
    _rbf_theta_cache[path] = table
    return table


def rbf_precomputed_shape(s, phase, theta_table, n_fine=500):
    """Drop-in replacement for lung_cilia_shape using the RBF theta table.

    Returns (x, y, z) in the normalised local filament frame (unit arc length).
    s     : float in [0, 1]
    phase : beat phase ψ in [0, 2π)
    """
    x_arr, y_arr, _ = rbf_precomputed_shape_batch(np.array([s]), phase, theta_table, n_fine)
    return float(x_arr[0]), float(y_arr[0]), 0.0


def rbf_precomputed_shape_batch(s_array, phase, theta_table, n_fine=500):
    """Vectorised version: compute positions for all s in s_array at once.

    Returns arrays (x_arr, y_arr, z_arr) of the same length as s_array.
    Use this in the animation loop to avoid redundant integration per segment.
    """
    s_grid, psi_grid_padded, theta_grid, n_s, n_psi = theta_table

    # Fine arc-length grid for integration
    s_fine = np.linspace(0.0, 1.0, n_fine)

    # Locate psi bracket (scalar)
    ip = int(np.searchsorted(psi_grid_padded, phase, side='right')) - 1
    ip = np.clip(ip, 0, n_psi - 1)
    dp = psi_grid_padded[ip + 1] - psi_grid_padded[ip]
    t_p = (phase - psi_grid_padded[ip]) / dp if dp != 0.0 else 0.0

    # Locate s brackets for each fine point (vectorised)
    is_arr = np.searchsorted(s_grid, s_fine, side='right') - 1
    is_arr = np.clip(is_arr, 0, n_s - 2)
    ds_arr = s_grid[is_arr + 1] - s_grid[is_arr]
    t_s_arr = np.where(ds_arr != 0.0, (s_fine - s_grid[is_arr]) / ds_arr, 0.0)

    # Bilinear interpolation → θ(s_fine, phase)
    v00 = theta_grid[is_arr,     ip]
    v01 = theta_grid[is_arr,     ip + 1]
    v10 = theta_grid[is_arr + 1, ip]
    v11 = theta_grid[is_arr + 1, ip + 1]
    theta_fine = ((1 - t_s_arr) * (1 - t_p) * v00 +
                  (1 - t_s_arr) *      t_p  * v01 +
                       t_s_arr  * (1 - t_p) * v10 +
                       t_s_arr  *      t_p  * v11)

    # Cumulative trapezoid → position curve
    x_fine = cumulative_trapezoid(np.cos(theta_fine), s_fine, initial=0.0)
    y_fine = cumulative_trapezoid(np.sin(theta_fine), s_fine, initial=0.0)

    # Interpolate to the requested s values
    x_arr = np.interp(s_array, s_fine, x_fine)
    y_arr = np.interp(s_array, s_fine, y_fine)
    return x_arr, y_arr, np.zeros_like(x_arr)


# ---------------------------------------------------------------------------
# Unified shape selector
# ---------------------------------------------------------------------------

def cilia_shape(s, phase, shape_type='lung', theta_table=None):
    """Return (x, y, z) for a single arc-length point.

    shape_type : 'lung'   – Fourier-polynomial lung cilia beat
                 'volvox' – Fourier-polynomial Volvox beat
                 'rbf'    – RBF 2D precomputed beat (requires theta_table)
    theta_table: result of load_rbf_theta_table(); only needed for 'rbf'.
    """
    if shape_type == 'rbf':
        if theta_table is None:
            raise ValueError("theta_table must be provided for shape_type='rbf'")
        return rbf_precomputed_shape(s, phase, theta_table)
    elif shape_type == 'volvox':
        return volvox_cilia_shape(s, phase)
    else:
        return lung_cilia_shape(s, phase)


def cilia_shape_batch(s_array, phase, shape_type='lung', theta_table=None):
    """Return (x_arr, y_arr, z_arr) for an array of arc-length values.

    shape_type : 'lung'   – Fourier-polynomial lung cilia beat
                 'volvox' – Fourier-polynomial Volvox beat
                 'rbf'    – RBF 2D precomputed beat (requires theta_table)
    theta_table: result of load_rbf_theta_table(); only needed for 'rbf'.
    """
    if shape_type == 'rbf':
        if theta_table is None:
            raise ValueError("theta_table must be provided for shape_type='rbf'")
        return rbf_precomputed_shape_batch(s_array, phase, theta_table)
    else:
        pts = [cilia_shape(s, phase, shape_type) for s in s_array]
        return (np.array([p[0] for p in pts]),
                np.array([p[1] for p in pts]),
                np.array([p[2] for p in pts]))
