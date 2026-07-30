#!/usr/bin/env python3
"""
Python translation of the WRITE_GENERALISED_FORCES simulation mode.

Runs a single prescribed-motion filament for 300 timesteps using the
Fulford-and-Blake beat model and wall-corrected RPY mobility, then writes
the generalised force reference files consumed by DYNAMIC_PHASE_EVOLUTION.

The output file names match the C++ naming convention exactly, so the
files are drop-in replacements for those produced by the C++ binary.

Usage:
    python3 write_generalised_forces.py
"""

import os
import numpy as np
from numpy.linalg import solve

try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

PI = np.pi

# =======================================================================
# Parameters  (edit these to match your simulation)
# =======================================================================

NSEG    = 20        # number of segments per filament
SEG_SEP = 2.6       # segment separation (in units of RSEG)
PLOT    = True      # show Q_phase / Q_angle plots after writing files

# Derived — do not edit unless you know what you are changing
MU      = 1.0               # dimensionless fluid viscosity  (config.hpp: MU 1.0)
RSEG    = 1.0               # dimensionless segment radius   (config.hpp: RSEG 1.0)
PERIOD  = 1.0               # beat period — forced to 1 by WRITE_GENERALISED_FORCES
STEPS   = 300               # STEPS_PER_PERIOD (config.hpp line 338)
DT      = PERIOD / STEPS
DL      = SEG_SEP * RSEG   # centre-to-centre segment distance
FIL_LEN = DL * (NSEG - 1)  # filament length  (= DL*(NSEG-1) for PRESCRIBED_CILIA)
omega0  = 2.0 * PI          # phase angular frequency (2π per period)

# Wall simulation (BODY_OR_SURFACE_TYPE=0 → INFINITE_PLANE_WALL=true in config.hpp).
# Set WALL=False only for free-space testing.
WALL    = True

# Body rotation matrix R: transforms the reference beat frame (xy-plane)
# to the lab frame.  For a wall simulation where the filament grows along z:
#   R = [[0, 0, -1],      <- reference-x maps to lab-z (height direction)
#        [0, 1,  0],      <- reference-y maps to lab-y (lateral direction)
#        [1, 0,  0]]
# This corresponds to a -90° rotation about the y-axis (quaternion (1/√2,0,-1/√2,0)).
# For free-space with the filament beating in the xy-plane, use np.eye(3).
if WALL:
    R    = np.array([[0., 0., -1.],
                     [0., 1.,  0.],
                     [1., 0.,  0.]])
    BASE = np.array([0.0, 0.0, 0.5 * DL])  # base segment half a step above wall
else:
    R    = np.eye(3)
    BASE = np.zeros(3)

# Output directory and file names — match C++ reference_file_name() for
# FULFORD_AND_BLAKE_BEAT.  C++ std::to_string(float) uses 6 decimal places.
OUT_DIR = 'input/forcing'
_sep    = f'{SEG_SEP:.6f}'
_stem   = f'fulford_and_blake_reference_{{t}}_NSEG={NSEG}_SEP={_sep}.dat'
FNAME_PHASE  = os.path.join(OUT_DIR, _stem.format(t='phase_generalised_forces'))
FNAME_ANGLE  = os.path.join(OUT_DIR, _stem.format(t='angle_generalised_forces'))
FNAME_SVALS  = os.path.join(OUT_DIR, _stem.format(t='s_values'))

# =======================================================================
# Fourier coefficients — Fulford-and-Blake beat  (SHAPE_SEQUENCE = 1)
#
# Matrix layout: row = Fourier mode index, col = polynomial degree index.
# The shape tangent at arc-parameter s and phase psi is:
#
#   tx = (cos_vec @ Ax + sin_vec @ Bx) @ s_vec_tangent
#   ty = (cos_vec @ Ay + sin_vec @ By) @ s_vec_tangent
#
# where:
#   cos_vec        = [1, cos(psi), cos(2psi), cos(3psi)]      shape (4,)
#   sin_vec        = [sin(psi), sin(2psi), sin(3psi)]          shape (3,)
#   s_vec_tangent  = [(i+1)*s^i  for i=0..2]                   shape (3,)
#   s_vec_position = [s^(i+1)    for i=0..2]                   shape (3,)
# =======================================================================
Ax = np.array([
    [ 9.7204e-01, -2.8315e-01,  4.9243e-02],
    [-1.8466e-02, -1.2926e-01,  2.6981e-01],
    [ 1.6209e-01, -3.4983e-01,  1.9082e-01],
    [ 1.0259e-02,  3.5907e-02, -6.8736e-02],
], dtype=float)

Ay = np.array([
    [-3.3547e-01,  4.0369e-01,  1.0362e-01],
    [ 4.0318e-01, -1.5553e+00,  7.3455e-01],
    [-9.9513e-02,  3.2829e-02, -1.2106e-01],
    [ 8.1046e-02, -3.0982e-01,  1.4568e-01],
], dtype=float)

Bx = np.array([
    [ 1.9697e-01, -5.1193e-01,  3.4778e-01],
    [-5.1295e-02,  4.3396e-01, -3.3547e-01],
    [ 1.2311e-02,  1.4157e-01, -1.1695e-01],
], dtype=float)

By = np.array([
    [ 2.9136e-01,  1.0721e+00, -1.0433e+00],
    [ 6.1554e-03,  3.2521e-01, -2.8315e-01],
    [-6.0528e-02,  2.3185e-01, -2.0108e-01],
], dtype=float)

NF = Ax.shape[0]   # number of Fourier modes  (= 4)
ND = Ax.shape[1]   # number of polynomial degrees (= 3)


# =======================================================================
# Beat-model functions  (port of filament.cpp, FIT_TO_DATA_BEAT branch)
# =======================================================================

def _fourier_vecs(psi):
    cos_pos = np.array([1.0] + [np.cos(n * psi) for n in range(1, NF)])
    sin_pos = np.array([np.sin(n * psi) for n in range(1, NF)])
    sin_vel = np.array([0.0] + [-n * np.sin(n * psi) for n in range(1, NF)])
    cos_vel = np.array([n * np.cos(n * psi) for n in range(1, NF)])
    return cos_pos, sin_pos, sin_vel, cos_vel


def fitted_shape_tangent(s, psi):
    """d/ds of the fitted shape at (s, psi) — matches filament::fitted_shape_tangent()."""
    s_vec = np.array([(i + 1) * s**i for i in range(ND)])
    cp, sp, _, _ = _fourier_vecs(psi)
    tx = (cp @ Ax + sp @ Bx) @ s_vec
    ty = (cp @ Ay + sp @ By) @ s_vec
    return tx, ty


def fitted_shape(s, psi):
    """Position [x, y, 0] in units of FIL_LEN — matches filament::fitted_shape()."""
    s_vec = np.array([s**(i + 1) for i in range(ND)])
    cp, sp, _, _ = _fourier_vecs(psi)
    px = (cp @ Ax + sp @ Bx) @ s_vec
    py = (cp @ Ay + sp @ By) @ s_vec
    return np.array([px, py, 0.0])


def fitted_shape_vel_dir(s, psi):
    """d/d(psi) of fitted_shape — matches filament::fitted_shape_velocity_direction()."""
    s_vec = np.array([s**(i + 1) for i in range(ND)])
    _, _, sv, cv = _fourier_vecs(psi)
    dx = (sv @ Ax + cv @ Bx) @ s_vec
    dy = (sv @ Ay + cv @ By) @ s_vec
    return np.array([dx, dy, 0.0])


def fitted_curve_length(s, psi):
    """
    Arc length from 0 to s via trapezoid rule on a fixed mesh of 10*NSEG points.
    Matches filament::fitted_curve_length() — the fixed mesh ensures monotonicity.
    """
    if s <= 0.0:
        return 0.0
    num_traps = 10 * NSEG
    dl        = 1.0 / num_traps
    floor_n   = int(np.floor(s / dl))
    ceil_n    = int(np.ceil(s / dl))

    tx, ty = fitted_shape_tangent(0.0, psi)
    f0 = np.hypot(tx, ty)
    length = 0.0
    for n in range(1, floor_n + 1):
        tx, ty = fitted_shape_tangent(n * dl, psi)
        f1 = np.hypot(tx, ty)
        length += 0.5 * dl * (f0 + f1)
        f0 = f1
    if ceil_n > floor_n:
        tx, ty = fitted_shape_tangent(ceil_n * dl, psi)
        f1 = np.hypot(tx, ty)
        length += 0.5 * (s - floor_n * dl) * (f0 + f1)
    return length


def find_s_to_use(psi):
    """
    Arc-length bisection: find s values so segments are equally spaced in arc length.
    Matches filament::find_fitted_shape_s() (WRITE_GENERALISED_FORCES branch).
    Returns s_arr of shape (NSEG,) with s_arr[0]=0, s_arr[-1]=1.
    """
    s_arr    = np.zeros(NSEG)
    s_arr[-1] = 1.0
    total    = fitted_curve_length(1.0, psi)
    tol      = 0.1 / NSEG

    for n in range(1, NSEG - 1):
        target = n / (NSEG - 1)
        lo, hi = s_arr[n - 1], 1.0
        curr   = 0.5 * (lo + hi)
        while True:
            frac = fitted_curve_length(curr, psi) / total
            if abs(frac - target) <= tol:
                break
            hi = curr if frac > target else hi
            lo = curr if frac <= target else lo
            curr = 0.5 * (lo + hi)
        s_arr[n] = curr

    return s_arr


# =======================================================================
# Mobility matrices
# =======================================================================

def rpy_mobility(pos):
    """
    Free-space Rotne-Prager-Yamakawa mobility matrix (3N × 3N).
    pos : (N, 3) segment positions.
    """
    N    = len(pos)
    M    = np.zeros((3 * N, 3 * N))
    mob0 = 1.0 / (6.0 * PI * MU * RSEG)

    for i in range(N):
        M[3*i:3*i+3, 3*i:3*i+3] = mob0 * np.eye(3)
        for j in range(i + 1, N):
            r_vec = pos[i] - pos[j]
            r     = np.linalg.norm(r_vec)
            r_hat = r_vec / r
            outer = np.outer(r_hat, r_hat)
            if r >= 2.0 * RSEG:
                a2r2 = (RSEG / r) ** 2
                Mij  = (1.0 / (8.0 * PI * MU * r)) * (
                    (1.0 + 2.0/3.0 * a2r2) * np.eye(3)
                    + (1.0 - 2.0 * a2r2)   * outer
                )
            else:
                ra  = r / RSEG
                Mij = mob0 * (
                    (1.0 - 9.0 * ra / 32.0) * np.eye(3)
                    + (3.0 * ra / 32.0)      * outer
                )
            M[3*i:3*i+3, 3*j:3*j+3] = Mij
            M[3*j:3*j+3, 3*i:3*i+3] = Mij

    return M


def rpy_mobility_wall(pos):
    """
    Wall-corrected RPY mobility (3N × 3N).  All z-positions must be > 0.

    Two corrections are applied:
    (1) Self-mobility — Brenner (1961) leading-order expansion:
            M_xx = M_yy = (1/6πμa)(1 − 9a/16h)   [parallel to wall]
            M_zz       = (1/6πμa)(1 − 9a/8h)    [normal to wall]
    (2) Cross-mobility — image-Stokeslet correction (leading-order Blake):
            ΔM_ij = −G^OS(x_i − x_j*)   where x_j* = (xj, yj, −zj)
    """
    N    = len(pos)
    mob0 = 1.0 / (6.0 * PI * MU * RSEG)
    M    = rpy_mobility(pos)

    for i in range(N):
        h_i = pos[i, 2]
        r_a = RSEG / h_i

        # (1) Brenner self-correction (replaces the free-space diagonal block)
        fac_xy = 1.0 - 9.0/16.0 * r_a
        fac_z  = 1.0 - 9.0/8.0  * r_a
        M[3*i:3*i+3, 3*i:3*i+3] = np.diag([mob0 * fac_xy,
                                             mob0 * fac_xy,
                                             mob0 * fac_z])

        # (2) Image-Stokeslet cross-correction
        for j in range(i + 1, N):
            R_vec = pos[i] - np.array([pos[j, 0], pos[j, 1], -pos[j, 2]])
            R     = np.linalg.norm(R_vec)
            img   = (1.0 / (8.0 * PI * MU * R)) * (
                np.eye(3) + np.outer(R_vec, R_vec) / R**2
            )
            M[3*i:3*i+3, 3*j:3*j+3] -= img
            M[3*j:3*j+3, 3*i:3*i+3] -= img.T

    return M


# =======================================================================
# Visualisation
# =======================================================================

def _setup_font():
    font_dir = os.path.expanduser("~/.local/share/fonts/cmu/cm-unicode-0.7.0")
    if os.path.isdir(font_dir):
        for font_file in os.listdir(font_dir):
            if font_file.endswith('.otf'):
                fm.fontManager.addfont(os.path.join(font_dir, font_file))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif']  = ['CMU Serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams.update({'font.size': 14})


def plot(q_phase_arr, q_angle_arr):
    """
    Two-panel figure:
      Left  — Q_phase and Q_angle vs beat phase ψ, normalised by ω η L³
      Right — Fourier amplitude spectrum of both signals
    """
    if not _HAS_MPL:
        print('matplotlib not available — skipping plot')
        return

    _setup_font()

    norm = omega0 * MU * FIL_LEN**3
    psi  = np.linspace(0, 2 * PI, STEPS + 1)[:-1]
    qp   = q_phase_arr / norm
    qa   = q_angle_arr / norm

    # Fourier amplitudes
    fft_p = np.fft.fft(qp)
    fft_a = np.fft.fft(qa)
    An_p  =  2 * np.real(fft_p[:STEPS // 2]) / STEPS
    Bn_p  = -2 * np.imag(fft_p[:STEPS // 2]) / STEPS
    An_a  =  2 * np.real(fft_a[:STEPS // 2]) / STEPS
    Bn_a  = -2 * np.imag(fft_a[:STEPS // 2]) / STEPS
    amp_p = np.sqrt(An_p**2 + Bn_p**2)
    amp_a = np.sqrt(An_a**2 + Bn_a**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Q vs ψ
    ax1.plot(psi, qp, label=r'$Q_1$', c='black', linestyle='solid')
    ax1.plot(psi, qa, label=r'$Q_2$', c='black', linestyle='dashed')
    ax1.set_xlabel(r'$\psi$')
    ax1.set_ylabel(r'$Q / \omega \eta L^3$')
    ax1.set_xticks(np.linspace(0, 2 * PI, 5),
                   [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax1.set_xlim(0, 2 * PI)
    ax1.legend(frameon=False)

    # Right: Fourier amplitude spectrum
    coeff_lim = 8
    nn = np.arange(coeff_lim)
    ax2.plot(nn, amp_p[:coeff_lim], label=r'$Q_1$', c='black', linestyle='solid',  marker='+')
    ax2.plot(nn, amp_a[:coeff_lim], label=r'$Q_2$', c='black', linestyle='dashed', marker='+')
    ax2.set_xlabel(r'$n$')
    ax2.set_ylabel(r'Amplitude')
    ax2.set_xlim(0, coeff_lim - 1)
    ax2.set_ylim(0)
    ax2.set_xticks(nn)
    ax2.legend(frameon=False)

    fig.tight_layout()
    plt.show()


# =======================================================================
# Main simulation loop
# =======================================================================

def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    fp = open(FNAME_PHASE, 'w')
    fa = open(FNAME_ANGLE, 'w')
    fs = open(FNAME_SVALS, 'w')

    fp.write(f'{STEPS} ')
    fa.write(f'{STEPS} ')
    fs.write(f'{STEPS} {NSEG} ')

    mob_fn = rpy_mobility_wall if WALL else rpy_mobility

    print(f'WRITE_GENERALISED_FORCES  NSEG={NSEG}  SEG_SEP={SEG_SEP}'
          f'  FIL_LEN={FIL_LEN:.4f}  WALL={WALL}')

    q_phase_arr = np.empty(STEPS)
    q_angle_arr = np.empty(STEPS)

    phase = 0.0
    for nt in range(STEPS):

        # 1. Arc-length parametrisation for this phase
        s_arr = find_s_to_use(phase)
        fs.write(' '.join(f'{s:.15e}' for s in s_arr) + ' ')

        # 2. Segment positions and velocity-direction vectors in the lab frame
        seg_pos = np.empty((NSEG, 3))
        vdp     = np.zeros((NSEG, 3))  # vel_dir_phase = FIL_LEN * R * d(shape)/d(psi)
        vda     = np.zeros((NSEG, 3))  # vel_dir_angle = FIL_LEN * R * cross(e_z, shape)

        seg_pos[0] = BASE
        # Segment 0 is the fixed base; vel_dir vectors remain zero there.

        for n in range(1, NSEG):
            sh          = fitted_shape(s_arr[n], phase)
            seg_pos[n]  = BASE + FIL_LEN * (R @ sh)

            vd          = fitted_shape_vel_dir(s_arr[n], phase)
            vdp[n]      = FIL_LEN * (R @ vd)

            # cross(e_z_ref, sh) = [-sh_y, sh_x, 0]  then rotate to lab frame
            cross_ref   = np.array([-sh[1], sh[0], 0.0])
            vda[n]      = FIL_LEN * (R @ cross_ref)

        # 3. Prescribed translational velocity V_n = omega0 * vel_dir_phase[n]
        V = (omega0 * vdp).ravel()

        # 4. Solve M * F = V for the constraint forces
        F = solve(mob_fn(seg_pos), V)

        # 5. Generalised forces Q = sum_n vel_dir[n] . F[n]
        q_phase = float(vdp.ravel() @ F)
        q_angle = float(vda.ravel() @ F)

        fp.write(f'{q_phase:.15e} ')
        fa.write(f'{q_angle:.15e} ')

        q_phase_arr[nt] = q_phase
        q_angle_arr[nt] = q_angle

        # 6. Advance phase (explicit Euler, omega0 * DT = 2pi/300)
        phase += omega0 * DT

        if (nt + 1) % 50 == 0:
            print(f'  step {nt+1:>3}/{STEPS}  phase={phase:.4f}'
                  f'  Q_phase={q_phase:.6e}  Q_angle={q_angle:.6e}')

    fp.write('\n'); fa.write('\n'); fs.write('\n')
    fp.close();     fa.close();     fs.close()

    print('Done. Wrote:')
    print(f'  {FNAME_PHASE}')
    print(f'  {FNAME_ANGLE}')
    print(f'  {FNAME_SVALS}')

    if PLOT:
        plot(q_phase_arr, q_angle_arr)


if __name__ == '__main__':
    run()