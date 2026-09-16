
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

import matplotlib.font_manager as fm

font_dir = os.path.expanduser("~/.local/share/fonts/cmu/cm-unicode-0.7.0")
if os.path.isdir(font_dir):
    for font_file in os.listdir(font_dir):
        if font_file.endswith('.otf'):
            fm.fontManager.addfont(os.path.join(font_dir, font_file))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['CMU Serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 24})


# ---------------------------------------------------------------------------
# Quaternion → rotation matrix (matches util.rot_mat)
# ---------------------------------------------------------------------------
def rot_mat(q):
    R = np.eye(3)
    R[1,1] -= 2*q[1]*q[1];  R[2,2] -= 2*q[1]*q[1]
    R[0,0] -= 2*q[2]*q[2];  R[2,2] -= 2*q[2]*q[2]
    R[0,0] -= 2*q[3]*q[3];  R[1,1] -= 2*q[3]*q[3]
    t = 2*q[1]*q[2]; R[1,0] = t; R[0,1] = t
    t = 2*q[1]*q[3]; R[2,0] = t; R[0,2] = t
    t = 2*q[2]*q[3]; R[1,2] = t; R[2,1] = t
    t = 2*q[0]*q[3]; R[1,0] += t; R[0,1] -= t
    t = 2*q[0]*q[2]; R[2,0] -= t; R[0,2] += t
    t = 2*q[0]*q[1]; R[2,1] += t; R[1,2] -= t
    return R


# ---------------------------------------------------------------------------
# Parse .par file
# ---------------------------------------------------------------------------
def read_par(par_path):
    pars = {}
    with open(par_path) as f:
        for line in f:
            parts = line.split('%%')
            if len(parts) == 2:
                pars[parts[1].strip()] = float(parts[0].strip())
    return pars


# ---------------------------------------------------------------------------
# Compute per-frame speed (along body axis) and dissipation
# ---------------------------------------------------------------------------
def compute_averages(sim_base, pars, skip_first=True):
    nfil   = int(pars['NFIL'])
    nseg   = int(pars['NSEG'])
    nblob  = int(pars['NBLOB'])
    nsteps = int(pars['TOTAL_TIME_STEPS'])
    plot_freq = int(pars['PLOT_FREQUENCY_IN_STEPS'])
    period = int(pars['STEPS_PER_PERIOD'])
    fillength = pars['FIL_LENGTH']
    ar = 15.0
    radius = ar * fillength

    n_frames = nsteps // plot_freq  # number of plotted frames after frame 0

    seg_forces_f    = open(sim_base + '_seg_forces.dat')
    seg_vels_f      = open(sim_base + '_seg_vels.dat')
    blob_forces_f   = open(sim_base + '_blob_forces.dat')
    blob_refs_f     = open(sim_base + '_blob_references.dat')
    body_states_f   = open(sim_base + '_body_states.dat')
    body_vels_f     = open(sim_base + '_body_vels.dat')

    # blob references are written once (first line)
    blob_refs = np.fromstring(blob_refs_f.readline(), sep=' ').reshape(nblob, 3)
    blob_refs_f.close()

    speed_list = []
    dissip_list = []

    for frame in range(n_frames + 1):
        sf_str  = seg_forces_f.readline()
        sv_str  = seg_vels_f.readline()
        bf_str  = blob_forces_f.readline()
        bv_str  = body_vels_f.readline()
        bs_str  = body_states_f.readline()

        if skip_first and frame == 0:
            continue

        seg_forces  = np.fromstring(sf_str,  sep=' ')[1:].reshape(nfil*nseg, 6)
        seg_vels    = np.fromstring(sv_str,  sep=' ')[1:].reshape(nfil*nseg, 6)
        blob_forces = np.fromstring(bf_str,  sep=' ')[1:].reshape(nblob, 3)
        body_vels   = np.fromstring(bv_str,  sep=' ')[1:]   # [vx,vy,vz, ox,oy,oz]
        body_states = np.fromstring(bs_str,  sep=' ')[1:]   # [x,y,z, q0,q1,q2,q3]

        R = rot_mat(body_states[3:7])
        body_axis = R @ np.array([0., 0., 1.])
        speed_axis = np.dot(body_vels[0:3], body_axis)

        blob_vels = body_vels[0:3] + np.cross(body_vels[3:6], blob_refs)
        dissipation = np.sum(blob_forces * blob_vels) + np.sum(seg_forces * seg_vels)

        speed_list.append(speed_axis)
        dissip_list.append(dissipation)

    for f in [seg_forces_f, seg_vels_f, blob_forces_f, body_states_f, body_vels_f]:
        f.close()

    speed_arr  = np.array(speed_list)
    dissip_arr = np.array(dissip_list)

    avg_speed  = np.mean(speed_arr)  / fillength
    avg_dissip = np.mean(dissip_arr) / fillength**3
    avg_efficiency = 6*np.pi*radius * np.mean(speed_arr)**2 / np.mean(dissip_arr)

    return avg_speed, avg_dissip, avg_efficiency


# ---------------------------------------------------------------------------
# Sweep parameters (match func.py 'volvox_bicilia_dp_sweep' preset)
# ---------------------------------------------------------------------------
path      = 'data/volvox/20260914_dp_sweep/'
dps       = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
wavnums   = np.array([-2.35, -1.0, 0.0, 1.0])
dp_n      = len(dps)
wavnum_n  = len(wavnums)
sim_n     = dp_n * wavnum_n   # 24

avg_speed_list      = np.full(sim_n, np.nan)
avg_dissipation_list = np.full(sim_n, np.nan)
avg_efficiency_list  = np.full(sim_n, np.nan)

for ind in range(sim_n):
    dp_i  = ind // wavnum_n
    wav_i = ind  % wavnum_n
    dp    = dps[dp_i]

    sim_base = (f"{path}ciliate_640fil_40962blob_15.00R_0.0050torsion"
                f"_0.0000tilt_{dp:.4f}dp_0.0000noise_0.0000ospread_{ind}index")
    par_path = sim_base + '.par'

    if not os.path.exists(par_path):
        print(f"  skipping ind={ind} (no .par)")
        continue

    print(f"processing ind={ind}  dp={dp:.1f}  k={wavnums[wav_i]:.2f} ...", end=' ', flush=True)
    try:
        pars = read_par(par_path)
        s, d, e = compute_averages(sim_base, pars)
        avg_speed_list[ind]       = s
        avg_dissipation_list[ind] = d
        avg_efficiency_list[ind]  = e
        print(f"V={s:.4f}  D={d:.3e}  eff={e:.4f}")
    except Exception as ex:
        print(f"ERROR: {ex}")


# ---------------------------------------------------------------------------
# Plots  (one line per wavnum, x-axis = dp)
# ---------------------------------------------------------------------------
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

for wi, wave in enumerate(wavnums):
    # fixed wavnum_j = wi → indices wi, wi+wavnum_n, wi+2*wavnum_n, ...
    idx = [dp_i * wavnum_n + wi for dp_i in range(dp_n)]
    ax1.plot(dps, avg_speed_list[idx],       marker='+', color='black',
             label=rf'$\kappa={wave}$', linestyle=linestyles[wi])
    ax2.plot(dps, avg_dissipation_list[idx], marker='+', color='black',
             label=rf'$\kappa={wave}$', linestyle=linestyles[wi])
    ax3.plot(dps, avg_efficiency_list[idx],  marker='+', color='black',
             label=rf'$\kappa={wave}$', linestyle=linestyles[wi])

ax1.legend(frameon=False, fontsize=20)
ax1.set_xlim(0, 0.5)
ax1.set_xlabel(r'$\Delta\psi_1/2\pi$')
ax1.set_ylabel(r'$\langle V \rangle T/L$')

ax2.legend(frameon=False, fontsize=20)
ax2.set_xlim(0, 0.5)
ax2.set_xlabel(r'$\Delta\psi_1/2\pi$')
ax2.set_ylabel(r'$\langle \mathcal{R} \rangle T^2/\mu L^3$')
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-1, 4))
ax2.yaxis.set_major_formatter(formatter)

ax3.legend(frameon=False, fontsize=20)
ax3.set_xlim(0, 0.5)
ax3.set_xlabel(r'$\Delta\psi_1/2\pi$')
ax3.set_ylabel(r'Efficiency')
formatter2 = mticker.ScalarFormatter(useMathText=True)
formatter2.set_powerlimits((-1, 4))
ax3.yaxis.set_major_formatter(formatter2)

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig1.savefig('fig/volvox_bicilia_dp_sweep_vel.png',  format='png', transparent=True)
fig2.savefig('fig/volvox_bicilia_dp_sweep_dis.png',  format='png', transparent=True)
fig3.savefig('fig/volvox_bicilia_dp_sweep_eff.png',  format='png', transparent=True)

plt.show()
