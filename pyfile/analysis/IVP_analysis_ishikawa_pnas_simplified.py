
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from matplotlib.ticker import ScalarFormatter
import matplotlib.font_manager as fm

# Path to the directory where fonts are stored
font_dir = os.path.expanduser("~/.local/share/fonts/cmu/cm-unicode-0.7.0")
# Choose the TTF or OTF version of CMU Serif Regular
font_path = os.path.join(font_dir, 'cmunrm.ttf')  # Or 'cmunrm.otf' if you prefer OTF
# Load the font into Matplotlib's font manager
prop = fm.FontProperties(fname=font_path)
# Register each font file with Matplotlib's font manager
for font_file in os.listdir(font_dir):
    if font_file.endswith('.otf'):
        fm.fontManager.addfont(os.path.join(font_dir, font_file))
# Set the global font family to 'serif' and specify CMU Serif
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif']
plt.rcParams['mathtext.fontset'] = 'cm'  # Use 'cm' for Computer Modern
plt.rcParams.update({'font.size': 24})

cmap_name = 'coolwarm'

trunc = 2

Ay = np.array([[-0.654, 0.787, 0.202], \
            [0.393, -1.516, 0.716], \
            [-0.097, 0.032, -0.118], \
            [0.079, -0.302, 0.142]])

Ax = np.array([[1.895, -0.552, 0.096], \
            [-0.018, -0.126, 0.263], \
            [0.158, -0.341, 0.186], \
            [0.010, 0.035, -0.067]])

By = np.array([[0, 0, 0], \
            [0.284, 1.045, -1.017], \
            [0.006, 0.317, -0.276], \
            [-0.059, 0.226, -0.196]])

Bx = np.array([[0, 0, 0], \
            [0.192, -0.499, 0.339], \
            [-0.050, 0.423, -0.327], \
            [0.012, 0.138, -0.114]])

Ay2 = np.array([[-3.3547e-01, 4.0369e-01, 1.0362e-01], \
            [4.0318e-01, -1.5553e+00, 7.3455e-01], \
            [-9.9513e-02, 3.2829e-02, -1.2106e-01], \
            [8.1046e-02, -3.0982e-01, 1.4568e-01]])

Ax2 = np.array([[9.7204e-01, -2.8315e-01, 4.9243e-02], \
            [-1.8466e-02, -1.2926e-01, 2.6981e-01], \
            [1.6209e-01, -3.4983e-01, 1.9082e-01], \
            [1.0259e-02, 3.5907e-02, -6.8736e-02]])

By2 = np.array([[0, 0, 0], \
            [2.9136e-01, 1.0721e+00, -1.0433e+00], \
            [6.1554e-03, 3.2521e-01, -2.8315e-01], \
            [-6.0528e-02, 2.3185e-01, -2.0108e-01]])

Bx2 = np.array([[0, 0, 0], \
            [1.9697e-01, -5.1193e-01, 3.4778e-01], \
            [-5.1295e-02, 4.3396e-01, -3.3547e-01], \
            [1.2311e-02, 1.4157e-01, -1.1695e-01]])

def original_shape_derivative(s, phase):
    svec = np.array([1, 2*s, 3*s**2])
    fourier_dim = np.shape(Ax)[0]
    cosvec = np.array([np.cos(n*phase) for n in range(fourier_dim)])
    sinvec = np.array([np.sin(n*phase) for n in range(fourier_dim)])
    cosvec[0] *= 0.5

    dx_ds = (cosvec @ Ax + sinvec @ Bx) @ svec
    dy_ds = (cosvec @ Ay + sinvec @ By) @ svec
    dz_ds = np.zeros(np.shape(dx_ds))

    return dx_ds, dy_ds, dz_ds

def integrand(s, phase):
    dx_ds, dy_ds, dz_ds = original_shape_derivative(s, phase)
    return np.sqrt(dx_ds**2 + dy_ds**2 + dz_ds**2)

def real_fillength(phase):
    arc_length, _ = quad(integrand, 0, 1, args=(phase))
    return arc_length

def original_shape_derivative2(s, phase):
    svec = np.array([1, 2*s, 3*s**2])
    fourier_dim = np.shape(Ax2)[0]
    cosvec = np.array([np.cos(n*phase) for n in range(fourier_dim)])
    sinvec = np.array([np.sin(n*phase) for n in range(fourier_dim)])

    dx_ds = (cosvec @ Ax2 + sinvec @ Bx2) @ svec
    dy_ds = (cosvec @ Ay2 + sinvec @ By2) @ svec
    dz_ds = np.zeros(np.shape(dx_ds))

    return dx_ds, dy_ds, dz_ds

def integrand2(s, phase):
    dx_ds, dy_ds, dz_ds = original_shape_derivative2(s, phase)
    return np.sqrt(dx_ds**2 + dy_ds**2 + dz_ds**2)

def real_fillength2(phase):
    arc_length, _ = quad(integrand2, 0, 1, args=(phase))
    return arc_length




fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)

# k_list = [-1, 0, 0.5, 1.0, 1.5, 2.0]
# labels = [r"$k=-1$",r"$k=0$",r"$k=0.5$",r"$k=1$",r"$k=1.5$",r"$k=2$",]
colors = ["black","red","green","blue","purple"]
N_list = [160, 640, 2560]


path = 'data/ishikawa/20240802_pnas_L0.975/'
# plot sim data
# for ind in range(3):
#     try:
#         time_array = np.load(f"{path}time_array_index{ind}.npy")
#         speed_array = np.load(f"{path}body_speed_array_index{ind}.npy")
#         rot_speed_array = np.load(f"{path}body_rot_speed_array_index{ind}.npy")
#         dissipation_array = np.load(f"{path}dissipation_array_index{ind}.npy")
#         efficiency_array = np.load(f"{path}efficiency_array_index{ind}.npy")

#         fillength_array = np.zeros(np.shape(time_array))
#         for fili in range(len(fillength_array)):
#             fillength_array[fili] = real_fillength(time_array[fili]*2*np.pi)

#         ax3.plot(time_array, efficiency_array, label = N_list[ind], c=colors[ind])
#         ax4.plot(time_array, fillength_array, label = N_list[ind], c='black')
#     except:
#         pass




# plot sim data
path = "data/ishikawa/20240731_pnas_L1/"
path = 'data/ishikawa/20240802_pnas_L0.975/'
# path = "data/ishikawa/20240805_volvox_beat/"
for ind in range(trunc):
    try:
        time_array = np.load(f"{path}time_array_index{ind}.npy")
        speed_array = np.load(f"{path}body_speed_array_index{ind}.npy")
        rot_speed_array = np.load(f"{path}body_rot_speed_array_index{ind}.npy")
        dissipation_array = np.load(f"{path}dissipation_array_index{ind}.npy")
        efficiency_array = np.load(f"{path}efficiency_array_index{ind}.npy")

        fillength_array = np.zeros(np.shape(time_array))
        for fili in range(len(fillength_array)):
            fillength_array[fili] = real_fillength2(time_array[fili]*2*np.pi)

        ax.plot(time_array, speed_array/fillength_array, ls = 'solid', alpha=1., c=colors[ind])
        ax2.plot(time_array, dissipation_array/fillength_array**3, ls = 'solid', c=colors[ind])

        ax3.plot(time_array, efficiency_array, ls = 'dashed', c=colors[ind])
        ax4.plot(time_array, fillength_array, ls = 'dashed', c='black')
    except:
        pass

# plot extracted data
directory = 'pyfile/analysis/ishikawa_data/'
files = ['vel_k0.0N162.csv', 'vel_k0.0N636.csv', 'vel_k0.0N2520.csv']

# files = ['k0.0N162.csv', 'k0.0N636.csv', 'k0.0N2520.csv']
for i, filename in enumerate(files[:trunc]):
    file = open(directory + filename, mode='r')
    df = pd.read_csv(directory + filename, header=None)
    data = df.to_numpy()
    x, y = data[:,0], data[:,1]

    ax.plot(x, y, ls = 'dashed', alpha=0.5, c=colors[i])

directory = 'pyfile/analysis/ishikawa_data/'
files = ['dissipation_k0.0N162.csv', 'dissipation_k0.0N636.csv', 'dissipation_k0.0N2520.csv']

# files = ['k0.0N162.csv', 'k0.0N636.csv', 'k0.0N2520.csv']
for i, filename in enumerate(files[:trunc]):
    file = open(directory + filename, mode='r')
    df = pd.read_csv(directory + filename, header=None)
    data = df.to_numpy()
    x, y = data[:,0], data[:,1]

    ax2.plot(x, y, ls = 'dashed', alpha=0.5, c=colors[i])

legend11 = ax.legend(loc='center', frameon=False)
# line1, = ax.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$L=0.975$' )
# line2, = ax.plot([-1, -1.1], [-1, -1.1], ls='dashdot', c='black', label=r'$L=1$' )
line3, = ax.plot([-1, -1.1], [-1, -1.1], ls='solid', c='black', label=r'Present data' )
line4, = ax.plot([-1, -1.1], [-1, -1.1], ls='dashed', c='black', label=r'Omori et al. (2020)')
legend21 = ax.legend(handles = [line3, line4], loc='upper right', fontsize=18, frameon=False)
ax.add_artist(legend11)
ax.set_xlim(0, 1)
ax.set_ylim(-0.6)
ax.set_xlabel(r'$t/T$')
ax.set_ylabel(r'$VT/L$')

legend21 = ax2.legend(loc='center', frameon=False)
# line1, = ax2.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$L=0.975$' )
# line2, = ax2.plot([-1, -1.1], [-1, -1.1], ls='dashdot', c='black', label=r'$L=1$' )
line3, = ax2.plot([-1, -1.1], [-1, -1.1], ls='solid', c='black', label=r'Present data' )
line4, = ax2.plot([-1, -1.1], [-1, -1.1], ls='dashed', c='black', label=r'Omori et al. (2020)')
legend22 = ax2.legend(handles = [line3, line4], loc='upper right', fontsize=18, frameon=False)
ax2.add_artist(legend21)
ax2.set_xlim(0, 1)
ax2.set_ylim(0)
ax2.set_xlabel(r'$t/T$')
ax2.set_ylabel(r'$\mathcal{R}T^2/\eta L^3$')

ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.ticklabel_format(axis='y', style='sci', scilimits=(4,4))

line1, = ax4.plot([-1, -1.1], [-1, -1.1], ls='dashed', c='black', label=r'$<L>=1$' )
line2, = ax4.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$<L>=0.975$' )
legend42 = ax4.legend(handles = [line1, line2])
ax4.set_xlim(0, 1)
ax4.set_ylim(0.92, 1.06)
ax4.set_xlabel(r'$t/T$')
ax4.set_ylabel(r'$L$')

ax.set_box_aspect(0.7) 
ax2.set_box_aspect(0.7) 

fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig.savefig(f'fig/ishikawa_pnas_comparison_vel.png', bbox_inches = 'tight', format='png', transparent=True)
fig2.savefig(f'fig/ishikawa_pnas_comparison_dissipation.png', bbox_inches = 'tight', format='png', transparent=True)
fig2.savefig(f'fig/ishikawa_pnas_comparison_dissipation.pdf', bbox_inches = 'tight', format='pdf', transparent=True)
# fig4.savefig(f'fig/ishikawa_pnas_comparison_real_length.png', bbox_inches = 'tight', format='png', transparent=True)
plt.show()