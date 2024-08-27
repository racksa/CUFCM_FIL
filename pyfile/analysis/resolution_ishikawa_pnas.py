
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import curve_fit

import configparser

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update({'font.size': 16})

cmap_name = 'coolwarm'

# Ay = np.array([[-0.654, 0.787, 0.202], \
#             [0.393, -1.516, 0.716], \
#             [-0.097, 0.032, -0.118], \
#             [0.079, -0.302, 0.142]])

# Ax = np.array([[1.895, -0.552, 0.096], \
#             [-0.018, -0.126, 0.263], \
#             [0.158, -0.341, 0.186], \
#             [0.010, 0.035, -0.067]])

# By = np.array([[0, 0, 0], \
#             [0.284, 1.045, -1.017], \
#             [0.006, 0.317, -0.276], \
#             [-0.059, 0.226, -0.196]])

# Bx = np.array([[0, 0, 0], \
#             [0.192, -0.499, 0.339], \
#             [-0.050, 0.423, -0.327], \
#             [0.012, 0.138, -0.114]])

# Ay2 = np.array([[-3.3547e-01, 4.0369e-01, 1.0362e-01], \
#             [4.0318e-01, -1.5553e+00, 7.3455e-01], \
#             [-9.9513e-02, 3.2829e-02, -1.2106e-01], \
#             [8.1046e-02, -3.0982e-01, 1.4568e-01]])

# Ax2 = np.array([[9.7204e-01, -2.8315e-01, 4.9243e-02], \
#             [-1.8466e-02, -1.2926e-01, 2.6981e-01], \
#             [1.6209e-01, -3.4983e-01, 1.9082e-01], \
#             [1.0259e-02, 3.5907e-02, -6.8736e-02]])

# By2 = np.array([[0, 0, 0], \
#             [2.9136e-01, 1.0721e+00, -1.0433e+00], \
#             [6.1554e-03, 3.2521e-01, -2.8315e-01], \
#             [-6.0528e-02, 2.3185e-01, -2.0108e-01]])

# Bx2 = np.array([[0, 0, 0], \
#             [1.9697e-01, -5.1193e-01, 3.4778e-01], \
#             [-5.1295e-02, 4.3396e-01, -3.3547e-01], \
#             [1.2311e-02, 1.4157e-01, -1.1695e-01]])

# def original_shape_derivative(s, phase):
#     svec = np.array([1, 2*s, 3*s**2])
#     fourier_dim = np.shape(Ax)[0]
#     cosvec = np.array([np.cos(n*phase) for n in range(fourier_dim)])
#     sinvec = np.array([np.sin(n*phase) for n in range(fourier_dim)])
#     cosvec[0] *= 0.5

#     dx_ds = (cosvec @ Ax + sinvec @ Bx) @ svec
#     dy_ds = (cosvec @ Ay + sinvec @ By) @ svec
#     dz_ds = np.zeros(np.shape(dx_ds))

#     return dx_ds, dy_ds, dz_ds

# def integrand(s, phase):
#     dx_ds, dy_ds, dz_ds = original_shape_derivative(s, phase)
#     return np.sqrt(dx_ds**2 + dy_ds**2 + dz_ds**2)

# def real_fillength(phase):
#     arc_length, _ = quad(integrand, 0, 1, args=(phase))
#     return arc_length

# def original_shape_derivative2(s, phase):
#     svec = np.array([1, 2*s, 3*s**2])
#     fourier_dim = np.shape(Ax2)[0]
#     cosvec = np.array([np.cos(n*phase) for n in range(fourier_dim)])
#     sinvec = np.array([np.sin(n*phase) for n in range(fourier_dim)])

#     dx_ds = (cosvec @ Ax2 + sinvec @ Bx2) @ svec
#     dy_ds = (cosvec @ Ay2 + sinvec @ By2) @ svec
#     dz_ds = np.zeros(np.shape(dx_ds))

#     return dx_ds, dy_ds, dz_ds

# def integrand2(s, phase):
#     dx_ds, dy_ds, dz_ds = original_shape_derivative2(s, phase)
#     return np.sqrt(dx_ds**2 + dy_ds**2 + dz_ds**2)

# def real_fillength2(phase):
#     arc_length, _ = quad(integrand2, 0, 1, args=(phase))
#     return arc_length

def fit_func(x, a, b, c):
    return a/(1+b*x)+c
    return a*(1/x**b) +c
    return a*np.exp(b*x) + c


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)
fig5 = plt.figure()
ax5 = fig5.add_subplot(1,1,1)
fig6 = plt.figure()
ax6 = fig6.add_subplot(1,1,1)

# k_list = [-1, 0, 0.5, 1.0, 1.5, 2.0]
# labels = [r"$k=-1$",r"$k=0$",r"$k=0.5$",r"$k=1$",r"$k=1.5$",r"$k=2$",]
colors = ["black","red","green","blue","purple", "pink", "brown", "cyan", "orange"]
N_list = [160, 640, 2560]




path = 'data/ishikawa/20240807_ishikawa_resolution5/'
path = 'data/ishikawa/20240822_ishikawa_resolution6/'
sim = configparser.ConfigParser()
sim.read(path+"rules.ini")
nblob_list = np.array([float(s) for s in sim["Parameter list"]['nblob'].split(', ')])
avg_speed_list = np.zeros(nblob_list.shape)
speed_error_list = np.zeros(nblob_list.shape)
dissipation_error_list = np.zeros(nblob_list.shape)
nsim = len(nblob_list)

ref_index = nsim - 4

ref_speed_array = np.load(f"{path}body_speed_array_index{ref_index}.npy")
ref_dissipation_array = np.load(f"{path}dissipation_array_index{ref_index}.npy")

# plot sim data
for ind in range(nsim):
    try:
        time_array = np.load(f"{path}time_array_index{ind}.npy")
        speed_array = np.load(f"{path}body_speed_array_index{ind}.npy")
        rot_speed_array = np.load(f"{path}body_rot_speed_array_index{ind}.npy")
        dissipation_array = np.load(f"{path}dissipation_array_index{ind}.npy")
        efficiency_array = np.load(f"{path}efficiency_array_index{ind}.npy")

        avg_speed_list[ind] = np.mean(np.abs(speed_array))
        # speed_error_list[ind] = np.mean(np.square((speed_array - ref_speed_array)/ref_speed_array))**.5
        speed_error_list[ind] = np.mean(np.abs(speed_array - ref_speed_array)/np.abs(ref_speed_array))
        
        dissipation_error_list[ind] = np.mean(np.abs(dissipation_array - ref_dissipation_array)/np.abs(ref_dissipation_array))

        # fillength_array = np.zeros(np.shape(time_array))
        # for fili in range(len(fillength_array)):
        #     fillength_array[fili] = real_fillength(time_array[fili]*2*np.pi)

        ax.plot(time_array, speed_array, label = rf'$M=$', alpha=1.)
        ax2.plot(time_array, dissipation_array, label = rf'$M=$')
        # ax2.plot(time_array, dissipation_array/fillength_array**3, label = rf'$M={N_list[ind]}$', c=colors[ind])
        ax3.plot(time_array, efficiency_array, label = rf'$M=$')
        
    except:
        pass



# plot extracted data
directory = 'pyfile/analysis/ishikawa_data/'
files = ['vel_k0.0N162.csv', 'vel_k0.0N636.csv', 'vel_k0.0N2520.csv']

for i, filename in enumerate(files):
    file = open(directory + filename, mode='r')
    df = pd.read_csv(directory + filename, header=None)
    data = df.to_numpy()
    x, y = data[:,0], data[:,1]

    ax.plot(x, y, ls = 'dotted', alpha=0.5, c=colors[i])

directory = 'pyfile/analysis/ishikawa_data/'
files = ['dissipation_k0.0N162.csv', 'dissipation_k0.0N636.csv', 'dissipation_k0.0N2520.csv']

for i, filename in enumerate(files):
    file = open(directory + filename, mode='r')
    df = pd.read_csv(directory + filename, header=None)
    data = df.to_numpy()
    x, y = data[:,0], data[:,1]

    ax2.plot(x, y, ls = 'dotted', alpha=0.5, c=colors[i])

# legend11 = ax.legend(loc='center', frameon=False)
# line1, = ax.plot([-1, -1.1], [-1, -1.1], ls='dashed', c='black', label=r'$<L>=1$' )
# line2, = ax.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$<L>=0.975$' )
# line3, = ax.plot([-1, -1.1], [-1, -1.1], ls='dotted', c='black', label=r'$Omori.\ (2020)$')
# legend21 = ax.legend(handles = [line1, line2, line3], loc='upper right')
# ax.add_artist(legend11)
ax.legend()
ax.set_xlim(0, 1)
ax.set_xlabel(r'$t/T$')
ax.set_ylabel(r'$V_z/L$')

# legend21 = ax2.legend(loc='center', frameon=False)
# line1, = ax2.plot([-1, -1.1], [-1, -1.1], ls='dashed', c='black', label=r'$<L>=1$' )
# line2, = ax2.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$<L>=0.975$' )
# line3, = ax2.plot([-1, -1.1], [-1, -1.1], ls='dotted', c='black', label=r'$Omori.\ (2020)$')
# legend22 = ax2.legend(handles = [line1, line2, line3], loc='upper right')
# ax2.add_artist(legend21)
ax2.set_xlim(0, 1)
ax2.set_xlabel(r'$t/T$')
ax2.set_ylabel(r'$PT^2/\mu L^3$')

# line1, = ax4.plot([-1, -1.1], [-1, -1.1], ls='dashed', c='black', label=r'$<L>=1$' )
# line2, = ax4.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$<L>=0.975$' )
# legend42 = ax4.legend(handles = [line1, line2, line3])
# ax4.set_xlim(0, 1)
# ax4.set_ylim(0.92, 1.06)

# cutoff = 2
# popt, pcov = curve_fit(fit_func, nblob_list[cutoff:], avg_speed_list[cutoff:])
# a, b, c = popt
# print(f"Fitted parameters: a = {a}, b = {b}, c = {c}")
# ax4.plot(nblob_list[cutoff:], fit_func(nblob_list[cutoff:], a, b, c), marker = '+')

ax4.plot(nblob_list, avg_speed_list, marker = '+')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel(r'$Number\ of\ rigid\ blobs$')
ax4.set_ylabel(r'$V$')
# ax4.set_ylabel(r'$\sqrt{\frac{|V-V_{ref}|^2}{|V_{ref}|^2}}$')



ax5.plot(nblob_list, speed_error_list, marker = '+', color='black')
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlabel(r'$Number\ of\ rigid\ blobs$')
ax5.set_ylabel(r'$\% error\ in\ swimming\ speed$')

ax6.plot(nblob_list, dissipation_error_list, marker = '+', color='black')
ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.set_xlabel(r'$Number\ of\ rigid\ blobs$')
ax6.set_ylabel(r'$\% error\ in\ dissipation$')


fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()

fig5.savefig(f'fig/resolution_ishikawa.pdf', bbox_inches = 'tight', format='pdf')
fig6.savefig(f'fig/resolution_ishikawa_dissipation.pdf', bbox_inches = 'tight', format='pdf')
plt.show()