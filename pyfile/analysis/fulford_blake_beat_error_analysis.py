import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.animation as animation
import configparser
from sklearn.cluster import KMeans
import time
import matplotlib as mpl
import os
from scipy.optimize import curve_fit
from scipy.integrate import quad

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

cmap_name = 'hsv'

# Fourier coeffs for the shape
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


A_y = np.array([[-3.3547e-01, 4.0369e-01, 1.0362e-01], \
            [4.0318e-01, -1.5553e+00, 7.3455e-01], \
            [-9.9513e-02, 3.2829e-02, -1.2106e-01], \
            [8.1046e-02, -3.0982e-01, 1.4568e-01]])

A_x = np.array([[9.7204e-01, -2.8315e-01, 4.9243e-02], \
            [-1.8466e-02, -1.2926e-01, 2.6981e-01], \
            [1.6209e-01, -3.4983e-01, 1.9082e-01], \
            [1.0259e-02, 3.5907e-02, -6.8736e-02]])

B_y = np.array([[0, 0, 0], \
            [2.9136e-01, 1.0721e+00, -1.0433e+00], \
            [6.1554e-03, 3.2521e-01, -2.8315e-01], \
            [-6.0528e-02, 2.3185e-01, -2.0108e-01]])

B_x = np.array([[0, 0, 0], \
            [1.9697e-01, -5.1193e-01, 3.4778e-01], \
            [-5.1295e-02, 4.3396e-01, -3.3547e-01], \
            [1.2311e-02, 1.4157e-01, -1.1695e-01]])

s_ref_filename = 'input/forcing/fulford_and_blake_reference_s_values_NSEG=20_SEP=2.600000.dat'
s_ref = np.loadtxt(s_ref_filename)
num_ref_phase = s_ref[0]
num_seg = int(s_ref[1])
num_frame = 20
radius = 1
L = (num_seg-1)*2.6
L = 1

def original_shape_s(phase):
    cycle = 0.5*phase/np.pi*num_ref_phase
    sfloor = int(np.floor(cycle))
    sceil = sfloor + 1 if sfloor < 299 else 0

    floor_w = (cycle - sfloor)
    ceil_w = (sceil - cycle) 

    s = s_ref[2:][num_seg*sfloor:num_seg*sfloor+num_seg]*floor_w + s_ref[2:][num_seg*sceil:num_seg*sceil+num_seg]*ceil_w

    return s

def original_shape(s, phase):
    svec = np.array([s, s**2, s**3])
    fourier_dim = np.shape(Ax)[0]
    cosvec = np.array([ np.cos(n*phase) for n in range(fourier_dim)])
    sinvec = np.array([ np.sin(n*phase) for n in range(fourier_dim)])
    cosvec[0] *= 0.5

    x = (cosvec@Ax + sinvec@Bx)@svec
    y = (cosvec@Ay + sinvec@By)@svec
    z = np.zeros(np.shape(x))

    return x, y, z

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

def ksi(s, A, B, phase):
    fourier_dim, degree = np.shape(A)
    svec = np.array([s**d for d in range(1, degree + 1)])

    cosvec = np.array([ np.cos(n*phase) for n in range(fourier_dim)])
    sinvec = np.array([ np.sin(n*phase) for n in range(fourier_dim)])

    return (cosvec@A + sinvec@B)@svec

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

total_lengths = np.zeros(num_frame)

for p in range(num_frame):
    phase = 2*np.pi/num_frame*p

    # color
    cmap = plt.get_cmap(cmap_name)
    fil_color = cmap(phase/(2*np.pi))

    # s for this phase
    s = np.linspace(0, L, num_seg)

    # calculate the real length
    total_lengths[p] = real_fillength(phase)

    # Plot fil line
    x_array, y_array, z_array = np.array(original_shape(s, phase))*L
    ax1.plot(y_array, x_array, c='black', alpha = 0.1+0.9*phase/(2*np.pi), zorder=1)

    # Add circles
    # for seg in range(num_seg):
    #     ax1.add_patch(plt.Circle((y_array[seg], x_array[seg]), 1, facecolor=fil_color, edgecolor=None, zorder=p, alpha = 1))

for p in range(num_frame):
    psi = 2*np.pi/num_frame*p
    s_value = np.linspace(0, L, num_seg)
    plot_x = ksi(s_value, A_x, B_x, psi)
    plot_y = ksi(s_value, A_y, B_y, psi)
    ax1.plot(plot_y, plot_x, c='blue')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_aspect('equal')
# ax1.axis('off')
ax1.set_xlim(-0.6, 0.8)
ax1.set_ylim(-0.0, 1.0)
legend1 = ax1.legend(loc='center', frameon=False)
line1, = ax1.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$<L>=0.97475$' )
line2, = ax1.plot([-1, -1.1], [-1, -1.1], ls='-', c='blue', label=r'$<L>=1$')
legend2 = ax1.legend(handles = [line1, line2], loc='upper right')
ax1.add_artist(legend1)


ax2.plot(np.linspace(0, 2*np.pi, num_frame+1)[:-1], total_lengths)
ax2.set_xlabel(r'$\psi$')
ax2.set_ylabel(r'$L/\psi$')
ax2.set_xlim(0, 2*np.pi)



plt.tight_layout()       
fig1.savefig(f'fig/fulford_blake_beat_error_anaysis.pdf', bbox_inches = 'tight', format='pdf')
fig1.savefig(f'fig/fulford_blake_beat_error_anaysis.png', bbox_inches = 'tight', format='png', transparent='True')
plt.show()