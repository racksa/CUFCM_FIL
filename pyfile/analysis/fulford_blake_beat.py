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

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

cmap_name = 'hsv'

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

s_ref_filename = 'input/forcing/fulford_and_blake_reference_s_values_NSEG=20_SEP=2.600000.dat'
s_ref = np.loadtxt(s_ref_filename)
num_ref_phase = s_ref[0]
num_seg = int(s_ref[1])
num_frame = 30
radius = 1
L = (num_seg-1)*2.6

def fitted_shape_s(phase):
    cycle = 0.5*phase/np.pi*num_ref_phase
    sfloor = int(np.floor(cycle))
    sceil = sfloor + 1 if sfloor < 299 else 0

    floor_w = (cycle - sfloor)
    ceil_w = (sceil - cycle) 

    s = s_ref[2:][num_seg*sfloor:num_seg*sfloor+num_seg]*floor_w + s_ref[2:][num_seg*sceil:num_seg*sceil+num_seg]*ceil_w

    return s

def fitted_shape(s, phase):
    pos = np.zeros(3)
    svec = np.array([s, s**2, s**3])
    fourier_dim = np.shape(Ax)[0]
    cosvec = np.array([ np.cos(n*phase) for n in range(fourier_dim)])
    sinvec = np.array([ np.sin(n*phase) for n in range(fourier_dim)])

    x = (cosvec@Ax + sinvec@Bx)@svec
    y = (cosvec@Ay + sinvec@By)@svec
    z = np.zeros(np.shape(x))

    return x, y, z



fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

for p in range(num_frame):
    phase = 2*np.pi/num_frame*p

    # color
    cmap = plt.get_cmap(cmap_name)
    fil_color = cmap(phase/(2*np.pi))

    # s for this phase
    s = fitted_shape_s(phase)

    # Plot fil line
    x_array, y_array, z_array = np.array(fitted_shape(s, phase))*L
    ax1.plot(y_array, x_array, color=fil_color, zorder=p)

    # Add circles
    # for seg in range(num_seg):
    #     ax1.add_patch(plt.Circle((y_array[seg], x_array[seg]), 1, facecolor=fil_color, edgecolor=None, zorder=p, alpha = 0.3))

origin = np.array([12, 2])
axisl = 8
ax1.scatter(0, 0, s=60, c = 'black', zorder = 1000)
ax1.arrow(origin[0], origin[1], axisl, 0.0, width=0.2, linewidth=0.2, color='black')
ax1.arrow(origin[0], origin[1], 0.0, axisl, width=0.2, linewidth=0.2, color='black' )
ax1.annotate(r'$x$', origin + np.array([axisl+3, 0]), fontsize=25, va='center')
ax1.annotate(r'$y$', origin + np.array([0, axisl+3.2]), fontsize=25, ha='center')
ax1.annotate(r'$x_b$', (1.5, 0.0), fontsize=25, va='center')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_aspect('equal')
# ax1.set_ylim(0)
ax1.axis('off')
fig1.savefig(f'fig/fulford_blake_beat.pdf', bbox_inches = 'tight', format='pdf')
fig1.savefig(f'fig/fulford_blake_beat.png', bbox_inches = 'tight', format='png')
plt.show()