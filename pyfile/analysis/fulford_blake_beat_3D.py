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
import matplotlib.font_manager as fm

try:
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
except:    
    print("WARNING: CMU font not found. Using default font.")


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
num_frame = 16
num_angle = 3
num_points = 30 # blob surface points
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


# Plotting
fig = plt.figure(dpi=200)
ax = fig.add_subplot(projection='3d')
ax.set_proj_type('ortho')
fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.set_proj_type('ortho')


u = np.linspace(0, 2 * np.pi, num_points)
v = np.linspace(0, np.pi, num_points)
x = radius * np.outer(np.cos(u), np.sin(v))
y = radius * np.outer(np.sin(u), np.sin(v))
z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

fil_phases = np.linspace(0, 2*np.pi, num_frame+1)[:-1]
fil_angles = np.linspace(-np.pi/6, 0/6, num_angle)
fil_angles = [-np.pi/12, 0, np.pi/4]


for i in range(num_frame):
    fil_data = np.zeros((num_seg, 3))

    # color
    cmap = plt.get_cmap(cmap_name)
    fil_color = cmap(fil_phases[i]/(2*np.pi))

    # s for this phase
    s = fitted_shape_s(fil_phases[i])

    # alpha
    alpha=0.1 + 0.9*i/num_frame

    for seg in range(num_seg):
        seg_pos = np.array(fitted_shape(s[seg], fil_phases[i]))*L
        fil_data[seg] = seg_pos
        ax.plot_surface(x+seg_pos[0], y+seg_pos[1], z+seg_pos[2], color=fil_color, alpha=0.5, zorder = i*1000)

    ax.plot(fil_data[:,0], fil_data[:,1], fil_data[:,2], c=fil_color, alpha = 1.0, zorder = i)

colors = ['r', 'g', 'b']
for j in range(num_angle):
    fil_data = np.zeros((num_seg, 3))
    normal = np.zeros((2,3))
    tangent = np.zeros((2,3))
    edge = np.zeros((5,3))
    phase = 0*np.pi/16

    # color
    cmap = plt.get_cmap(cmap_name)
    fil_color = cmap(phase/(2*np.pi))

    # s for this phase
    s = fitted_shape_s(phase)

    # rotation angle
    rotation_matrix = np.array([
        [np.cos(fil_angles[j]), -np.sin(fil_angles[j]), 0],
        [np.sin(fil_angles[j]), np.cos(fil_angles[j]), 0],
        [0, 0, 1]
    ])

    for seg in range(num_seg):
        seg_pos = np.array(fitted_shape(s[seg], phase))*L
        seg_pos = np.dot(rotation_matrix, seg_pos)
        fil_data[seg] = seg_pos
        # ax2.plot_surface(x+seg_pos[0], y+seg_pos[1], z+seg_pos[2], color=fil_color, alpha=0.5)

    ax2.plot(fil_data[:,0], fil_data[:,1], fil_data[:,2], c=fil_color, zorder = 100, alpha = 1.0)

    edge_l = 20
    edge_h = 60
    
    edge[0] = np.dot(rotation_matrix, np.array([0,-edge_l,0]))
    edge[1] = np.dot(rotation_matrix, np.array([0,edge_l,0]))
    edge[2] = np.dot(rotation_matrix, np.array([edge_h,edge_l,0]))
    edge[3] = np.dot(rotation_matrix, np.array([edge_h,-edge_l,0]))
    edge[4] = np.dot(rotation_matrix, np.array([0,-edge_l,0]))

    normal[1] = np.dot(rotation_matrix, np.array([edge_l,0,0]))
    tangent[1] = np.dot(rotation_matrix, np.array([0,edge_l,0]))

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
vmin = 0
vmax = 2*np.pi
norm = Normalize(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=cmap_name, norm=norm)
sm.set_array([])
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
cbar =plt.colorbar(sm, cax=cax)
cbar.ax.set_yticks(np.linspace(vmin, vmax, 7), [r'$0$', r'$\pi/3$', r'$2\pi/3$', r'$\pi$', r'$4\pi/3$', r'$5\pi/3$', r'$2\pi$'])
cbar.set_label(r"$\psi_1$")

ax2.set_ylim(-30, 30)
ax2.set_xlim(0, 60)

ax.axis('off')
ax.set_aspect('equal')
ax.view_init(elev=100000., azim=0)
ax.dist=7
fig.tight_layout()
fig.savefig(f'fig/single_fil.png', bbox_inches = 'tight', format='png', transparent=True)

ax2.axis('off')
ax2.set_aspect('equal')
ax2.view_init(elev=100000., azim=0)
fig2.tight_layout()
            
plt.show()
