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
import util




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
num_frame = 8
num_angle = 2
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
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')

for p in range(num_frame):
    phase = 2*np.pi/num_frame*p

    # color
    cmap = plt.get_cmap(cmap_name)
    fil_color = cmap(phase/(2*np.pi))

    # s for this phase
    s = fitted_shape_s(phase)

    # Plot fil line
    x_array, y_array, z_array = np.array(fitted_shape(s, phase))*L
    # ax1.plot(y_array, x_array, color=fil_color, zorder=p)
    ax1.plot(y_array, x_array, c='black', alpha = 0.1+0.9*phase/(2*np.pi), zorder=1)


    # Add circles
    # for seg in range(num_seg):
    #     ax1.add_patch(plt.Circle((y_array[seg], x_array[seg]), 1, facecolor=fil_color, edgecolor=None, zorder=p, alpha = 1))

fil_angles = [0, np.pi/3.5]
angle_colors = ['black', 'r']
angle_axis_colors = ['black', 'r']
num_frame2 = 15
fil_data = np.zeros((num_seg, 3))

for p in range(num_frame):
    phase = 2*np.pi/num_frame*p

    # color
    cmap = plt.get_cmap(cmap_name)
    fil_color = cmap(phase/(2*np.pi))

    # s for this phase
    s = fitted_shape_s(phase)

    rotation_matrix = np.array([
            [np.cos(fil_angles[-1]), -np.sin(fil_angles[-1]), 0],
            [np.sin(fil_angles[-1]), np.cos(fil_angles[-1]), 0],
            [0, 0, 1]
        ])

    for seg in range(num_seg):
        seg_pos = np.array(fitted_shape(s[seg], phase))*L
        seg_pos = np.dot(rotation_matrix, seg_pos)
        fil_data[seg] = seg_pos
    ax2.plot(fil_data[:,1], fil_data[:,0], c='green', alpha = 0.1+0.9*phase/(2*np.pi), zorder=100, linewidth=3.0)

    for seg in range(num_seg):
        seg_pos = np.array(fitted_shape(s[seg], phase))*L
        fil_data[seg] = seg_pos
    ax2.plot(fil_data[:,1], fil_data[:,0], c='black', alpha = 0.1+0.9*phase/(2*np.pi), zorder=2, linewidth=1.0)

# Draw annotations
# origin = np.array([12, 2])
# axisl_x = 8
# axisl_y = 8
# ax1.arrow(origin[0], origin[1], axisl_x, 0.0, width=0.5, linewidth=0.2, color='black', zorder=1000)
# ax1.arrow(origin[0], origin[1], 0.0, axisl_y, width=0.5, linewidth=0.2, color='black', zorder=1000)
# ax1.annotate(r'$x$', origin + np.array([axisl_x+3, 0]), fontsize=25, va='center')
# ax1.annotate(r'$y$', origin + np.array([0, axisl_y+3.2]), fontsize=25, ha='center')

origin = np.array([0, 0])
axisl_x = 35
axisl_y = 53
# ax1.arrow(origin[0], origin[1], axisl_x, 0.0, width=0.5, linewidth=0.2, color='black', zorder=1000)

ax2.plot(axisl_y*np.array([0, np.sin(fil_angles[-1])]), axisl_y*np.array([0, np.cos(fil_angles[-1])]),\
         linestyle='dashed', c='green', zorder=1000, linewidth=1.0)
ax2.plot(axisl_y*np.array([0, 0]), axisl_y*np.array([0, 1]),\
         linestyle='dashed', c='black', zorder=1000, linewidth=1.0)

# # Draw x_b
# ax1.scatter(0, 0, s=60, c = 'black', zorder = 1000)
# ax1.annotate(r'$x_b$', (-8, 0.0), fontsize=25, va='center')

# ax2.scatter(0, 0, s=60, c = 'black', zorder = 1000)
# ax2.annotate(r'$x_b$', (-6, 0.0), fontsize=25, va='center')

# # Draw curved arrows
import matplotlib.patches as patches
import matplotlib.path as mpath
def rotation_2D(angle):
    return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

# # psi
start = np.array([-12.7, 45.4])
control1 = np.array([44, 47])
control2 = np.array([53, 25])
end = np.array([6, 36])

path_data = [
    (mpath.Path.MOVETO, start),
    (mpath.Path.CURVE4, control1),
    (mpath.Path.CURVE4, control2),
    (mpath.Path.CURVE4, end)
]
codes, verts = zip(*path_data)
path = mpath.Path(verts, codes)
arrow = patches.FancyArrowPatch(path=path,
                                arrowstyle='->',  # arrowstyle can be adjusted
                                color='black',
                                linewidth=1,       # Adjust thickness
                                mutation_scale=20)
ax1.add_patch(arrow)
ax1.annotate(r'$\psi_1(t)$', (5, 40.0), fontsize=25, va='center')

# # theta
start_t = np.array([0, 50])
control1_t = rotation_2D(-fil_angles[-1]/3.)@start_t
control2_t = rotation_2D(-fil_angles[-1]*2./3.)@start_t
end_t = rotation_2D(-fil_angles[-1])@start_t


path_data_t = [
    (mpath.Path.MOVETO, start_t),
    (mpath.Path.CURVE4, control1_t),
    (mpath.Path.CURVE4, control2_t),
    (mpath.Path.CURVE4, end_t)
]
codes, verts = zip(*path_data_t)
path = mpath.Path(verts, codes)
arrow = patches.FancyArrowPatch(path=path,
                                arrowstyle='->',  # arrowstyle can be adjusted
                                color='black',
                                linewidth=1,       # Adjust thickness
                                mutation_scale=20)
ax2.add_patch(arrow)
ax2.annotate(r'$\psi_2(t)$', (15, 50), fontsize=25, va='center')


# # plot rot_ref


rot_angle = np.pi*0.3
rot_axis = np.array([1,1,1])


offset = np.array([0, 0, 0])
radius = 30
xb = util.spherical_to_cartesian(30, 0, 2.5)
quaternion = util.point_to_quaternion(xb[0], xb[1], xb[2],)
rot_matrix = util.rot_mat(quaternion)

for p in range(num_frame):
    phase = 2*np.pi/num_frame*p
    fil_data = np.zeros((num_seg, 3))

    # s for this phase
    s = fitted_shape_s(phase)

    # alpha
    alpha=0.1 + 0.9*p/num_frame

    for seg in range(num_seg):
        seg_pos = np.array(fitted_shape(s[seg], phase))*L
        fil_data[seg] = seg_pos
    # ax3.plot(fil_data[:,2], fil_data[:,1], fil_data[:,0], c='black', alpha = 1.0, zorder = p)

    for seg in range(num_seg):
        seg_pos = np.array(fitted_shape(s[seg], phase))*L
        seg_pos = np.dot(rot_matrix, seg_pos) + offset + xb
        fil_data[seg] = seg_pos
    ax3.plot(fil_data[:,2], fil_data[:,1], fil_data[:,0], c='green', alpha = 1.0, zorder = p)

axis_length = 80
x_axis = np.array([[0, 0, 0], [1, 0, 0]])*axis_length
y_axis = np.array([[0, 0, 0], [0, 1, 0]])*axis_length
z_axis = np.array([[0, 0, 0], [0, 0, 1]])*axis_length

rotated_x_axis = np.array([np.dot(rot_matrix, x_axis[0]), np.dot(rot_matrix, x_axis[1])]) + offset + xb
rotated_y_axis = np.array([np.dot(rot_matrix, y_axis[0]), np.dot(rot_matrix, y_axis[1])]) + offset + xb
rotated_z_axis = np.array([np.dot(rot_matrix, z_axis[0]), np.dot(rot_matrix, z_axis[1])]) + offset + xb

ax3.plot(x_axis[:,2],x_axis[:,1],x_axis[:,0],\
         linestyle='dashed', c='black', zorder=1000, linewidth=1.0)
ax3.plot(y_axis[:,2],y_axis[:,1],y_axis[:,0],\
         linestyle='dashed', c='black', zorder=1000, linewidth=1.0)
ax3.plot(z_axis[:,2],z_axis[:,1],z_axis[:,0],\
         linestyle='dashed', c='black', zorder=1000, linewidth=1.0)

ax3.plot(rotated_x_axis[:,2],rotated_x_axis[:,1],rotated_x_axis[:,0],\
         linestyle='dashed', c='black', zorder=1000, linewidth=1.0)
ax3.plot(rotated_y_axis[:,2],rotated_y_axis[:,1],rotated_y_axis[:,0],\
         linestyle='dashed', c='black', zorder=1000, linewidth=1.0)
ax3.plot(rotated_z_axis[:,2],rotated_z_axis[:,1],rotated_z_axis[:,0],\
         linestyle='dashed', c='black', zorder=1000, linewidth=1.0)

resolution = 10
theta_range = [0, np.pi / 2]  # Theta range (0 to 90 degrees)
phi_range = [0, np.pi]    
theta = np.linspace(theta_range[0], theta_range[1], resolution)
phi = np.linspace(phi_range[0], phi_range[1], resolution)
theta, phi = np.meshgrid(theta, phi)

# Convert spherical coordinates to Cartesian coordinates
x = radius * np.sin(theta) * np.cos(phi) + offset[0]
y = radius * np.sin(theta) * np.sin(phi) + offset[1]
z = radius * np.cos(theta) + offset[2]

# Plot the surface
ax3.plot_surface(z, y, x, alpha=0.8, edgecolor='k')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_aspect('equal')
ax1.axis('off')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_aspect('equal')
ax2.axis('off')

ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.set_aspect('equal')
ax3.view_init(elev=15, azim=10) 

plt.tight_layout()       
fig1.savefig(f'fig/fulford_blake_beat_psi.png', bbox_inches = 'tight', format='png', transparent=True)
fig2.savefig(f'fig/fulford_blake_beat_theta.png', bbox_inches = 'tight', format='png', transparent=True)
# fig1.savefig(f'fig/fulford_blake_beat.png', bbox_inches = 'tight', format='png')
plt.show()