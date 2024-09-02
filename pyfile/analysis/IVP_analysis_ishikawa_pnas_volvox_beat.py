
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update({'font.size': 16})

cmap_name = 'coolwarm'

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

def calculate_enclosed_areas(x, y):
    """
    Calculate the areas enclosed by the curve and the x-axis
    separately for the segments where the curve is above (y > 0)
    and below (y < 0) the x-axis.

    Parameters:
    x (np.ndarray): 1-D array of x-coordinates.
    y (np.ndarray): 1-D array of corresponding y = f(x) values.

    Returns:
    tuple: A tuple containing two elements:
        - area_above (float): Total area where y > 0.
        - area_below (float): Total area where y < 0.
    """

    # Initialize areas
    area_above = 0
    area_below = 0

    # Find indices where sign changes or at boundaries
    sign_changes = np.where(np.diff(np.sign(y)))[0] + 1
    segments = np.split(x, sign_changes)  # Split x array at sign changes
    y_segments = np.split(y, sign_changes)  # Split y array at sign changes

    for x_seg, y_seg in zip(segments, y_segments):
        if np.all(y_seg > 0):  # If the entire segment is above x-axis
            area_above += np.trapz(y_seg, x_seg)
        elif np.all(y_seg < 0):  # If the entire segment is below x-axis
            area_below += np.trapz(y_seg, x_seg)
        else:
            # Handle mixed segments with interpolation
            # Find zero crossing using linear interpolation
            for i in range(len(y_seg) - 1):
                if y_seg[i] * y_seg[i+1] < 0:  # Sign change between points
                    # Linear interpolation to find the x-coordinate of zero crossing
                    x_zero = x_seg[i] + (x_seg[i+1] - x_seg[i]) * (-y_seg[i]) / (y_seg[i+1] - y_seg[i])
                    y_zero = 0  # At zero crossing

                    # Split the segment at the zero crossing
                    x_seg_above = np.append(x_seg[:i+1], x_zero)
                    y_seg_above = np.append(y_seg[:i+1], y_zero)
                    x_seg_below = np.insert(x_seg[i+1:], 0, x_zero)
                    y_seg_below = np.insert(y_seg[i+1:], 0, y_zero)

                    # Compute the areas separately for above and below
                    if np.any(y_seg_above > 0):
                        area_above += np.trapz(y_seg_above[y_seg_above > 0], x_seg_above[y_seg_above > 0])
                    if np.any(y_seg_below < 0):
                        area_below += np.trapz(y_seg_below[y_seg_below < 0], x_seg_below[y_seg_below < 0])

    return area_above, area_below


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
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
avg_speed_original_list = np.zeros(len(N_list))
avg_speed_volvox_list = np.zeros(len(N_list))


path = 'data/ishikawa/20240829_pnas_volvox_beat/'
# plot sim data
for ind in range(2,3):
    try:
        time_array = np.load(f"{path}time_array_index{ind}.npy")
        speed_array = np.load(f"{path}body_speed_array_index{ind}.npy")
        rot_speed_array = np.load(f"{path}body_rot_speed_array_index{ind}.npy")
        dissipation_array = np.load(f"{path}dissipation_array_index{ind}.npy")
        efficiency_array = np.load(f"{path}efficiency_array_index{ind}.npy")

        fillength_array = np.zeros(np.shape(time_array))
        for fili in range(len(fillength_array)):
            fillength_array[fili] = real_fillength(time_array[fili]*2*np.pi)

        area_above, area_below = calculate_enclosed_areas(time_array, speed_array)
        print(f"Area above the x-axis (y > 0): {area_above}")
        print(f"Area below the x-axis (y < 0): {area_below}")
        ax.fill_between(time_array, speed_array, where=(speed_array > 0), interpolate=False, color=colors[ind], facecolor=None, hatch='/', alpha=0.3, label=f'Forward')
        ax.fill_between(time_array, speed_array, where=(speed_array < 0), interpolate=False, color=colors[ind], facecolor='b', hatch='\\', alpha=0.3, label=f'Backward')
        ax.annotate(rf'$A_1={area_above:.3f}$', (0.13, 0.25))
        ax.annotate(rf'$A_2={-area_below:.3f}$', (0.63, -0.2))

        ax.plot(time_array, np.zeros(time_array.shape), alpha=1., c='black')
        ax.plot(time_array, speed_array, alpha=1., c=colors[ind])
        # ax.plot(time_array, np.ones(time_array.shape)*np.mean(speed_array), alpha=1., c=colors[ind])
        avg_speed_volvox_list[ind] = np.mean(speed_array)

        # ax2.plot(time_array, dissipation_array, label = rf'$M={N_list[ind]}$', c=colors[ind])
        # ax2.plot(time_array, dissipation_array/fillength_array**3, label = rf'$M={N_list[ind]}$', c=colors[ind])
        # ax4.plot(time_array, fillength_array, label = N_list[ind], c='black')
    except:
        pass


# plot sim data
path = "data/ishikawa/20240802_pnas_L0.975/"
# path = "data/ishikawa/20240805_volvox_beat/"
for ind in range(2,3):
    try:
        time_array = np.load(f"{path}time_array_index{ind}.npy")
        speed_array = np.load(f"{path}body_speed_array_index{ind}.npy")
        rot_speed_array = np.load(f"{path}body_rot_speed_array_index{ind}.npy")
        dissipation_array = np.load(f"{path}dissipation_array_index{ind}.npy")
        efficiency_array = np.load(f"{path}efficiency_array_index{ind}.npy")

        fillength_array = np.zeros(np.shape(time_array))
        for fili in range(len(fillength_array)):
            fillength_array[fili] = real_fillength2(time_array[fili]*2*np.pi)

        area_above, area_below = calculate_enclosed_areas(time_array, speed_array)
        print(f"Area above the x-axis (y > 0): {area_above}")
        print(f"Area below the x-axis (y < 0): {area_below}")
        ax1.fill_between(time_array, speed_array, where=(speed_array > 0), interpolate=False, color=colors[ind], facecolor=None, hatch='/', alpha=0.3, label=f'Forward')
        ax1.fill_between(time_array, speed_array, where=(speed_array < 0), interpolate=False, color=colors[ind], facecolor='b', hatch='\\', alpha=0.3, label=f'Backward')
        ax1.annotate(rf'$A_1={area_above:.3f}$', (0.0, 0.5))
        ax1.annotate(rf'$A_2={-area_below:.3f}$', (0.5, -0.2))


        ax1.plot(time_array, np.zeros(time_array.shape), alpha=1., c='black')
        ax1.plot(time_array, speed_array, ls = 'dashed', alpha=1., c=colors[ind])
        # ax.plot(time_array, np.ones(time_array.shape)*np.mean(speed_array), ls = 'dashed', alpha=1., c=colors[ind])
        avg_speed_original_list[ind] = np.mean(speed_array)

        # ax2.plot(time_array, dissipation_array, ls = 'dashed', c=colors[ind])
        # ax2.plot(time_array, dissipation_array/fillength_array**3, ls = 'dashed', c=colors[ind])
        # ax4.plot(time_array, fillength_array, ls = 'dashed', c='black')
    except:
        pass



# plot extracted data
# directory = 'pyfile/analysis/ishikawa_data/'
# files = ['vel_k0.0N162.csv', 'vel_k0.0N636.csv', 'vel_k0.0N2520.csv']

# # files = ['k0.0N162.csv', 'k0.0N636.csv', 'k0.0N2520.csv']
# for i, filename in enumerate(files):
#     file = open(directory + filename, mode='r')
#     df = pd.read_csv(directory + filename, header=None)
#     data = df.to_numpy()
#     x, y = data[:,0], data[:,1]

#     ax.plot(x, y, ls = 'dotted', alpha=0.5, c=colors[i])

# directory = 'pyfile/analysis/ishikawa_data/'
# files = ['dissipation_k0.0N162.csv', 'dissipation_k0.0N636.csv', 'dissipation_k0.0N2520.csv']

# # files = ['k0.0N162.csv', 'k0.0N636.csv', 'k0.0N2520.csv']
# for i, filename in enumerate(files):
#     file = open(directory + filename, mode='r')
#     df = pd.read_csv(directory + filename, header=None)
#     data = df.to_numpy()
#     x, y = data[:,0], data[:,1]

#     ax2.plot(x, y, ls = 'dotted', alpha=0.5, c=colors[i])

legend11 = ax.legend(frameon=False)
# line1, = ax.plot([-1, -1.1], [-1, -1.1], ls='dashed', c='black', label=r'$<L>=1$' )
# line2, = ax.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$<L>=0.975$' )
# line3, = ax.plot([-1, -1.1], [-1, -1.1], ls='dotted', c='black', label=r'$Omori.\ (2020)$')
# legend21 = ax.legend(handles = [line1, line2, line3], loc='upper right')
ax.add_artist(legend11)
ax.set_xlim(0, 1)
ax.set_xlabel(r'$t/T$')
ax.set_ylabel(r'$V_z/L$')

legend11 = ax1.legend(frameon=False)
# line1, = ax1.plot([-1, -1.1], [-1, -1.1], ls='dashed', c='black', label=r'$<L>=1$' )
# line2, = ax1.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$<L>=0.975$' )
# line3, = ax1.plot([-1, -1.1], [-1, -1.1], ls='dotted', c='black', label=r'$Omori.\ (2020)$')
# legend21 = ax1.legend(handles = [line1, line2, line3], loc='upper right')
ax1.add_artist(legend11)
ax1.set_xlim(0, 1)
ax1.set_xlabel(r'$t/T$')
ax1.set_ylabel(r'$V_z/L$')


# legend21 = ax2.legend(loc='center', frameon=False)
line1, = ax2.plot([-1, -1.1], [-1, -1.1], ls='dashed', c='black', label=r'$<L>=1$' )
line2, = ax2.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$<L>=0.975$' )
line3, = ax2.plot([-1, -1.1], [-1, -1.1], ls='dotted', c='black', label=r'$Omori.\ (2020)$')
legend22 = ax2.legend(handles = [line1, line2, line3], loc='upper right')
# ax2.add_artist(legend21)
ax2.set_xlim(0, 1)
ax2.set_xlabel(r'$t/T$')
ax2.set_ylabel(r'$PT^2/\mu L^3$')

ax3.plot(N_list, avg_speed_original_list,  c='black', label='Lung cilia' )
ax3.plot(N_list, avg_speed_volvox_list, ls='dashed', c='black', label='Volvox cilia')
ax3.legend()
ax3.set_xlabel(r'$M$')
ax3.set_ylabel(r'$V/L$')


line1, = ax4.plot([-1, -1.1], [-1, -1.1], ls='dashed', c='black', label=r'$<L>=1$' )
line2, = ax4.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$<L>=0.975$' )
legend42 = ax4.legend(handles = [line1, line2])
ax4.set_xlim(0, 1)
ax4.set_ylim(0.92, 1.06)
ax4.set_xlabel(r'$t/T$')
ax4.set_ylabel(r'$L$')

fig.tight_layout()
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig.savefig(f'fig/ishikawa_pnas_volvox_vel_area.pdf', bbox_inches = 'tight', format='pdf')
fig1.savefig(f'fig/ishikawa_pnas_original_vel_area.pdf', bbox_inches = 'tight', format='pdf')
# fig2.savefig(f'fig/ishikawa_pnas_comparison_dissipation.pdf', bbox_inches = 'tight', format='pdf')
fig3.savefig(f'fig/ishikawa_pnas_volvox_speed.pdf', bbox_inches = 'tight', format='pdf')
# fig4.savefig(f'fig/ishikawa_pnas_comparison_real_length.pdf', bbox_inches = 'tight', format='pdf')
plt.show()