
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update({'font.size': 16})

cmap_name = 'coolwarm'

path = "data/ishikawa/20240731_jfm/"

time_array_symplectic = np.load(f"{path}time_array_index0.npy")
r_array_symplectic = np.load(f"{path}r_array_index0.npy")

time_array_diaplectic = np.load(f"{path}time_array_index1.npy")
r_array_diaplectic = np.load(f"{path}r_array_index1.npy")


r_data = np.load(f"{path}r_data.npy")
k_data = np.load(f"{path}k_data.npy")
tilt_data = np.load(f"{path}tilt_data.npy")
avg_vz_data = np.load(f"{path}avg_vz_data.npy")
avg_speed_data = np.load(f"{path}avg_speed_data.npy")
avg_speed_along_axis_data = np.load(f"{path}avg_speed_along_axis_data.npy")
avg_rot_speed_data = np.load(f"{path}avg_rot_speed_data.npy")
avg_rot_speed_along_axis_data = np.load(f"{path}avg_rot_speed_along_axis_data.npy")

# n_folder_heldfixed = r_data_heldfixed.shape[0]
n_folder = r_data.shape[0]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)

# for fi in range(n_folder_heldfixed):
#     plot_x = k_data_heldfixed[fi] 
#     plot_y = r_data_heldfixed[fi]
#     indices_symplectic = np.where(plot_y > .4)[0]
#     indices_diaplectic = np.where((plot_y  < .4) & (plot_y > 0.04))[0]
#     indices_diaplectic_k2 = np.where(plot_y  < 0.04)[0]

#     ax.scatter(plot_x[indices_symplectic], plot_y[indices_symplectic], s=100, marker='x', c='r')
#     ax.scatter(plot_x[indices_diaplectic], plot_y[indices_diaplectic], s=100, marker='+', c='r')
#     ax.scatter(plot_x[indices_diaplectic_k2], plot_y[indices_diaplectic_k2], s=100, marker='P', c='r')



for fi in range(n_folder):
    plot_x = k_data[fi]
    plot_y = tilt_data[fi]

    cmap = plt.get_cmap(cmap_name)
    color_data = r_data[fi]
    color = cmap(color_data/max(color_data))
    indices_meridional = np.where(r_data[fi][:] > .4)
    indices_zonal = np.where(r_data[fi][:] < .4)

    # variable = avg_rot_speed_along_axis_data[fi]
    # variable_label = r"$<Ω⋅e_1>/L$"
    variable = avg_speed_along_axis_data[fi]
    variable_label = r"$<V⋅e_1>/L$"
    vmin2 = np.min(variable)
    vmax2 = np.max(variable)
    color2 = cmap((variable-vmin2)/(vmax2-vmin2))


    ax.scatter(plot_x, plot_y, s=100, marker='+', c=color)
    ax2.scatter(plot_x, plot_y, s=100, marker='+', c=color2)


    # ax3.scatter(tilt_data[fi][:][indices_meridional], variable[:][indices_meridional], color = 'r', label=r'$<r>$>0.4')
    # ax3.scatter(tilt_data[fi][:][indices_zonal], variable[:][indices_zonal], color = 'b', label=r'$<r>$<0.4')
    ax3.scatter(plot_x[:][indices_meridional], variable[:][indices_meridional], color = 'r', label=r'$<r>$>0.4')
    ax3.scatter(plot_x[:][indices_zonal], variable[:][indices_zonal], color = 'b', label=r'$<r>$<0.4')


    # indices_symplectic = np.where(plot_y > .4)[0]
    # indices_diaplectic = np.where(plot_y  < .4)[0]

    # ax.scatter(plot_x[indices_symplectic], plot_y[indices_symplectic], s=100, marker='x', c='b')
    # ax.scatter(plot_x[indices_diaplectic], plot_y[indices_diaplectic], s=100, marker='+', c='b')

# colorbar
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# vmin = 0
# vmax = np.max(r_data)
# norm = Normalize(vmin=vmin, vmax=vmax)
# sm = ScalarMappable(cmap=cmap_name, norm=norm)
# sm.set_array([])
# cbar = fig.colorbar(sm)
# # cbar.ax.set_yticks(np.linspace(vmin, vmax, 3), ['0', 'π/4', 'π/2'])
# cbar.set_label(r"<r>")    

norm = Normalize(vmin=vmin2, vmax=vmax2)
sm = ScalarMappable(cmap=cmap_name, norm=norm)
sm.set_array([])
cbar = fig2.colorbar(sm)
cbar.set_label(variable_label)   

# legend
# ax.scatter(-1, -1, marker='x', c='black', s=100, label='Symplectic')
# ax.scatter(-1, -1, marker='+', c='black', s=100, label='Diaplectic')
# ax.scatter(-1, -1, marker='P', c='black', s=100, label='Diaplectic(#k=2)')
# ax.scatter(-1, -1, marker='s', c='r', s=100, label='Held fixed')
# ax.scatter(-1, -1, marker='s', c='b', s=100, label='Free')

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'tilt angle')
# ax.set_ylim(0)
# ax.set_xlim(0, 0.06)
# ax.legend()

ax2.set_xlabel(r'$k$')
ax2.set_ylabel(r'tilt angle')

ax3.set_xlabel(r'tilt angle')
ax3.set_ylabel(variable_label)  
ax3.legend()

fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig.savefig(f'fig/order_parameter_tilt.pdf', bbox_inches = 'tight', format='pdf')
fig2.savefig(f'fig/avg_rot_speed_along_axis_data_tilt.pdf', bbox_inches = 'tight', format='pdf')
fig3.savefig(f'fig/rot_speed_vs_tilt.pdf', bbox_inches = 'tight', format='pdf')
plt.show()