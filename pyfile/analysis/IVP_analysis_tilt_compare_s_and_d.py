
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

path = "data/tilt_test/makeup_pattern/"


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
colors = ['r', 'b']
labels = ['Meridional MCW', 'Zonal MCW']

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


n_tilt = 5
tilt_angle = np.linspace(0, np.pi/4, n_tilt+1)[:-1]
tilt_angle = tilt_angle*180/np.pi


for fi in range(n_folder):

    
    avg_speed_over_angle = np.zeros(tilt_angle.shape)
    avg_rot_speed_over_angle = np.zeros(tilt_angle.shape)
    

    for ti in range(n_tilt):

        k = k_data[fi][ti::n_tilt]
        r = r_data[fi][ti::n_tilt]
        avg_speed = avg_speed_along_axis_data[fi][ti::n_tilt]
        avg_rot_speed = avg_rot_speed_along_axis_data[fi][ti::n_tilt]

        avg_speed_over_angle[ti] = np.mean(avg_speed)
        avg_rot_speed_over_angle[ti] = np.mean(avg_rot_speed)

    # cmap = plt.get_cmap(cmap_name)
    # color_data = r_data[fi]
    # color = cmap(color_data/max(color_data))
    # indices_meridional = np.where(r_data[fi][:] > .4)
    # indices_zonal = np.where(r_data[fi][:] < .4)

    # variable = avg_rot_speed_along_axis_data[fi]
    # variable_label = r"$<Ω⋅e_1>/L$"

    # variable = avg_speed_along_axis_data[fi]
    # variable_label = r"$<V⋅e_1>/L$"
    # vmin2 = np.min(variable)
    # vmax2 = np.max(variable)
    # color2 = cmap((variable-vmin2)/(vmax2-vmin2))


        ax.plot(k, avg_speed, marker='+', c=colors[fi])

        ax2.plot(k, avg_rot_speed, marker='+', c=colors[fi])

        ax5.scatter(k, r, marker='+', c=colors[fi])

    ax3.plot(tilt_angle, avg_speed_over_angle, marker='+', c=colors[fi], label=labels[fi])

    ax4.plot(tilt_angle, avg_rot_speed_over_angle, marker='+', c=colors[fi], label=labels[fi])

        
    
    # ax2.scatter(k, plot_y, s=100, marker='+', c=color2)


    # ax3.scatter(tilt_data[fi][:][indices_meridional], variable[:][indices_meridional], color = 'r', label=r'$<r>$>0.4')
    # ax3.scatter(tilt_data[fi][:][indices_zonal], variable[:][indices_zonal], color = 'b', label=r'$<r>$<0.4')
    # ax3.scatter(k[:][indices_meridional], variable[:][indices_meridional], color = 'r', label=r'$<r>$>0.4')
    # ax3.scatter(k[:][indices_zonal], variable[:][indices_zonal], color = 'b', label=r'$<r>$<0.4')


    # indices_symplectic = np.where(plot_y > .4)[0]
    # indices_diaplectic = np.where(plot_y  < .4)[0]

    # ax.scatter(k[indices_symplectic], plot_y[indices_symplectic], s=100, marker='x', c='b')
    # ax.scatter(k[indices_diaplectic], plot_y[indices_diaplectic], s=100, marker='+', c='b')

# colorbar
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# norm = Normalize(vmin=vmin2, vmax=vmax2)
# sm = ScalarMappable(cmap=cmap_name, norm=norm)
# sm.set_array([])
# cbar = fig2.colorbar(sm)
# cbar.set_label(variable_label)   

# legend
# ax.scatter(-1, -1, marker='x', c='black', s=100, label='Symplectic')
# ax.scatter(-1, -1, marker='+', c='black', s=100, label='Diaplectic')
# ax.scatter(-1, -1, marker='P', c='black', s=100, label='Diaplectic(#k=2)')
# ax.scatter(-1, -1, marker='s', c='r', s=100, label='Held fixed')
# ax.scatter(-1, -1, marker='s', c='b', s=100, label='Free')

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$\chi$')
# ax.set_ylim(0)
# ax.set_xlim(0, 0.06)
# ax.legend()

ax2.set_xlabel(r'$k$')
ax2.set_ylabel(r'$\chi$')

ax3.set_xlabel(r'$\chi$')
ax3.set_ylabel(r'$<V>T/L$')
ax3.legend()
# ax3.set_xticks(tilt_angle, ['0', 'π/20', '2π/20', '3π/20', '4π/20'])
ax3.set_xlim(tilt_angle[0], tilt_angle[-1])

ax4.set_xlabel(r'$\chi$')
ax4.set_ylabel(r'$<\Omega>$')
ax4.legend()
# ax4.set_xticks(tilt_angle, ['0', 'π/20', '2π/20', '3π/20', '4π/20'])
ax4.set_xlim(tilt_angle[0], tilt_angle[-1])

ax5.scatter(-.1, -.1, marker='+', c='red', label='Symplectic')
ax5.scatter(-.1, -.1, marker='+', c='blue', label='Diaplectic')
ax5.set_xlabel(r'$k$')
ax5.set_ylabel(r'$<r>$')
ax5.legend()
ax5.set_ylim(0)
ax5.set_xlim(0)

fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
# fig.savefig(f'fig/order_parameter_tilt.pdf', bbox_inches = 'tight', format='pdf')
# fig2.savefig(f'fig/avg_rot_speed_along_axis_data_tilt.pdf', bbox_inches = 'tight', format='pdf')
fig3.savefig(f'fig/tilt_speed_vs_tilt.pdf', bbox_inches = 'tight', format='pdf')
fig4.savefig(f'fig/tilt_rot_speed_vs_tilt.pdf', bbox_inches = 'tight', format='pdf')
fig5.savefig(f'fig/tilt_order_parameter.pdf', bbox_inches = 'tight', format='pdf')
plt.show()