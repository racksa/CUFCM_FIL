
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update({'font.size': 24})

cmap_name = 'bwr'

path = "data/ic_hpc_sim/20240311_8/"

time_array = np.load(f"{path}time_array_index0.npy")
r_array = np.load(f"{path}r_array_index0.npy")

stamp_index = np.array([0, 85, 350, 584, 905])
stamp_x = time_array[stamp_index]
stamp_y = r_array[stamp_index]

fig = plt.figure(figsize=(6,3), dpi=200)
ax = fig.add_subplot(1,1,1)

ax.plot(time_array, r_array, c='black')
ax.scatter(stamp_x, stamp_y, marker='^', s= 100, color='black')
ax.set_xticks(np.linspace(0, 40, 5))
ax.set_yticks(np.linspace(0, 0.7, 3))

for si, x in enumerate(stamp_x):
    ax.annotate(rf'{si+1}', (stamp_x[si]-1, stamp_y[si]+0.1), fontsize=25, va='center')


# for fi in range(n_folder_free):
#     plot_x = k_data_free[fi]
#     plot_y = r_data_free[fi]

#     cmap = plt.get_cmap(cmap_name)
#     color = cmap(tilt_data_free[fi]/max(tilt_data_free[fi]))
    
#     print(min(plot_x), max(plot_x))
#     ax.scatter(plot_x, plot_y, s=100, marker='+', c=color)

    # indices_symplectic = np.where(plot_y > .4)[0]
    # indices_diaplectic = np.where(plot_y  < .4)[0]

    # ax.scatter(plot_x[indices_symplectic], plot_y[indices_symplectic], s=100, marker='x', c='b')
    # ax.scatter(plot_x[indices_diaplectic], plot_y[indices_diaplectic], s=100, marker='+', c='b')

# colorbar
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
# vmin = 0
# vmax = np.max(tilt_data_free)
# norm = Normalize(vmin=vmin, vmax=vmax)
# sm = ScalarMappable(cmap=cmap_name, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm)
# cbar.ax.set_yticks(np.linspace(vmin, vmax, 3), ['0', 'π/4', 'π/2'])
# cbar.set_label(r"Tilt angle")    

# ax.scatter(-1, -1, marker='+', c='r', label='Held fixed - Symplectic')
# ax.scatter(-1, -1, marker='x', c='r', label='Held fixed - Diaplectic')
# ax.scatter(-1, -1, marker='X', c='r', label='Held fixed - Diaplectic(#k=2)')
# ax.scatter(-1, -1, marker='+', c='b', label='Free - Symplectic')
# ax.scatter(-1, -1, marker='x', c='b', label='Free - Diaplectic')

ax.set_xlabel(r'$t/T_0$')
ax.set_ylabel(r'$r$')
ax.set_ylim(0,0.7)
# ax.set_xlim(0, 0.06)
# ax.legend()

plt.tight_layout()
fig.savefig(f'fig/order_parameter_roadmap.pdf', bbox_inches = 'tight', format='pdf')
fig.savefig(f'fig/order_parameter_roadmap.png', bbox_inches = 'tight', format='png', transparent=True)
plt.show()