
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
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

path = "data/ic_hpc_sim_free/20240311_1/"
path = "data/ic_hpc_sim_free_with_force/20240311_1/"

indices = [0, 1]

index = 1
index2 = 0
force = True


r_data = np.load(f"{path}r_array_index{index}.npy")
time_data = np.load(f"{path}time_array_index{index}.npy")
body_speed_data = np.load(f"{path}body_speed_array_index{index}.npy")
body_rot_speed_data = np.load(f"{path}body_rot_speed_array_index{index}.npy")
num_eff_beat_data = np.load(f"{path}num_eff_beat_array_index{index}.npy")
if force:
    dissipation_data = np.load(f"{path}dissipation_array_index{index}.npy")
    efficiency_data = np.load(f"{path}efficiency_array_index{index}.npy")

r_data2 = np.load(f"{path}r_array_index{index2}.npy")
time_data2 = np.load(f"{path}time_array_index{index2}.npy")
body_speed_data2 = np.load(f"{path}body_speed_array_index{index2}.npy")
body_rot_speed_data2 = np.load(f"{path}body_rot_speed_array_index{index2}.npy")
num_eff_beat_data2 = np.load(f"{path}num_eff_beat_array_index{index2}.npy")
if force:
    dissipation_data2 = np.load(f"{path}dissipation_array_index{index2}.npy")
    efficiency_data2 = np.load(f"{path}efficiency_array_index{index2}.npy")

num_frame = len(time_data)

time_data = np.linspace(0, 1+1./num_frame, num_frame)

# n_folder_heldfixed = r_data_heldfixed.shape[0]
n_folder = r_data.shape[0]

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax12 = ax1.twinx()

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


# for fi in range(n_folder_heldfixed):
#     plot_x = k_data_heldfixed[fi] 
#     plot_y = r_data_heldfixed[fi]
#     indices_symplectic = np.where(plot_y > .4)[0]
#     indices_diaplectic = np.where((plot_y  < .4) & (plot_y > 0.04))[0]
#     indices_diaplectic_k2 = np.where(plot_y  < 0.04)[0]

#     ax.scatter(plot_x[indices_symplectic], plot_y[indices_symplectic], s=100, marker='x', c='r')
#     ax.scatter(plot_x[indices_diaplectic], plot_y[indices_diaplectic], s=100, marker='+', c='r')
#     ax.scatter(plot_x[indices_diaplectic_k2], plot_y[indices_diaplectic_k2], s=100, marker='P', c='r')


# cmap = plt.get_cmap(cmap_name)
# color_data = r_data[fi]
# color = cmap(color_data/max(color_data))
# indices_meridional = np.where(r_data[fi][:] > .4)
# indices_zonal = np.where(r_data[fi][:] < .4)

# variable = avg_rot_speed_along_axis_data[fi]
# variable_label = r"$<Ω⋅e_1>$"
# variable = avg_speed_along_axis_data[fi]
# variable_label = r"$<V⋅e_1>/L$"
# vmin2 = np.min(variable)
# vmax2 = np.max(variable)
# color2 = cmap((variable-vmin2)/(vmax2-vmin2))


ax1.plot(time_data[:-1], r_data[:-1], marker='+', c='black')
ax12.plot(time_data[:-1], r_data2[:-1], marker='x', c='blue')

ax2.plot(time_data[:-1], num_eff_beat_data[:-1], marker='+', c='black', label='Symplectic')
ax2.plot(time_data[:-1], num_eff_beat_data2[:-1], marker='x', c='blue', label='Diaplectic')

ax3.plot(time_data[:-1], body_speed_data[:num_frame-1], marker='+', c='black', label='Symplectic')
ax3.plot(time_data[:-1], body_speed_data2[:num_frame-1], marker='x', c='blue', label='Diaplectic')

ax4.plot(time_data[:-1], body_rot_speed_data[:num_frame-1], marker='+', c='black', label='Symplectic')
ax4.plot(time_data[:-1], body_rot_speed_data2[:num_frame-1], marker='x', c='blue', label='Diaplectic')

if force:
    ax5.plot(time_data[:-1], dissipation_data[:num_frame-1], marker='+', c='black', label='Symplectic')
    ax5.plot(time_data[:-1], dissipation_data2[:num_frame-1], marker='x', c='blue', label='Diaplectic')

    ax6.plot(time_data[:-1], efficiency_data[:num_frame-1], marker='+', c='black', label='Symplectic')
    ax6.plot(time_data[:-1], efficiency_data2[:num_frame-1], marker='x', c='blue', label='Diaplectic')


# ax3.scatter(tilt_data[fi][:][indices_meridional], variable[:][indices_meridional], color = 'r', label=r'$<r>$>0.4')
# ax3.scatter(tilt_data[fi][:][indices_zonal], variable[:][indices_zonal], color = 'b', label=r'$<r>$<0.4')
# ax3.scatter(plot_x[:][indices_meridional], variable[:][indices_meridional], color = 'r', label=r'$<r>$>0.4')
# ax3.scatter(plot_x[:][indices_zonal], variable[:][indices_zonal], color = 'b', label=r'$<r>$<0.4')

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

ax1.set_xlabel(r'$t/T$')
ax1.set_ylabel(r'r')
ax1.plot(-1, -1, marker='+', c='black', label='Symplectic')
ax1.plot(-1, -1, marker='x', c='blue', label='Diaplectic')
ax1.legend(loc='upper left')
ax1.set_xlim((0, 1))
ax1.set_ylim((0, 1))

ax2.set_xlabel(r'$t/T$')
ax2.set_ylabel(r'No. of effective strokes')
ax2.legend()
ax2.set_xlim((0, 1))

ax3.set_xlabel(r'$t/T$')
ax3.set_ylabel(r"$VT/L$")
ax3.legend()
ax3.set_xlim((0, 1))
# ax3.set_ylim((np.min(body_speed_data)-0.1*np.ptp(body_speed_data), np.max(body_speed_data)+0.1*np.ptp(body_speed_data)))

ax4.set_xlabel(r'$t/T$')
ax4.set_ylabel(r"$Ω$")
ax4.legend()
ax4.set_xlim((0, 1))

ax5.set_xlabel(r'$t/T$')
ax5.set_ylabel(r'$\mathcal{R}T^2/\mu L^3$')
ax5.legend()
ax5.set_xlim((0, 1))

ax6.set_xlabel(r'$t/T$')
ax6.set_ylabel(r'$Efficiency$')
ax6.legend()
ax6.set_xlim((0, 1))

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()
fig1.savefig(f'fig/r_1T_index{index}.pdf', bbox_inches = 'tight', format='pdf')
fig2.savefig(f'fig/num_beat_1T_index{index}.pdf', bbox_inches = 'tight', format='pdf')
fig3.savefig(f'fig/speed_1T_index{index}.pdf', bbox_inches = 'tight', format='pdf')
fig4.savefig(f'fig/rot_speed_1T_index{index}.pdf', bbox_inches = 'tight', format='pdf')
fig5.savefig(f'fig/dissipation_1T_index{index}.pdf', bbox_inches = 'tight', format='pdf')
fig6.savefig(f'fig/efficiency_1T_index{index}.pdf', bbox_inches = 'tight', format='pdf')
plt.show()