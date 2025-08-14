
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

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

# path = "data/ic_hpc_sim_free_with_force/20240311_1/"
# index = 1 # symplectic
# index2 = 0 # diaplectic

path1 = "data/for_paper/hydrodynamics_in_one_period/20250302/"
path2 = "data/for_paper/hydrodynamics_in_one_period/20250302/"
index1 = 0 # symplectic
index2 = 1 # diaplectic
force = True

path1 = "data/for_paper/flowfield_example/20250522_flowfield_free/"
path2 = "data/for_paper/flowfield_example/20250522_flowfield_free/"
index1 = 0 # symplectic
index2 = 1 # diaplectic
force = True

plot_squirmer = False

r_data = np.load(f"{path1}r_array_index{index1}.npy")
time_data = np.load(f"{path1}time_array_index{index1}.npy")
body_speed_data = np.load(f"{path1}body_speed_array_index{index1}.npy")
body_rot_speed_data = np.load(f"{path1}body_rot_speed_array_index{index1}.npy")
num_eff_beat_data = np.load(f"{path1}num_eff_beat_array_index{index1}.npy")
if force:
    dissipation_data = np.load(f"{path1}dissipation_array_index{index1}.npy")
    efficiency_data = np.load(f"{path1}efficiency_array_index{index1}.npy")
if plot_squirmer:
    try:
        squirmer_speed_data = np.load(f"{path1}squirmer_speed_array_index{index1}.npy")
    except:
        pass

r_data2 = np.load(f"{path2}r_array_index{index2}.npy")
time_data2 = np.load(f"{path2}time_array_index{index2}.npy")
body_speed_data2 = np.load(f"{path2}body_speed_array_index{index2}.npy")
body_rot_speed_data2 = np.load(f"{path2}body_rot_speed_array_index{index2}.npy")
num_eff_beat_data2 = np.load(f"{path2}num_eff_beat_array_index{index2}.npy")
if force:
    dissipation_data2 = np.load(f"{path2}dissipation_array_index{index2}.npy")
    efficiency_data2 = np.load(f"{path2}efficiency_array_index{index2}.npy")
if plot_squirmer:
    try:
        squirmer_speed_data2 = np.load(f"{path2}squirmer_speed_array_index{index2}.npy")
    except:
        pass

print(dissipation_data.shape)
num_frame = len(time_data)

time_data = np.linspace(0, 1+1./num_frame, num_frame)

# n_folder_heldfixed = r_data_heldfixed.shape[0]
n_folder = r_data.shape[0]

figsize=(6,6)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax12 = ax1.twinx()

fig2 = plt.figure()
# ax2 = fig2.add_subplot(1,1,1)

fig3 = plt.figure()
# ax3 = fig3.add_subplot(1,1,1)

fig4 = plt.figure()
# ax4 = fig4.add_subplot(1,1,1)

fig5 = plt.figure()
# ax5 = fig5.add_subplot(1,1,1)

fig6 = plt.figure()
ax6 = fig6.add_subplot(1,1,1)

fig, ((ax3, ax4), (ax2, ax5)) = plt.subplots(2, 2, figsize=(10, 8))


# for fi in range(n_folder_heldfixed):
#     plot_x = k_data_heldfixed[fi] 
#     plot_y = r_data_heldfixed[fi]
#     indices_symplectic = np.where(plot_y > .4)[0]
#     indices_diaplectic = np.where((plot_y  < .4) & (plot_y > 0.04))[0]
#     indices_diaplectic_k2 = np.where(plot_y  < 0.04)[0]

#     ax.scatter(plot_x[indices_symplectic], plot_y[indices_symplectic], s=100, marker=dia_marker, c='r')
#     ax.scatter(plot_x[indices_diaplectic], plot_y[indices_diaplectic], s=100, marker=sym_marker, c='r')
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

sym_marker = None
dia_marker = None


ax1.plot(time_data[:-1], r_data[:-1], marker=sym_marker, c='black')
ax12.plot(time_data[:-1], r_data2[:-1], marker=dia_marker, c='blue')

ax2.plot(time_data[:-1], num_eff_beat_data[:-1], marker=sym_marker, c='black')
ax2.plot(time_data[:-1], num_eff_beat_data2[:-1], marker=dia_marker, c='blue')

# Plot the lines
line_sym = ax3.plot(time_data[:-1], body_speed_data[:num_frame-1], marker=sym_marker, c='black', label='Symplectic')[0]
line_dia = ax3.plot(time_data[:-1], body_speed_data2[:num_frame-1], marker=dia_marker, c='blue', label='Diaplectic')[0]
if plot_squirmer:
    try:
        ax3.scatter(time_data[0:-1:10], squirmer_speed_data[:num_frame-1:10]/49.4, marker="^", c='black')
        ax3.scatter(time_data[0:-1:10], squirmer_speed_data2[:num_frame-1:10]/49.4, marker="^", c='blue')

        squirmer_legend1 = ax3.scatter([], [], marker='^', c='black')
        squirmer_legend2 = ax3.scatter([], [], marker='^', c='blue')
        from matplotlib.legend_handler import HandlerTuple
        ax3.legend([line_sym, line_dia, (squirmer_legend1, squirmer_legend2)],
                [ 'Symplectic', 'Diaplectic', 'Squirmer',],
                fontsize=16, frameon=False,
                handler_map={tuple: HandlerTuple(ndivide=None)})
    except:
        ax3.legend(fontsize=16, frameon=False)


ax4.plot(time_data[:-1], body_rot_speed_data[:num_frame-1], marker=sym_marker, c='black')
ax4.plot(time_data[:-1], body_rot_speed_data2[:num_frame-1], marker=dia_marker, c='blue')

if force:
    ax5.plot(time_data[:-1], dissipation_data[:num_frame-1], marker=sym_marker, c='black')
    ax5.plot(time_data[:-1], dissipation_data2[:num_frame-1], marker=dia_marker, c='blue')

    ax6.plot(time_data[:-1], efficiency_data[:num_frame-1], marker=sym_marker, c='black')
    ax6.plot(time_data[:-1], efficiency_data2[:num_frame-1], marker=dia_marker, c='blue')

# from matplotlib.lines import Line2D
# legend_handles = [
#     Line2D([0], [0], color='black', lw=1.5, label='symplectic'),
#     Line2D([0], [0], color='b', lw=1.5, label='diaplectic'),
# ]

# fig.legend(handles=legend_handles, loc='upper center', ncol=2, frameon=False,
#            bbox_to_anchor=(0.5, 0.985))   # keep within [0,1] so it shows
# fig.subplots_adjust(top=0.80)            # create space at the top


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
# ax.scatter(-1, -1, marker=dia_marker, c='black', s=100, label='Symplectic')
# ax.scatter(-1, -1, marker=sym_marker, c='black', s=100, label='Diaplectic')
# ax.scatter(-1, -1, marker='P', c='black', s=100, label='Diaplectic(#k=2)')
# ax.scatter(-1, -1, marker='s', c='r', s=100, label='Held fixed')
# ax.scatter(-1, -1, marker='s', c='b', s=100, label='Free')

ax1.set_xlabel(r'$t/T$')
ax1.set_ylabel(r'r')
ax1.plot(-1, -1, marker=sym_marker, c='black', label='Symplectic')
ax1.plot(-1, -1, marker=dia_marker, c='blue', label='Diaplectic')
ax1.legend(loc='upper left', fontsize=16, frameon=False)
ax1.set_xlim((0, 1))
ax1.set_ylim((0, 1))
ax1.set_box_aspect(1) 

ax2.set_xlabel(r'$t/T$')
ax2.set_ylabel(r'Effective beat num')
ax2.legend(fontsize=16, frameon=False)
ax2.set_xlim((0, 1))
ax2.set_box_aspect(1) 

ax3.set_xlabel(r'$t/T$')
ax3.set_ylabel(r"$VT/L$")
ax3.set_xlim((0, 1))
ax3.set_box_aspect(1) 
# ax3.set_ylim((np.min(body_speed_data)-0.1*np.ptp(body_speed_data), np.max(body_speed_data)+0.1*np.ptp(body_speed_data)))

ax4.set_xlabel(r'$t/T$')
ax4.set_ylabel(r"$\Omega$")
ax4.legend(fontsize=16, frameon=False)
ax4.set_xlim((0, 1))
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-4, -4))  # Forces 10^-5 notation
ax4.yaxis.set_major_formatter(formatter)
ax4.set_box_aspect(1) 


ax5.set_xlabel(r'$t/T$')
ax5.set_ylabel(r'$\mathcal{R}T^2/\eta L^3$')
ax5.legend(fontsize=16, frameon=False)
ax5.set_xlim((0, 1))
ax5.set_box_aspect(1) 


formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-1, 4))  # Forces 10^4 notation when values are large
ax5.yaxis.set_major_formatter(formatter)

ax6.set_xlabel(r'$t/T$')
ax6.set_ylabel(r'$Efficiency$')
ax6.legend(fontsize=16, frameon=False)
ax6.set_xlim((0, 1))
ax6.set_box_aspect(1) 

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()
fig.tight_layout()
fig.savefig(f'fig/4figs.png', bbox_inches='tight', format='png', transparent=True)
fig1.savefig(f'fig/r_1T.png', bbox_inches = 'tight', format='png', transparent=True)
fig2.savefig(f'fig/num_beat_1T.png', bbox_inches = 'tight', format='png', transparent=True)
fig3.savefig(f'fig/speed_1T.png', bbox_inches = 'tight', format='png', transparent=True)
fig4.savefig(f'fig/rot_speed_1T.png', bbox_inches = 'tight', format='png', transparent=True)
fig5.savefig(f'fig/dissipation_1T.png', bbox_inches = 'tight', format='png', transparent=True)
fig6.savefig(f'fig/efficiency_1T.png', bbox_inches = 'tight', format='png', transparent=True)
plt.show()