
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

path_heldfixed = "data/ic_hpc_sim/"
path_free = "data/ic_hpc_sim_free_with_force/"

r_data_heldfixed = np.load(f"{path_heldfixed}r_data.npy")
k_data_heldfixed = np.load(f"{path_heldfixed}k_data.npy")

r_data_free = np.load(f"{path_free}r_data.npy")
k_data_free = np.load(f"{path_free}k_data.npy")
avg_speed_data_free = np.load(f"{path_free}avg_speed_data.npy")
avg_speed_along_axis_data_free = np.load(f"{path_free}avg_speed_along_axis_data.npy")
avg_rot_speed_along_axis_data_free = np.load(f"{path_free}avg_rot_speed_along_axis_data.npy")
avg_vz_data_free = np.load(f"{path_free}avg_vz_data.npy")
# eff_data_free = np.load(f"{path_free}eff_data.npy")

n_folder_heldfixed = r_data_heldfixed.shape[0]
n_folder_free = r_data_free.shape[0]

dpi = 100

fig = plt.figure(dpi=dpi)
ax = fig.add_subplot(1,1,1)
fig2 = plt.figure(dpi=dpi)
ax2 = fig2.add_subplot(1,1,1)
fig3 = plt.figure(dpi=dpi)
ax3 = fig3.add_subplot(1,1,1)
fig4 = plt.figure(dpi=dpi)
ax4 = fig4.add_subplot(1,1,1)

x_scale_offset = 1e-2

for fi in range(n_folder_heldfixed):
    plot_x = k_data_heldfixed[fi] / x_scale_offset
    plot_y = r_data_heldfixed[fi]
    indices_symplectic = np.where(plot_y > .4)[0]
    indices_diaplectic = np.where((plot_y  < .4) & (plot_y > 0.04))[0]
    indices_diaplectic_k2 = np.where(plot_y  < 0.04)[0]

    marker = '.'
    color = 'r'
    s = 50

    ax.scatter(plot_x[indices_symplectic], plot_y[indices_symplectic], s=s, marker=marker, c=color)
    ax.scatter(plot_x[indices_diaplectic], plot_y[indices_diaplectic], s=s, marker=marker, c=color)
    ax.scatter(plot_x[indices_diaplectic_k2], plot_y[indices_diaplectic_k2], s=s, marker=marker, c=color)

    #special points for paper (made up)
    if fi==9: # symplectic
        ax.scatter(plot_x[1], plot_y[1], s=100, marker='o', c='black', linewidths=1, zorder=300)
    if fi==3: # asym_symplectic
        ax.scatter(plot_x[0], plot_y[0], s=100, marker='s', c='black', linewidths=1, zorder=300)
    if fi==3:
        ax.scatter(plot_x[9], plot_y[9], s=100, marker='x', c='black', linewidths=2, zorder=300)
    if fi==8:
        ax.scatter(plot_x[9], plot_y[9], s=100, marker='^', c='black', linewidths=1, zorder=300)

for fi in range(n_folder_free):
    plot_x = k_data_free[fi] / x_scale_offset
    plot_y = r_data_free[fi]
    indices_symplectic = np.where(plot_y > .4)[0]
    indices_diaplectic = np.where(plot_y  < .4)[0]

    marker = '.'
    color = 'b'
    s = 50

    ax.scatter(plot_x[indices_symplectic], plot_y[indices_symplectic], s=s, marker=marker, c=color)
    ax.scatter(plot_x[indices_diaplectic], plot_y[indices_diaplectic], s=s, marker=marker, c=color)

    plot_y2 = avg_speed_along_axis_data_free[fi]
    ax2.scatter(plot_x[indices_symplectic], plot_y2[indices_symplectic], s=100, marker='+', c='black')
    ax2.scatter(plot_x[indices_diaplectic], plot_y2[indices_diaplectic], s=50, marker='x', c='b')
    
    plot_y3 = avg_rot_speed_along_axis_data_free[fi]
    ax3.scatter(plot_x[indices_symplectic], plot_y3[indices_symplectic], s=100, marker='+', c='black')
    ax3.scatter(plot_x[indices_diaplectic], plot_y3[indices_diaplectic], s=50, marker='x', c='b')


# ax.scatter(-1, -1, marker='+', c='r', label='Held fixed - Symplectic')
# ax.scatter(-1, -1, marker='x', c='r', label='Held fixed - Diaplectic')
# ax.scatter(-1, -1, marker='X', c='r', label='Held fixed - Diaplectic(#k=2)')
# ax.scatter(-1, -1, marker='+', c='b', label='Free - Symplectic')
# ax.scatter(-1, -1, marker='x', c='b', label='Free - Diaplectic')

# ax.scatter(-1, -1, marker='x', c='black', s=100, label='Symplectic')
# ax.scatter(-1, -1, marker='+', c='black', s=100, label='Diaplectic')
# ax.scatter(-1, -1, marker='P', c='black', s=100, label='Diaplectic(#k=2)')
ax.scatter(-1, -1, marker='.', c='r', s=50, label='Fixed')
ax.scatter(-1, -1, marker='.', c='b', s=50, label='Free')

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$<r>$')
ax.set_ylim(0)
ax.set_xlim(0, 0.09 / x_scale_offset)
ax.legend(fontsize=16, frameon=False)


ax.annotate(r'$\times 10^{-2}$', xy=(1, -0.20), xycoords='axes fraction', 
             fontsize=20, ha='right')

ax2.set_xlabel(r'$k$')
ax2.set_ylabel(r"$<V>T/L$")
ax2.scatter(None, None, marker='+', c='black', label='Symplectic')
ax2.scatter(None, None,  marker='x', c='b', label='Diaplectic')
ax2.legend(fontsize=16, frameon=False)

ax3.set_xlabel(r'$k$')
ax3.set_ylabel(r"$<\Omega>T$")
ax3.scatter(None, None, marker='+', c='black', label='Symplectic')
ax3.scatter(None, None,  marker='x', c='b', label='Diaplectic')
ax3.legend(fontsize=16, frameon=False)


ax4.scatter(None, None, s=100, marker='o', c='black', linewidths=1, zorder=300, label='Symplectic')
ax4.scatter(None, None, s=100, marker='s', c='black', linewidths=1, zorder=300, label='Asymmetric Symplectic')
ax4.scatter(None, None, s=100, marker='x', c='black', linewidths=2, zorder=300, label='Diaplectic ($\kappa=1$)')
ax4.scatter(None, None, s=100, marker='^', c='black', linewidths=1, zorder=300, label='Diaplectic ($\kappa=2$)')
ax4.legend(fontsize=16, frameon=False)
ax4.set_axis_off()


fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig.savefig(f'fig/order_parameter.pdf', bbox_inches = 'tight', format='pdf', transparent=True)
fig.savefig(f'fig/order_parameter.png', bbox_inches = 'tight', format='png', transparent=True)
fig2.savefig(f'fig/IVP_velocities_free.pdf', bbox_inches = 'tight', format='pdf', transparent=True)
fig3.savefig(f'fig/IVP_efficiencies_free.pdf', bbox_inches = 'tight', format='pdf', transparent=True)
fig4.savefig(f'fig/IVP_symbols.pdf', bbox_inches = 'tight', format='pdf', transparent=True)
plt.show()