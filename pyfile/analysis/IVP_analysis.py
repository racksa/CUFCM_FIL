
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

def box(x, box_size):
    return x - np.floor(x/box_size)*box_size

def cartesian_to_spherical(x):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    theta = np.arctan2(x[1], x[0])
    phi = np.arccos(x[2] / r)
    
    return r, theta, phi

path_heldfixed = "data/ic_hpc_sim_rerun/"
path_free = "data/ic_hpc_sim_free_with_force3/"
# path_free = "data/ic_hpc_sim_free_continue/"
fillength = 2.6*19
radius = fillength*7.5

fil_references = np.load(f"{path_heldfixed}fil_ref_data.npy")
nfil = int(np.shape(fil_references)[0]/3)
fil_references_sphpolar = np.zeros((nfil,3))
for i in range(nfil):
    fil_references_sphpolar[i] = cartesian_to_spherical(fil_references[3*i: 3*i+3])

phase_data_heldfixed = np.load(f"{path_heldfixed}phase_data.npy")
phase_data_free = np.load(f"{path_free}phase_data.npy")
r_data_heldfixed = np.load(f"{path_heldfixed}r_data.npy")
k_data_heldfixed = np.load(f"{path_heldfixed}k_data.npy")

phase_data_free = np.load(f"{path_free}phase_data.npy")
r_data_free = np.load(f"{path_free}r_data.npy")
k_data_free = np.load(f"{path_free}k_data.npy")
avg_speed_data_free = np.load(f"{path_free}avg_speed_data.npy")
avg_speed_along_axis_data_free = np.load(f"{path_free}avg_speed_along_axis_data.npy")
avg_rot_speed_along_axis_data_free = np.load(f"{path_free}avg_rot_speed_along_axis_data.npy")
avg_vz_data_free = np.load(f"{path_free}avg_vz_data.npy")
try:
    dis_data = np.load(f"{path_free}dis_data.npy")
except:
    pass

n_folder_heldfixed = r_data_heldfixed.shape[0]
n_folder_free = r_data_free.shape[0]

dpi = 100

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)
fig6 = plt.figure()
ax6 = fig6.add_subplot(1,1,1)

plot_phase_data = 'free'
plot_phase_data = None

if plot_phase_data == 'fixed':
    ncol = np.shape(phase_data_heldfixed)[1]
    nrow = np.shape(phase_data_heldfixed)[0]
elif plot_phase_data == 'free':
    ncol = np.shape(phase_data_free)[1]
    nrow = np.shape(phase_data_free)[0]
else:
    ncol = 1
    nrow = 1

if plot_phase_data:
    fig5, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True)
    axs_flat = axs.ravel()
    for ax5 in axs_flat:
        ax5.set_xticklabels([])
        ax5.set_yticklabels([])
        ax5.set_xlabel('')
        ax5.set_ylabel('')

x_scale_offset = 1

colormap = 'hsv'
colormap = 'twilight_shifted'
vmin, vmax = 0, 2*np.pi
cmap = mpl.colormaps[colormap]

for fi in range(n_folder_heldfixed):
    plot_x = k_data_heldfixed[fi] / x_scale_offset
    plot_y = r_data_heldfixed[fi]
    indices_symplectic = np.where(plot_y > .4)[0]
    indices_diaplectic = np.where((plot_y  < .4) & (plot_y > 0.04))[0]
    indices_diaplectic_k2 = np.where(plot_y  < 0.04)[0]

    marker_fixed = '^'
    marker_free = 'x'
    color = 'r'
    s = 30

    ax.scatter(plot_x[indices_symplectic], plot_y[indices_symplectic], s=s, facecolor='none', marker=marker_fixed, edgecolor='black')
    ax.scatter(plot_x[indices_diaplectic], plot_y[indices_diaplectic], s=s, facecolor='none', marker=marker_fixed, edgecolor='b')
    ax.scatter(plot_x[indices_diaplectic_k2], plot_y[indices_diaplectic_k2], s=s, facecolor='none', marker=marker_fixed, edgecolor='b')

    #special points for paper
    # fii = 2
    # if fi==fii: # symplectic
    #     ax.scatter(plot_x[0], plot_y[0], s=100, marker='o', c='r', linewidths=1, zorder=300)
    # if fi==fii: # symplectic 2
    #     ax.scatter(plot_x[2], plot_y[2], s=100, marker='s', c='r', linewidths=1, zorder=300)
    # if fi==fii: # symplectic 3
    #     ax.scatter(plot_x[7], plot_y[7], s=200, marker='+', c='r', linewidths=3, zorder=300)
    # if fi==6: #k=1
    #     ax.scatter(plot_x[10], plot_y[10], s=100, marker='^', c='r', linewidths=1, zorder=300)
    # if fi==6: #k=2
    #     ax.scatter(plot_x[9], plot_y[9], s=100, marker='x', c='r', linewidths=2, zorder=300)

    
    # # plot fil_phases
    if plot_phase_data == 'fixed':
        for simi in range(ncol):
            if simi in indices_diaplectic or simi in indices_diaplectic_k2:
                colors = cmap(box(phase_data_heldfixed[fi][simi], vmax)/vmax)
                axs_flat[int(fi*ncol+simi)].scatter(fil_references_sphpolar[:,1], fil_references_sphpolar[:,2], c=colors)
        

for fi in range(n_folder_free):
    plot_x = k_data_free[fi] / x_scale_offset
    plot_y = r_data_free[fi]
    indices_symplectic = np.where(plot_y > .4)[0]
    indices_diaplectic = np.where(plot_y  < .4)[0]

    marker_free = 'o'
    color = 'b'
    s = 30

    ax.scatter(plot_x[indices_symplectic], plot_y[indices_symplectic], s=s, facecolor='none', marker=marker_free, edgecolors='black')
    ax.scatter(plot_x[indices_diaplectic], plot_y[indices_diaplectic], s=s, facecolor='none', marker=marker_free, edgecolors='blue')

    plot_y2 = avg_speed_along_axis_data_free[fi]
    ax2.scatter(plot_x[indices_symplectic], plot_y2[indices_symplectic], s=100, marker='+', c='black')
    ax2.scatter(plot_x[indices_diaplectic], plot_y2[indices_diaplectic], s=50, marker='x', c='b')

    plot_y3 = avg_rot_speed_along_axis_data_free[fi]
    ax3.scatter(plot_x[indices_symplectic], plot_y3[indices_symplectic], s=100, marker='+', c='black')
    ax3.scatter(plot_x[indices_diaplectic], plot_y3[indices_diaplectic], s=50, marker='x', c='b')

    try:
        plot_y6 = 6*np.pi*radius*avg_speed_along_axis_data_free[fi]**2/dis_data[fi]/fillength
        ax6.scatter(plot_x[indices_symplectic], plot_y6[indices_symplectic], s=100, marker='+', c='black')
        ax6.scatter(plot_x[indices_diaplectic], plot_y6[indices_diaplectic], s=50, marker='x', c='b')
    except:
        pass

    # plot fil_phases
    if plot_phase_data == 'free':
        for simi in range(ncol):
            if simi in indices_symplectic:
                colors = cmap(box(phase_data_free[fi][simi], vmax)/vmax)
                axs_flat[int(fi*ncol+simi)].scatter(fil_references_sphpolar[:,1], fil_references_sphpolar[:,2], c=colors)
        

# ax.scatter(-1, -1, marker='+', c='r', label='Held fixed - Symplectic')
# ax.scatter(-1, -1, marker='x', c='r', label='Held fixed - Diaplectic')
# ax.scatter(-1, -1, marker='X', c='r', label='Held fixed - Diaplectic(#k=2)')
# ax.scatter(-1, -1, marker='+', c='b', label='Free - Symplectic')
# ax.scatter(-1, -1, marker='x', c='b', label='Free - Diaplectic')

# ax.scatter(-1, -1, marker='x', c='black', s=100, label='Symplectic')
# ax.scatter(-1, -1, marker='+', c='black', s=100, label='Diaplectic')
# ax.scatter(-1, -1, marker='P', c='black', s=100, label='Diaplectic(#k=2)')

from matplotlib.legend_handler import HandlerTuple
fixed_legend1 = ax.scatter(-1, -1, marker='^', facecolor='none', edgecolors='black', s=30)
fixed_legend2 = ax.scatter(-1, -1, marker='^', facecolor='none', edgecolors='blue', s=30)
free_legend1 = ax.scatter(-1, -1, marker='o', facecolor='none', edgecolors='black', s=30)
free_legend2 = ax.scatter(-1, -1, marker='o', facecolor='none', edgecolors='blue', s=30)
ax.legend([ (free_legend1, free_legend2), (fixed_legend1, fixed_legend2)], ['free', 'fixed'],\
           fontsize=16, frameon=False, handler_map={tuple: HandlerTuple(ndivide=None)})


ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$ \langle r \rangle $')
ax.set_ylim(0)
ax.set_xlim(0, 0.09 / x_scale_offset)



ax.annotate(r'$\times 10^{-2}$', xy=(1, -0.20), xycoords='axes fraction', 
             fontsize=20, ha='right')

ax2.set_xlabel(r'$k$')
ax2.set_ylabel(r"$\langle V \rangle T/L$")
ax2.scatter(None, None, marker='+', c='black', label='Symplectic')
ax2.scatter(None, None,  marker='x', c='b', label='Diaplectic')
ax2.legend(fontsize=16, frameon=False)
ax2.set_xlim(0,)
ax2.set_ylim(0, 0.17)
ax2.set_xticks(np.array([0, 0.02, 0.04, 0.06, 0.08]) / x_scale_offset)
ax2.set_box_aspect(0.7) 
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 4))  # Forces 10^4 notation when values are large
ax2.yaxis.set_major_formatter(formatter)
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 2))  # Forces 10^4 notation when values are large
ax2.xaxis.set_major_formatter(formatter)

indices_symplectic = np.where(r_data_free > .4)
indices_diaplectic = np.where(r_data_free  < .4)
sym_k = k_data_free[indices_symplectic]
dia_k = k_data_free[indices_diaplectic]
sym_speed = avg_speed_along_axis_data_free[indices_symplectic]
dia_speed = avg_speed_along_axis_data_free[indices_diaplectic]
sym_eff = 6*np.pi*radius*sym_speed**2/dis_data[indices_symplectic]/fillength
dia_eff = 6*np.pi*radius*dia_speed**2/dis_data[indices_diaplectic]/fillength

# Symplectic: sort by k
sym_sort_idx = np.argsort(sym_k)
sym_k = sym_k[sym_sort_idx]
sym_speed = sym_speed[sym_sort_idx]
sym_eff = sym_eff[sym_sort_idx]

# Diaplectic: sort by k
dia_sort_idx = np.argsort(dia_k)
dia_k = dia_k[dia_sort_idx]
dia_speed = dia_speed[dia_sort_idx]
dia_eff = dia_eff[dia_sort_idx]





# Fit a straight line
sym_p = np.polyfit(sym_k, sym_speed, 1)  # slope, intercept
sym_speed_fit = np.polyval(sym_p, sym_k)
dia_p = np.polyfit(dia_k, dia_speed, 1)  # slope, intercept
dia_speed_fit = np.polyval(dia_p, dia_k)

# Spread: residual std
sym_residuals = sym_speed - sym_speed_fit
spread = 2*np.std(sym_residuals)
dia_residuals = dia_speed - dia_speed_fit
dia_spread = 2*np.std(dia_residuals)
ax2.plot(sym_k, sym_speed_fit, c='black', linestyle='dashed',)
ax2.plot(dia_k, dia_speed_fit, c='blue', linestyle='dashed',)
ax2.fill_between(sym_k, sym_speed_fit - spread, sym_speed_fit + spread, color='black', alpha=0.2)
ax2.fill_between(dia_k, dia_speed_fit - dia_spread, dia_speed_fit + dia_spread, color='blue', alpha=0.2)



sym_p = np.polyfit(sym_k, sym_eff, 1)  # slope, intercept
sym_eff_fit = np.polyval(sym_p, sym_k)
dia_p = np.polyfit(dia_k, dia_eff, 1)  # slope, intercept
dia_eff_fit = np.polyval(dia_p, dia_k)

sym_residuals = sym_eff - sym_eff_fit
spread = 2*np.std(sym_residuals)
dia_residuals = dia_eff - dia_eff_fit
dia_spread = 2*np.std(dia_residuals)
ax6.plot(sym_k, sym_eff_fit, c='black', linestyle='dashed',)
ax6.plot(dia_k, dia_eff_fit, c='blue', linestyle='dashed',)
ax6.fill_between(sym_k, sym_eff_fit - spread, sym_eff_fit + spread, color='black', alpha=0.2)
ax6.fill_between(dia_k, dia_eff_fit - dia_spread, dia_eff_fit + dia_spread, color='blue', alpha=0.2)


ax3.set_xlabel(r'$k$')
ax3.set_ylabel(r"$\langle \Omega \rangle T$")
ax3.scatter(None, None, marker='+', c='black', label='Symplectic')
ax3.scatter(None, None,  marker='x', c='b', label='Diaplectic')
ax3.legend(fontsize=16, frameon=False)


ax4.scatter(None, None, s=100, marker='o', c='r', linewidths=1, zorder=300, label='Symplectic ($k=0.005$)')
ax4.scatter(None, None, s=100, marker='s', c='r', linewidths=1, zorder=300, label='Symplectic ($k=0.015$)')
ax4.scatter(None, None, s=200, marker='+', c='r', linewidths=3, zorder=300, label='Symplectic ($k=0.04$)')
ax4.scatter(None, None, s=100, marker='^', c='r', linewidths=1, zorder=300, label='Diaplectic ($\kappa=1$)')
ax4.scatter(None, None, s=100, marker='x', c='r', linewidths=2, zorder=300, label='Diaplectic ($\kappa=2$)')
ax4.legend(fontsize=16, frameon=False)
ax4.set_axis_off()


ax6.set_xlabel(r'$k$')
ax6.set_ylabel(r"Efficiency")
ax6.scatter(None, None, marker='+', c='black', label='Symplectic')
ax6.scatter(None, None,  marker='x', c='b', label='Diaplectic')
ax6.legend(fontsize=16, frameon=False)
ax6.set_xlim(0,)
ax6.set_ylim(0, 5e-4)
ax6.set_xticks(np.array([0, 0.02, 0.04, 0.06, 0.08]) / x_scale_offset)
ax6.set_box_aspect(0.7) 
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-1, 4))  # Forces 10^4 notation when values are large
ax6.yaxis.set_major_formatter(formatter)
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 2))  # Forces 10^4 notation when values are large
ax6.xaxis.set_major_formatter(formatter)

fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig6.tight_layout()
# fig.savefig(f'fig/order_parameter.pdf', bbox_inches = 'tight', format='pdf', transparent=True)
fig.savefig(f'fig/order_parameter.png', bbox_inches = 'tight', format='png', transparent=True)
fig2.savefig(f'fig/IVP_velocities_free.png', bbox_inches = 'tight', format='png', transparent=True)
fig3.savefig(f'fig/IVP_rot_velocities_free.png', bbox_inches = 'tight', format='png', transparent=True)
fig4.savefig(f'fig/IVP_symbols.png', bbox_inches = 'tight', format='png', transparent=True)
fig6.savefig(f'fig/IVP_efficiencies_free.png', bbox_inches = 'tight', format='png', transparent=True)
plt.show()