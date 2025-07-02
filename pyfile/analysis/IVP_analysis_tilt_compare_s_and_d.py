
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

path = "data/tilt_test/makeup_pattern_with_force/"
# path = "data/tilt_test/makeup_pattern/"

force = True

r_data = np.load(f"{path}r_data.npy")
k_data = np.load(f"{path}k_data.npy")
tilt_data = np.load(f"{path}tilt_data.npy")
avg_vz_data = np.load(f"{path}avg_vz_data.npy")
avg_speed_data = np.load(f"{path}avg_speed_data.npy")
avg_speed_along_axis_data = np.load(f"{path}avg_speed_along_axis_data.npy")
avg_rot_speed_data = np.load(f"{path}avg_rot_speed_data.npy")
avg_rot_speed_along_axis_data = np.load(f"{path}avg_rot_speed_along_axis_data.npy")

print(tilt_data[0], k_data[0])
if force:
    avg_dis_data = np.load(f"{path}dis_data.npy")

# n_folder_heldfixed = r_data_heldfixed.shape[0]
n_folder = r_data.shape[0]
colors = [ 'b', 'black']
labels = [ 'Zonal MCW', 'Meridional MCW',]
markers = ['+', 'x']

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
fig6 = plt.figure()
ax6 = fig6.add_subplot(1,1,1)
fig7 = plt.figure()
ax7 = fig7.add_subplot(1,1,1)

plottype = 'avg'
# plottype = 'individual'
n_k = 1
if plottype == 'avg':
    n_k = 1
elif plottype == 'individual':
    n_k = 10

n_tilt = 5
tilt_angle = np.linspace(0, 50*np.pi/180, n_tilt+1)[:-1]
tilt_angle = tilt_angle*180/np.pi

print(tilt_angle)


# print(avg_dis_data[0][0::n_tilt], np.mean(avg_dis_data[0][0::n_tilt]))
# print(avg_dis_data[0][1::n_tilt], np.mean(avg_dis_data[0][1::n_tilt]))


for fi in range(n_folder):
    avg_speed_over_k = np.zeros(tilt_angle.shape)
    avg_rot_speed_over_k = np.zeros(tilt_angle.shape)
    if force:
        avg_dis_over_k = np.zeros(tilt_angle.shape)
    
    for ki in range(n_k):

        for ti in range(n_tilt):

            cmap_name = 'Greys'
            cmap = plt.get_cmap(cmap_name)
            color = cmap(ti/(n_tilt-1))

            k = k_data[fi][ti::n_tilt]
            r = r_data[fi][ti::n_tilt]
            avg_speed = avg_speed_along_axis_data[fi][ti::n_tilt]
            avg_rot_speed = avg_rot_speed_along_axis_data[fi][ti::n_tilt]
            if force:
                avg_dis = avg_dis_data[fi][ti::n_tilt]

            if plottype == 'avg':
                avg_speed_over_k[ti] = np.mean(avg_speed)
                avg_rot_speed_over_k[ti] = np.mean(avg_rot_speed)
                if force:
                    avg_dis_over_k[ti] = np.mean(avg_dis)

            if plottype == 'individual':
                avg_speed_over_k[ti] = avg_speed[ki] #np.mean(avg_speed)
                avg_rot_speed_over_k[ti] = avg_rot_speed[ki] #np.mean(avg_rot_speed)
                if force:
                    avg_dis_over_k[ti] = avg_dis[ki] #np.mean(avg_rot_speed

            # ax.plot(k, avg_speed, marker='+', c=color)

            ax2.plot(k, avg_rot_speed, marker=markers[fi], c=color)

            ax5.scatter(k[:-2], r[:-2], marker='.', s=50, c=colors[fi])

        ax3.plot(tilt_angle, avg_speed_over_k, marker=markers[fi], c=colors[fi])

        ax4.plot(tilt_angle, avg_rot_speed_over_k, marker=markers[fi], c=colors[fi])

        if force:
            ax6.plot(tilt_angle, avg_dis_over_k, marker=markers[fi], c=colors[fi])

            ax7.plot(tilt_angle, avg_speed_over_k**2/avg_dis_over_k, marker=markers[fi], c=colors[fi])

    std3 = np.array([np.std(avg_speed_along_axis_data[fi][ti::n_tilt]) for ti in range(n_tilt)])
    ax3.scatter(tilt_data[fi]*180/np.pi, avg_speed_along_axis_data[fi], marker=markers[fi], c=colors[fi])
    ax3.fill_between(tilt_angle, avg_speed_over_k - std3,
                 avg_speed_over_k + std3, color=colors[fi], alpha=0.2)

    std4 = np.array([np.std(avg_rot_speed_along_axis_data[fi][ti::n_tilt]) for ti in range(n_tilt)])
    ax4.scatter(tilt_data[fi]*180/np.pi, avg_rot_speed_along_axis_data[fi], marker=markers[fi], c=colors[fi])
    ax4.fill_between(tilt_angle, avg_rot_speed_over_k - std4,
                 avg_rot_speed_over_k + std4, color=colors[fi], alpha=0.2)

    
    ax3.plot(np.nan, np.nan, marker=markers[fi], c=colors[fi], label = labels[fi])
    ax4.plot(np.nan, np.nan, marker=markers[fi], c=colors[fi], label = labels[fi])

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
# ax3.plot(None, None, marker='x', c='black', s=100, label='Symplectic')
# ax.scatter(-1, -1, marker='+', c='black', s=100, label='Diaplectic')
# ax.scatter(-1, -1, marker='P', c='black', s=100, label='Diaplectic(#k=2)')
# ax.scatter(-1, -1, marker='s', c='r', s=100, label='Held fixed')
# ax.scatter(-1, -1, marker='s', c='b', s=100, label='Free')

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$<V>T/L$')
# ax.set_ylim(0)
# ax.set_xlim(0, 0.06)
# ax.legend()

ax2.set_xlabel(r'$k$')
ax2.set_ylabel(r'$<\Omega>T$')

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 2))  # Forces 10^4 notation when values are large
ax3.yaxis.set_major_formatter(formatter)
ax3.set_xlabel(r'$\chi(deg)$')
ax3.set_ylabel(r'$<V>T/L$')
ax3.legend(fontsize=16, frameon=False)
# ax3.set_xticks(tilt_angle, ['0', 'π/20', '2π/20', '3π/20', '4π/20'])
ax3.set_xlim(tilt_angle[0], tilt_angle[-1])

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-1, 4))  # Forces 10^4 notation when values are large
ax4.yaxis.set_major_formatter(formatter)
ax4.set_xlabel(r'$\chi(deg)$')
ax4.set_ylabel(r'$<\Omega>T$')
ax4.legend(fontsize=16, frameon=False)
# ax4.set_xticks(tilt_angle, ['0', 'π/20', '2π/20', '3π/20', '4π/20'])
ax4.set_xlim(tilt_angle[0], tilt_angle[-1])

ax5.scatter(-.1, -.1, marker='.', c=colors[1], s=50, label='Symplectic')
ax5.scatter(-.1, -.1, marker='.', c=colors[0], s=50, label='Diaplectic')
ax5.set_xlabel(r'$k$')
ax5.set_ylabel(r'$<r>$')
ax5.legend(fontsize=16, frameon=False)
ax5.set_ylim(0)
ax5.set_xlim(0)

fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()
fig7.tight_layout()
# fig.savefig(f'fig/order_parameter_tilt.pdf', bbox_inches = 'tight', format='pdf')
# fig2.savefig(f'fig/avg_rot_speed_along_axis_data_tilt.pdf', bbox_inches = 'tight', format='pdf')
fig3.savefig(f'fig/tilt_speed_vs_tilt.pdf', bbox_inches = 'tight', format='pdf')
fig4.savefig(f'fig/tilt_rot_speed_vs_tilt.pdf', bbox_inches = 'tight', format='pdf')
fig5.savefig(f'fig/tilt_order_parameter.pdf', bbox_inches = 'tight', format='pdf')
plt.show()