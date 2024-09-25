
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



fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(1,1,1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)
fig5 = plt.figure()
ax5 = fig5.add_subplot(1,1,1)



# k_list = [-1, 0, 0.5, 1.0, 1.5, 2.0]
# labels = [r"$k=-1$",r"$k=0$",r"$k=0.5$",r"$k=1$",r"$k=1.5$",r"$k=2$",]
colors = ["black","red","green","blue","purple"]
sim_n = 15

k_list = np.zeros(sim_n)
avg_r_list = np.zeros(sim_n)
avg_speed_list = np.zeros(sim_n)
avg_dis_list = np.zeros(sim_n)
avg_eff_list = np.zeros(sim_n)

paths = list()

for i in range(1, 8):
    paths.append(f'data/ic_hpc_sim_free_continue/20240311_{i}/')

# plot sim data
for pind, path in enumerate(paths):
    cmap_name = 'hsv'
    cmap = plt.get_cmap(cmap_name)
    color = cmap(pind/len(paths))
    for ind in range(sim_n):
        try:
            time_array = np.load(f"{path}time_array_index{ind}.npy")
            speed_array = np.load(f"{path}body_speed_array_index{ind}.npy")
            rot_speed_array = np.load(f"{path}body_rot_speed_array_index{ind}.npy")
            dissipation_array = np.load(f"{path}dissipation_array_index{ind}.npy")
            efficiency_array = np.load(f"{path}efficiency_array_index{ind}.npy")
            r_array = np.load(f"{path}r_array_index{ind}.npy")
            k_array = np.load(f"{path}k_index{ind}.npy")

            time_array -= time_array[0]

            k_list[ind] = k_array
            avg_r_list[ind] = np.mean(r_array)

            ax.plot(time_array, speed_array, alpha=1.,)
            avg_speed_list[ind] = np.mean(speed_array)
            avg_dis_list[ind] = np.mean(dissipation_array)

            ax2.plot(time_array, dissipation_array,)
        except:
            pass

    indices_symplectic = np.where(avg_r_list > .4)[0]
    indices_diaplectic = np.where((avg_r_list  < .4))[0]

    print(avg_r_list)

    ax3.scatter(k_list[indices_symplectic], avg_speed_list[indices_symplectic] , marker='+', c='r')
    ax3.scatter(k_list[indices_diaplectic], avg_speed_list[indices_diaplectic] , marker='+', c='b')

    ax4.scatter(k_list[indices_symplectic], avg_r_list[indices_symplectic] , marker='+', c='r')
    ax4.scatter(k_list[indices_diaplectic], avg_r_list[indices_diaplectic] , marker='+', c='b')

    L=19*2.6
    radius = 7.5*L

    ax5.scatter(k_list[indices_symplectic], 6*np.pi*radius/L*avg_speed_list[indices_symplectic]**2/avg_dis_list[indices_symplectic] , marker='+', c='r')
    ax5.scatter(k_list[indices_diaplectic], 6*np.pi*radius/L*avg_speed_list[indices_diaplectic]**2/avg_dis_list[indices_diaplectic] , marker='+', c='b')
    


legend11 = ax.legend(frameon=False)
ax.add_artist(legend11)
ax.set_xlim(0, 1)
ax.set_xlabel(r'$t/T$')
ax.set_ylabel(r'$V_zT/L$')

ax2.set_xlim(0, 1)
ax2.set_xlabel(r'$t/T$')
ax2.set_ylabel(r'$\mathcal{R}T^2/\mu L^3$')


ax3.set_xlabel(r'$k$')
ax3.set_ylabel(r'$<V>T/L$')
ax3.legend()

ax4.set_xlabel(r'$k$')
ax4.set_ylabel(r'$<r>$')

ax5.set_xlabel(r'$k$')
ax5.set_ylabel(r'$Efficiency$')


# line1, = ax4.plot([-1, -1.1], [-1, -1.1], ls='dashed', c='black', label=r'$<L>=1$' )
# line2, = ax4.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label=r'$<L>=0.975$' )
# legend42 = ax4.legend(handles = [line1, line2])
# ax4.set_xlim(0, 1)
# ax4.set_ylim(0.92, 1.06)
# ax4.set_xlabel(r'$t/T$')
# ax4.set_ylabel(r'$L$')

fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
# fig.savefig(f'fig/ishikawa_pnas_volvox_vel_area.pdf', bbox_inches = 'tight', format='pdf')
# fig1.savefig(f'fig/ishikawa_pnas_original_vel_area.pdf', bbox_inches = 'tight', format='pdf')
# fig2.savefig(f'fig/dissipation.pdf', bbox_inches = 'tight', format='pdf')
# fig3.savefig(f'fig/dp_speed.pdf', bbox_inches = 'tight', format='pdf')
plt.show()