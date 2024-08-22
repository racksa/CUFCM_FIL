
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import curve_fit

import configparser

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update({'font.size': 16})

cmap_name = 'coolwarm'

def fit_func(x, a, b, c):
    return a/(1+b*x)+c
    return a*(1/x**b) +c
    return a*np.exp(b*x) + c


# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(1,1,1)
# fig3 = plt.figure()
# ax3 = fig3.add_subplot(1,1,1)
fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)
fig5 = plt.figure()
ax5 = fig5.add_subplot(1,1,1)

# k_list = [-1, 0, 0.5, 1.0, 1.5, 2.0]
# labels = [r"$k=-1$",r"$k=0$",r"$k=0.5$",r"$k=1$",r"$k=1.5$",r"$k=2$",]
colors = ["black","red","green","blue","purple", "pink", "brown", "cyan", "orange"]
N_list = [160, 640, 2560]


path = 'data/resolution/20240822_sangani_boxsize2/'
sim = configparser.ConfigParser()
sim.read(path+"rules.ini")
nblob_list = np.array([float(s) for s in sim["Parameter list"]['nblob'].split(', ')])
L_list = np.array([float(s) for s in sim["Parameter list"]['boxsize'].split(', ')])
avg_speed_list = np.zeros(nblob_list.shape)
speed_error_list = np.zeros(nblob_list.shape)
dissipation_error_list = np.zeros(nblob_list.shape)
nsim = len(nblob_list)



def sangani(c_13):
    return 1 - 1.7601*c_13 + c_13**3 - 1.5593*c_13**6 + 3.9799*c_13**8 - 3.0734*c_13**10

fil_length = 2.6*20
ref_speed = 3.4206784569e+00/fil_length # for ar=6, nblob=9000
ref_speed = 1.0277908446e+00/fil_length # for ar=20, nblob=40961

swimmer_R = fil_length*10
swimmer_V = 4./3*np.pi*swimmer_R**3
c_3list = (swimmer_V/L_list**3)**(1./3)
sangani_speed_list = sangani(c_3list)

print(swimmer_R)

# plot sim data
for ind in range(nsim):
    try:
        time_array = np.load(f"{path}time_array_index{ind}.npy")
        speed_array = np.load(f"{path}body_speed_array_index{ind}.npy")
        rot_speed_array = np.load(f"{path}body_rot_speed_array_index{ind}.npy")
        dissipation_array = np.load(f"{path}dissipation_array_index{ind}.npy")
        efficiency_array = np.load(f"{path}efficiency_array_index{ind}.npy")


        avg_speed_list[ind] = np.mean(np.abs(speed_array))
        
        # fillength_array = np.zeros(np.shape(time_array))
        # for fili in range(len(fillength_array)):
        #     fillength_array[fili] = real_fillength(time_array[fili]*2*np.pi)

        # ax.plot(time_array, speed_array, label = rf'$M=$', alpha=1.)
        # ax2.plot(time_array, dissipation_array, label = rf'$M=$')
        # ax3.plot(time_array, efficiency_array, label = rf'$M=$')
        
    except:
        pass


speed_error_list = np.abs(avg_speed_list/ref_speed - sangani_speed_list)/np.abs(sangani_speed_list)

ax4.scatter(c_3list, avg_speed_list/ref_speed, color='black', facecolors='none', edgecolors='black', label='Rigid sphere')
plot_c3 = np.linspace(min(c_3list), max(c_3list), 100)
ax4.plot(plot_c3, sangani(plot_c3), color = 'black', label='Sangani & Acrivos(1982)')
ax4.set_xlabel(r'$c^{1/3}$')
ax4.set_ylabel(r'$V/W$')
ax4.legend()

print(speed_error_list)
ax5.plot(c_3list, speed_error_list, color='black', marker = '+')
ax5.set_xlabel(r'$c^{1/3}$')
ax5.set_ylabel(r'$\%error$')
ax5.set_yscale('log')



# fig.tight_layout()
# fig2.tight_layout()
# fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()

fig4.savefig(f'fig/sangani1982.pdf', bbox_inches = 'tight', format='pdf')
fig5.savefig(f'fig/sangani1982_error.pdf', bbox_inches = 'tight', format='pdf')
plt.show()