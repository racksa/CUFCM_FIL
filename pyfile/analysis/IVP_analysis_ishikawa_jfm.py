
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update({'font.size': 16})

cmap_name = 'coolwarm'

path = "data/ishikawa/20240731_jfm/"
path = "data/ishikawa/20240827_ishikawa_jfm2/"

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)

k_list = [-1, 0, 0.5, 1.0, 1.5, 2.0]
labels = [r"$k=-1$",r"$k=0$",r"$k=0.5$",r"$k=1$",r"$k=1.5$",r"$k=2$",]
colors = ["brown","black","red","green","blue","purple"]

# plot sim data
for ind in range(1, 6):
    try:
        time_array = np.load(f"{path}time_array_index{ind}.npy")
        speed_array = np.load(f"{path}body_speed_array_index{ind}.npy")
        rot_speed_array = np.load(f"{path}body_rot_speed_array_index{ind}.npy")
        dissipation_array = np.load(f"{path}dissipation_array_index{ind}.npy")
        efficiency_array = np.load(f"{path}efficiency_array_index{ind}.npy")

        ax.plot(time_array, speed_array, label = rf'$k={k_list[ind]}$', alpha=1., c=colors[ind])
        ax2.plot(time_array, dissipation_array, label = k_list[ind])
        ax3.plot(time_array, efficiency_array, label = k_list[ind])
    except:
        pass

# plot extracted data
directory = 'pyfile/analysis/ishikawa_data/'
files = ['k0.0.csv', 'k0.5.csv', 'k1.0.csv', 'k1.5.csv', 'k2.0.csv']

# files = ['k0.0N162.csv', 'k0.0N636.csv', 'k0.0N2520.csv']
for i, filename in enumerate(files):
    file = open(directory + filename, mode='r')
    df = pd.read_csv(directory + filename, header=None)
    data = df.to_numpy()
    x, y = data[:,0], data[:,1]

    ax.plot(x, y, ls = 'dotted', alpha=0.5, c=colors[i+1])

legend1 = ax.legend(frameon=False)
line1, = ax.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label='data' )
line2, = ax.plot([-1, -1.1], [-1, -1.1], ls='dotted', c='black', label=r'$Ito.\ (2019)$')
legend2 = ax.legend(handles = [line1, line2], loc='upper right')
ax.add_artist(legend1)
ax.set_xlim(0, 1)
ax.set_xlabel(r'$t/T$')
ax.set_ylabel(r'$V_z/L$')


fig.tight_layout()
fig.savefig(f'fig/ishikawa_jfm_comparison.pdf', bbox_inches = 'tight', format='pdf')
plt.show()