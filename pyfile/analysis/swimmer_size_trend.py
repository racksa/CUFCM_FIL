
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update({'font.size': 16})

cmap_name = 'coolwarm'

path = "data/giant_swimmer/combined_analysis_force_rerun/"


# keywords = ['time_array_index', 'wavenumber_array_index', 'body_speed_array_index']

arrays = {'time_array_index': [],
          'wavenumber_array_index': [],
          'body_speed_array_index': [],
          'dissipation_array_index': []}



filnumbers = np.array([159, 639, 1128, 1763, 2539, 4291])
L = 2.6*19
cilia_array_length = 0.5*(np.array([8, 15, 20, 25, 30, 39])*L)*np.pi

avg_wavenumbers = np.zeros(len(filnumbers))
avg_body_speed = np.zeros(len(filnumbers))
avg_dissipation = np.zeros(len(filnumbers))
indices = []

def sort_key_frame(s):
    # Split the string by the underscore and convert the second part to an integer
    return int(s[len(keyword):-4])


# Regular expression pattern to extract the frame number
pattern = re.compile(rf"time_array_index(\d+)\.npy")
for filename in os.listdir(path):
    match = pattern.match(filename)
    if match:
        index = int(filename[len('time_array_index'):-4])
        indices.append(index)
        for keyword in arrays:
            data = np.load(f"{path}{keyword}{index}.npy")
            arrays[keyword].append(data)

            # if keyword == 'wavenumber_array_index':
            #     print(index, filename,  np.mean(data))
                

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)

# print(arrays['wavenumber_array_index'])


for si in indices:
    nfil = filnumbers[si]
    time_array = arrays['time_array_index'][si]
    time_array -= time_array[0]
    wavenumber_array = arrays['wavenumber_array_index'][si]
    avg_wavenumbers[si] = np.mean(wavenumber_array)

    body_speed_array = arrays['body_speed_array_index'][si]
    avg_body_speed[si] = np.mean(body_speed_array)

    dissipation_array = arrays['dissipation_array_index'][si]
    avg_dissipation[si] = np.mean(dissipation_array)

    print(si, nfil, np.mean(wavenumber_array))
    
    # ax.plot(time_array, wavenumber_array)

sorted_indices = np.argsort(indices)
avg_wavenumbers = avg_wavenumbers[sorted_indices]
avg_body_speed = avg_body_speed[sorted_indices]
avg_dissipation = avg_dissipation[sorted_indices]


ax.plot(cilia_array_length, avg_body_speed, color='black', marker='+')
ax2.plot(cilia_array_length, avg_wavenumbers, color='black', marker='+')
ax3.plot(filnumbers, avg_dissipation, color='black', marker='+')



from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ax.set_xlabel(r'$Semi-circumference\ of\ swimmer$')
ax.set_ylabel(r'$V_zT/L$')
# ax.set_ylim(0)
# ax.set_xlim(0, 0.06)
# ax.legend()

ax2.set_xlabel(r'$Semi-circumference\ of\ swimmer$')
ax2.set_ylabel(r'$Wavenumber$')


ax3.set_xlabel(r'$M$')
ax3.set_ylabel(r'$PT^2/\mu L^3$')

# ax3.set_xlabel(r'tilt angle')
# ax3.set_ylabel(variable_label)  

fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig.savefig(f'fig/speed_trend.pdf', bbox_inches = 'tight', format='pdf')
fig2.savefig(f'fig/wavenumber_trend.pdf', bbox_inches = 'tight', format='pdf')
fig3.savefig(f'fig/dissipation_trend.pdf', bbox_inches = 'tight', format='pdf')
plt.show()