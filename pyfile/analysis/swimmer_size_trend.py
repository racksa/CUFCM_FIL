
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

path = "data/giant_swimmer/combined_analysis/"


keywords = ['time_array_fil', 'wavenumber_array_fil']

arrays = {'fil_array': [],
          'time_array_fil': [],
          'wavenumber_array_fil': []}

filnumbers = []

def sort_key_frame(s):
    # Split the string by the underscore and convert the second part to an integer
    return int(s[len(keyword):-4])




# Regular expression pattern to extract the frame number
pattern = re.compile(rf"time_array_fil(\d+)\.npy")
for filename in os.listdir(path):
    match = pattern.match(filename)
    if match:
        nfil = int(filename[len('time_array_fil'):-4])
        for ki, keyword in enumerate(keywords):
            data = np.load(f"{path}{keyword}{nfil}.npy")
            arrays[keyword].append(data)

            if keyword == 'wavenumber_array_fil':
                print(ki, filename,  np.mean(arrays[keyword]))
        filnumbers.append(nfil)
        arrays['fil_array'].append(nfil)
                

L = 2.6*19
cilia_array_length = 0.5*(np.array([8.0, 15, 20, 30, 39])*L)*np.pi
avg_wavenumbers = np.zeros(len(arrays['fil_array']))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)

print(np.mean(arrays['wavenumber_array_fil'], axis=1))


for si in range(len(arrays['time_array_fil'])):
    nfil = arrays['fil_array'][si]
    time_array = arrays['time_array_fil'][si]
    wavenumber_array = arrays['wavenumber_array_fil'][si]
    time_array -= time_array[0]
    avg_wavenumbers[si] = np.mean(wavenumber_array)

    print(si, nfil, np.mean(wavenumber_array))
    
    ax.plot(time_array, wavenumber_array)

sorted_indices = np.argsort(filnumbers)
avg_wavenumbers = avg_wavenumbers[sorted_indices]
filnumbers = np.array(filnumbers)[sorted_indices]


ax2.plot(cilia_array_length, avg_wavenumbers, color='black', marker='+')



from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ax.set_xlabel(r'$t/T$')
ax.set_ylabel(r'$\Delta \psi$')
# ax.set_ylim(0)
# ax.set_xlim(0, 0.06)
# ax.legend()

ax2.set_xlabel(r'$Semi-circumference\ of\ swimmer$')
ax2.set_ylabel(r'Wavenumber')

# ax3.set_xlabel(r'tilt angle')
# ax3.set_ylabel(variable_label)  

fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
# fig.savefig(f'fig/order_parameter_tilt.pdf', bbox_inches = 'tight', format='pdf')
fig2.savefig(f'fig/wavenumbers.pdf', bbox_inches = 'tight', format='pdf')
# fig3.savefig(f'fig/rot_speed_vs_tilt.pdf', bbox_inches = 'tight', format='pdf')
plt.show()