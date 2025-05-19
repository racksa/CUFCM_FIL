
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re
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

path = "data/giant_swimmer/combined_analysis_force_rerun/"

path = "data/for_paper/giant_swimmer_rerun/20250516_force/"


# keywords = ['time_array_index', 'wavenumber_array_index', 'body_speed_array_index']

arrays = {'time_array_index': [],
          'wavenumber_array_index': [],
          'body_speed_array_index': [],
          'dissipation_array_index': []}



filnumbers = np.array([159, 639, 1128, 1763, 2539, 4291])
L = 2.6*19
RoL = 0.5*np.array([8, 15, 20, 25, 30, 39])
cilia_array_length = RoL*L*np.pi

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
# fig3 = plt.figure()
# ax3 = fig3.add_subplot(1,1,1)
fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)

ax3 = ax.twinx()  


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

print(avg_wavenumbers)

ax.plot(RoL, avg_body_speed, color='black', marker='+')
ax2.plot(RoL, avg_wavenumbers, color='black', marker='+')
ax3.plot(RoL, avg_dissipation/1e4, color='blue', marker='+')
ax4.plot(RoL, cilia_array_length/avg_wavenumbers/L, color='black', marker='+')



from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ax.set_xlabel(r'$R/L$')
ax.set_ylabel(r'$<V>T/L$')  
ax.set_ylim(0.0, 0.2)
# ax.set_xlim(0, 0.06)
# ax.legend()

ax2.set_xlabel(r'$R/L$')
ax2.set_ylabel(r'$Wavenumber$')

ax3.annotate(r'$\times 10^{4}$', 
             xy=(1, 1), xycoords='axes fraction', 
             fontsize=20, ha='left', va='bottom', color='b')


ax3.set_xlabel(r'$R/L$')
ax3.set_ylabel(r'$<\mathcal{R}>T^2/\mu L^3$', color='blue')
ax3.tick_params(axis="y", labelcolor='blue')

ax4.set_xlabel(r'$R/L$')
ax4.set_ylabel(r'$\lambda/L$')
ax4.set_ylim(23, 70)

# ax3.set_xlabel(r'tilt angle')
# ax3.set_ylabel(variable_label)  

fig.tight_layout()
fig2.tight_layout()
# fig3.tight_layout()
fig4.tight_layout()
fig.savefig(f'fig/speed_trend.pdf', bbox_inches = 'tight', format='pdf')
fig2.savefig(f'fig/wavenumber_trend.pdf', bbox_inches = 'tight', format='pdf')
# fig3.savefig(f'fig/dissipation_trend.pdf', bbox_inches = 'tight', format='pdf')
fig4.savefig(f'fig/wavelength_trend.pdf', bbox_inches = 'tight', format='pdf')
fig4.savefig(f'fig/wavelength_trend.png', bbox_inches = 'tight', format='png', transparent=True)
plt.show()