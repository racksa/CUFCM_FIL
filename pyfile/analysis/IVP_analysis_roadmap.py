
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

cmap_name = 'bwr'

# path = "data/ic_hpc_sim/20240311_8/"
# index = 0
# path2 = "data/ic_hpc_sim/20240311_10/"
# index2 = 12

path = "data/for_paper/roadmap/20250728/"
index = 0
path2 = "data/for_paper/roadmap/20250728/"
index2 = 1


time_array_symplectic = np.load(f"{path}time_array_index{index}.npy")
r_array_symplectic = np.load(f"{path}r_array_index{index}.npy")

time_array_diaplectic = np.load(f"{path2}time_array_index{index2}.npy")
r_array_diaplectic = np.load(f"{path2}r_array_index{index2}.npy")

# stamp_index_symplectic = np.array([0, 50, 480, 942, 2526])
# stamp_x_symplectic = time_array_symplectic[stamp_index_symplectic]
# stamp_y_symplectic = r_array_symplectic[stamp_index_symplectic]

# stamp_index_diaplectic = np.array([0, 40, 210, 535, 2517])
# stamp_x_diaplectic = time_array_diaplectic[stamp_index_diaplectic]
# stamp_y_diaplectic = r_array_diaplectic[stamp_index_diaplectic]
# roman_symbols = ['I', 'II', 'III', 'IV', 'V']

line_indices = np.array([0, 120, 260, 535, 2517])

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1,1,1)
fig2 = plt.figure(figsize=(7, 4.8))
ax2 = fig2.add_subplot(1,1,1)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)

emerging_stop_index = 3000

sym_steady_start = 6000
sym_steady_stop = 6200
dia_steady_start = 6000
dia_steady_stop = 6200

ax.plot(time_array_symplectic, r_array_symplectic, c='black')
ax.plot(time_array_diaplectic, r_array_diaplectic, c='lightgrey')
ax.set_xticks(np.linspace(0, 300, 5))
ax.set_yticks(np.linspace(0, 0.7, 3))

ax.set_xlabel(r'$t/T$')
ax.set_ylabel(r'$r$')
ax.set_ylim(0,0.7)
ax.set_xlim(None, 300)
# ax.legend()

ax2.plot(time_array_symplectic[:emerging_stop_index], r_array_symplectic[:emerging_stop_index], c='black')
ax2.plot(time_array_diaplectic[:emerging_stop_index], r_array_diaplectic[:emerging_stop_index], c='lightgrey')
# ax2.scatter(stamp_x_symplectic, stamp_y_symplectic, marker='^', facecolor='none', s= 100, color='red', zorder=100)
# ax2.scatter(stamp_x_diaplectic, stamp_y_diaplectic, marker='o', facecolor='none', s= 100, color='red', zorder=100)
ymin, ymax= 0, 0.7
for i in range(5):
    ax2.vlines(time_array_diaplectic[line_indices[i]], ymin, ymax, color='black', linestyle='--', linewidth=1)
    ax2.text(time_array_symplectic[line_indices[i]], ymax, rf'$t_{i+1}$', ha='center', va='bottom', fontsize=24)
ax2.set_xticks(np.linspace(0, 100, 5))
ax2.set_yticks(np.linspace(ymin, ymax, 3))
ax2.set_xlim(-5, 100)
ax2.set_ylim(ymin, ymax)
# ax2.set_xlabel(r'$t/T$')
# ax2.set_ylabel(r'$r$')

ax3.plot(time_array_symplectic[sym_steady_start:sym_steady_stop], r_array_symplectic[sym_steady_start:sym_steady_stop], c='black')
ax3.set_xlim(time_array_symplectic[sym_steady_start], time_array_symplectic[sym_steady_stop])
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlabel('')
ax3.set_ylabel('')
# ax3.set_xlabel(r'$t/T$')
# ax3.set_ylabel(r'$r$')

ax4.plot(time_array_diaplectic[dia_steady_start:dia_steady_stop], r_array_diaplectic[dia_steady_start:dia_steady_stop], c='lightgrey')
ax4.set_xlim(time_array_symplectic[sym_steady_start], time_array_symplectic[sym_steady_stop])
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_xlabel('')
ax4.set_ylabel('')
# ax4.set_xlabel(r'$t/T$')
# ax4.set_ylabel(r'$r$')

fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig.savefig(f'fig/order_parameter_roadmap.png', bbox_inches = 'tight', format='png', transparent=True)
fig2.savefig(f'fig/order_parameter_roadmap_zoomin.png', bbox_inches = 'tight', format='png', transparent=True)
fig3.savefig(f'fig/order_parameter_roadmap_symplectic.png', bbox_inches = 'tight', format='png', transparent=True)
fig4.savefig(f'fig/order_parameter_roadmap_diaplectic.png', bbox_inches = 'tight', format='png', transparent=True)
plt.show()