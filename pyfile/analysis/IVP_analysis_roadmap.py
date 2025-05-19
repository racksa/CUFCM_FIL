
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

path = "data/ic_hpc_sim/20240311_8/"
index = 0
path2 = "data/ic_hpc_sim/20240311_10/"
index2 = 12

path = "data/for_paper/roadmap/20250514/"
index = 0
path2 = "data/for_paper/roadmap/20250514/"
index2 = 1


time_array_symplectic = np.load(f"{path}time_array_index{index}.npy")
r_array_symplectic = np.load(f"{path}r_array_index{index}.npy")

time_array_diaplectic = np.load(f"{path2}time_array_index{index2}.npy")
r_array_diaplectic = np.load(f"{path2}r_array_index{index2}.npy")

stamp_index_symplectic = np.array([0, 50, 480, 942, 2526])
stamp_x_symplectic = time_array_symplectic[stamp_index_symplectic]
stamp_y_symplectic = r_array_symplectic[stamp_index_symplectic]

# stamp_index_diaplectic = np.array([0, 105, 194, 366, 918])
stamp_index_diaplectic = np.array([0, 40, 210, 535, 2517])
stamp_x_diaplectic = time_array_diaplectic[stamp_index_diaplectic]
stamp_y_diaplectic = r_array_diaplectic[stamp_index_diaplectic]
roman_symbols = ['I', 'II', 'III', 'IV', 'V']

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(time_array_symplectic, r_array_symplectic, c='black')
ax.plot(time_array_diaplectic, r_array_diaplectic, c='blue')
ax.scatter(stamp_x_symplectic, stamp_y_symplectic, marker='^', facecolor='none', s= 100, color='black')
ax.scatter(stamp_x_diaplectic, stamp_y_diaplectic, marker='o', facecolor='none', s= 100, color='black')
ax.set_xticks(np.linspace(0, 100, 5))
ax.set_yticks(np.linspace(0, 0.7, 3))

# for si, x in enumerate(stamp_x_symplectic):
#     if si < 1:
#         ax.plot([stamp_x_symplectic[si], -2], [stamp_y_symplectic[si], 0.1], c='black', linestyle='dashed')
#         ax.annotate(rf'{si+1}', (-2-1, 0.1+0.05), fontsize=22, va='center')
#         ax.plot([stamp_x_diaplectic[si], 8], [stamp_y_diaplectic[si], 0.03], c='black', linestyle='dashed')
#         ax.annotate(roman_symbols[si], (8, 0.03), fontsize=22, va='center')
#     else:
#         ax.annotate(rf'{si+1}', (stamp_x_symplectic[si]-1, stamp_y_symplectic[si]+0.05), fontsize=22, va='center')
#         ax.annotate(roman_symbols[si], (stamp_x_diaplectic[si]+1.3, stamp_y_diaplectic[si]), fontsize=22, va='center')


# for fi in range(n_folder_free):
#     plot_x = k_data_free[fi]
#     plot_y = r_data_free[fi]

#     cmap = plt.get_cmap(cmap_name)
#     color = cmap(tilt_data_free[fi]/max(tilt_data_free[fi]))
    
#     print(min(plot_x), max(plot_x))
#     ax.scatter(plot_x, plot_y, s=100, marker='+', c=color)

    # indices_symplectic = np.where(plot_y > .4)[0]
    # indices_diaplectic = np.where(plot_y  < .4)[0]

    # ax.scatter(plot_x[indices_symplectic], plot_y[indices_symplectic], s=100, marker='x', c='b')
    # ax.scatter(plot_x[indices_diaplectic], plot_y[indices_diaplectic], s=100, marker='+', c='b')

# colorbar
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
# vmin = 0
# vmax = np.max(tilt_data_free)
# norm = Normalize(vmin=vmin, vmax=vmax)
# sm = ScalarMappable(cmap=cmap_name, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm)
# cbar.ax.set_yticks(np.linspace(vmin, vmax, 3), ['0', 'π/4', 'π/2'])
# cbar.set_label(r"Tilt angle")    

# ax.scatter(-1, -1, marker='+', c='r', label='Held fixed - Symplectic')
# ax.scatter(-1, -1, marker='x', c='r', label='Held fixed - Diaplectic')
# ax.scatter(-1, -1, marker='X', c='r', label='Held fixed - Diaplectic(#k=2)')
# ax.scatter(-1, -1, marker='+', c='b', label='Free - Symplectic')
# ax.scatter(-1, -1, marker='x', c='b', label='Free - Diaplectic')

ax.set_xlabel(r'$t/T$')
ax.set_ylabel(r'$r$')
ax.set_ylim(0,0.7)
ax.set_xlim(None, 100)
# ax.legend()

plt.tight_layout()
fig.savefig(f'fig/order_parameter_roadmap.pdf', bbox_inches = 'tight', format='pdf')
fig.savefig(f'fig/order_parameter_roadmap.png', bbox_inches = 'tight', format='png', transparent=True)
plt.show()