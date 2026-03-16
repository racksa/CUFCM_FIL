from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import os
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
plt.rcParams.update({'font.size': 16})


tolerance_array = np.array([1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
cpu_ptps = np.array([49193, 36642, 21412, 17265, 9745])
gpu_ptps_best = np.array([1363010, 888726, 566342, 398849, 256872 ])
gpu_ptps_linklist = np.array([772771, 704082, 529951, 380010, 254849])
gpu_ptps_precompute = np.array([408469, 221378, 129920, 81896, 46341])

gpu_ptps_BPP = np.array([1363010, 888726, 566342, 398849, 256872 ])


ratio = gpu_ptps_best / cpu_ptps


fig, ax1 = plt.subplots()


# line2, = ax1.plot(tolerance_array, gpu_ptps_best, color='red', marker='o', linestyle='-', lw=2, label = 'GPU best')
# line3, = ax1.plot(tolerance_array, gpu_ptps_linklist, color='grey', marker='o', linestyle='dashed', lw=1, label = 'GPU linklist')
# line4, = ax1.plot(tolerance_array, gpu_ptps_precompute, color='blue', marker='o', linestyle='dotted', lw=1, label = 'GPU pre-compute')
line1, = ax1.plot(tolerance_array, cpu_ptps, color='black', marker='o', linestyle='-', lw=2, label = 'CPU MPI')
line5, = ax1.plot(tolerance_array, gpu_ptps_best, color='red', marker='o', linestyle='-', lw=2, label = 'GPU CUDA-BPP')

ax1.legend(frameon=False)
ax1.set_ylabel(r'$PTPS$')
ax1.set_xlabel(r'$Tolerance$')
ax1.set_xscale('log')
ax1.set_yscale('log')
fig.tight_layout()
# fig.savefig("compare_gpu_cpu_time.eps", format="eps", bbox_inches='tight')
fig.savefig("fig/compare_gpu_cpu_time.png", format="png", bbox_inches='tight', transparent=True)

fig, ax1 = plt.subplots()

line3, = ax1.plot(tolerance_array, ratio, color='black', lw=1,)

ax1.set_ylabel(r'$PTPS_{gpu}/PTPS_{cpu}$')
ax1.set_xlabel(r'TOL')
ax1.set_xscale('log')
# ax2.set_yscale('log')

fig.tight_layout()
# fig.savefig("ratio.eps", format="eps", bbox_inches='tight')
fig.savefig("fig/ratio.png", format="png", bbox_inches='tight', transparent=True)
plt.show()
















#