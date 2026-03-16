import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
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

def box(x, box_size):
    return x - np.floor(x/box_size)*box_size

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(1,1,1)

mrow = 3000
accuracy_list = ['1e-4', '1e-5', '1e-6', '1e-7', '1e-8']
linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted', 'solid' ]
marker_list = ['', '', '', '', '']
group = 'group_single'
for ai, accuracy in enumerate(accuracy_list):
    try:
        a = np.loadtxt(f"data/numeric_error/{group}/run_{accuracy}_1/ciliate_159fil_9000blob_8.00R_0.0800torsion_true_states.dat", max_rows=mrow)
        b = np.loadtxt(f"data/numeric_error/{group}/run_{accuracy}_2/ciliate_159fil_9000blob_8.00R_0.0800torsion_true_states.dat", max_rows=mrow)

        a = a[2:]
        b = b[2:]

        nfil = int(a.shape[1]/2)
        length = a.shape[0]
        aux = np.ones(nfil)*np.pi

        print(accuracy, length, a.shape)

        # print(a.shape, b.shape)
        error_norm_array = np.zeros(a.shape[0])
        error_avg_array = np.zeros(a.shape[0])

        for i in range(length):
            diff = a[i] - b[i]

            a[i][:nfil] = box(a[i][:nfil], 2*np.pi)

            error_avg = np.linalg.norm(diff, ord=1)/nfil
            error_avg_array[i] = error_avg
            error_norm = np.linalg.norm(diff)
            error_norm_array[i] = error_norm
            error = np.linalg.norm(diff) / np.linalg.norm(a[i])

        ax.plot(np.linspace(0,length/300,length), error_norm_array, c='black', linestyle=linestyle_list[ai], marker=marker_list[ai], label = f"TOL={accuracy}")
        # ax2.plot(np.linspace(0,length/300,length), error_avg_array, c='black', linestyle=linestyle_list[ai], marker=marker_list[ai], label = f"TOL={accuracy}")
    except:
        pass

ax.set_xlim(0)
ax.set_yscale('log')
ax.set_ylabel(r'$\|\mathbf{\Phi}_1 - \mathbf{\Phi}_2\|_2$')
ax.set_xlabel(r'$t/T$')

# ax2.set_xlim(0)
# ax2.set_yscale('log')
# ax2.set_ylabel(r'$\frac{\|\mathbf{x}_1 - \mathbf{x}_2\|_1}{N}$')
# ax2.set_xlabel(r'$t/T$')

ax.legend(fontsize=16, frameon=False)
# ax2.legend()
fig.tight_layout()
# fig2.tight_layout()
fig.savefig(f'fig/numeric_error_{group}.pdf', bbox_inches = 'tight', format='pdf')
            
plt.show()