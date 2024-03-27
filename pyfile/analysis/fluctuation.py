import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update({'font.size': 16})

def box(x, box_size):
    return x - np.floor(x/box_size)*box_size

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(1,1,1)

mrow = 100
accuracy_list = ['1e-4', '1e-5', '1e-6', '1e-7', '1e-8']
linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted', 'solid' ]
marker_list = ['', '', '', '', '']
group = 'group6'
for ai, accuracy in enumerate(accuracy_list):
    try:
        a = np.loadtxt(f"data/numeric_error/{group}/run_{accuracy}_1/ciliate_159fil_9000blob_8.00R_0.0140torsion_true_states.dat", max_rows=mrow)
        b = np.loadtxt(f"data/numeric_error/{group}/run_{accuracy}_2/ciliate_159fil_9000blob_8.00R_0.0140torsion_true_states.dat", max_rows=mrow)

        a = a[2:]
        b = b[2:]

        nfil = int(a.shape[1]/2)
        length = a.shape[0]
        aux = np.ones(nfil)*np.pi

        print(accuracy, length)

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
ax.set_ylabel(r'$\|\mathbf{x}_1 - \mathbf{x}_2\|_2$')
ax.set_xlabel(r'$t/T$')

# ax2.set_xlim(0)
# ax2.set_yscale('log')
# ax2.set_ylabel(r'$\frac{\|\mathbf{x}_1 - \mathbf{x}_2\|_1}{N}$')
# ax2.set_xlabel(r'$t/T$')

ax.legend()
# ax2.legend()
fig.tight_layout()
# fig2.tight_layout()
fig.savefig(f'fig/numeric_error_{group}.pdf', bbox_inches = 'tight', format='pdf')
            
plt.show()