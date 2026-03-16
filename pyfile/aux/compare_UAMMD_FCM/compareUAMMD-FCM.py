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

a = np.array([
    0.02594395, 0.02559936, 0.02610200, 0.02603981, 0.02578173, 0.02614055, 0.02609002
])

NPTS = np.array([
    308, 360, 400, 448, 500, 540, 588
])

ngd = np.array([
    6, 8, 10, 12, 14, 16, 19
])

tolerance = np.array([
    1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8
])

tolerance_txt = [
    'TOL=1e-2', 'TOL=1e-3', 'TOL=1e-4', 'TOL=1e-5', 'TOL=1e-6', 'TOL=1e-7', 'TOL=1e-8'
]

v_percentage_error = np.array([
    0.25569558615117755, 0.03744556709010038, 0.004944955511937988, 0.0006082118093558622, 7.496046054081896e-05, 7.650313005545048e-06, 6.844941950438487e-07
])

w_percentage_error = np.array([
   0.05975866795297148, 0.017394730201607287, 0.0045918599762769284, 0.0010432102786717746, 0.00020241769647921445, 3.3754897629114046e-05, 4.8444808805084595e-06
])



fig, ax1 = plt.subplots()
ax2 = ax1.twiny()
ax1.scatter(ngd, v_percentage_error, color='r')
ax1.plot(ngd, v_percentage_error, label=r"$M_G$", color='r')
ax2.scatter(NPTS, v_percentage_error, color='b')
ax2.plot(NPTS, v_percentage_error, label=r"$M_x$", color='b')
# ax1.set_title("Linear V relative error between UAMMD-FCM and C-FCM")
ax1.set_yscale("log")
ax1.set_xlabel(r"$M_G$")
# ax1.set_ylabel(r"$\langle | \frac{V_{FFCM}-V_{UAMMD}}{V_{UAMMD}} | \rangle $")
ax1.set_ylabel(r"$\epsilon_c$")
ax1.set_xticks(ngd)
ax2.set_xlabel(r"$M_x$")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0, frameon=False)

for i, txt in enumerate(tolerance_txt):
    ax2.annotate(txt, (NPTS[i], v_percentage_error[i]))

fig.tight_layout()
fig.savefig("fig/compare_v.png", format="png", bbox_inches='tight', transparent=True)



fig2, ax1 = plt.subplots()
ax2 = ax1.twiny()
ax1.scatter(ngd, w_percentage_error, color='r')
ax1.plot(ngd, w_percentage_error, label=r"$M_G$", color='r')
ax2.scatter(NPTS, w_percentage_error, color='b')
ax2.plot(NPTS, w_percentage_error, label=r"$M_x$", color='b')
# ax1.set_title("Angular W relative error between UAMMD-FCM and C-FCM")
ax1.set_yscale("log")
ax1.set_xlabel(r"$M_G$")
# ax1.set_ylabel(r"$\langle | \frac{V_{FFCM}-V_{UAMMD}}{V_{UAMMD}} | \rangle $")
ax1.set_ylabel(r"$\epsilon_c$")
ax1.set_xticks(ngd)
ax2.set_xlabel(r"$M_x$")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0, frameon=False)

for i, txt in enumerate(tolerance_txt):
    ax2.annotate(txt, (NPTS[i], w_percentage_error[i]))

# fig2.savefig("compare_w.eps", format="eps", bbox_inches='tight')
fig2.tight_layout()
fig2.savefig("fig/compare_w.png", format="png", bbox_inches='tight', transparent=True)
plt.show()






















#