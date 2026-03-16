import numpy as np
import matplotlib.pyplot as plt
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


def read_complex_numbers(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                # Splitting the line into individual complex numbers
                complex_numbers = np.array([np.complex128(num) for num in line.split()])
                data.append(complex_numbers)
    return np.array(data)

# File path
file_path = 'data/JFNK/soln/eigenvalues.dat'
data = read_complex_numbers(file_path)

k_array = data[:, 0]



fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.scatter(k_array.real, data[:, 1].real, c='blue', marker='v', label=r'$Re(\lambda_0)$')
ax1.scatter(k_array.real, data[:, 1].imag, c='blue', marker='*', label=r'$Im(\lambda_0)$')

ax1.scatter(k_array.real, data[:, 2].real, c='black', marker='v', label=r'$Re(\lambda_1)$')
ax1.scatter(k_array.real, data[:, 2].imag, c='black', marker='*', label=r'$Im(\lambda_1)$')
ax1.set_xlabel(r'$k$')
ax1.set_ylabel(r'$Re(\lambda)$,$Im(\lambda)$')
ax1.legend(fontsize=16, frameon=False)
fig1.tight_layout()
fig1.savefig(f'fig/eigenvalues.pdf', bbox_inches = 'tight', format='pdf')
plt.show()
