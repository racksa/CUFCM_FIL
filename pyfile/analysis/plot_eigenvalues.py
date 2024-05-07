import numpy as np
import matplotlib.pyplot as plt


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
ax1.legend()
fig1.tight_layout()
fig1.savefig(f'fig/eigenvalues.pdf', bbox_inches = 'tight', format='pdf')
plt.show()
