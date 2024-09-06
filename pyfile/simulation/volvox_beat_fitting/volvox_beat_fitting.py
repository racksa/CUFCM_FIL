import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
import matplotlib as mpl
from numpy.linalg import lstsq
import matplotlib.animation as animation
import os
import re
from scipy.integrate import quad

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

cmap_name = 'hsv'



# Fourier coeffs for the shape
# Ay = np.array([[-3.3547e-01, 4.0369e-01, 1.0362e-01], \
#             [4.0318e-01, -1.5553e+00, 7.3455e-01], \
#             [-9.9513e-02, 3.2829e-02, -1.2106e-01], \
#             [8.1046e-02, -3.0982e-01, 1.4568e-01]])

# Ax = np.array([[9.7204e-01, -2.8315e-01, 4.9243e-02], \
#             [-1.8466e-02, -1.2926e-01, 2.6981e-01], \
#             [1.6209e-01, -3.4983e-01, 1.9082e-01], \
#             [1.0259e-02, 3.5907e-02, -6.8736e-02]])

# By = np.array([[0, 0, 0], \
#             [2.9136e-01, 1.0721e+00, -1.0433e+00], \
#             [6.1554e-03, 3.2521e-01, -2.8315e-01], \
#             [-6.0528e-02, 2.3185e-01, -2.0108e-01]])

# Bx = np.array([[0, 0, 0], \
#             [1.9697e-01, -5.1193e-01, 3.4778e-01], \
#             [-5.1295e-02, 4.3396e-01, -3.3547e-01], \
#             [1.2311e-02, 1.4157e-01, -1.1695e-01]])


# Directory containing the CSV files
save = False
directory = "pyfile/simulation/volvox_beat_fitting/"
keyword = 'frame'

# save = True

# Regular expression pattern to extract the frame number
pattern = re.compile(rf"{keyword}(\d+)\.csv")
def sort_key_frame(s):
    # Split the string by the underscore and convert the second part to an integer
    return int(s[len(keyword):-4])

datafiles = []
for filename in os.listdir(directory):
    match = pattern.match(filename)
    if match:
        datafiles.append(filename)

datafiles = sorted(datafiles, key=sort_key_frame)

num_phases = len(datafiles)
phase_diff = 2*np.pi/num_phases

psi_values = np.arange(0, phase_diff*num_phases, phase_diff)
num_s = 20
data = np.zeros((num_phases, num_s, 2))


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)


degree = 4  # Degree of the polynomial basis
fourier_dim = 4 # Number of Fourier terms


def calculate_real_length(path):
    distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    s = np.insert(np.cumsum(distances), 0, 0)
    return s[-1]

# Read data
# xy_ratio = 1.46
xy_ratio = 1
for indi, datafile in enumerate(datafiles):
    xy = np.genfromtxt(directory + datafile, delimiter=',', skip_header=1)
    # Interchange x and y
    x = xy[:, 1]
    y = xy[:, 0]/xy_ratio
    x -= x[0]
    y -= y[0]

    tck, u = splprep([x, y], s=0)

    # Generate new interpolated points
    unew = np.linspace(0, 1, num_s)
    out = np.array(splev(unew, tck))

    total_length = calculate_real_length(out.T)
    out /= total_length

    data[indi] = out.T

    ax1.plot(out[1], out[0], marker="+", label='Interpolated path', c='blue')


def ksi(s, A, B, phase):
    fourier_dim, degree = np.shape(A)
    # svec = np.array([s, s**2, s**3])
    svec = np.array([s**d for d in range(1, degree + 1)])

    cosvec = np.array([ np.cos(n*phase) for n in range(fourier_dim)])
    sinvec = np.array([ np.sin(n*phase) for n in range(fourier_dim)])

    return (cosvec@A + sinvec@B)@svec

def calculate_arc_length(path):
    distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    s = np.insert(np.cumsum(distances), 0, 0)
    s /= s[-1]  # Normalize to range [0, 1]
    return s

def construct_S(s, degree):
    return np.vstack([s**d for d in range(1, degree + 1)]).T

def fourier_design_matrix(S, psi, fourier_dim):
    N = len(psi)
    basis_size = S.shape[1]
    design_matrix = np.zeros((N, 2*fourier_dim*basis_size))
    for n in range(fourier_dim):
        cos_term = np.cos((n) * psi)[:, None] * S
        sin_term = np.sin((n) * psi)[:, None] * S
        design_matrix[:, n*basis_size:(n+1)*basis_size] = cos_term
        design_matrix[:, (fourier_dim + n)*basis_size:(fourier_dim + n + 1)*basis_size] = sin_term
    return design_matrix

s_values = [calculate_arc_length(path) for path in data]
S_values = [construct_S(s, degree) for s in s_values]



s_flat = np.concatenate(s_values)
psi_flat = np.concatenate([[psi]*num_s for psi in psi_values])
x_flat = np.concatenate([path[:, 0] for path in data])
y_flat = np.concatenate([path[:, 1] for path in data])

S_flat = np.concatenate(S_values)

X_design = fourier_design_matrix(S_flat, psi_flat, fourier_dim)

# Fit for x and y coordinates
coefficients_x, _, _, _ = lstsq(X_design, x_flat, rcond=None)
coefficients_y, _, _, _ = lstsq(X_design, y_flat, rcond=None)

# Extract A and B matrices
basis_size = S_flat.shape[1]
A_x = coefficients_x[:fourier_dim*basis_size].reshape((fourier_dim, basis_size))
B_x = coefficients_x[fourier_dim*basis_size:].reshape((fourier_dim, basis_size))
A_y = coefficients_y[:fourier_dim*basis_size].reshape((fourier_dim, basis_size))
B_y = coefficients_y[fourier_dim*basis_size:].reshape((fourier_dim, basis_size))

def print_matrixA(matrix_name, matrix):
    print(f"{matrix_name} = matrix({fourier_dim},{degree});")
    rows, cols = matrix.shape
    for col in range(cols):
        for row in range(rows):
            print(f"{matrix_name}({row},{col}) = {matrix[row, col]:.8f};")

def print_matrixB(matrix_name, matrix):
    print(f"{matrix_name} = matrix({fourier_dim-1},{degree});")
    rows, cols = matrix.shape
    for col in range(cols):
        for row in range(1, rows):
            print(f"{matrix_name}({row-1},{col}) = {matrix[row, col]:.8f};")

# Output matrices in the desired format
print_matrixA('Ay', A_y)
print_matrixA('Ax', A_x)
print_matrixB('By', B_y)
print_matrixB('Bx', B_x)

print(repr(A_y))
print(repr(A_x))
print(repr(B_y))
print(repr(B_x))

portion = 0.52
num_eff_beat = 32
num_rec_beat = 32
recon_phases = np.linspace(0, 2*np.pi*portion, num_eff_beat)
diff = recon_phases[1]-recon_phases[0]
recon_phases = np.append(recon_phases, np.linspace(2*np.pi*portion + diff, 2*np.pi, num_rec_beat)[:-1])

for ind, psi in enumerate(recon_phases):
    s_value = np.linspace(0, 1, num_s)
    plot_x = ksi(s_value, A_x, B_x, psi)
    plot_y = ksi(s_value, A_y, B_y, psi)
    color = 'r'
    if psi > 2*np.pi*portion:
        color = 'b'
    # ax2.plot(plot_y, plot_x, c=color, marker="+",)
    ax2.plot(plot_y, plot_x, c='black', alpha=0.1+0.9*psi/np.pi/2.)

    path = np.array([plot_y, plot_x]).T

    if save:
        np.savetxt(f"fitted_frame{ind}.csv", np.array([plot_y, plot_x]).T, delimiter=",")


ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax2.set_axis_off()

fig1.tight_layout()
fig2.tight_layout()
# fig1.savefig(f'extracted_raw_data.pdf', bbox_inches = 'tight', format='pdf', transparent=True)
# fig2.savefig(f'fitted_fourier_series.pdf', bbox_inches = 'tight', format='pdf', transparent=True)

def animation_func(t):
    ax4.cla()
    ax4.set_title(rf"${t}$")
    # ax.set_ylabel(r"$\theta$")
    # ax.set_xlabel(r"$\phi$")
    ax4.set_xlim(-0.5, 0.9)
    ax4.set_ylim(0, 1.5)
    ax4.set_aspect('equal')

    s_value = np.linspace(0, 1, num_s)
    psi = recon_phases[t]
    plot_x = ksi(s_value, A_x, B_x, psi)
    plot_y = ksi(s_value, A_y, B_y, psi)
    ax4.plot(plot_y, plot_x, c='black')

for psi in recon_phases:
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    ani = animation.FuncAnimation(fig4, animation_func, frames=len(recon_phases), interval=1, repeat=True)
    plt.show()

plt.show()
