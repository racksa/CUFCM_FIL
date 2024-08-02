import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import splprep, splev
import matplotlib as mpl
from numpy.linalg import lstsq



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

# print("Ax matrix:\n", Ax)
# print("Bx matrix:\n", Bx)
# print("Ay matrix:\n", Ay)
# print("By matrix:\n", By)


mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

cmap_name = 'hsv'

img_indices = np.arange(1, 29, 2)
num_phases = len(img_indices)

psi_values = np.arange(0, 2*np.pi/14*num_phases, 2*np.pi/14)
num_s = 40
data = np.zeros((num_phases, num_s, 2))
# print(psi_values, num_phases, 2*np.pi/14)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

# Read data
xy_ratio = 1.47
for indi, ind in enumerate(img_indices):
    xy = np.genfromtxt(f'frame{ind}.csv', delimiter=',', skip_header=1)
    # Interchange x and y
    x = xy[:, 1]
    y = xy[:, 0]/xy_ratio
    x -= x[0]
    y -= y[0]

    tck, u = splprep([x, y], s=0)

    # Generate new interpolated points
    unew = np.linspace(0, 1, num_s)
    out = np.array(splev(unew, tck))

    data[indi] = out.T

    ax1.plot(out[1], out[0], marker="+", label='Interpolated path', c='blue')

def ksi(s, A, B, phase):
    fourier_dim, degree = np.shape(A)
    # svec = np.array([s, s**2, s**3])
    svec = np.array([s**d for d in range(1, degree + 1)])

    cosvec = np.array([ np.cos(n*phase) for n in range(fourier_dim)])
    sinvec = np.array([ np.sin(n*phase) for n in range(fourier_dim)])
    cosvec[0] *= 0.5
    
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

degree = 5  # Degree of the polynomial basis
s_values = [calculate_arc_length(path) for path in data]
S_values = [construct_S(s, degree) for s in s_values]




s_flat = np.concatenate(s_values)
psi_flat = np.concatenate([[psi]*num_s for psi in psi_values])
x_flat = np.concatenate([path[:, 0] for path in data])
y_flat = np.concatenate([path[:, 1] for path in data])

S_flat = np.concatenate(S_values)
fourier_dim = 4 # Number of Fourier terms

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


print("Ax matrix:\n", A_x)
print("Bx matrix:\n", B_x)
print("Ay matrix:\n", A_y)
print("By matrix:\n", B_y)


num_recon_phase = num_phases*2
for p in range(num_recon_phase):
    s_value = np.linspace(0, 1, num_s)
    psi = 2*np.pi/num_recon_phase*p
    plot_x = ksi(s_value, A_x, B_x, psi)
    plot_y = ksi(s_value, A_y, B_y, psi)
    ax2.plot(plot_y, plot_x, c='black', alpha = 0.1+0.9*p/num_recon_phase)

ax1.set_aspect('equal')
ax2.set_aspect('equal')

fig1.tight_layout()
fig2.tight_layout()
fig1.savefig(f'extracted_raw_data.pdf', bbox_inches = 'tight', format='pdf', transparent=True)
fig2.savefig(f'fitted_fourier_series.pdf', bbox_inches = 'tight', format='pdf', transparent=True)

plt.show()
