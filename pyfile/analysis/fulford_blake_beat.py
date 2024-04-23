import numpy as np
import matplotlib.pyplot as plt

# Fourier coeffs for the shape
Ay = np.array([[-3.3547e-01, 4.0369e-01, 1.0362e-01], \
            [4.0318e-01, -1.5553e+00, 7.3455e-01], \
            [-9.9513e-02, 3.2829e-02, -1.2106e-01], \
            [8.1046e-02, -3.0982e-01, 1.4568e-01]])

Ax = np.array([[9.7204e-01, -2.8315e-01, 4.9243e-02], \
            [-1.8466e-02, -1.2926e-01, 2.6981e-01], \
            [1.6209e-01, -3.4983e-01, 1.9082e-01], \
            [1.0259e-02, 3.5907e-02, -6.8736e-02]])

By = np.array([[0, 0, 0], \
            [2.9136e-01, 1.0721e+00, -1.0433e+00], \
            [6.1554e-03, 3.2521e-01, -2.8315e-01], \
            [-6.0528e-02, 2.3185e-01, -2.0108e-01]])

Bx = np.array([[0, 0, 0], \
            [1.9697e-01, -5.1193e-01, 3.4778e-01], \
            [-5.1295e-02, 4.3396e-01, -3.3547e-01], \
            [1.2311e-02, 1.4157e-01, -1.1695e-01]])

def fitted_shape(s, phase):
    pos = np.zeros(3)
    svec = np.array([s, s**2, s**3])
    fourier_dim = np.shape(Ax)[0]
    cosvec = np.array([ np.cos(n*phase) for n in range(fourier_dim)])
    sinvec = np.array([ np.sin(n*phase) for n in range(fourier_dim)])

    x = (cosvec@Ax + sinvec@Bx)@svec
    y = (cosvec@Ay + sinvec@By)@svec
    z = np.zeros(np.shape(x))

    return x, y, z



fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

division = 20
Nseg = 20
phase_shift = -np.pi
for p in range(division):
    phase = 2*np.pi/division*p + phase_shift
    s  = np.linspace(0, 1, Nseg)
    x_array, y_array, z_array = fitted_shape(s, phase)

    ax1.plot(y_array, x_array, color='black', alpha=1-(0.9*p/division+0.1))

origin = np.array([0.3, 0.03])
axisl = 0.2
ax1.scatter(0, 0, s=60, c = 'black')
ax1.arrow(origin[0], origin[1], axisl, 0.0, width=0.008, linewidth=0.01, color='black')
ax1.arrow(origin[0], origin[1], 0.0, axisl, width=0.008, linewidth=0.01, color='black' )
ax1.annotate('x', origin + np.array([axisl+0.07, 0]), fontsize=25, va='center')
ax1.annotate('y', origin + np.array([0, axisl+0.07]), fontsize=25, ha='center')
ax1.annotate(r'$\mathbf{x}_b$', (0.03, 0.0), fontsize=25, va='center')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_aspect('equal')
# ax1.set_ylim(0)
ax1.axis('off')
fig1.savefig(f'fig/fulford_blake_beat.pdf', bbox_inches = 'tight', format='pdf')
plt.show()