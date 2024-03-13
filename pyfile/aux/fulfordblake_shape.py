import numpy as np
import matplotlib.pyplot as plt



def fitted_shape(s, phase):
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

    pos = np.zeros(3)
    svec = np.array([s, s**2, s**3])
    fourier_dim = np.shape(Ax)[0]
    cosvec = np.array([ np.cos(n*phase) for n in range(fourier_dim)])
    sinvec = np.array([ np.sin(n*phase) for n in range(fourier_dim)])

    x = (cosvec@Ax + sinvec@Bx)@svec
    y = (cosvec@Ay + sinvec@By)@svec
    z = np.zeros(np.shape(x))

    return x, y, z



s = np.linspace(0, 1, 20)
print(s)
x, y, z = fitted_shape(s, 2)
print(f'x shape = {x.shape}')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y)

plt.show()