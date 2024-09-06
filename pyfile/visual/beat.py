import numpy as np

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

def lung_cilia_shape(s, phase):
    pos = np.zeros(3)
#     svec = np.array([s, s**2, s**3])
#     fourier_dim = np.shape(Ax)[0]
    degrees = np.shape(Ax)[1]
    fourier_dim = np.shape(Ax)[0]
    svec = np.array([s**(i+1) for i in range(degrees)])
    cosvec = np.array([ np.cos(n*phase) for n in range(fourier_dim)])
    sinvec = np.array([ np.sin(n*phase) for n in range(fourier_dim)])
    cosvec[0] *= 0.5

    x = (cosvec@Ax + sinvec@Bx)@svec
    y = (cosvec@Ay + sinvec@By)@svec
    z = np.zeros(np.shape(x))

    return x, y, z


Ay_volvox = np.array([[-0.01359477, -0.2933222 ,  1.11787568, -0.5797987 ],
       [ 0.17304993, -2.07743633,  1.96507494, -0.48418322],
       [-0.13139014,  1.67275679, -3.30501852,  1.75899013],
       [-0.16029509,  0.86911573, -1.43711565,  0.70427363]])
Ax_volvox = np.array([[ 1.00173453e+00, -1.93437790e-01, -2.09032251e-01, 1.09657224e-01],
       [ 1.77973257e-03, -2.19512900e-01,  1.18549520e+00, -8.75601570e-01],
       [ 1.82160747e-01, -1.19973029e+00,  1.85563800e+00, -7.38088036e-01],
       [-1.18146940e-01,  1.11157299e+00, -2.22516778e+00, 1.24886035e+00]])
By_volvox = np.array([[-1.65423231e-14,  2.22044605e-16, -4.60742555e-15, -1.27675648e-15],
       [-7.16106264e-02,  2.20709826e+00, -4.29445659e+00, 2.11206054e+00],
       [-2.81241335e-01,  1.40574837e+00, -1.48026680e+00, 3.96486282e-01],
       [ 2.78457057e-02, -3.71722651e-01,  1.03114541e+00, -7.11831064e-01]])
Bx_volvox = np.array([[ 4.24660307e-15,  5.89719246e-15, -5.20417043e-15, -3.55271368e-15],
       [ 9.13270401e-02, -5.35979006e-01,  1.37873237e+00, -5.83970315e-01],
       [-9.93255062e-02,  1.16560579e+00, -2.64682150e+00, 1.58062759e+00],
       [-1.57456039e-01,  9.40433425e-01, -1.39914889e+00, 5.95880263e-01]])

def volvox_cilia_shape(s, phase):
    pos = np.zeros(3)
    # svec = np.array([s, s**2, s**3])
    degrees = np.shape(Ax_volvox)[1]
    fourier_dim = np.shape(Ax_volvox)[0]
    svec = np.array([s**(i+1) for i in range(degrees)])
    
    cosvec = np.array([ np.cos(n*phase) for n in range(fourier_dim)])
    sinvec = np.array([ np.sin(n*phase) for n in range(fourier_dim)])

    x = (cosvec@Ax_volvox + sinvec@Bx_volvox)@svec
    y = (cosvec@Ay_volvox + sinvec@By_volvox)@svec
    z = np.zeros(np.shape(x))

    return x, y, z
