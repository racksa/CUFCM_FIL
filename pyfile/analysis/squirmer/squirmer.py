import numpy as np
import scipy as sp
from scipy.special import legendre
from scipy.special import lpmv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os

#: Hyperparameter N
N = 17
zigma = 1.
epsilon = 0.05
radius = 1.
U = .05


#: Mode coefficients
B_1 = 1.
B_2 = 5. * B_1

def P(N, x):
    return legendre(N)(x)

def V(N, x):
    return -2./N/(N+1)*lpmv(1, N, x)

#: Number of points to visulise
no_of_points = 300

#: Original altitude angles of points
org_r = np.ones(no_of_points)
org_theta = np.linspace(0, np.pi, no_of_points)

#: A;titude angles of points
r = org_r
theta = org_theta


def u_mode( N, theta, r, A, B ):
    '''
    Return radial velocity mode
    '''
    # return (.5 * N * radius**N / r**N - (.5*N-1) * radius**(N+2)/r**(N+2) ) * A_N * P(N, np.cos(theta)) + \
    #        (radius**(N+2)/r**(N+2) - radius**N/r**N) * B_N * P(N, np.cos(theta))
    return radius**N/r**N * B * ( (radius**2/r**2 - 1) * P(N, np.cos(theta)) )


def v_mode( N, theta, r, A, B ):
    '''
    Return altitude velocity mode
    '''
    # return (.5 * N * radius**(N+2) / r**(N+2) - (.5*N-1) * radius**(N)/r**(N) ) * B_N * V(N, np.cos(theta)) + \
    #        .5*N*(.5*N-1)*(radius**(N)/r**(N) - radius**(N+2)/r**(N+2)) * A_N * V(N, np.cos(theta))
    return radius**N/r**N * B * ( ( .5*N*radius**N/r**N - (.5*N-1) )* V(N, np.cos(theta)) )

def r_mode( N, theta, r, a ):
    '''
    Return the displacement from r=radius
    '''
    return radius * epsilon * a * P(N, np.cos(theta))

def theta_mode( N, theta, r, b ):
    '''
    Return the angle displacement from their original position
    '''
    return epsilon * b * V(N, np.cos(theta))

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (phi, rho)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z
    
def spherical_to_cartesian_field(ur, utheta, uphi, theta, phi):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    ux = ur * sin_theta * cos_phi + utheta * cos_theta * cos_phi - uphi * sin_phi
    uy = ur * sin_theta * sin_phi + utheta * cos_theta * sin_phi + uphi * cos_phi
    uz = ur * cos_theta - utheta * sin_theta

    return ux, uy, uz

def rot_mat(quaternion):
    ret = np.zeros((3,3))
    ret[0, 0] = 1.0
    ret[1, 1] = 1.0
    ret[2, 2] = 1.0

    temp = 2.0*quaternion[1]*quaternion[1]
    ret[1, 1] -= temp;
    ret[2, 2] -= temp

    temp = 2.0*quaternion[2]*quaternion[2]
    ret[0, 0] -= temp
    ret[2, 2] -= temp

    temp = 2.0*quaternion[3]*quaternion[3]
    ret[0, 0] -= temp
    ret[1, 1] -= temp

    temp = 2.0*quaternion[1]*quaternion[2]
    ret[1, 0] = temp
    ret[0, 1] = temp

    temp = 2.0*quaternion[1]*quaternion[3]
    ret[2, 0] = temp
    ret[0, 2] = temp

    temp = 2.0*quaternion[2]*quaternion[3]
    ret[1, 2] = temp
    ret[2, 1] = temp

    temp = 2.0*quaternion[0]*quaternion[3]
    ret[1, 0] += temp;
    ret[0, 1] -= temp;

    temp = 2.0*quaternion[0]*quaternion[2]
    ret[2, 0] -= temp;
    ret[0, 2] += temp;

    temp = 2.0*quaternion[0]*quaternion[1];
    ret[2, 1] += temp;
    ret[1, 2] -= temp;

    return ret

def plot_swimmer( N ):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'ref.dat')

    poss = np.loadtxt(file_path)
    poss = poss.reshape(-1,3)

    #: Configure the plot
    # plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    body_pos = np.array([0, 0, 0])
    q = np.array([0.992782,0.0918451,0.0769908,-0.00528911])
    Q = rot_mat(q)

    axis_start = Q@np.array([0, 0, -35]) + body_pos
    axis_end = Q@np.array([0, 0, 35]) + body_pos
    ax.plot([axis_start[0], axis_end[0]], [axis_start[1], axis_end[1]], [axis_start[2], axis_end[2]], c='black')

    num_points = 300
    radius = np.linalg.norm(poss[0])
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x+body_pos[0], y+body_pos[1], z+body_pos[2], color='grey', alpha=0.5)
    
    for pos in poss:
        A_1 = 300
        B_1 = 50.
        phi = np.arctan2(pos[1], pos[0]);
        theta = np.arccos(pos[2]/np.linalg.norm(pos));
        
        ur = A_1 * P(1, np.cos(theta));
        utheta = B_1 * V(1, np.cos(theta));

        v = spherical_to_cartesian_field(ur, utheta, 0.0, theta, phi)
        v = Q@v
        pos = body_pos + Q@pos
        norm = 20
        ax.scatter(pos[0], pos[1], pos[2], c='r')
        ax.quiver(pos[0], pos[1], pos[2], v[0]/norm, v[1]/norm, v[2]/norm)
    
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.savefig(f'fig/squirming_mode.pdf', bbox_inches = 'tight', format='pdf')
    plt.show()
    

plot_swimmer( N )

