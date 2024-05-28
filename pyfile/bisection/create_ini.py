#!/usr/bin/python3

import driver
import numpy as np
import util
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import os
import sys


# Initialisation
NSEG = 20      # Number of segments
NFIL = 159       # Number of filaments
NBLOB = 9000
AR = 8
T = 1
sim_length = 0.00

k = 0.010

k_string = f'ini_states'
iteration_string = 'view_ini'

def read_fil_references(fileName):
    try:
        with open(fileName, 'r') as file:
            lines = file.readlines()
            data = [line.strip().split() for line in lines]
            data = np.array(data, dtype=float)  # Assuming your data is numeric
            return data[:, :].reshape(-1)
    except Exception as e:
        print(f"Error: {e}")
        return []

# Read fiament position
fil_references = read_fil_references('data/bisection/ini_states/fil_references.dat')
fil_references_sphpolar = np.zeros((NFIL,3))
for i in range(NFIL):
    fil_references_sphpolar[i] = util.cartesian_to_spherical(fil_references[3*i: 3*i+3])

# Creating artificial basis states
leftstate = np.zeros(2*NFIL)
rightstate = np.zeros(2*NFIL)
for i in range(NFIL):
    leftstate[i] = fil_references_sphpolar[i,1]
    rightstate[i] = fil_references_sphpolar[i,2]

x = np.insert( leftstate, 0, [k, T])
np.savetxt(f'data/bisection/ini_states' + f"leftstate.dat", x, newline = " ")
x = np.insert( rightstate, 0, [k, T])
np.savetxt(f'data/bisection/ini_states' + f"rightstate.dat", x, newline = " ")


# Plotting
interpolate = True
video = True
num_alpha = 300

# colormap = 'cividis'
colormap = 'twilight_shifted'
# colormap = 'hsv'

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
vmin = 0
vmax = 2*np.pi
norm = Normalize(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.ax.set_yticks(np.linspace(vmin, vmax, 7), ['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])
cbar.set_label(r"phase")    

global frame
frame = 0
import scipy.interpolate

def animation_func(t):
    ax.cla()

    alpha = t/num_alpha * 0.6 + 0.2

    initial_condition = np.zeros(np.shape(leftstate))
    initial_condition = alpha*leftstate + (1-alpha)*rightstate
    initial_condition[:NFIL] = np.arctan2((alpha*np.sin(leftstate[:NFIL]) + (1-alpha)*np.sin(rightstate[:NFIL])),
                                (alpha*np.cos(leftstate[:NFIL]) + (1-alpha)*np.cos(rightstate[:NFIL])))

    variables = initial_condition[:NFIL]
    variables = util.box(variables, 2*np.pi)

    ax.set_title(rf"$\alpha={alpha:.3f}$")
    ax.set_ylabel(r"$\theta$")
    ax.set_xlabel(r"$\phi$")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, np.pi)
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5), ['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_yticks(np.linspace(0, np.pi, 5), ['0', 'π/4', 'π/2', '3π/4', 'π'])
    ax.invert_yaxis()
    fig.tight_layout()

    cmap = mpl.colormaps[colormap]
    colors = cmap(variables/vmax)

    # Interpolation
    if (interpolate):
        n1, n2 = 128, 128
        offset = 0.2
        azim_grid = np.linspace(min(fil_references_sphpolar[:,1])+offset, max(fil_references_sphpolar[:,1])-offset, n1)
        polar_grid = np.linspace(min(fil_references_sphpolar[:,2])+offset, max(fil_references_sphpolar[:,2])-offset, n2)
        xx, yy = np.meshgrid(azim_grid, polar_grid)
        xx, yy = xx.ravel(), yy.ravel()

        
        colors_inter = scipy.interpolate.griddata((fil_references_sphpolar[:,1],fil_references_sphpolar[:,2]), colors, (xx, yy), method='nearest')
        ax.scatter(xx, yy, c=colors_inter)

                
    else:
    # Individual filaments
        ax.scatter(fil_references_sphpolar[:,1], fil_references_sphpolar[:,2], c=colors)


if(video):
    for i in range(num_alpha):
        frame = i
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        ani = animation.FuncAnimation(fig, animation_func, frames=num_alpha, interval=1, repeat=False)
        plt.show()    
        # FFwriter = animation.FFMpegWriter(fps=16)
        # ani.save(f'fig/fil_phase_index{self.index}_{self.date}_anim.mp4', writer=FFwriter)
        ## when save, need to comment out plt.show() and be patient!
else:
    animation_func(alpha)
        
    # plt.savefig(f'fig/fil_phase_index{self.index}_{self.date}.pdf', bbox_inches = 'tight', format='pdf')
plt.show()