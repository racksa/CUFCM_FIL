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

# debug
lower, upper = (0, 1)
print(f'lower = {lower}; upper = {upper}')
alpha_range = upper-lower
alpha = float(sys.argv[1])

# Initialise the driver
d = driver.DRIVER()
d.cuda_device = int(sys.argv[2])
d.category = f'bisection/{k_string}/'
d.date = f'alpha{alpha}'
d.dir = f"data/{d.category}{d.date}/"
os.system(f'mkdir -p {d.dir}')
d.change_variables(NFIL, NSEG, NBLOB, AR, k, T, 1.)
d.update_globals_file()


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

def read_input_state(filename):
    with open(filename, 'r') as file:
        num_lines = sum(1 for line in file)

        if num_lines == 1:
            full_input = np.loadtxt(filename)
        else:
            full_input = np.loadtxt(filename)[-1]
            
        full_input[2:2+NFIL] = util.box(full_input[2:2+NFIL], 2*np.pi)

        return full_input[2:]

def run(d):
    d.change_variables(NFIL, NSEG, NBLOB, AR, k, T, sim_length)
    d.update_globals_file()
    d.run()

def move_output_file(d, iter):
    body_states_filename = d.dir + d.simName + "_body_states.dat"
    fil_states_filename = d.dir + d.simName + "_true_states.dat"
    seg_states_filename = d.dir + d.simName + "_seg_states.dat"

    if os.path.exists(seg_states_filename):
        os.remove(seg_states_filename)

    if os.path.exists(body_states_filename):
        os.remove(body_states_filename)

    if os.path.exists(fil_states_filename):
        os.rename(fil_states_filename, d.dir + d.simName + f"_true_states_{iter}.dat")
        print(f"File '{fil_states_filename}' has been renamed successfully.")
    
def calculate_r(d):
    plot_end_frame_setting = 3000
    frames_setting = 3000

    plot_end_frame = min(plot_end_frame_setting, sum(1 for line in open(d.dir + d.simName + '_true_states.dat')))
    plot_start_frame = max(0, plot_end_frame-frames_setting)
    frames = plot_end_frame - plot_start_frame
    fil_references = read_fil_references(d.dir + d.simName + '_fil_references.dat')
    fil_states_f = open(d.dir + d.simName + '_true_states.dat', "r")

    time_array = np.arange(plot_start_frame, plot_end_frame )
    r_array = np.zeros(frames)

    for i in range(plot_end_frame):
        print(" frame ", i, "/", plot_end_frame, "          ", end="\r")
        fil_states_str = fil_states_f.readline()

        if(i>=plot_start_frame):
            fil_states = np.array(fil_states_str.split()[2:], dtype=float)
            fil_phases = fil_states[:NFIL]
            fil_phases = util.box(fil_phases, 2*np.pi)
            
            r_array[i-plot_start_frame] = np.abs(np.sum(np.exp(1j*fil_phases))/NFIL)

    return time_array, r_array




leftstate = read_input_state(f'data/{d.category}' + f"leftstate.dat")
rightstate = read_input_state(f'data/{d.category}' + f"rightstate.dat")


# initial_condition = np.zeros(np.shape(leftstate))
# initial_condition[NFIL:] = alpha*leftstate[NFIL:] + (1-alpha)*rightstate[NFIL:]
# initial_condition[:NFIL] = alpha*leftstate[:NFIL] + (1-alpha)*rightstate[:NFIL]
# initial_condition[:NFIL] = np.arctan2((alpha*np.sin(leftstate[:NFIL]) + (1-alpha)*np.sin(rightstate[:NFIL])),
#                             (alpha*np.cos(leftstate[:NFIL]) + (1-alpha)*np.cos(rightstate[:NFIL])))

# x = np.insert( initial_condition, 0, [k, T])
# np.savetxt(d.dir + "psi.dat", x, newline = " ")
# d.write_rules()
# move_output_file(d, 0)
# run(d)

# Plotting
# fil_references = read_fil_references(d.dir + d.simName + '_fil_references.dat')
fil_references = read_fil_references('data/bisection/ini_states/fil_references.dat')
fil_references_sphpolar = np.zeros((NFIL,3))
for i in range(NFIL):
    fil_references_sphpolar[i] = util.cartesian_to_spherical(fil_references[3*i: 3*i+3])
        
interpolate = False
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

    alpha = t/num_alpha

    initial_condition = np.zeros(np.shape(leftstate))
    initial_condition = alpha*leftstate + (1-alpha)*rightstate
    initial_condition[:NFIL] = np.arctan2((alpha*np.sin(leftstate[:NFIL]) + (1-alpha)*np.sin(rightstate[:NFIL])),
                                (alpha*np.cos(leftstate[:NFIL]) + (1-alpha)*np.cos(rightstate[:NFIL])))

    variables = initial_condition[:NFIL]
    variables = util.box(variables, 2*np.pi)

    ax.set_title(rf"$\alpha={t/num_alpha}$")
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

        
        colors_inter = scipy.interpolate.griddata((fil_references_sphpolar[:,1],fil_references_sphpolar[:,2]), colors, (xx, yy), method='linear')
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