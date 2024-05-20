#!/usr/bin/python3

import driver
import numpy as np
import util
import matplotlib.pyplot as plt
import os
import sys


# Initialisation
NSEG = 20      # Number of segments
NFIL = 159       # Number of filaments
NBLOB = 9000
AR = 8
T = 1
sim_length = 0.0034
sim_length = 100

k = 0.010

k_string = f'ini_states'
iteration_string = 'view_ini'

# Bisection
# sec = int(sys.argv[1]) # index of evaluation points, [1, par-1]
# par = int(sys.argv[2]) # num of sections

# bisection_indices = np.loadtxt(f'data/bisection/ini_states/bisection_indices.dat', dtype='int')
# if bisection_indices.size==0:
#     print("empty")
#     bisection_indices = [[0, 1]]

# bisection_indices = [[0, 1], [7, 8]]

# intervals = [[0, 1]]
# for ite, pair in enumerate(bisection_indices):
#     section, num_sec = pair
#     alpha_range = intervals[ite][1] - intervals[ite][0]
#     section_length = alpha_range/(num_sec)
#     print(f'section_length = {section_length}')
    

#     lower = (section)*section_length + intervals[ite][0]
#     upper = (section+1)*section_length + intervals[ite][0]
#     intervals.append([lower, upper])

#     print(f'interval = {intervals[-1]}')
#     print('--------------------')



# lower, upper = intervals[-1]
# print(f'lower = {lower}; upper = {upper}')
# alpha_range = upper-lower
# alpha = (sec)/(par)*alpha_range + lower

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


initial_condition = np.zeros(np.shape(leftstate))
initial_condition[NFIL:] = alpha*leftstate[NFIL:] + (1-alpha)*rightstate[NFIL:]
initial_condition[:NFIL] = alpha*leftstate[:NFIL] + (1-alpha)*rightstate[:NFIL]
initial_condition[:NFIL] = np.arctan2((alpha*np.sin(leftstate[:NFIL]) + (1-alpha)*np.sin(rightstate[:NFIL])),
                            (alpha*np.cos(leftstate[:NFIL]) + (1-alpha)*np.cos(rightstate[:NFIL])))

# print(np.shape(((alpha*np.sin(leftstate[:NFIL]) + (1-alpha)*np.sin(rightstate[:NFIL])),
#                             (alpha*np.cos(leftstate[:NFIL]) + (1-alpha)*np.cos(rightstate[:NFIL])))))

# initial_condition = alpha*leftstate + (1-alpha)*rightstate


x = np.insert( initial_condition, 0, [k, T])
np.savetxt(d.dir + "psi.dat", x, newline = " ")
d.write_rules()
move_output_file(d, 0)
run(d)
