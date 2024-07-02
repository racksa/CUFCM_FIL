
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update({'font.size': 16})


def box(x, box_size):
    return x - np.floor(x/box_size)*box_size

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

        return full_input[2:]

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


fig = plt.figure()
ax = fig.add_subplot(1,1,1)

bisection_dir = 'data/bisection/k0.020/'
true_state_file = 'ciliate_159fil_9000blob_8.00R_0.0200torsion_true_states.dat'

sections = [section for section in os.listdir(bisection_dir) if os.path.isdir(os.path.join(bisection_dir, section))]
sections = sections

t_array = np.arange(len(sections))
r_array = np.zeros(t_array.shape)

cut_frames = np.loadtxt(f'{bisection_dir}cut_frame.dat', dtype='int')

colormap = 'jet'
cmap = mpl.colormaps[colormap]

last_t = 0
for si, section in enumerate(sections[:-1]):
    iterations = [iteration for iteration in os.listdir(bisection_dir+section) if os.path.isdir(os.path.join(bisection_dir+section, iteration))]
    print(f"---------------{section}")
    bisection_indices_filename = 'bisection_indices_1e-7.dat'
    bisection_indices = np.loadtxt(f'{bisection_dir}{section}/{bisection_indices_filename}', dtype='int')

    leftstate = read_input_state(f'{bisection_dir}{section}/' + f"leftstate.dat")
    rightstate = read_input_state(f'{bisection_dir}{section}/' + f"rightstate.dat")

    next_leftstate = read_input_state(f'{bisection_dir}{sections[si+1]}/' + f"leftstate.dat")
    next_rightstate = read_input_state(f'{bisection_dir}{sections[si+1]}/' + f"rightstate.dat")

    # calculate r based on the initial condition only
    # section_length = 1
    # intervals = [[0, 1]]
    # for ite, iteration in enumerate(iterations):
    #     pair = bisection_indices[ite]
    #     sec, num_sec = pair

    #     alpha_range = intervals[ite][1] - intervals[ite][0]
    #     section_length = alpha_range/(num_sec)
    #     lower = (sec)*section_length + intervals[ite][0]
    #     upper = (sec+1)*section_length + intervals[ite][0]
    #     intervals.append([lower, upper])
        
    #     print(iteration, pair, lower, upper)
    # alpha = lower

    # initial_condition = np.zeros(np.shape(leftstate))
    # NFIL = int(len(leftstate)/2)

    # initial_condition[NFIL:] = alpha*leftstate[NFIL:] + (1-alpha)*rightstate[NFIL:]
    # initial_condition[:NFIL] = np.arctan2((alpha*np.sin(leftstate[:NFIL]) + (1-alpha)*np.sin(rightstate[:NFIL])),
    #                             (alpha*np.cos(leftstate[:NFIL]) + (1-alpha)*np.cos(rightstate[:NFIL])))
    # initial_condition[:NFIL] = box(initial_condition[:NFIL], 2*np.pi)
    # r_array[si] = np.abs(np.sum(np.exp(1j*initial_condition[:NFIL]))/NFIL)

        
    # find where I cut them..
    for ite, iteration in enumerate(iterations[-1:]):
        sim_indices = [sim_index for sim_index in os.listdir(f"{bisection_dir}{section}/{iteration}") \
                   if os.path.isdir(os.path.join(f"{bisection_dir}{section}/{iteration}", sim_index))]
    
        plot_left = False
        plot_right = False
        cut_t = 0
        
        for sim in sim_indices:
            fil_states_f = open(f'{bisection_dir}{section}/{iteration}/{sim}/{true_state_file}', "r")
            sim_length = sum(1 for line in open(f'{bisection_dir}{section}/{iteration}/{sim}/{true_state_file}'))

            if plot_left and plot_right:
                last_t += cut_t + 1
                break
            plot_this = False

            t_array = (np.arange(sim_length) + last_t)/30
            r_array = np.zeros(sim_length)
            r_array_right = np.zeros(sim_length)
            break_counter = 0
            break_limit = 90000

            for t in range(sim_length):
                fil_states_str = fil_states_f.readline()
                fil_states = np.array(fil_states_str.split()[2:], dtype=float)
                NFIL = int(len(fil_states)/2)
                fil_states[:NFIL] = box(fil_states[:NFIL], 2*np.pi)
                phases = fil_states[:NFIL]

                if np.allclose(next_leftstate, fil_states):
                    print(sim, t)
                    plot_left = True
                    plot_this = True
                    cut_t = t
                if np.allclose(next_rightstate, fil_states):
                    print(sim, t)
                    plot_right = True
                    plot_this = True
                    cut_t = t
                
                if plot_this:
                    break_counter += 1
                if break_counter > break_limit or t == sim_length-1:
                    break

                r_array[t] = np.abs(np.sum(np.exp(1j*phases))/NFIL)
            
            if plot_this:
                color = cmap(si/len(sections))
                ax.plot(t_array[:t], r_array[:t], c = color, alpha =
                         0.5)
                ax.plot(t_array[:cut_t], r_array[:cut_t], c = 'black', alpha = 0.5)
                print(len(t_array[:t]))


    # calculate r based on the data
    # sim_indices = [sim_index for sim_index in os.listdir(f"{bisection_dir}{section}/{iterations[-1]}") \
    #                if os.path.isdir(os.path.join(f"{bisection_dir}{section}/{iterations[-1]}", sim_index))]
    
    # pair = bisection_indices[-1]
    # sec, num_sec = pair
        
    # fil_states_f = open(f'{bisection_dir}{section}/{iterations[-1]}/{sim_indices[sec]}/{true_state_file}', "r")
    # sim_length = sum(1 for line in open(f'{bisection_dir}{section}/{iterations[-1]}/{sim_indices[sec]}/{true_state_file}'))

    # for t in range(sim_length):
    #     fil_states_str = fil_states_f.readline()
    #     fil_states = np.array(fil_states_str.split()[2:], dtype=float)
    #     NFIL = int(len(fil_states)/2)
    #     fil_states[:NFIL] = box(fil_states[:NFIL], 2*np.pi)
    #     phases = fil_states[:NFIL]

        # r = np.abs(np.sum(np.exp(1j*phases))/NFIL)
        # r_array[si] += r/30
        
    # print(iterations[-1], bisection_indices[-1])
    # sim_indices = [sim_index for sim_index in os.listdir(f"{bisection_dir}{section}/{iterations[-1]}") \
    #                if os.path.isdir(os.path.join(f"{bisection_dir}{section}/{iterations[-1]}", sim_index))]
    # fil_states_f = open(f'{bisection_dir}{section}/{iterations[-1]}/{sim_indices[-1]}/{true_state_file}', "r")
    
    # sim_length = sum(1 for line in open(f'{bisection_dir}{section}/{iterations[-1]}/{sim_indices[-1]}/{true_state_file}'))
    # for t in range(30):
    #     fil_states_str = fil_states_f.readline()
    #     fil_states = np.array(fil_states_str.split()[2:], dtype=float)
    #     NFIL = int(len(fil_states)/2)
    #     fil_states[:NFIL] = box(fil_states[:NFIL], 2*np.pi)
    #     phases = fil_states[:NFIL]

    #     r = np.abs(np.sum(np.exp(1j*phases))/NFIL)
    #     r_array[si] += r/30
    


ax.set_ylim(0, 1)
ax.set_xlabel(r't/T')
ax.set_ylabel(r'$<r>$')
fig.savefig(f'fig/edge.pdf', bbox_inches = 'tight', format='pdf')
plt.show()
