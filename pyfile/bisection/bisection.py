#!/usr/bin/python3

import driver
import numpy as np
import util


# Initialisation
NSEG = 20      # Number of segments
NFIL = 159       # Number of filaments
NBLOB = 9000
AR = 8
k = 0.005
T = 1

# Number of time steps (ndts) and fixT
ndts = 300
Max_iterations = 10
Ustar = np.zeros(2*NFIL)

# Initialise the driver
d = driver.DRIVER()
d.cuda_device = 5
d.category = 'bisection/'
d.date = 'k=0.005'
d.dir = f"data/{d.category}{d.date}/"

leftstate_filename = d.dir + f"leftstate.dat"
rightstate_filename = d.dir + f"rightstate.dat"



def read_input_state(filename):
    with open(filename, 'r') as file:
        num_lines = sum(1 for line in file)

        if num_lines == 1:
            full_input = np.loadtxt(filename)
        else:
            full_input = np.loadtxt(filename)[-1]
            
        full_input[2:2+NFIL] = util.box(full_input[2:2+NFIL], 2*np.pi)

        return full_input[1:]

def run(d):
    d.change_variables(NFIL, NSEG, NBLOB, AR, k, T, 1.)
    d.update_globals_file()
    d.run()

def read_output_state(d):
    output_filename = d.dir + d.simName + "_true_states.dat"
    U = np.loadtxt(output_filename)[-1][2:]
    return U

def identify_states(d):
    return 0

for i in range(Max_iterations):
    leftstate = read_input_state(leftstate_filename)
    rightstate = read_output_state(rightstate_filename)

    output_state = read_output_state(d)
    
    print(output_state)

    # Add stopping criteria here
    # Add update to the left or right state here


