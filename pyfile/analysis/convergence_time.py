
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

k_array = np.array([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08,\
                    0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08,\
                    0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08,\
                    0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08,\
                    0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08,\
                    0.007, 0.012, 0.017, 0.022, 0.027, 0.032, 0.037, 0.042, 0.047, 0.052, 0.057, 0.062, 0.067, 0.072, 0.077, 0.082,\
                    0.007, 0.012, 0.017, 0.022, 0.027, 0.032, 0.037, 0.042, 0.047, 0.052, 0.057, 0.062, 0.067, 0.072, 0.077, 0.082,\
                    0.007, 0.012, 0.017, 0.022, 0.027, 0.032, 0.037, 0.042, 0.047, 0.052, 0.057, 0.062, 0.067, 0.072, 0.077, 0.082,\
                    0.006, 0.011, 0.016, 0.021, 0.026, 0.031, 0.036, 0.041, 0.046, 0.051, 0.056, 0.061, 0.066, 0.071, 0.076, 0.081,\
                    0.006, 0.011, 0.016, 0.021, 0.026, 0.031, 0.036, 0.041, 0.046, 0.051, 0.056, 0.061, 0.066, 0.071, 0.076, 0.081,\
                    
                                ])

convergence_time = np.array([40, 90, 160, 75, 65, 150, 100, 210, 160, 260, 830, 210, 90, 120, 370, 500,\
                             50, 210, 150, 70, 100, 120, 100, 120, 160, 100, 730, 110, 2000, 120, 370, 130,\
                             80, 150, 150, 40, 120, 125, 140, 150, 175, 270, 140, 2000, 550, 640, 110, 400, \
                             55, 50, 60, 120, 70, 120, 160, 70, 210, 190, 250, 760, 130, 440, 3400, 130,\
                             50, 150, 30, 80, 100, 200, 125, 730, 110, 120, 760, 140, 120, 210, 4000, 1500,\
                             60, 60, 100, 190, 50, 100, 120, 40, 210, 90, 160, 140, 690, 680, 1200, 80,\
                             40, 90, 40, 100, 40, 150, 130, 70, 80, 100, 110, 230, 115, 130, 470, 3300,\
                             30, 100, 70, 70, 40, 75, 210, 100, 100, 130, 510, 370, 100, 480, 150, 300,\
                             50, 40, 40, 105, 120, 110, 35, 80, 25, 125, 70, 1080, 1610, 95, 225, 2840, \
                             40, 55, 55, 145, 90, 60, 50, 1450, 190, 2260, 100, 130, 50, 320, 700, 300
                              ])

symplectic_array = np.array([True, True, False, False, True, True, True, True, False, False, False, False, False, False, False, False,\
        True, False, True, True, True, True, True, True, False, False, False, False, False, False, False, False,\
        True, True, True, True, False, True, False, False, True, False, False, False, False, False, False, False,\
        True, True, True, False, True, False, True, True, False, False, False, False, False, False, False, False,\
        True, False, True, False, True, False, True, False, False, False, False, False, False, False, False, False,\
        False, True, True, True, False, True, True, True, False, False, False, False, False, False, False, False,\
        True, True, True, True, True, True, False, True, False, False, False, False, False, False, False, False,\
        True, False, True, True, True, True, True, False, False, False, False, False, False, False, False, False,\
        True, True, True, False, False, True, True, True, True, False, False, False, False, False, False, False,\
        True, True, True, False, True, True, True, False, False, False, False, False, False, False, False, False,\
    ])

color_array = np.empty(symplectic_array.shape, dtype='object')
color_array[symplectic_array] = 'r'
color_array[~symplectic_array] = 'b'

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

# print(k_array.shape, convergence_time.shape)
# ax.scatter(k_array, convergence_time, color=color_array)

ax.scatter(k_array[symplectic_array], convergence_time[symplectic_array], color='r', label='Symplectic')
ax.scatter(k_array[~symplectic_array], convergence_time[~symplectic_array], color='b', label='Diaplectic')


ax.set_xlabel(r'$k$')
ax.set_ylabel(r'Convergence time ($t/T$)')
ax.set_ylim(0, 1000)
ax.legend()

plt.tight_layout()
fig.savefig(f'fig/convergence_time.pdf', bbox_inches = 'tight', format='pdf')
plt.show()