import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.signal import savgol_filter

directory = 'pyfile/analysis/ishikawa_data/'
files = ['k0.0.csv', 'k0.5.csv', 'k1.0.csv', 'k1.5.csv', 'k2.0.csv']

# files = ['k0.0N162.csv', 'k0.0N636.csv', 'k0.0N2520.csv']
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlim(0, 1)
# ax.set_ylim(-1.4, 3.4)
for i, filename in enumerate(files):
    file = open(directory + filename, mode='r')
    df = pd.read_csv(directory + filename, header=None)
    data = df.to_numpy()
    x, y = data[:,0], data[:,1]

    ax.plot(x, y, ls = 'dotted', alpha=0.5, label=f'{i}')

legend1 = ax.legend()
line1, = ax.plot([-1, -1.1], [-1, -1.1], ls='-', c='black', label='data' )
line2, = ax.plot([-1, -1.1], [-1, -1.1], ls='dotted', c='black', label=r'$Ito.\ etc.\ (2019)$')
legend2 = ax.legend(handles = [line1, line2], loc='upper left')
ax.add_artist(legend1)
plt.show()