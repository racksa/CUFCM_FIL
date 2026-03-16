import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

# Path to the directory where fonts are stored
font_dir = os.path.expanduser("~/.local/share/fonts/cmu/cm-unicode-0.7.0")
# Choose the TTF or OTF version of CMU Serif Regular
font_path = os.path.join(font_dir, 'cmunrm.ttf')  # Or 'cmunrm.otf' if you prefer OTF
# Load the font into Matplotlib's font manager
prop = fm.FontProperties(fname=font_path)
# Register each font file with Matplotlib's font manager
for font_file in os.listdir(font_dir):
    if font_file.endswith('.otf'):
        fm.fontManager.addfont(os.path.join(font_dir, font_file))
# Set the global font family to 'serif' and specify CMU Serif
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif']
plt.rcParams['mathtext.fontset'] = 'cm'  # Use 'cm' for Computer Modern
plt.rcParams.update({'font.size': 24})

def interpolate_curve(points):
    # Separate x and y coordinates of points
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    
    # Create a cubic spline interpolation
    cs = CubicSpline(x, y)
    
    # Generate a smooth curve
    smooth_x = np.linspace(min(x), max(x), 1000)
    smooth_y = cs(smooth_x)
    
    return smooth_x, smooth_y

# Example usage:
# points = [(2.5, 6), (4, 8), (4.5, -2), (5, 7), (6, -1), (8, 2)]
points = [(5, 3), (8, 4), (10, 3.5), (12, -1), (16, 1)]
smooth_x, smooth_y = interpolate_curve(points)

fig1 = plt.figure()
ax1 = fig1.add_subplot()

A = np.array([6, 0.5])
B = np.array([14, 1])
ax1.scatter(A[0], A[1], c='black')
ax1.scatter(B[0], B[1], c='black')
# ax1.annotate(r'$\mathbf{A}$', xy=A, xytext=(A[0]-1, A[1] - 0.5))
# ax1.annotate(r'$\mathbf{B}$', xy=B, xytext=(B[0]+0.2, B[1] ))

alpha = 0.6
C = alpha*A+(1-alpha)*B
# ax1.annotate(r'$\{\alpha\mathbf{A} + (1-\alpha)\mathbf{B} \mid 0<\alpha<1\}$', xy=C, xytext=(C[0]-4, C[1] - 3),
            #  arrowprops=dict(facecolor='black', arrowstyle='->'))


ax1.plot([A[0], B[0]], [A[1], B[1]], c = 'black', linestyle='dashed')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')


ax1.plot(smooth_x, smooth_y, c ='blue')
ax1.annotate('$f(\mathbf{x})=0$', xy=points[2], xytext=(points[2][0]+0.15, points[2][1]), c ='blue')
ax1.set_aspect('equal')
fig1.tight_layout()
# fig1.savefig(f'bisection_illustration.pdf', bbox_inches = 'tight', format='pdf')
fig1.savefig(f'bisection_illustration.png', bbox_inches = 'tight', format='png', transparent=True)
plt.show()