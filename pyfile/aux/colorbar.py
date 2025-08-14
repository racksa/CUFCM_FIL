import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib as mpl
import os
import matplotlib.font_manager as fm

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
plt.rcParams.update({'font.size': 36})
# Generate an array of values
x = np.linspace(0, 1, 100)

# Create a colormap using a built-in matplotlib colormap
cmap_name = 'hsv'
cmap = plt.get_cmap(cmap_name)

# Convert the colormap to BGR hexadecimal colors
bgr_colors = []
for value in x:
    rgb_color = cmap(value)[:3]  # Get the RGB color tuple
    bgr_color = mcolors.rgb2hex(rgb_color)  # Convert RGB to BGR hexadecimal format
    bgr_colors.append(bgr_color)
    


# Plot the colormap
fig, ax = plt.subplots(figsize=(12, 1))
ax.imshow([x], aspect='equal', cmap=cmap)
x_ticks = np.linspace(-0.5, 99.5, 5)
# x_ticks_label = [r"{:.2f}$\pi$".format((x+0.5)*2*np.pi/100./np.pi) for x in x_ticks]
x_ticks_label = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
ax.set_xticks(x_ticks, x_ticks_label)
ax.set_yticks([])
# ax.set_xlabel(r"Phase")
fig.tight_layout()
fig.savefig('cmap.pdf', bbox_inches = 'tight', format='pdf')
fig.savefig('cmap.png', bbox_inches = 'tight', format='png')


fig2, ax2 = plt.subplots(figsize=(5, 6))  # Adjust figure size as needed
ax2.set_visible(False)  # Hide the axis

# Define a colormap and normalize
cmap = 'hsv'  # Choose a colormap (e.g., viridis, plasma, jet, etc.)
vmin = 0
vmax = 2*np.pi
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap = mpl.colormaps[cmap]
# Create a ScalarMappable and add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy array for the colorbar

# Add colorbar
cbar = plt.colorbar(sm, ax=ax2, orientation='vertical')
cbar.ax.set_yticks(np.linspace(0, 2*np.pi, 7), [r'$0$', r'$\pi/3$', r'$2\pi/3$', r'$\pi$', r'$4\pi/3$', r'$5\pi/3$', r'2$\pi$'])
cbar.set_label(rf"$\psi_1$")

# Customize colorbar label
# cbar.set_label("Color Scale", fontsize=12)
fig2.tight_layout()
fig2.savefig('colorbar.png', bbox_inches = 'tight', format='png', transparent=True)

plt.show()

# Print the BGR hexadecimal colors
for i, color in enumerate(bgr_colors):
    print(f"Value: {x[i]:.2f}   BGR Color: {color}")