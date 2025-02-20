import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# Path to the directory where fonts are stored
font_dir = os.path.expanduser("~/.local/share/fonts/cmu/cm-unicode-0.7.0")

# Choose the TTF or OTF version of CMU Serif Regular
font_path = os.path.join(font_dir, 'cmunrm.ttf')  # Or 'cmunrm.otf' if you prefer OTF

# Load the font into Matplotlib's font manager
prop = fm.FontProperties(fname=font_path)

print( prop.get_name())
# Set the font for all plot elements
plt.rcParams['font.serif'] = prop.get_name()
plt.rcParams['mathtext.fontset'] = 'cm'  # Ensure math uses Computer Modern

# Sample plot to test the font
plt.plot([0, 1], [0, 1], label=r"$y = x$")
plt.title('Sample Plot with CMU Serif Font')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()