
import numpy as np
from mayavi import mlab

# Create a grid of points
x, y, z = np.mgrid[-10:10:10j, -10:10:10j, -10:10:10j]

# Create a sample 3D vector field (velocity field)
# Example: a simple vortex
u = -x
v = x
w = np.zeros_like(z)

# Start the figure
mlab.figure(size=(800, 600))

# Plot the streamlines
streamlines = mlab.flow(x, y, z, u, v, w, line_width=1.0)

# Show the plot
mlab.show()
