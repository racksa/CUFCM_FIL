import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Poincaré section
def poincare_section(states, plane_normal, plane_point):
    poincare_points = []
    for state in states:
        if np.isclose(np.dot(plane_normal, state), np.dot(plane_normal, plane_point), atol=1e-5):
            poincare_points.append(state)
    return np.array(poincare_points)

# Generate example data
t = 1000  # Number of frames
n = 100   # Dimension of the system
states = np.random.rand(t, n)  # Example random states, replace this with your actual data

# Define normal vector and a point on the plane
plane_normal = np.random.rand(n)
plane_point = np.random.rand(n)

# Plotting the Poincaré section

poincare_points = poincare_section(states, plane_normal, plane_point)
print(np.shape(poincare_points))

plt.figure(figsize=(8, 6))
plt.plot(poincare_points[:, 0], poincare_points[:, 1], '.', markersize=0.5)
plt.xlabel('Component 1')
plt.ylabel('Component 2')