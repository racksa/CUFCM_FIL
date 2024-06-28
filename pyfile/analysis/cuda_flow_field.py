import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib as mpl
from numba import cuda, float32, float64
import time


def stokeslet(x, x0, f0):
    r = np.linalg.norm(x-x0)
    return f0/(8.*np.pi*r) + np.dot(f0, x-x0) * (x - x0) / (8.*np.pi*r**3)

fig = plt.figure()
ax = fig.add_subplot()

# sphere
radius = 100
y_lower, y_upper = -1.5*radius, 1.5*radius, 
z_lower, z_upper = -1.5*radius, 1.5*radius, 

x_list = [0]
y_list = np.arange(y_lower, y_upper+0.01, 20)
z_list = np.arange(z_lower, z_upper+0.01, 20)

y_mesh, z_mesh = np.meshgrid(y_list, z_list)
y_flat = y_mesh.flatten()
z_flat = z_mesh.flatten()

pos_list = np.column_stack((np.ones(y_flat.shape)*x_list[0], y_flat, z_flat))

ax.cla()
ax.set_xlim(y_lower, y_upper)
ax.set_ylim(z_lower, z_upper)

v_list = np.zeros(np.shape(pos_list))

np.random.seed(0)
source_n = 10000
source_pos_list = (np.random.rand(source_n, 3)-0.5)*(y_upper-y_lower)
source_pos_list[:,0] = 0
source_force_list = np.ones(source_pos_list.shape)

ax.scatter( source_pos_list[:,1], source_pos_list[:,2], s=1)


# computing v list
def compute_v_list(pos_list, source_pos_list, source_force_list, v_list):
    for pi, pos in enumerate(pos_list):
        for si, source_pos in enumerate(source_pos_list):
            source_force = source_force_list[si]
            v_list[pi] += stokeslet(pos, source_pos, source_force)

#
@cuda.jit(device=True)
def stokeslet_device(x, x0, f0, result):
    dis_x = x[0] - x0[0]
    dis_y = x[1] - x0[1]
    dis_z = x[2] - x0[2]
    r = (dis_x**2 + dis_y**2 + dis_z**2)**0.5
    
    scalar_term = 1 / (8. * np.pi * r)
    dot_product = f0[0] * dis_x + f0[1] * dis_y + f0[2] * dis_z
    vector_term_x = dot_product * dis_x / (8. * np.pi * r**3)
    vector_term_y = dot_product * dis_y / (8. * np.pi * r**3)
    vector_term_z = dot_product * dis_z / (8. * np.pi * r**3)
    
    result[0] = scalar_term*f0[0] + vector_term_x
    result[1] = scalar_term*f0[1] + vector_term_y
    result[2] = scalar_term*f0[2] + vector_term_z

@cuda.jit
def compute_v_list_cuda(pos_list, source_pos_list, source_force_list, v_list):
    pi = cuda.grid(1)
    
    if pi < pos_list.shape[0]:
        pos = pos_list[pi]
        v_temp = cuda.local.array(3, float64)  # Local array for temporary storage
        v_temp[0] = 0.0
        v_temp[1] = 0.0
        v_temp[2] = 0.0
        
        for si in range(source_pos_list.shape[0]):
            source_pos = source_pos_list[si]
            source_force = source_force_list[si]

            temp_result = cuda.local.array(3, dtype=float64)
            stokeslet_device(pos, source_pos, source_force, temp_result)
            
            v_temp[0] += temp_result[0]
            v_temp[1] += temp_result[1]
            v_temp[2] += temp_result[2]

        v_list[pi, 0] += v_temp[0]
        v_list[pi, 1] += v_temp[1]
        v_list[pi, 2] += v_temp[2]

# main body
start_time = time.time()

# Move data to GPU
d_pos_list = cuda.to_device(pos_list)
d_source_pos_list = cuda.to_device(source_pos_list)
d_source_force_list = cuda.to_device(source_force_list)
d_v_list = cuda.to_device(v_list)
# Define the grid and block dimensions
threads_per_block = 256
blocks_per_grid = (pos_list.shape[0] + threads_per_block - 1) // threads_per_block
# Launch the kernel
compute_v_list_cuda[blocks_per_grid, threads_per_block](d_pos_list, d_source_pos_list, d_source_force_list, d_v_list)
# Copy the result back to the host
v_list = d_v_list.copy_to_host()

# compute_v_list(pos_list, source_pos_list, source_force_list, v_list)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

# plotting
colormap = 'jet'
cmap = mpl.colormaps[colormap]
speed_list = np.linalg.norm(v_list, axis=1)
max_speed = max(speed_list)
normalised_v_list = v_list/5
print('maxspeed', max_speed)
colors = cmap(np.clip(speed_list/max_speed, None, 0.999))

speed_mesh = speed_list.reshape(y_mesh.shape)
phi_var_plot = ax.imshow(speed_mesh, cmap='jet', origin='upper', extent=[y_lower, y_upper, z_lower, z_upper])

# # ax.scatter(pos_list[:,1], pos_list[:,2], color=colors)
# # ax.quiver(pos_list[:,1], pos_list[:,2], normalised_v_list[:,1], normalised_v_list[:,2], scale_units='xy',scale=1.)

# vy_mesh = normalised_v_list[:,1].reshape(y_mesh.shape)
# vz_mesh = normalised_v_list[:,2].reshape(y_mesh.shape)
# ax.streamplot(y_mesh, z_mesh, vy_mesh, vz_mesh)

ax.set_aspect('equal')
# plt.savefig(f'fig/flowfield2D_example.pdf', bbox_inches = 'tight', format='pdf')
plt.show()