import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
from mayavi.api import Engine
from mayavi.plugins.app import main
from tvtk.api import tvtk

e = Engine()
e.start()

index = 1
t = 2994

print(f'vx_mesh_index{index}_{t}.npy')

vx = np.load(f'vx_mesh_index{index}_{t}.npy')
vy = np.load(f'vy_mesh_index{index}_{t}.npy')
vz = np.load(f'vz_mesh_index{index}_{t}.npy')
x = np.load(f'x_mesh_index{index}_{t}.npy')
y = np.load(f'y_mesh_index{index}_{t}.npy')
z = np.load(f'z_mesh_index{index}_{t}.npy')
fil_phases = np.load(f'fil_phases_index{index}_{t}.npy')
fil_data = np.load(f'fil_data_index{index}_{t}.npy')

nfil = fil_data.shape[0]
nseg = fil_data.shape[1]

colormap_name = 'YlOrRd'  # Choose a colormap (e.g., viridis, plasma, coolwarm, etc.)
colormap_name_fil = 'hsv'
cmap = plt.get_cmap(colormap_name_fil)
norm = plt.Normalize(vmin=0, vmax=2*np.pi)
fil_colors = cmap(norm(fil_phases))


grid_spacing = z[0][0][1] - z[0][0][0]
grid_point = x.shape[0]

speed_field = np.sqrt(vx**2 + vy**2 + vz**2)
speed_limit = 30

print(np.max(speed_field))
print(np.mean(speed_field))

scale_factors = np.where(speed_field > speed_limit, speed_limit / speed_field, 1)
vx = vx * scale_factors
vy = vy * scale_factors
vz = vz * scale_factors

speed_field = np.sqrt(vx**2 + vy**2 + vz**2)

radius = 19*2.6*15/2
# r_ratio = 2.
# y_lower, y_upper = -r_ratio*radius, r_ratio*radius, 
# z_lower, z_upper = -r_ratio*radius, r_ratio*radius, 

x_lower, x_upper = np.min(x), np.max(x)
y_lower, y_upper = np.min(y), np.max(y)
z_lower, z_upper = np.min(z), np.max(z)
print(x_lower, x_upper, y_lower, y_upper, z_lower, z_upper)

# x_list = np.arange(y_lower, y_upper+0.01, 20)
# y_list = np.arange(y_lower, y_upper+0.01, 20)
# z_list = np.arange(z_lower, z_upper+0.01, 20)


# # Using np.mgrid to achieve the same result
# x_mgrid, y_mgrid, z_mgrid = np.mgrid[x_lower:x_upper+1:grid_spacing, x_lower:x_upper+1:grid_spacing, x_lower:x_upper+1:grid_spacing,]


fig = mlab.figure(size=(600, 600), bgcolor = (238./256,238./256,236./256  ),)




# Plot the streamlines
src = mlab.pipeline.vector_field(x, y, z, vx, vy, vz)
magnitude = mlab.pipeline.extract_vector_norm(src)


mlab.points3d(0,0,0,resolution=64, scale_factor=radius*2)
for fil in range(nfil):
    fil_x, fil_y, fil_z = fil_data[fil][:,0], fil_data[fil][:,1], fil_data[fil][:,2]
    mlab.plot3d(fil_x, fil_y, fil_z,
                color=tuple(fil_colors[fil][:3]),
                tube_radius = 2,)

iso_surface = 28
mask = speed_field > 18
vx_masked = vx*mask
vy_masked = vy*mask
vz_masked = vz*mask
x_masked = x*mask
y_masked = y*mask
z_masked = z*mask
print(vx_masked.shape)
print(vy_masked.shape)
print(speed_field.shape)

mlab.quiver3d(x_masked, y_masked, z_masked, vx_masked, vy_masked, vz_masked,
              colormap=colormap_name,
              scale_factor=1.,
              mask_points=200,
              scalars=speed_field,
              vmax=speed_limit,
              opacity=.8)

iso = mlab.pipeline.iso_surface(magnitude, contours=[iso_surface,], opacity=0.1, colormap='Greys',)



# vcp = mlab.pipeline.vector_cut_plane(magnitude, mask_points=2,
#                                         scale_factor=40,
#                                         colormap='jet',
#                                         plane_orientation='x_axes')



for x_value in np.linspace(-radius*2, radius*2, 3):
    st = mlab.flow(x, y, z, vx, vy, vz,
                            seedtype='plane',
                            seed_visible=True,
                            integration_direction='both',
                            vmax=speed_limit,
                            colormap=colormap_name,
                            opacity=1)
    
    st.stream_tracer.maximum_propagation = 100000.0
    st.streamline_type = 'tube'
    st.tube_filter.radius = 2.5
    st.tube_filter.number_of_sides = 6
    st.tube_filter.vary_radius = 'vary_radius_by_scalar'
    st.seed.widget.center = np.array([ x_value,  0,  0])
    st.seed.widget.normal = np.array([ 1,  2,  0])
    st.seed.widget.origin = np.array([ x_value, y_lower,  z_lower])
    st.seed.widget.point1 = np.array([ x_value, y_upper,  z_lower])
    st.seed.widget.point2 = np.array([ x_value, y_lower,  z_upper])

    # st.seed.widget.center = np.array([ 0,  0,  0])
    # st.seed.widget.normal = np.array([ 0,  0,  1])
    # st.seed.widget.origin = np.array([ x_lower, y_lower,  z_value])
    # st.seed.widget.point1 = np.array([ x_lower, y_upper,  z_value])
    # st.seed.widget.point2 = np.array([ x_upper, y_lower,  z_value])

    st.seed.widget.resolution = 3
    st.seed.widget.enabled = False

    streamline_actor = st.actor.actors[0]
    streamline_actor.property.ambient = 1.0  # Set ambient lighting to full
    streamline_actor.property.diffuse = 0.0  # Set diffuse lighting to none
    streamline_actor.property.specular = 0.0  # Set specular lighting to none


fig.scene.x_plus_view()
mlab.show()



#---------------------------------------------------------
