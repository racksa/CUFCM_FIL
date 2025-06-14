import numpy as np
import math
import os

def rot_mat(quaternion):
    ret = np.zeros((3,3))
    ret[0, 0] = 1.0
    ret[1, 1] = 1.0
    ret[2, 2] = 1.0

    temp = 2.0*quaternion[1]*quaternion[1]
    ret[1, 1] -= temp;
    ret[2, 2] -= temp

    temp = 2.0*quaternion[2]*quaternion[2]
    ret[0, 0] -= temp
    ret[2, 2] -= temp

    temp = 2.0*quaternion[3]*quaternion[3]
    ret[0, 0] -= temp
    ret[1, 1] -= temp

    temp = 2.0*quaternion[1]*quaternion[2]
    ret[1, 0] = temp
    ret[0, 1] = temp

    temp = 2.0*quaternion[1]*quaternion[3]
    ret[2, 0] = temp
    ret[0, 2] = temp

    temp = 2.0*quaternion[2]*quaternion[3]
    ret[1, 2] = temp
    ret[2, 1] = temp

    temp = 2.0*quaternion[0]*quaternion[3]
    ret[1, 0] += temp;
    ret[0, 1] -= temp;

    temp = 2.0*quaternion[0]*quaternion[2]
    ret[2, 0] -= temp;
    ret[0, 2] += temp;

    temp = 2.0*quaternion[0]*quaternion[1];
    ret[2, 1] += temp;
    ret[1, 2] -= temp;

    return ret

def find_t(quaternion):
    q = quaternion
    qsq = q**2
    return np.array([1-2*(qsq[2]+qsq[3]), 2*(q[1]*q[2]+q[3]*q[0]), 2*(q[1]*q[3]-q[2]*q[0])])

def blob_point_from_data(body_states, blob_references):
    blob_pos = np.matmul(rot_mat(body_states[3:7]), blob_references)
    # print('blobref, blobpos', blob_references, blob_pos)

    x = body_states[0] + blob_pos[0]
    y = body_states[1] + blob_pos[1]
    z = body_states[2] + blob_pos[2]

    return x, y, z


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def box(x, box_size):
    return x - np.floor(x/box_size)*box_size


def two_points_at_boundary(two_points_x, two_points_y, two_points_z, rod_length):
    points = np.array((two_points_x, two_points_y, two_points_z))
    vector = points[:,1] - points[:,0]
    if np.linalg.norm(vector) > rod_length:
        return [0,0], [0,0], [0,0]
        # two_points_x, two_points_y, two_points_z = [], [], []
    return two_points_x, two_points_y, two_points_z
    
def cartesian_to_spherical(x):
    """
    Convert Cartesian coordinates to spherical polar coordinates.
    
    Args:
        x (float, float, float): cartesian-coordinate.
    
    Returns:
        tuple: (r, theta, phi), where r is the radial distance, phi is the azimuthal angle,
               and theta is the elevation angle (zenith angle).
    """
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    phi = np.arctan2(x[1], x[0])
    theta = np.arccos(x[2] / r)
    
    return r, phi, theta

def spherical_to_cartesian(r, phi, theta):
    """
    Convert spherical coordinates (r, phi, theta) to Cartesian coordinates (x, y, z).
    
    Args:
        r (float): Radial distance from the origin.
        theta (float): Angle in radians measured counterclockwise from the positive x-axis to the projection
                      of the point onto the xy-plane.
        phi (float): Angle in radians measured from the positive z-axis to the line connecting the origin
                    and the point.
    
    Returns:
        tuple: A tuple containing the Cartesian coordinates (x, y, z).
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def create_3d_cell_list(positions, cell_size):
    pos1, pos2= np.min(positions, axis=0), np.max(positions, axis=0)

    domain_size = pos2-pos1
    # Calculate the dimensions of the 3D grid based on cell size
    grid_shape = np.ceil(domain_size/ cell_size).astype(int)
    
    # Create an empty 3D cell list
    cell_list = [[] for _ in range(grid_shape[0] * grid_shape[1] * grid_shape[2])]
    
    # Assign particles to 3D cells
    for i, pos in enumerate(positions):
        cell_x = int((pos[0]-pos1[0]) / cell_size)
        cell_y = int((pos[1]-pos1[1]) / cell_size)
        cell_z = int((pos[2]-pos1[2]) / cell_size)
        cell_index = cell_x + cell_y * grid_shape[0] + cell_z * grid_shape[0] * grid_shape[1]
        cell_list[cell_index].append(i)
    
    return cell_list, grid_shape

def label_colliding_particles_with_3d_cell_list(positions, cell_size, radius):
    N = positions.shape[0]
    
    # Create the 3D cell list
    cell_list, grid_shape = create_3d_cell_list(positions, cell_size)
    
    # Calculate the squared distances only within cells and neighboring cells
    colliding_particles = []
    colliding_indices = []

    for cell_x in range(grid_shape[0]):
        for cell_y in range(grid_shape[1]):
            for cell_z in range(grid_shape[2]):
                cell_index = cell_x + cell_y * grid_shape[0] + cell_z * grid_shape[0] * grid_shape[1]
                cell = cell_list[cell_index]

                for i in cell:
                    for j in cell:
                        if i != j:
                            # Calculate squared distance between particles i and j
                            distance_sq = np.sum((positions[i] - positions[j]) ** 2)
                            
                            if distance_sq < 4*radius**2:
                                colliding_indices.append(i)
                                colliding_particles.append((i, j))

    return set(colliding_indices), colliding_particles

def label_colliding_particles(positions, radius):
    N = positions.shape[0]

    # Calculate all pairwise squared Euclidean distances efficiently
    squared_distances = np.sum((positions[:, np.newaxis] - positions)**2, axis=2)

    # Set the diagonal elements (self-distances) to a large value to avoid counting them
    np.fill_diagonal(squared_distances, np.inf)

    # Find the indices of particles that collide with others
    colliding_indices = np.where(squared_distances < 4*radius**2)

    # Create a list of colliding particle pairs
    colliding_particles = []
    for i, j in zip(*colliding_indices):
        colliding_particles.append((i, j))

    return colliding_indices, colliding_particles

def list_folders(path):
    try:
        folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
        return folders
    except OSError as e:
        print(f"Error: {e}")
        return []
    
def get_member_functions(obj):
    try:
        # Get all attributes of the object
        attributes = dir(obj)
        
        # Filter out methods (functions)
        methods = [attr for attr in attributes if callable(getattr(obj, attr))]
        
        return methods
    except Exception as e:
        print(f"Error: {e}")
        return []

def normalize_quaternions(quaternions):
    """Normalize an array of quaternions."""
    norms = np.linalg.norm(quaternions, axis=1)
    return quaternions / norms[:, np.newaxis]

def quaternion_conjugate(q):
    """Compute the conjugate of a quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q, r):
    """Quaternion multiplication (q * r)."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def compute_angular_velocity(quaternions, dt):
    """Compute angular velocity from an array of quaternions."""
    # Ensure quaternions are normalized
    quaternions = normalize_quaternions(quaternions)

    # Number of quaternions
    n = quaternions.shape[0]
    
    # Initialize angular velocity array
    angular_velocity = np.zeros((n-1, 3))
    
    # Compute angular velocity
    for i in range(n-1):        
        # Quaternion difference
        dq = quaternion_multiply(quaternions[i+1], quaternion_conjugate(quaternions[i]))
        
        # Quaternion derivative
        dq_dt = dq / dt
        
        # Angular velocity (extract vector part and multiply by 2)
        omega = 2 * dq_dt[1:4]
        
        # Store angular velocity
        angular_velocity[i, :] = omega
    
    return angular_velocity

#