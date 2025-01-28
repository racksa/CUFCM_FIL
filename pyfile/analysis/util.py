import numpy as np

# Define a function to generate a quaternion for rotation
def rotation_quaternion(rot_axis, rot_angle):
    # Normalize the rotation axis
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    x, y, z = rot_axis

    # Compute the quaternion components
    w = np.cos(rot_angle / 2)
    sin_half_angle = np.sin(rot_angle / 2)
    x = x * sin_half_angle
    y = y * sin_half_angle
    z = z * sin_half_angle

    # Return the quaternion as [w, x, y, z]
    return np.array([w, x, y, z])


# Function to compute quaternion from a point with reference to the y-axis
def point_to_quaternion(x, y, z):
    # Normalize the point
    v = np.array([x, y, z])
    v = v / np.linalg.norm(v)

    # Reference vector (x-axis)
    x_axis = np.array([1, 0, 0])

    # Compute rotation axis (cross product)
    u = np.cross(x_axis, v)
    u_norm = np.linalg.norm(u)

    # If the vector is already aligned, return identity quaternion
    if u_norm == 0:
        if np.allclose(v, x_axis):  # Same as x-axis
            return np.array([1, 0, 0, 0])  # Identity quaternion
        else:  # Opposite of x-axis
            return np.array([0, 0, 1, 0])  # 180-degree rotation around y-axis

    # Normalize the rotation axis
    u = u / u_norm

    # Compute rotation angle
    cos_angle = np.dot(x_axis, v)
    angle = np.arccos(np.clip(cos_angle, -1, 1))

    # Quaternion components
    w = np.cos(angle / 2)
    sin_half_angle = np.sin(angle / 2)
    x = u[0] * sin_half_angle
    y = u[1] * sin_half_angle
    z = u[2] * sin_half_angle

    return np.array([w, x, y, z])

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

def cartesian_to_spherical(x):
    """
    Convert Cartesian coordinates to spherical polar coordinates.
    
    Args:
        x (float, float, float): cartesian-coordinate.
    
    Returns:
        tuple: (r, theta, phi), where r is the radial distance, theta is the polar angle (azimuthal angle),
               and phi is the elevation angle (zenith angle).
    """
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    theta = np.arctan2(x[1], x[0])
    phi = np.arccos(x[2] / r)
    
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).
    
    Args:
        r (float): Radial distance from the origin.
        theta (float): Angle in radians measured counterclockwise from the positive x-axis to the projection
                      of the point onto the xy-plane.
        phi (float): Angle in radians measured from the positive z-axis to the line connecting the origin
                    and the point.
    
    Returns:
        tuple: A tuple containing the Cartesian coordinates (x, y, z).
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z