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



def point_to_quaternion(x, y, z):
    # Normalize the point
    v = np.array([x, y, z])
    v = v / np.linalg.norm(v)
    
    # Reference vector (z-axis)
    z_axis = np.array([0, 0, 1])
    
    # Compute rotation axis (cross product)
    u = np.cross(z_axis, v)
    u_norm = np.linalg.norm(u)
    
    # If the vector is already aligned, return identity quaternion
    if u_norm == 0:
        if np.allclose(v, z_axis):  # Same as z-axis
            return np.array([1, 0, 0, 0])  # Identity quaternion
        else:  # Opposite of z-axis
            return np.array([0, 1, 0, 0])  # 180-degree rotation around x-axis
    
    # Normalize the rotation axis
    u = u / u_norm
    
    # Compute rotation angle
    cos_angle = np.dot(z_axis, v)
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

def quaternion_normal(quaternion):
    """
    Computes the normal vector from the given quaternion.

    Args:
        quaternion (list or array): A list or array of 4 floats [scalar_part, v0, v1, v2],
                                    where scalar_part is the real part and v0, v1, v2 are the vector parts.

    Returns:
        list: A 3D normal vector as a list of floats [n0, n1, n2].
    """
    scalar_part = quaternion[0]
    vector_part = quaternion[1:4]  # Extract vector part (v0, v1, v2)

    n = [0.0, 0.0, 0.0]

    n[0] = 2.0 * (vector_part[0] * vector_part[1] - scalar_part * vector_part[2])
    n[1] = 1.0 - 2.0 * (vector_part[0] ** 2 + vector_part[2] ** 2)
    n[2] = 2.0 * (vector_part[1] * vector_part[2] + scalar_part * vector_part[0])

    return n

def sqrt_in_place(quaternion):
    scalar_part = quaternion[0]
    vector_part = np.array(quaternion[1:4])

    if scalar_part < -0.999999999:
        # Handle the special case where scalar_part is close to -1
        scalar_part = 0.0
        vector_part = np.array([0.0, 0.0, 1.0])  # Any unit vector would be correct here
    else:
        # Compute the square root of the quaternion
        scalar_part = np.sqrt(0.5 * (1.0 + scalar_part))
        temp = 2.0 * scalar_part
        vector_part /= temp

    # Return the modified quaternion
    return [scalar_part] + vector_part.tolist()

def quaternion_multiply(q1, q2):
    # Extract components from the input arrays
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    
    # Perform quaternion multiplication
    a = a1*a2 - b1*b2 - c1*c2 - d1*d2
    b = a1*b2 + b1*a2 + c1*d2 - d1*c2
    c = a1*c2 - b1*d2 + c1*a2 + d1*b2
    d = a1*d2 + b1*c2 - c1*b2 + d1*a2
    
    return np.array([a, b, c, d])

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