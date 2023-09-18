"""Utils containing basic transformations and rotations
"""
from typing import Union

from pytransform3d import rotations, transformations
import numpy as np


def matrix_from_quat(quaternion):
    """Quaternions to rotation matrix

    Args:
        quaternion (array_like, (4,)):  quaternions of format (w,x,y,z)

    Returns:
        ndarray (3,3): rotation matrix
    """
    return rotations.matrix_from_quaternion(quaternion)


def quat_from_matrix(matrix):
    """Rotation matrix to quaternions

    Args:
        matrix (ndarray, (3,3)): Rotation matrix

    Returns:
        ndarray (4,):  quaternions of format (w,x,y,z)
    """
    return rotations.quaternion_from_matrix(matrix)


def euler_xyz_from_matrix(matrix):
    """Takes roll, pitch, yaw angles (Cardan Extrinsic Euler XYZ) from rotation matrix

    Args:
        matrix (ndarray, (3,3)): Rotation matrix

    Returns:
        ndarray (3,1): Euler angles roll, pitch, yaw or extrinsic xyz convention
    """
    return rotations.extrinsic_euler_xyz_from_active_matrix(matrix)


def matrix_from_euler_xyz(euler):
    """Takes roll, pitch, yaw angles (Cardan Extrinsic Euler XYZ) and turns into rotation matrix

    Roll, pitch, yaw is the same as extrinsic euler XYZ angles
    Source: https://dfki-ric.github.io/pytransform3d/_modules/pytransform3d/rotations/_conversions.html#active_matrix_from_extrinsic_roll_pitch_yaw

    Args:
        euler (array_like, (3,)): roll, pitch, yaw or extrinsic xyz convention

    Returns:
        ndarray (3,3): rotation matrix

    """
    return rotations.active_matrix_from_extrinsic_euler_xyz(euler)


def euler_from_quat(quaternion):
    """Quaternions to Eulers

    Args:
        quaternion (array_like, (4,)): quaternions of format (w,x,y,z)

    Returns:
        euler (ndarray, (3,)): Euler angles (roll, pitch, yaw)
    """
    matrix = matrix_from_quat(quaternion)

    return euler_xyz_from_matrix(matrix)


def quat_from_euler(euler):
    """Euler XYZ to Quaternions

    Args:
        euler (ndarray, (3,)): Euler angles (roll, pitch, yaw)

    Returns:
       quaternion (array_like, (4,)): quaternions of format (w,x,y,z)
    """
    rotation = matrix_from_euler_xyz(euler)
    rotation_quat = rotations.quaternion_from_matrix(rotation)

    return rotation_quat


def check_pq_pose(pose):
    try:
        pose = transformations.check_pq(pose)
    except Exception as e:
        raise ValueError(
            "Pose is not in np.array((*translation, *quaternion)) format"
        ) from e


def check_quaternion(quat):
    """Checks quaternion and ensures unit quaternion

    Args:
        quat (array_like): (4,) of any format (x,y,z,w) or (w,x,y,z)

    Raises:
        ValueError: If quaternion is not the right format
    """

    try:
        _ = rotations.check_quaternion(quat)
    except Exception as e:
        raise ValueError(f"Quaternion is not in the right format {quat}") from e

    if not np.isclose(np.linalg.norm(quat), 1.0):
        raise ValueError(f"Quaternion is not a unit vector {quat}")


def pq_from_hmatrix(matrix):
    """Wrapper for pq_from_transform
    Args:
        matrix (array_like): (4,4) Transformation matrix from frame A to frame B
    Returns:
        pose (array_like): Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)
    """
    return transformations.pq_from_transform(matrix)


def hmatrix_from_pq(pose):
    """Wrapper for transform_from_pq

    Args:
        pose (array_like): Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    Returns:
        matrix (array_like): Transformation matrix from frame A to frame B
    """
    return transformations.transform_from_pq(pose)


def pose_in_a_to_pose_in_b(pose_in_a, pose_a_in_b):
    """Pose transformation for pq (translation, quaternion) poses

    Converts a pq pose corresponding to pose C in frame A (T_A^C)
    to a tq pose corresponding to the same pose C in frame B (T_B^C)

    Args:
        pose_in_a: numpy array of shape (7,) corresponding to the pose of C in frame A (T_A^C)
        pose_a_in_b: numpy array of shape (7,) corresponding to the pose of A in frame B (T_B^A)
        or transformation of b to a

    Returns:
        numpy array of shape (7,) corresponding to the pose of C in frame B (T_B^C)
    """

    pose_a_matrix = hmatrix_from_pq(pose_in_a)
    pose_a_in_b_matrix = hmatrix_from_pq(pose_a_in_b)

    # matrix_results is (4,4)
    matrix_results = hmatrix_in_a_to_hmatrix_in_b(pose_a_matrix, pose_a_in_b_matrix)

    pq_pose = transformations.pq_from_transform(matrix_results)

    return pq_pose


def hmatrix_in_a_to_hmatrix_in_b(pose_in_a, pose_a_in_b):
    """Pose transformation for either rotation or homogenous matrices

    Converts a matrix corresponding to pose or orientation C in frame A (T_A^C)
    to a homogenous matrix corresponding to the same pose or orientation of C in frame B (T_B^C)

    pose of C in B = pose of A in B * pose of C in A
    Take a point in C, transform it to A, then to B
    T_B^C = T_B^A * T_A^C

    Args:
        pose_in_a: numpy array of shape (3,3) or (4,4) corresponding to the orientation/pose of C in frame A (T_A^C)
        pose_a_in_b: numpy array of shape (3,3) or (4,4) corresponding to the orientation/pose of A in frame B (T_B^A)
        or transformation of b to a
    Returns:
        numpy array of shape (3,3) or (4,4) corresponding to the orientation or pose of C in frame B (T_B^C)
    """
    return pose_a_in_b @ pose_in_a


def hmatrix_inverse(hmatrix):
    """Computes the inverse of a pose

    Computes the inverse of a homogenous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    Note, the inverse of a pose matrix is the following
    [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    Intuitively, this makes sense.
    The original pose matrix translates by t, then rotates by R.
    We just invert the rotation by applying R-1 = R.T, and also translate back.
    Since we apply translation first before rotation, we need to translate by
    -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    R-1 to align the axis again.

    Args:
        hmatrix: numpy array of shape (4,4) for the pose to inverse
    Returns:
        numpy array of shape (4,4) for the inverse pose
    """

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = hmatrix[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(hmatrix[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


def check_transform(hmatrix):
    """Validates the transform input

    We enable strict_check since it attempts to evaluate if the matrix is numerically close enough to a real transform

    Args:
        hmatrix (array_like): (4,4) shape numpy
    """

    try:
        _ = transformations.check_transform(hmatrix, strict_check=True)
    except Exception as e:
        raise ValueError(f"Matrix is not a valid transformation {hmatrix}") from e


def position_euler_xyz_from_matrix(hmatrix):
    """Converts a homogeneous matrix to a 6-element list [x, y, z, r, p, y]

    Args:
        hmatrix (array): A (4,4) homogeneous matrix representing a transformation or pose

    Returns:
        list: len = 6 list representing [x, y, z, r, p, y]
    """
    translation = hmatrix[:3, 3]
    rotation_mat = hmatrix[:3, :3]
    euler_xyz = euler_xyz_from_matrix(rotation_mat)

    new_position = np.zeros(6)
    new_position[:3] = translation
    new_position[3:] = euler_xyz
    return list(new_position)


def quat_xyzw_from_wxyz(orientation):
    """Go from quaternion in xyzw to wxyz

    Args:
        orientation (array_like, shape (4,)): Quaternion in form wxyz

    Returns:
        orientation (array_like, shape (4,)): Quaternion in form xyzw
    """

    return rotations.quaternion_xyzw_from_wxyz(orientation)


def quat_wxyz_from_xyzw(orientation):
    """Go from quaternion in xyzw to wxyz

    Args:
        orientation (array_like, shape (4,)): Quaternion in form wxyz

    Returns:
        orientation (array_like, shape (4,)): Quaternion in form xyzw
    """

    return rotations.quaternion_wxyz_from_xyzw(orientation)


def matrix_from_axis_angle(axis_angle):
    """Rotation matrix from axis angle

    Args:
        axis_angle (array_like, shape (4,)): Angle-axis in form (x, y, z, angle), where (x, y, z) is a vector

    Returns:
        matrix (np.array, shape(3,3)): Rotation matrix
    """

    return rotations.matrix_from_axis_angle(axis_angle)


def axis_angle_from_matrix(matrix):
    """Axis angle from rotation matrix

    Args:
        matrix (np.array, shape (3,3)): Rotation matrix

    Returns:
        axis_angle (np.array, shape(4,)):  Angle-axis in form (x, y, z, angle), where (x, y, z) is a vector
    """

    return rotations.axis_angle_from_matrix(matrix)


def quaternion_slerp(quat_start, quat_end, ratio):
    """Quaternion Spherical Linear Interpolation

    Args:
        quat_start (array_like): (w,x,y,z) quaternion for the start of the interpolation
        quat_end (array_like): (w,x,y,z) quaternion for the end of the interpolation
        ratio (float): Fraction along the interpolation (0.5 would be the midpoint)

    Returns:
        quat_interpolated: (w,x,y,z) interpolated quaternion along the start to end
    """
    return rotations.quaternion_slerp(quat_start, quat_end, ratio)


def pick_closest_quaternion(quaternion, target_quaternion):
    """Picks the closest quaternion in cases where quaternion and -quaternion conflict.

    This is important because while quat and -quat represent the same orientation, conversions to euler angles are imprecise and might flip the rotation.

    Args:
        quaternion (array_like): (w,x,y,z) quaternion that could be quat or -quat
        target_quaternion (array_like): (w,x,y,z) quaternion to be closest to

    Returns:
        _type_: _description_
    """
    return rotations.pick_closest_quaternion(quaternion, target_quaternion)


def matrix_from_position_euler_xyz(pose: Union[list, np.ndarray]) -> np.ndarray:
    """Calculates the homogeneous matrix for a [x, y, z, r, p, y] pose
    Args:
        pose (list): [x, y, z, r, p, y] to convert. Length = 6.
    Returns:
        array: Homogeneous matrix representing this pose. Shape = (4,4).
    """
    xyz = pose[:3]
    rpy = pose[3:]
    rot = matrix_from_euler_xyz(rpy)
    trans = xyz
    hmatrix = np.eye(4)
    hmatrix[:3, :3] = rot
    hmatrix[:3, 3] = trans
    return hmatrix
