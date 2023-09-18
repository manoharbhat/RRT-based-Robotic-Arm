"""General helper functions"""

import numpy as np
from heapq import nsmallest


def quaternion_distance(q1, q2):
    """Quaternion Distance Approximation

    https://math.stackexchange.com/questions/90081/quaternion-distance

    We use: 1-⟨q1,q2⟩^2
    In particular, it gives 0 whenever the quaternions represent the same orientation, and it gives 1 whenever the two orientations are 180∘ apart.
    """
    distance = 1 - np.inner(q1, q2) ** 2
    return distance


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2':
    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arccos(np.dot(v1_u, v2_u))


def normalize(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)
