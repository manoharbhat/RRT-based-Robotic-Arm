"""script_1 should implement the functions below

The script is meant to test the straight line functionality. Feel free to add any utilities that you need here either from your utils or from our utils
"""

import time
from medra_robotics.scripts.script_utils import *
from medra_robotics.utils.transformations import *
from medra_robotics.utils.helper_functions import *
from medra_robotics.arm_control.xarm_constants import *
import numpy as np
import heapq
import math
# Normally we would have this in arm_control, but for the purposes of grader readability we'll stick this functionality here
# def move_around_obstacle(arm, **kwargs):

#     return []


def astar(start, goal, obstacle_list, distance_func):
    """
    A* algorithm for finding the shortest path between start and goal nodes
    while avoiding obstacles. The distance between two nodes is calculated
    using distance_func. Returns a list of nodes in the shortest path.
    """
    open_list = [(0, start)]
    closed_set = set()
    g_score = {start: 0}
    f_score = {start: distance_func(start, goal)}

    while open_list:
        current_f, current = heapq.heappop(open_list)

        if current == goal:
            # goal reached
            path = []
            while current in g_score:
                path.append(current)
                current = g_score[current]
            return path[::-1]

        closed_set.add(current)

        for neighbor in get_neighbors(current, obstacle_list):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + distance_func(current, neighbor)

            if neighbor not in [i[1] for i in open_list]:
                heapq.heappush(open_list, (tentative_g_score + distance_func(neighbor, goal), neighbor))
            elif tentative_g_score >= g_score[neighbor]:
                continue

            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = tentative_g_score + distance_func(neighbor, goal)

    return None

def get_neighbors(node, obstacle_list):
    """
    Returns a list of neighboring nodes that are not in the obstacle_list
    """
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        neighbor = (node[0] + dx, node[1] + dy)
        if neighbor not in obstacle_list:
            neighbors.append(neighbor)
    return neighbors

def move_around_obstacle(arm, first_pose, second_pose, pybullet):
    path = []
    path.append(first_pose)

    obstacle_positions = [(0, 0.6, 0)]

    for pos in obstacle_positions:
        obstacle_id = pybullet.loadURDF(
            "cube.urdf",
            basePosition=pos,
            useFixedBase=True,
            globalScaling=0.5,
        )

    is_path_clear_here = is_path_clear(obstacle_positions, first_pose, second_pose)

    if not is_path_clear_here:
        print("Path not clear. Cannot move arm to second pose.")
        return []

    mid_point = (first_pose + second_pose) / 2
    ee_pose = arm.get_ee_pose()
    orientation = ee_pose[3:]  # or ee_pose[-3:]


    angle_between_poses = angle_between(first_pose, second_pose)

    rotations = np.linspace(0, angle_between_poses, 10)

    for r in rotations:
        pose = mid_point.copy()

        pose[3:] = orientation

        pose[3] += r

        path.append(pose)
    path.append(second_pose)
    path = np.clip(path, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
    # path = np.ndarray.tolist()
    

    return path

def angles_clipping(path):
    """ 
    This function clips the joint angles so that they stay within the joint limits.

    Args:
        path (List): List of numpy arrays representing the poses of the end-effector.
    """
    for i in range(len(path)):
        path[i] = np.clip(path[i], JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
    return path

def is_path_clear(obstacle_positions, first_pose, second_pose):
    """
    Check if the straight line between two poses is clear of obstacles.

    Args:
        obstacle_positions (List[Tuple[float, float, float]]): List of obstacle positions as tuples of (x, y, z) coordinates.
        first_pose (numpy.ndarray): 1D numpy array of length 7 representing the first pose.
        second_pose (numpy.ndarray): 1D numpy array of length 7 representing the second pose.

    Returns:
        bool: True if the path is clear, False otherwise.
    """
    for obstacle_pos in obstacle_positions:
        # Compute the vector between the two poses
        vec = second_pose[:3] - first_pose[:3]

        # Compute the distance between the two poses
        dist = np.linalg.norm(vec)

        # Compute the direction of the vector
        dir_vec = vec / dist

        # Compute the position of the obstacle in the coordinate system of the line segment
        obstacle_pos_local = np.dot(np.linalg.inv(matrix_from_euler_xyz(first_pose[3:])), np.array(obstacle_pos) - first_pose[:3])

        # If the obstacle is not between the two poses, continue to the next obstacle
        if obstacle_pos_local[0] < 0 or obstacle_pos_local[0] > dist:
            continue

        # Compute the closest point on the line segment to the obstacle
        closest_point = first_pose[:3] + dir_vec * obstacle_pos_local[0]

        # Compute the distance between the closest point and the obstacle
        dist_to_obstacle = np.linalg.norm(np.array(obstacle_pos) - closest_point)

        # If the distance is less than a threshold value, the path is not clear
        if dist_to_obstacle < 0.5:
            return False

    # If none of the obstacles are too close to the line segment, the path is clear
    return True

