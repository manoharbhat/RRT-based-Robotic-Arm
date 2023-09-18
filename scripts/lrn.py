"""script_1 should implement the functions below

The script is meant to test the straight line functionality. Feel free to add any utilities that you need here either from your utils or from our utils
"""
import medra_robotics.arm_control.xarm_constants as consts

import time
from medra_robotics.scripts.script_utils import *
from medra_robotics.utils import transformations

import numpy as np

# Normally we would have this in arm_control, but for the purposes of grader readability we'll stick this functionality here
import heapq
import numpy as np

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

def a_star(start, goal, obstacles):
    open_list = []
    closed_list = []

    start_node = Node(start)
    goal_node = Node(goal)
    obstacles = get_cube_vertices([0,1.2,0],2000)
    print(obstacles)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node == goal_node:
            path = []
            while current_node is not None:
                path.append(np.array(current_node.position))
                current_node = current_node.parent
            return path[::-1]

        closed_list.append(current_node)

        for neighbor in get_neighbors(current_node, obstacles):
            if neighbor in closed_list:
                continue

            tentative_g = current_node.g + distance(current_node.position, neighbor.position)

            if neighbor not in open_list:
                neighbor.g = tentative_g
                neighbor.h = distance(neighbor.position, goal_node.position)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current_node
                heapq.heappush(open_list, neighbor)
            elif tentative_g < neighbor.g:
                neighbor.g = tentative_g
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current_node

    return None

def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

def get_neighbors(node, obstacles):
    neighbors = []

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if i == 0 and j == 0 and k == 0:
                    continue

                neighbor_pos = (node.position[0] + i, node.position[1] + j, node.position[2] + k)

                if neighbor_pos in obstacles:
                    continue

                neighbor_node = Node(neighbor_pos)
                neighbors.append(neighbor_node)

    return neighbors
def get_cube_vertices(pose, size):
    # Define the center of the cube based on the pose
    center = np.array([pose[0], pose[1], pose[2]])

    # Calculate the half length of the cube based on the size
    half_length = size / 2.0

    # Define the eight vertices of the cube
    vertices = np.array([
        [center[0] - half_length, center[1] - half_length, center[2] - half_length],
        [center[0] + half_length, center[1] - half_length, center[2] - half_length],
        [center[0] + half_length, center[1] + half_length, center[2] - half_length],
        [center[0] - half_length, center[1] + half_length, center[2] - half_length],
        [center[0] - half_length, center[1] - half_length, center[2] + half_length],
        [center[0] + half_length, center[1] - half_length, center[2] + half_length],
        [center[0] + half_length, center[1] + half_length, center[2] + half_length],
        [center[0] - half_length, center[1] + half_length, center[2] + half_length]
    ])
    # print(vertices)
    return vertices


def move_around_obstacle(arm, **kwargs):

    start = (370, -67, 481)
    goal = (370, 300, 481)
    obstacles = get_cube_vertices([0,1.2,0],500)

    # Run the A* algorithm to find the path
    path = a_star(start, goal, obstacles)
    quat1 = transformations.quat_from_euler([3.14159, 0, 4.11245])
    quat2 =  transformations.quat_from_euler([ -1.14159, 1, 2.11245])
    k = transformations.quaternion_slerp(quat1,quat2,0.5)
    quat3 = transformations.euler_from_quat(k)
    if path is None:
        print("No path found")
    else:
        for i in range(len(path)):
            if i < len(path)-1:
                path[i] =np.hstack([path[i], quat3])
            else:
                path[i] =np.hstack([path[i], [ -1.14159, 1, 2.11245]])
                print("last one",path[i])

            # print("vertex",path[i])
            path[i] = arm.get_inverse_kinematics(path[i])
        print("Path found:") #, path)

        return path
   