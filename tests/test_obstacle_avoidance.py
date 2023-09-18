"""xarm_sim_test demonstrates how the xarm sim is meant to be started and run in real time

The script is meant to execute as many functionalities that are availability for the XArm simulation
"""
import time
from turtle import distance
from medra_robotics.arm_control.xarm import XArm
from medra_robotics.arm_control.xarm_sim import (
    pybullet_gui_setup,
)
import medra_robotics.arm_control.xarm_constants as consts

from medra_robotics.scripts.script_1 import move_around_obstacle
import numpy as np

from medra_robotics.utils.helper_functions import angle_between

def test_obstacle_avoid(arm, pybullet):
    first_pose = np.array([370.37, -67.51, 481.06, 3.14159, 0, 4.11245])
    second_pose = np.array([370.37, 300.51, 481.06, -1.14159, 1, 2.11245])
    arm.set_ee_pose(first_pose, wait=True)

    is_near_second_pose = False

    pybullet.setRealTimeSimulation(1)

    traj_path = move_around_obstacle(
        arm, first_pose=first_pose, second_pose=second_pose, pybullet=pybullet
    )

    if len(traj_path) == 0:
        raise Exception("Trajectory path is empty")

    arm.set_ee_pose(first_pose, wait=True)

    pybullet.setRealTimeSimulation(1)

    index = 0
    while True:
        if ((traj_path is not None) and index<=len(traj_path)):
            try:
                
                arm.set_joint_pos(traj_path[index], wait=True)
            except:
                print(index)
                pass
        else: 
            break
        index += 1

        pybullet.stepSimulation()

        is_near_second_pose = arm.is_near_ee_pose(second_pose)
        time.sleep(1.0 / 60.0)

        if is_near_second_pose:
            break

        if index == len(traj_path):
            print("\n\n\n")
            print("DIDN'T REACH SECOND POSE")
            print("\n\n\n")
            break

    print(f"Desired Pose: {second_pose}")
    print(f"Actual Pose: {arm.get_ee_pose()}")


if __name__ == "__main__":
    pybullet = pybullet_gui_setup()

    xarm = XArm(ip_or_gui=pybullet, use_ik_nullspace=True)

    # install a cube as an obstacle with at x=0, y=0.6, z=0 and set it to a default orientation
    cubeStartPos = [0, 0.6, 0]
    cubeStartOrientation = xarm.arm.bullet_client.getQuaternionFromEuler([0, 0, 0])

    # Scale the cube to be only 0.5 m
    boxId = xarm.arm.bullet_client.loadURDF(
        "cube.urdf",
        cubeStartPos,
        cubeStartOrientation,
        useFixedBase=True,
        globalScaling=0.5,
    )

    test_obstacle_avoid(xarm, pybullet)
