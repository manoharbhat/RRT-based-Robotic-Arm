"""xarm_sim_test demonstrates how the xarm sim is meant to be started and run in real time

The script is meant to execute as many functionalities that are availability for the XArm simulation
"""
from medra_robotics.arm_control.xarm import XArm
from medra_robotics.arm_control.xarm_sim import pybullet_gui_setup
import numpy as np

from medra_robotics.utils.transformations import euler_from_quat

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

    while True:
        joint_pos = [-0.143092, -0.02105, -1.413996, -0.0, 1.435045, 3.112614]

        print("Set Joint Pos")
        xarm.set_joint_pos(joint_pos, wait=True)

        print("Calculate the EE Forward Kinematics")
        ee_pose = xarm.bullet_kinematics.get_fk(joint_pos)
        ee_pose_euler = np.hstack((ee_pose[:3], euler_from_quat(ee_pose[3:])))

        print(f"Forward Kinematics Pose: {ee_pose_euler}")
        print(f"XArm Pose: {xarm.get_ee_pose()}")

        is_near_ee_pose = xarm.is_near_ee_pose(ee_pose_euler)
        print(f"Is Arm Close to Forward Kinematics Solution? {is_near_ee_pose}")

        next_pose = [470.37, -67.51, 481.06, 3.14159, 0, 4.11245]

        print("Set EE Pose")
        xarm.set_ee_pose(next_pose, wait=True)

        print(np.linalg.norm(np.subtract(xarm.get_ee_pose()[:3], next_pose[:3])))

        # "Tests" to make sure the functions are good
        print(f"XArm Pose: {xarm.get_ee_pose()}")
        print(f"XArm Pose in degrees: {xarm.get_ee_pose(is_radian=False)}")

        # Sets the endpoint transformation to be a transform in the Z-axis by 226 mm
        print("Set EE Pose Delta")
        simple_z_transform = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 226], [0, 0, 0, 1]]
        )
        # The same rotations shown above with endpoint transform should be different 90 deg
        print("Rotate at the original ee")
