"""Contains XArm class based on the Python XArmAPI
"""
import signal
import sys
from typing import Optional, Union

from pybullet_utils.bullet_client import BulletClient
import numpy as np

from medra_robotics.arm_control.xarm_sim import XArmSim
from medra_robotics.utils.transformations import (
    quat_from_euler,
)
from medra_robotics.arm_control.kinematics import BulletKinematics
import medra_robotics.arm_control.xarm_constants as consts
from medra_robotics.utils.helper_functions import quaternion_distance
from medra_robotics.arm_control.types import XArmAPIProtocol


class XArm:
    """XArm class that wraps the XArmAPI with additonal functions

    Args:
        config: RobotConfig that should contain relevant information
        ip (str, optional): ip address of robot. Defaults to "192.168.1.11".
        use_bullet_ik (bool, optional): whether to use pybullet's inverse kinematics solver. Defaults to True.
        ik_use_last_joint_pos (bool, optional): use last desired joint pos as input to pybullet ik solver.
            Makes IK output more consistent.Defaults to True.

    TODO:
        * Set up joint speed when in ik mode
        * Set default acceleration
    """

    def __init__(
        self,
        ip_or_gui: Union[str, BulletClient] = "192.168.1.11",
        use_bullet_ik=True,
        use_ik_nullspace=False,
        ik_use_last_joint_pos=True,
        logger=None,
    ):

        # flag for using the last set desired joint pos for ik (rather than sensed joint pos)
        self.use_bullet_ik = use_bullet_ik
        self.ik_use_last_joint_pos = ik_use_last_joint_pos

        self.logger = logger

        self.dof = 6  # degrees of freedom of robot

        # tracking the last desired joint pos
        self.last_des_joint_pos = None
        # tracking the last desired gripper pos for grasp verification
        self.last_des_gripper_pos = None

        self.arm: XArmAPIProtocol = XArmSim(ip_or_gui, is_radian=True)

        if self.use_bullet_ik:
            self.bullet_kinematics = BulletKinematics(nullspace=use_ik_nullspace)

        self.set_speed_acc_jerk()

    def set_speed_acc_jerk(
        self,
        gripper_speed=consts.DEFAULT_GRIPPER_SPEED,
        ee_speed=consts.DEFAULT_EE_SPEED,
        ee_acc=consts.DEFAULT_EE_ACC,
        ee_jerk=consts.DEFAULT_EE_JERK,
        joint_speed=consts.DEFAULT_JOINT_SPEED,
        joint_acc=consts.DEFAULT_JOINT_ACC,
        joint_jerk=consts.DEFAULT_JOINT_JERK,
    ):
        """Sets the speed/acceleration/jerk properties to desired values
        If an argument is not specified, it will default to the the values specified in xarm_constants
        See the relevant @properties and @setters for more info
        Args:
            gripper_speed (float, optional): Defaults to consts.DEFAULT_GRIPPER_SPEED.
            ee_speed (float, optional): Defaults to consts.DEFAULT_EE_SPEED.
            ee_acc (float, optional): Defaults to consts.DEFAULT_EE_ACC.
            ee_jerk (float, optional): Defaults to consts.DEFAULT_EE_JERK.
            joint_speed (float, optional): Defaults to consts.DEFAULT_JOINT_SPEED.
            joint_acc (float, optional): Defaults to consts.DEFAULT_JOINT_ACC.
            joint_jerk (float, optional): Defaults to consts.DEFAULT_JOINT_JERK.
        """
        self.gripper_speed = gripper_speed
        self.ee_speed = ee_speed
        self.ee_acc = ee_acc
        self.ee_jerk = ee_jerk
        self.joint_speed = joint_speed
        self.joint_acc = joint_acc
        self.joint_jerk = joint_jerk

    def set_ee_pose(self, pose, speed=None, acc=None, wait=True, timeout=None):
        """Sets position of the robot based on the desired end-effector pose
        Args:
            pose (list[float]): Pose in x, y, z, roll, pitch, yaw (unit: mm, rad).
                If pose is not of length 6, the command errors
                If any components are None, these will be set to the current ee position component
            speed (float, optional): move speed (mm/s, rad/s). Defaults to None, meaning the current default is used
            acc (float, optional): move acceleration (mm/s^2, rad/s^2), Default is self.arm.last_used_tcp_acc
            wait (bool, optional): whether to wait for the arm to complete, default is True
            timeout (float, optional): maximum waiting time(unit: second), default is None(no timeout),
                only valid if wait is True
        Returns:
            int: code, 0 if it went through
        TODO:
            Throw more descriptive errors for each of these failed assertions
            Setting the xarm ik style of ee pose is really weird with how we define the speeds and accelerations!!
                - Make a different function call altogether - set_ee_pose_xarm()??
        """

        assert isinstance(pose, (np.ndarray, list))
        assert len(pose) == 6

        # If any elements in the desired pose are None, fill them with the current pose data
        if any(elem is None for elem in pose):
            current_pose = self.get_ee_pose()
            pose = np.where(np.equal(pose, None), current_pose, pose)  # type: ignore
        # Ensure that the above process removed all Nones from the commanded pose
        assert not any(elem is None for elem in pose)

        new_pose = pose

        if self.use_bullet_ik:
            joint_poses = self.get_inverse_kinematics(new_pose)

            # joint speed is just approximate
            # #TODO: actually set joint speed
            if speed is not None:
                joint_speed = speed / 500.0
            else:
                joint_speed = self.joint_speed
            return self.set_joint_pos(
                joint_poses, speed=joint_speed, mvacc=self.joint_acc, wait=wait
            )

        # Right now this will use XArm motion planning as well
        code = self.arm.set_position(
            *new_pose,
            speed=speed,
            mvacc=acc,
            wait=wait,
            timeout=timeout,
            is_radian=True,
        )

        return code

    def get_joint_pos(self):
        """Get robot joint positions
        Returns:
            list[float]: joint positions for the 6 joints in radians
        """
        positions, _, _ = self.get_joint_states()

        return positions

    def set_joint_pos(
        self,
        angles,
        speed=None,
        mvacc=None,
        wait=True,
        timeout=None,
    ):
        """Sets all servo angles on the robot
        This wrapper hides some other options from the xarm that are currently unimportant. Potentially in
        the future, we can un-hide some of these options to increase functionality (TODO)
        Args:
            angles (list, length=6): Servo angles in radians
            speed (float, optional): Move speed in rad/s. Defaults to None, in which case the current default is used
            mvacc (float, optional): Joint acceleration, rad/s^2. Defaults to last-used value
            wait (bool, optional):  Whether to wait until achieving the position. Defaults to True.
            timeout (float, optional): Wait time, units: second. Defaults to None.
        Returns:
            int: code. From XArm docs:
                code < 0: the last_used_angles/last_used_joint_speed/last_used_joint_acc will not be modified
                code >= 0: the last_used_angles/last_used_joint_speed/last_used_joint_acc will be modified
        """
        assert len(angles) == 6

        if not np.all(angles > consts.JOINT_LIMITS_LOWER) or not np.all(
            angles < consts.JOINT_LIMITS_UPPER
        ):
            raise ValueError(f"Joint angles were exceeded {angles}",angles)

        self.last_des_joint_pos = angles

        if speed is None:
            speed = self.joint_speed

        angles = angles + [0]  # Adding dummy angle for non-existent "7th xarm6 angle"

        code = self.arm.set_servo_angle(
            servo_id=None,
            angle=angles,
            speed=speed,
            mvacc=mvacc,
            mvtime=None,
            relative=False,
            is_radian=True,
            wait=wait,
            timeout=timeout,
            radius=None,
        )

    def get_joint_states(self, is_radian=True):
        """Returns joint states
        Args:
            is_radian (bool, optional): If true, returns units in radian.
            If false, returns units in degrees.
            Defaults to True.
        Returns:
            positions (list[float]): Joint positions
            velocities (list[float]): Joint velocities
            efforts (list[float]): Joint effort
        """

        code, [positions, velocities, efforts] = self.arm.get_joint_states(
            is_radian=is_radian
        )

        if code != 0:
            self._log(f"get_joint_states error, code {code}", verbose=True)
            raise Exception("SDK error")

        positions = positions[: self.dof]
        velocities = velocities[: self.dof]
        efforts = efforts[: self.dof]

        return positions, velocities, efforts

    def is_near_joint_pos(self, joint_pos, tol=0.0001):
        """Checks if the robot is already in some gripper position, to a specified tolerance
        Args:
            ee_pose (list): [x, y, z, roll, pitch, yaw]: The EE pose to check if the robot is near
            tol (float, optional): Gripper tolerance (mm). Defaults to 0.5.
        Returns:
            bool: True if the robot is near the specified pose, else False
        """

        current_joint_pos = self.get_joint_pos()
        joint_pos_check = np.all(np.isclose(current_joint_pos, joint_pos, atol=tol))
        return

    def get_ee_pose(self, is_radian=True):
        """Get x, y, z, roll, pitch, yaw of the end effector
        TODO: make into pq pose -- make everything in this manner
        Args:
            is_radian (bool, optional):
            Returns radians as default (True, None), returns degrees if false
            Defaults to self.arm.default_is_radian, which is True
        Returns:
           list[float]: x, y, z, roll, pitch, yaw
        """
        code, positions = self.arm.get_position(is_radian)  # can get as degrees

        return positions

    def is_near_ee_pose(self, ee_pose, xyz_tol=0.5, ori_tol=0.1):
        """Checks if the robot is already in some pose, to a specified tolerance
        Args:
            ee_pose (list): [x, y, z, roll, pitch, yaw]: The EE pose to check if the robot is near
            xyz_tol (float, optional): X, Y, and Z positional tolerance (mm). Defaults to 0.5.
            rpy_tol (float, optional): Roll, Pitch, and Yaw angular tolerance (rad). Defaults to 0.1.
        Returns:
            bool: True if the robot is near the specified pose, else False
        """
        assert len(ee_pose) == 6
        current_pose = self.get_ee_pose()
        xyz_check = np.all(np.isclose(current_pose[:3], ee_pose[:3], atol=xyz_tol))
        current_pose_quat = quat_from_euler(current_pose[3:])
        desired_pose_quat = quat_from_euler(ee_pose[3:])
        ori_check = quaternion_distance(current_pose_quat, desired_pose_quat) < ori_tol

        return xyz_check and ori_check

    def get_inverse_kinematics(self, pose: Union[list, np.ndarray]):
        """Returns the joint angles to achieve as calculated from the inverse kinematics
        Args:
            pose (list): [x, y, z, r, p, y] of the desired ee pose
        Returns:
            list: Six joint angles representing the IK solution
        """
        if self.use_bullet_ik:
            current_joint_pos = self.get_joint_pos()
            if self.ik_use_last_joint_pos:
                if self.last_des_joint_pos is None:
                    ik_joint_pos = current_joint_pos
                elif not self.is_near_joint_pos(self.last_des_joint_pos):
                    ik_joint_pos = current_joint_pos
                else:
                    ik_joint_pos = self.last_des_joint_pos
            else:
                ik_joint_pos = current_joint_pos
            angles = self.bullet_kinematics.get_ik(
                dpose=pose, current_joint_pos=list(ik_joint_pos)
            )
            return np.array(angles)
        else:
            code, angles = self.arm.get_inverse_kinematics(
                pose, input_is_radian=True, return_is_radian=True
            )
            if code != 0:
                self._log(f"get_inverse_kinematics error, code {code}", verbose=True)
                raise Exception("SDK error")
            # Return the first 6 angles only because we don't have the xarm 7
            return np.array(angles[:6])
