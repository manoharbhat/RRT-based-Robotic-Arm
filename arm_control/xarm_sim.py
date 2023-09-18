"""Contains XArm class based on the Python XArmAPI
"""
from typing import Optional, Union

import numpy as np
import pybullet as p
from pybullet_utils import bullet_client as bc
import pybullet_data as pd

import warnings
import time


from medra_robotics.utils.transformations import (
    hmatrix_from_pq,
    hmatrix_in_a_to_hmatrix_in_b,
    hmatrix_inverse,
    position_euler_xyz_from_matrix,
    quat_wxyz_from_xyzw,
)


import medra_robotics.arm_control.xarm_constants as consts
from medra_robotics.arm_control.kinematics import BulletKinematics


def pybullet_gui_setup():
    """One time PyBullet GUI Client Setup

    TODO: Write XArmSim in a way that allows us to run this multi-threaded headless

    You only need to initialize this once since only one GUI thread can be run

    Returns:
        PyBullet Client: Instance of PyBullet Client
    """

    pgui = bc.BulletClient(connection_mode=p.GUI)
    pgui.setGravity(0, 0, -9.8)
    time_step = 1.0 / 60  # Running sim at 60Hz
    pgui.setTimeStep(time_step)

    # set search path (so we can directly load xarm urdf from pybullet repo)
    pgui.setAdditionalSearchPath(pd.getDataPath())

    # Move camera closer to robot
    cam_debug_info = p.getDebugVisualizerCamera()
    camera_scale = 1.5
    pgui.resetDebugVisualizerCamera(
        cameraDistance=camera_scale,
        cameraYaw=cam_debug_info[-4],
        cameraPitch=cam_debug_info[-3],
        cameraTargetPosition=cam_debug_info[-1],
    )

    pgui.setRealTimeSimulation(1)

    return pgui


class XArmSim:
    """A Simulated version of the XArm API
    Do not use this class directly. It can be used as an alternative backend for the XArm class


    TODO:
        * Test nullspace mode
        * Set default acceleration
    """

    default_is_radian = True

    def __init__(self, bullet_client: bc.BulletClient, use_dynamics=True, **kwargs):
        self.dof = 6  # degrees of freedom of robot
        self.use_dynamics = use_dynamics  # whether or not using dynamics to run robot
        self.bullet_client = bullet_client  # pybullet client
        self.use_bullet_ik = True
        self.bullet_kinematics = BulletKinematics()
        self.ik_use_last_joint_pos = False
        self._ee_speed = None
        flags = (
            self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
            or self.bullet_client.URDF_USE_SELF_COLLISION
        )

        # Loading xarm urdf from pybullet's repo
        self.arm = self.bullet_client.loadURDF(
            "xarm/xarm6_with_gripper.urdf",  # TODO: have our own locally stored urdf
            useFixedBase=True,
            flags=flags,
        )

        self.robot_index = [*range(1, 7)]  # robot joint indices 1-7
        self.gripper_index = [*range(8, 14)]  # robot gripper indices 8-13
        self._endpoint_transform_mat = np.eye(4)

        self.robot_forces = 2400  # default values

        # Setting linear and angular damping for the robot
        for j in range(self.bullet_client.getNumJoints(self.arm)):
            self.bullet_client.changeDynamics(
                self.arm, j, linearDamping=0, angularDamping=0
            )
            info = p.getJointInfo(self.arm, j)

        self.nullspace = False  # setting it as false directly

        # Trajectory Playback Parameters
        self.history_index = 0  # which index we are on in trajectory history
        self.finished_traj = False  # flag for when trjectory is done

        # This is not fully tested yet
        if self.nullspace:
            # joint lower limits (used for null space)
            self.joint_limits_lower = (
                np.array([-360.0, -118.0, -225.0, -360.0, -97.0, -360.0])
                * np.pi
                / 180.0
            )
            # joint upper limits (used for null space)
            self.joint_limits_upper = (
                np.array([360.0, 120.0, 11.0, 260.0, 180.0, 360.0]) * np.pi / 180.0
            )
            # joint ranges for null space
            self.joint_ranges = self.joint_limits_upper - self.joint_limits_lower

            # restposes for null space, current pose for using camera
            self.rest_pose = [
                -0.190844,
                -0.452549,
                -1.355301,
                0.065357,
                1.465232,
                -3.426718,
            ]  # todo: make this dynamic or set it in config

    def _is_near_joint_pos(self, joint_pos, tol=0.001):
        """Checks if the robot is already in some gripper position, to a specified tolerance

        Args:
            ee_pose (list): [x, y, z, roll, pitch, yaw]: The EE pose to check if the robot is near
            tol (float, optional): Gripper tolerance (mm). Defaults to 0.5.

        Returns:
            bool: True if the robot is near the specified pose, else False
        """

        _, [current_positions, *_] = self.get_joint_states()
        delta = np.sum(np.subtract(current_positions, joint_pos[:6]))
        joint_pos_check = np.all(np.isclose(current_positions, joint_pos[:6], atol=tol))
        return joint_pos_check, delta

    def set_servo_angle(
        self,
        angle: Optional[list[float]] = None,
        wait=False,
        timeout=None,
        position_gain=0.1,
        **kwargs,
    ):
        """Control the robot joints
        Args:
            angle (list[float]): List of robot joint positions, length 6
            wait (bool, optional):  Whether to wait until achieving the position. Defaults to False.
            timeout (float, optional): Wait time, units: second. Defaults to None.
            position_gain (float, optional): Position gain (kp) for controller. Changes speed. Defaults to 0.01.
        """
        if angle is None:
            return
        if self.use_dynamics:
            self.bullet_client.setJointMotorControlArray(
                self.arm,
                [*range(1, self.dof + 1)],
                self.bullet_client.POSITION_CONTROL,
                angle[:6],
                forces=[self.robot_forces] * self.dof,
                positionGains=[position_gain] * self.dof,
            )

        else:
            for index, robot_index in enumerate(self.robot_index):
                self.bullet_client.resetJointState(self.arm, robot_index, angle[index])

        if wait:
            self.wait(angle, self._is_near_joint_pos, timeout=timeout)

        return 0

    def _gripper_real2sim(self, gripper_pos, gripper_speed=None):
        """Convert gripper positions on real robot to pybullet sim robot

        Args:
            gripper_pos (float): Gripper position on real robot.
                If None will return gripper open position.
            gripper_speed (float, optional): Gripper speed on real robot.
                Defaults to None, which returns maximum gripper speed.

        Returns:
            sim_gripper_pos (float): Gripper position on sim robot
            sim_gripper_speed (float): Gripper speed on sim robot
        """
        gripper_pos_slope = (consts.SIM_OPEN_GRIPPER - consts.SIM_CLOSE_GRIPPER) / (
            consts.MAX_GRIPPER - consts.MIN_GRIPPER
        )

        if gripper_pos is None:
            sim_gripper_pos = consts.SIM_OPEN_GRIPPER  # just open gripper
        else:
            sim_gripper_pos = consts.SIM_CLOSE_GRIPPER + gripper_pos_slope * (
                gripper_pos - consts.MIN_GRIPPER
            )

        sim_gripper_pos = np.clip(
            sim_gripper_pos,
            np.minimum(consts.SIM_MIN_GRIPPER, consts.SIM_MAX_GRIPPER),
            np.maximum(consts.SIM_MIN_GRIPPER, consts.SIM_MAX_GRIPPER),
        )

        # convert real speed to sim speed

        speed_slope = (consts.SIM_MAX_GRIPPER_SPEED - consts.SIM_MIN_GRIPPER_SPEED) / (
            consts.MAX_GRIPPER_SPEED - consts.MIN_GRIPPER_SPEED
        )

        if gripper_speed is None:
            sim_gripper_speed = consts.SIM_MAX_GRIPPER_SPEED
        else:
            sim_gripper_speed = consts.SIM_MIN_GRIPPER_SPEED + speed_slope * (
                gripper_speed - consts.MIN_GRIPPER_SPEED
            )

        sim_gripper_speed = np.clip(
            sim_gripper_speed,
            consts.SIM_MIN_GRIPPER_SPEED,
            consts.SIM_MAX_GRIPPER_SPEED,
        )

        return sim_gripper_pos, sim_gripper_speed

    def _gripper_sim2real(self, gripper_pos, gripper_speed=None):
        """Convert gripper positions on real robot to pybullet sim robot

        Args:
            gripper_pos (float): Gripper position on real robot. If None will return gripper open position.
            gripper_speed (float, optional): Gripper speed on real robot.
                Defaults to None, which returns maximum gripper speed.

        Returns:
            real_gripper_pos (float): Gripper position on sim robot
            real_gripper_speed (float): Gripper speed on sim robot
        """

        # convert real gripper pos to sim gripper pos
        gripper_pos_slope = (consts.MAX_GRIPPER - consts.MIN_GRIPPER) / (
            consts.SIM_OPEN_GRIPPER - consts.SIM_CLOSE_GRIPPER
        )

        if gripper_pos is None:
            real_gripper_pos = consts.MAX_GRIPPER  # just open gripper
        else:
            real_gripper_pos = consts.MIN_GRIPPER + gripper_pos_slope * (
                gripper_pos - consts.SIM_CLOSE_GRIPPER
            )

        real_gripper_pos = np.clip(
            real_gripper_pos, consts.MIN_GRIPPER, consts.MAX_GRIPPER
        )

        speed_slope = (consts.MAX_GRIPPER_SPEED - consts.MIN_GRIPPER_SPEED) / (
            consts.SIM_MAX_GRIPPER_SPEED - consts.SIM_MIN_GRIPPER_SPEED
        )

        if gripper_speed is None:
            real_gripper_speed = consts.SIM_MAX_GRIPPER_SPEED
        else:
            real_gripper_speed = consts.MIN_GRIPPER_SPEED + speed_slope * (
                gripper_speed - consts.SIM_MIN_GRIPPER_SPEED
            )

        real_gripper_speed = np.clip(
            real_gripper_speed,
            consts.MIN_GRIPPER_SPEED,
            consts.MAX_GRIPPER_SPEED,
        )

        return real_gripper_pos, real_gripper_speed

    def get_position(self, is_radian=None, **kwargs):
        eef_pos_in_world = (
            np.array(self.bullet_client.getLinkState(self.arm, self.dof)[4]) * 1000.0
        )

        eef_orn_in_world = np.array(
            self.bullet_client.getLinkState(
                self.arm,
                self.dof,
            )[5]
        )

        eef_orn_in_world = quat_wxyz_from_xyzw(eef_orn_in_world)

        eef_pose_in_world = hmatrix_from_pq(
            np.hstack([eef_pos_in_world, eef_orn_in_world])
        )

        base_pos_in_world = (
            np.array(
                self.bullet_client.getBasePositionAndOrientation(
                    self.arm,
                )[0]
            )
            * 1000.0
        )
        base_orn_in_world = np.array(
            self.bullet_client.getBasePositionAndOrientation(
                self.arm,
            )[1]
        )
        base_orn_in_world = quat_wxyz_from_xyzw(base_orn_in_world)
        base_pose_in_world = hmatrix_from_pq(
            np.hstack([base_pos_in_world, base_orn_in_world])
        )

        world_pose_in_base = hmatrix_inverse(base_pose_in_world)

        eef_pose_in_base = hmatrix_in_a_to_hmatrix_in_b(
            eef_pose_in_world, world_pose_in_base
        )

        euler_pose = position_euler_xyz_from_matrix(eef_pose_in_base)

        return 0, euler_pose

    def get_joint_states(self, **kwargs) -> tuple[int, list]:
        """Returns joint states in simulation

        Args:
            is_radian (bool, optional): If true, returns units in radian.
            If false, returns units in degrees.
            Defaults to True.

        Returns:
            positions (list[float]): Joint positions
            velocities (list[float]): Joint velocities
            efforts (list[float]): Joint effort
        """

        joint_states = self.bullet_client.getJointStates(self.arm, self.robot_index)

        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        efforts = [state[3] for state in joint_states]

        return 0, [positions, velocities, efforts]

    # TODO do we need this?
    def _get_gripper_state(self):
        """Obtains the joint positions of the gripper

        Returns:
            positions: (6,) joint positions of the gripper

        """
        joint_states = self.bullet_client.getJointStates(self.arm, self.gripper_index)

        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        efforts = [state[3] for state in joint_states]

        return positions, velocities, efforts

    def get_gripper_position(self, **kwargs) -> tuple[int, int]:
        """Get the gripper position"""

        positions, _, _ = self._get_gripper_state()

        # Gripper position in simulation
        # TODO: confirm this is right
        pos = positions[-1]

        # Convert gripper position in simulation to real
        pos, _ = self._gripper_sim2real(pos)

        return 0, pos

    def _is_near_gripper_pos(self, gripper_pos, tol=1):
        """Checks if the robot is already in some gripper position, to a specified tolerance

        Args:
            ee_pose (list): [x, y, z, roll, pitch, yaw]: The EE pose to check if the robot is near
            tol (float, optional): Gripper tolerance (mm). Defaults to 0.5.

        Returns:
            bool: True if the robot is near the specified pose, else False
        """

        _, position = self.get_gripper_position()
        delta = np.sum(position - gripper_pos)
        gripper_check = np.all(np.isclose(position, gripper_pos, atol=tol))
        return gripper_check, delta

    # Should be same order as xarm.py, but need to change when this is called to make this work!
    def set_gripper_position(
        self,
        pos,
        wait=False,
        speed=consts.DEFAULT_GRIPPER_SPEED,
        timeout=None,
        **kwargs,
    ):
        """Control gripper position and speed

        Args:
            sim_gripper_pos (float): Gripper position command
            sim_gripper_speed (float, optional): Gripper speed command (which changes position kp)
            wait (bool, optional): Whether to wait until achieving the position. Defaults to False.
            speed (float, domain [1000, 5000], optional): units: r/min. Defaults to DEFAULT_GRIPPER_SPEED.
            timeout (float, optional): Wait time, units: second. Defaults to None.
        """

        if speed is not None:
            self._gripper_speed = speed

        pos = np.clip(pos, consts.MIN_GRIPPER, consts.MAX_GRIPPER)

        sim_gripper_pos, sim_gripper_speed = self._gripper_real2sim(pos, speed)

        print(sim_gripper_pos)

        if self.use_dynamics:
            self.bullet_client.setJointMotorControlArray(
                self.arm,
                self.gripper_index,
                self.bullet_client.POSITION_CONTROL,
                [sim_gripper_pos] * len(self.gripper_index),
                forces=[self.robot_forces] * len(self.gripper_index),
                positionGains=[sim_gripper_speed] * len(self.gripper_index),
            )
        else:
            for i in self.gripper_index:
                self.bullet_client.resetJointState(self.arm, i, sim_gripper_pos)
        if wait:
            self.wait(pos, self._is_near_gripper_pos, timeout=timeout)

        return 0

    def enable_gripper(self):
        """Simple wrapper function to enable the gripper"""
        warnings.warn("Enable gripper is unsupported in simulation")

    def disconnect(self):
        """Disconnects from PyBullet Client"""
        self.bullet_client.disconnect()

    def set_state(self, state: int):
        pass

    def set_mode(self, mode: int = 0) -> int:
        return mode

    def motion_enable(self, enable: bool):
        pass

    def set_position(self, *args, **kwargs):
        pass

    def set_gripper_enable(self, enable: bool) -> int:
        return 0

    def get_tgpio_analog(self, ionum: int) -> tuple[int, list]:
        return 0, []

    def get_tgpio_digital(self, ionum: int) -> tuple[int, Union[list, int]]:
        return 0, []

    def set_tgpio_digital(self, ionum: int, value: int) -> int:
        return 0

    def set_cgpio_digital(self, ionum: int, value: int) -> int:
        return 0

    def set_collision_sensitivity(self, value) -> int:
        return 0

    def set_collision_rebound(self, on: bool) -> list:
        return []

    def set_tcp_offset(self, offset: list[float]) -> int:
        return 0

    def set_joint_jerk(self, jerk: float) -> int:
        return 0

    def set_gripper_speed(self, speed: float) -> int:
        return 0

    def set_tcp_jerk(self, jerk: float) -> int:
        return 0

    def get_inverse_kinematics(
        self,
        pose: list,
        input_is_radian: Optional[bool] = None,
        return_is_radian: Optional[bool] = None,
    ) -> tuple[int, list]:
        return 0, [0, 0, 0, 0, 0, 0]

    def clean_error(self) -> int:
        return 0

    def clean_gripper_error(self) -> int:
        return 0

    @property
    def mode(self) -> int:
        return 0

    @property
    def state(self) -> int:
        return 0

    @property
    def last_used_tcp_speed(self) -> Optional[int]:
        pass

    @property
    def last_used_tcp_acc(self) -> Optional[int]:
        pass

    @property
    def tcp_jerk(self) -> float:
        return 0

    @property
    def last_used_joint_speed(self) -> Optional[float]:
        pass

    @property
    def last_used_joint_acc(self) -> Optional[float]:
        pass

    @property
    def joint_jerk(self) -> float:
        return 0

    @property
    def gripper_speed(self):
        return self._gripper_speed

    @property
    def operating_state(self):
        """XArm state

        Returns:
            1: in motion
            2: sleeping
            3: suspended
            4: stopping
        """
        warnings.warn("Operating state is unsupported in simulation")

    @property
    def tcp_offset(self):
        """The currently set TCP (tool center point) offset for the XArm

        This is measured with respect to the center of the last link of the arm

        Returns:
            list: [x, y, z, roll, pitch, yaw] TCP offset
        """
        warnings.warn("Reading TCP offset is unsupported in simulation")
        return []

    @tcp_offset.setter
    def tcp_offset(self, offset):
        """Sets the TCP offset for the XArm

        Args:
            offset (list, length 6): [x, y, z, roll, pitch, yaw] desired TCP offset
        """
        warnings.warn("Setting TCP offset is unsupported in simulation")

    def wait(self, pose, is_near, timeout: Optional[float] = None, **kwargs):
        if timeout is not None:
            current_timeout = time.time() + timeout
        else:
            current_timeout = None
        delta = 10000
        # Lazy evaluation means first conditional triggers
        while True:
            # Caution: If you disable setRealTimeSimulation, you need to stepSimulation here or nothing will work
            time.sleep(1.0 / 60)
            near, new_delta = is_near(pose, **kwargs)
            converged = abs(delta - new_delta) < 0.000002
            if near:
                break
            if converged:
                warnings.warn("did not get close, but converged")
                break
            delta = new_delta
            if current_timeout is not None:
                if time.time() > current_timeout:
                    break

    @property
    def current_mode(self):
        return "position"

    # TODO this should probably live elsewhere
    def _run_real_robot_traj(
        self, joint_pos_history, gripper_pos_history=None, gripper_speed_history=None
    ):
        """Run real robot trajectory with histories of joint pos, gripper pos, and gripper speed

        Note that gripper pos and speed are based off of real robot data

        Args:
            joint_pos_history (list[list[float]], ): List of a list of robot joints positions (length 6)
            gripper_pos_history (list[float], optional): List of gripper positions. Defaults to None.
            gripper_speed_history (list[float], optional): List of gripper speeds. Defaults to None.
        """
        joint_pos_command = joint_pos_history[self.history_index]
        assert len(joint_pos_command) == self.dof
        if gripper_pos_history is not None:
            gripper_pos_command = gripper_pos_history[self.history_index]
        else:
            gripper_pos_command = None
        if gripper_speed_history is not None:
            gripper_speed_command = gripper_speed_history[self.history_index]
        else:
            gripper_speed_command = None

        gripper_pos_command, gripper_speed_command = self._gripper_real2sim(
            gripper_pos_command, gripper_speed_command
        )

        # Get joint states for robot and gripper
        robot_joint_states = self.bullet_client.getJointStates(
            self.arm, self.robot_index
        )

        sensed_joint_pos = np.asarray([state[0] for state in robot_joint_states])

        # Check if robot is in commanded pose
        threshold = 5e-2
        norm = np.linalg.norm(abs(joint_pos_command - sensed_joint_pos))
        # if robot is in commanded pose, go to next index
        if norm < threshold:
            if self.history_index < len(joint_pos_history) - 1:
                self.history_index += 1
            else:
                if not self.finished_traj:
                    print("Trajectory finished. Press ESC from GUI to exit")
                self.finished_traj = True

        self.set_joint_pos(joint_pos_command)
        if gripper_pos_command:
            self.set_gripper_position(gripper_pos_command, speed=gripper_speed_command)
