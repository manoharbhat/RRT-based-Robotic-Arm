""" Bullet IK class specifically for XArm
"""
import numpy as np

try:
    import pybullet as p
    import pybullet_data as pdata
    from pybullet_utils import bullet_client as bc
except ImportError as exc:
    raise Exception(
        print(f"{exc}, please make sure this is installed. Run `pip install pybullet`")
    ) from exc

from medra_robotics.utils.transformations import (
    quat_xyzw_from_wxyz,
    quat_wxyz_from_xyzw,
)
from medra_robotics.utils.helper_functions import quaternion_distance


class BulletKinematics:
    """Bullet Kinematics class for XArm

    TODO:
    * can directly import urdf/not xarm specific
    * Test nullspace

    Args:
    bullet_client (pybullet_client, optional): If you want to use a specific pybullet client, you can pass it in.
        Defaults to None. If None, will set self.bullet_client as pybullet (from import)
    """

    def __init__(self, bullet_client=None, nullspace=False):

        # setting up pybullet
        if bullet_client is None:
            # Headless mode
            self.bullet = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            self.bullet = bullet_client

        # This allows us to load xarm urdf
        self.bullet.setAdditionalSearchPath(pdata.getDataPath())

        # load arm into simulation
        self.sim_arm = self.bullet.loadURDF(
            "xarm/xarm6_robot.urdf", useFixedBase=True
        )  # note: we are using pybullet's xarm urdf
        self.dof = 6  # degrees of freedom

        self.residual_threshold = 1e-6  # todo: set in config

        self.nullspace = nullspace  # setting it as false directly

        # Meta IK Loop Params; loop around the IK solver to confirm IK is doing as advertised
        self.check_l2_threshold = (
            1e-5  # l2 threshold is the convergence position threshold for meta IK
        )
        self.check_orientation_threshold = (
            1e-3  # threshold for the convergence orientation threshold for meta IK
        )
        self.max_iterations = 10  # number of iterations for meta IK loop

        self.robot_index = [*range(1, 7)]

        if self.nullspace:
            (
                self.joint_limits_lower,
                self.joint_limits_upper,
                self.joint_ranges,
            ) = self.get_joint_ranges()

    def get_joint_states(self):
        joint_states = []
        for i in self.robot_index:
            rp = self.bullet.getJointState(self.sim_arm, i)[0]
            joint_states.append(rp)
        return joint_states

    def get_joint_ranges(self):

        lower_limits, upper_limits, joint_ranges = [], [], []

        for i in self.robot_index:
            joint_info = self.bullet.getJointInfo(self.sim_arm, i)
            ll, ul = joint_info[8:10]
            jr = ul - ll

            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(jr)

        return lower_limits, upper_limits, joint_ranges

    def _set_joints(self, joint_poses):
        for i, num in enumerate(self.robot_index):
            self.bullet.resetJointState(self.sim_arm, num, joint_poses[i])

    def get_ik(self, dpose, current_joint_pos=None):
        """Get inverse kinematics of

        Args:
            dpose (array_like, (6,) or (7,)): Desired pose either in (x, y, z, roll, pitch, yaw)/(x, y, z, w, x, y, z)
                We assume this is using XArm, units are in mm for position.
            current_joint_pos (array_like, (6,)): Current joint positions of XArm.
                Used if you want IK to find local solution to current joint position. Units radian. Defaults to None.

        Returns:
            list(float), shape (6,): Desired joint positions, units in radian.
        """
        assert len(dpose) in [6, 7]

        pos = np.array(dpose[0:3]) / 1000.0  # convert mm into meters
        orientation = dpose[3:]
        if len(orientation) == 3:  # euler angles
            orientation = self.bullet.getQuaternionFromEuler(orientation)
        elif len(orientation) == 4:  # quaternions
            orientation = quat_xyzw_from_wxyz(orientation)
            # pybullet is x, y, z, w

        kwargs = {
            "bodyUniqueId": self.sim_arm,
            "endEffectorLinkIndex": self.dof,
            "targetPosition": pos,
            "targetOrientation": orientation,
            "maxNumIterations": 1000,
            "residualThreshold": self.residual_threshold,
        }
        # first update the pose on bullet
        if current_joint_pos is not None:
            kwargs["currentPositions"] = list(current_joint_pos)

        if self.nullspace:
            kwargs["lowerLimits"] = self.joint_limits_lower
            kwargs["upperLimits"] = self.joint_limits_upper
            kwargs["jointRanges"] = self.joint_ranges
            kwargs["restPoses"] = self.get_joint_states()

        found_ik_solution = False

        joint_poses = [None for _ in range(self.dof)]

        for _ in range(self.max_iterations):
            joint_poses = self.bullet.calculateInverseKinematics(**kwargs)

            self._set_joints(joint_poses)

            if self.nullspace:
                kwargs["restPoses"] = list(joint_poses)
            else:
                kwargs["currentPositions"] = list(joint_poses)

            ls = self.bullet.getLinkState(self.sim_arm, self.dof)
            new_position = ls[4]
            new_orientation = ls[5]

            pos_check = np.linalg.norm(new_position - pos) < self.check_l2_threshold

            # Quaternion distance is 0-1 for 0 to 180deg
            ori_check = (
                quaternion_distance(orientation, new_orientation)
                < self.check_orientation_threshold
            )

            if ori_check and pos_check:
                found_ik_solution = True
                break

        if not found_ik_solution:
            raise ValueError(
                "IK solution was not found. It is unsafe to move to this solution!"
            )

        return list(joint_poses)

    def get_fk(self, joint_poses):
        """Forward Kinematics using PyBullet

        Args:
            joint_poses (array_like): array with self.dof values that correspond to the desired joint positions

        Returns:
            pose: (x, y, z, qw, qx, qy, qz)
        """
        if isinstance(joint_poses, np.ndarray):
            if joint_poses.shape != (self.dof,):
                raise ValueError(
                    f"Numpy array of pose is an invalid shape {joint_poses.shape}"
                )

        if isinstance(joint_poses, list):
            assert len(joint_poses) == self.dof

        self._set_joints(joint_poses)

        ee_link_state = self.bullet.getLinkState(
            self.sim_arm, self.dof, 0, computeForwardKinematics=True
        )

        quaternion_wxyz = quat_wxyz_from_xyzw(ee_link_state[5])

        # Convert meters to mm
        ee_link_pose = np.hstack((np.array(ee_link_state[4]) * 1000.0, quaternion_wxyz))

        return ee_link_pose
