from typing import Optional, Protocol, Union


class XArmAPIProtocol(Protocol):
    def set_state(self, state: int):
        ...

    def set_mode(self, mode: int = 0) -> int:
        ...

    def motion_enable(self, enable: bool):
        ...

    def get_position(self, is_radian: bool) -> tuple[int, list]:
        ...

    def get_joint_states(
        self, is_radian: Optional[bool] = None, num=3
    ) -> tuple[int, list]:
        ...

    def set_position(
        self,
        x=None,
        y=None,
        z=None,
        roll=None,
        pitch=None,
        yaw=None,
        radius=None,
        speed=None,
        mvacc=None,
        mvtime=None,
        relative=False,
        is_radian=None,
        wait=False,
        timeout=None,
        **kwargs
    ):
        ...

    def set_gripper_position(self, *args, **kwargs) -> int:
        ...

    def get_gripper_position(self, **kwargs) -> tuple[int, Optional[int]]:
        ...

    def set_gripper_enable(self, enable: bool) -> int:
        ...

    def set_servo_angle(
        self,
        servo_id,
        angle,
        speed,
        mvacc,
        mvtime,
        relative,
        is_radian,
        wait,
        timeout,
        radius,
    ) -> int:
        ...

    def disconnect(self):
        ...

    def get_tgpio_analog(self, ionum: int) -> tuple[int, list]:
        ...

    def get_tgpio_digital(self, ionum: int) -> tuple[int, Union[list, int]]:
        ...

    def set_tgpio_digital(self, ionum: int, value: int) -> int:
        ...

    def set_cgpio_digital(self, ionum: int, value: int) -> int:
        ...

    def set_collision_sensitivity(self, value) -> int:
        ...

    def set_collision_rebound(self, on: bool) -> list:
        ...

    def set_tcp_offset(self, offset: list[float]) -> int:
        ...

    def set_joint_jerk(self, jerk: float) -> int:
        ...

    def set_gripper_speed(self, speed: float) -> int:
        ...

    def set_tcp_jerk(self, jerk: float) -> int:
        ...

    def get_inverse_kinematics(
        self,
        pose: list,
        input_is_radian: Optional[bool] = None,
        return_is_radian: Optional[bool] = None,
    ) -> tuple[int, list]:
        ...

    def clean_error(self) -> int:
        ...

    def clean_gripper_error(self) -> int:
        ...

    @property
    def mode(self) -> int:
        ...

    @property
    def state(self) -> int:
        ...

    @property
    def tcp_offset(self) -> list[float]:
        ...

    @property
    def last_used_tcp_speed(self) -> Optional[int]:
        ...

    @property
    def last_used_tcp_acc(self) -> Optional[int]:
        ...

    @property
    def tcp_jerk(self) -> float:
        ...

    @property
    def last_used_joint_speed(self) -> Optional[float]:
        ...

    @property
    def last_used_joint_acc(self) -> Optional[float]:
        ...

    @property
    def joint_jerk(self) -> float:
        ...
