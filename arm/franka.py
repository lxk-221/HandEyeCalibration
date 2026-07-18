"""
Franka Emika (FR3/FCI) 机械臂实现 (franky SDK, 无 ROS)。

franky 是 libfranka/FCI 的 pybind11 封装, 通过 TCP 直连机器人, 不依赖 ROS。
参考: prior session 中 replay_pose_sequence.py 的验证。

关键 API (已在本机离线验证):
  - Robot(ip)
  - robot.recover_from_errors()
  - robot.move(JointMotion(JointState(np.array([7 个关节]))))
  - robot.move(CartesianMotion(Affine(translation, quaternion)))  # 或 Affine(4x4)
  - robot.current_joint_positions.q     -> 7-关节角 (rad)
  - robot.current_pose.end_effector_pose.matrix -> 4x4 T_ee2base (平移 m)
"""
import numpy as np
import franky

from arm.arm import Arm


class Franka(Arm):
    def __init__(self, robot_ip: str = "172.16.0.8",
                 relative_dynamics_factor: float = 0.3,
                 automatic_recovery: bool = True):
        """
        robot_ip: FCI 主机地址 (机器人须处于 FCI exec 模式, 蓝灯)。
        relative_dynamics_factor: 速度/加速度缩放 (0~1), 越小越稳。默认 0.3 与遥操一致。
        automatic_recovery: move 失败时自动 recover_from_errors 重试一次。
        """
        self._robot = franky.Robot(robot_ip)
        self._dyn = float(relative_dynamics_factor)
        self._auto_recovery = automatic_recovery
        self._robot.relative_dynamics_factor = self._dyn
        if self._robot.has_errors and self._auto_recovery:
            self._robot.recover_from_errors()
        super().__init__()

    # ---- 只读状态 ----
    @property
    def joints(self) -> np.ndarray:
        """当前 7 关节角, 单位 rad。"""
        return np.array(self._robot.current_joint_positions.q, dtype=np.float64)

    @property
    def T_ee2base(self) -> np.ndarray:
        """T_gripper2base, 4x4 齐次矩阵, 平移单位 m。"""
        return np.array(
            self._robot.current_pose.end_effector_pose.matrix, dtype=np.float64
        ).reshape(4, 4)

    # ---- 移动 ----
    def move2joints(self, joints) -> None:
        """移动到给定关节角 (rad, 7 维)。"""
        q = np.asarray(joints, dtype=np.float64).reshape(7)
        self._run_motion(franky.JointMotion(
            franky.JointState(q), relative_dynamics_factor=self._dyn))

    def move2pose(self, T_ee2base: np.ndarray) -> None:
        """移动到给定末端位姿 (4x4, 平移 m)。"""
        T = np.asarray(T_ee2base, dtype=np.float64).reshape(4, 4)
        self._run_motion(franky.CartesianMotion(
            franky.Affine(T), relative_dynamics_factor=self._dyn))

    def _run_motion(self, motion) -> None:
        """执行一次运动, 失败时按配置自动 recover 重试一次。"""
        try:
            self._robot.move(motion)
        except Exception as e:
            if not self._auto_recovery:
                raise
            print(f"[Franka] move 失败 ({e}), 尝试 recover_from_errors 后重试...")
            self._robot.recover_from_errors()
            self._robot.move(motion)

    def close(self) -> None:
        """franky 无显式断开, 留空。子类可按需覆写。"""
        pass
