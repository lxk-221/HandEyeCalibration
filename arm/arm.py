"""
Arm 抽象基类。

手眼标定只需要两类能力:
  - 读取当前状态: 关节角 `joints` 和末端位姿 `T_ee2base`  (必须实现)
  - 移动 (可选):   `move2joints` / `move2pose`
    * 交互式标定不需要任何移动方法 (人通过 guiding/遥操移动, 程序只读取)。
    * 序列式 (非交互) 标定按序列类型自动选择调用哪一个。
    * 子类按硬件便利二选一实现, 或两者都实现; 未实现的会抛 NotImplementedError。

约定:
  - 平移单位: 米 (m)。
  - `T_ee2base` 遵循 OpenCV 手眼标定约定:
        p_base = T_ee2base @ p_ee   (即 T_gripper2base, 4x4 齐次矩阵)。
"""
import numpy as np


class Arm:
    def __init__(self) -> None:
        pass

    # ---- 必须实现: 只读状态 (标定的核心输入) ----
    @property
    def joints(self) -> np.ndarray:
        """当前关节角, np.ndarray, 单位 rad。"""
        raise NotImplementedError

    @property
    def T_ee2base(self) -> np.ndarray:
        """
        T_gripper2base: 4x4 齐次变换矩阵, 平移单位 m。
        把末端 (gripper) 系下的点变到 base 系: p_base = T @ p_ee。
        参考: https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
        """
        raise NotImplementedError

    # ---- 可选: 移动 (序列式标定按需调用) ----
    def move2joints(self, joints) -> None:
        """移动到给定关节角 (rad)。未实现则抛 NotImplementedError。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 move2joints")

    def move2pose(self, T_ee2base: np.ndarray) -> None:
        """移动到给定末端位姿 (4x4, 平移 m)。未实现则抛 NotImplementedError。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 move2pose")
