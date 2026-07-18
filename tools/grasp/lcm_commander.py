#!/usr/bin/env python3
"""LCM 指令收发封装 (主控侧)。

四个独立 channel (无嵌套 struct):
  arm_command          主控 -> 接收端 (机械臂运动)
  hand_command         主控 -> 接收端 (灵巧手运动)
  execution_feedback   接收端 -> 主控 (执行反馈)
  ee_pose              接收端 -> 主控 (末端位姿实时转发, 替代主控直连 ROS 读 T_gripper2base)

阻塞协议: 每条指令带自增 cmd_id; 接收端执行完后用相同 cmd_id 回 execution_feedback。
move_arm()/move_hand() 发完指令后阻塞等待对应 cmd_id 的 feedback。

末端位姿: 后台 pump 线程持续处理 LCM 消息, 把最新 ee_pose 缓存到成员变量;
get_ee_pose() 取最新值即实时位姿 (拍照时刻直接取用, 无需依赖 ROS)。

用法:
    with LcmCommander() as cmd:
        cmd.move_hand([255,10,255,255,255,255])   # 张开手, 阻塞到接收端反馈完成
        T = cmd.get_ee_pose()                       # 取当前末端位姿 (实时)
        cmd.move_arm(T_ee2base, speed=0.8)          # 移动臂, 阻塞到位
        cmd.move_hand([0,5,70,80,80,70])            # 抓取
"""
import os
import sys
import threading
import time
from typing import Optional

import numpy as np

# lcm-gen 生成的模块用平级文件 (模块名==类名, 用 模块.类 访问)。注入本目录到 sys.path。
_LCM_TYPES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcm_types")
if _LCM_TYPES_DIR not in sys.path:
    sys.path.insert(0, _LCM_TYPES_DIR)

import arm_command          # noqa: E402
import hand_command         # noqa: E402
import execution_feedback   # noqa: E402
import ee_pose              # noqa: E402
import joint_command        # noqa: E402
import lcm                  # noqa: E402

CHANNEL_ARM = "arm_command"
CHANNEL_HAND = "hand_command"
CHANNEL_FEEDBACK = "execution_feedback"
CHANNEL_EE_POSE = "ee_pose"
CHANNEL_JOINT = "joint_command"      # 关节空间运动 (WARMUP/COOLDOWN, 避奇异)


def _create_lcm(lcm_url, retries=5):
    """创建 LCM, self test 偶发失败时重试。彻底失败给出诊断+修复指引。"""
    import time
    last_err = None
    for i in range(retries):
        try:
            return lcm.LCM(lcm_url)
        except Exception as e:
            last_err = e
            print(f"[commander] LCM 创建失败 ({i+1}/{retries}): {e}，重试...")
            time.sleep(0.5)
    raise RuntimeError(
        f"LCM 创建失败 ({last_err})。LCM self test 依赖多播 loopback。\n"
        "通常原因: 系统多播路由走了无线网卡导致回环不可靠。修复:\n"
        "  sudo ip link set lo multicast\n"
        "  sudo ip route add 224.0.0.0/4 dev lo\n"
        f"URL={lcm_url}"
    )


class LcmCommander:
    """主控侧的 LCM 指令收发器。
    后台线程持续 pump LCM (更新 ee_pose 缓存 + 收 feedback);
    move_arm/move_hand 发指令后阻塞等对应 cmd_id 的 feedback。"""

    def __init__(self, lcm_url: str = "udpm://224.0.0.1?ttl=0",
                 default_timeout: float = 30.0):
        self._lc = _create_lcm(lcm_url)
        self._default_timeout = default_timeout
        self._next_cmd_id = 1
        self._lock = threading.Lock()
        self._feedbacks = {}              # cmd_id -> execution_feedback
        self._feedback_event = threading.Event()
        self._latest_ee_pose = None       # 最近一帧 ee_pose 的 4x4 (np.float64)
        self._stop_event = threading.Event()
        self._lc.subscribe(CHANNEL_FEEDBACK, self._on_feedback)
        self._lc.subscribe(CHANNEL_EE_POSE, self._on_ee_pose)
        # 后台 pump 线程: 持续处理 LCM 消息 (更新 ee_pose + 收 feedback)
        self._pump_thread = threading.Thread(target=self._pump_loop, daemon=True)
        self._pump_thread.start()

    # ---- 后台 pump ----
    def _pump_loop(self):
        while not self._stop_event.is_set():
            self._lc.handle_timeout(50)   # 50ms 非阻塞, 让 stop_event 能及时退出

    def _on_feedback(self, _channel, data):
        fb = execution_feedback.execution_feedback.decode(data)
        with self._lock:
            self._feedbacks[fb.cmd_id] = fb
        self._feedback_event.set()

    def _on_ee_pose(self, _channel, data):
        msg = ee_pose.ee_pose.decode(data)
        T = np.asarray(msg.t_ee2base, dtype=np.float64).reshape(4, 4)
        with self._lock:
            self._latest_ee_pose = T

    # ---- 内部 ----
    def _alloc_cmd_id(self) -> int:
        with self._lock:
            cid = self._next_cmd_id
            self._next_cmd_id += 1
            return cid

    def _wait_feedback(self, cmd_id: int, timeout: float) -> execution_feedback:
        """阻塞等指定 cmd_id 的 feedback (pump 线程在后台收)。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._feedback_event.wait(timeout=min(0.1, max(0, deadline - time.monotonic()))):
                continue
            self._feedback_event.clear()
            with self._lock:
                if cmd_id in self._feedbacks:
                    return self._feedbacks.pop(cmd_id)
        raise TimeoutError(f"cmd_id={cmd_id} 等待 feedback 超时 ({timeout}s)")

    # ---- 公开接口 ----
    def get_ee_pose(self, timeout: float = 3.0) -> np.ndarray:
        """取最新末端位姿 T_gripper2base (4x4, 平移米)。未收到过则阻塞等首帧。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_ee_pose is not None:
                    return self._latest_ee_pose.copy()
            time.sleep(0.02)
        raise TimeoutError(f"未在 {timeout}s 内收到 ee_pose (桥是否在运行?)")

    def move_arm(self, T_ee2base, speed=0.8, accel=0.8, block=True,
                 timeout: Optional[float] = None) -> bool:
        """发 arm 运动指令并阻塞等完成 (block=True 时)。返回是否成功。"""
        cmd_id = self._alloc_cmd_id()
        msg = arm_command.arm_command()
        msg.cmd_id = cmd_id
        msg.t_ee2base = list(np.asarray(T_ee2base, dtype=np.float64).reshape(16))
        msg.speed = float(speed)
        msg.accel = float(accel)
        msg.block = bool(block)
        self._lc.publish(CHANNEL_ARM, msg.encode())
        if not block:
            return True
        fb = self._wait_feedback(cmd_id, timeout or self._default_timeout)
        if not fb.success:
            raise RuntimeError(f"arm 指令 cmd_id={cmd_id} 执行失败: {fb.error}")
        return True

    def move_joints(self, joints, speed=0.8, accel=0.8, block=True,
                    timeout: Optional[float] = None) -> bool:
        """发关节空间运动指令 (WARMUP/COOLDOWN 用, 避开笛卡尔插值的奇异/碰撞)。
        joints: 7 关节角 (弧度)。阻塞等 feedback。"""
        cmd_id = self._alloc_cmd_id()
        msg = joint_command.joint_command()
        msg.cmd_id = cmd_id
        msg.joints = [float(v) for v in joints[:7]]
        msg.speed = float(speed)
        msg.accel = float(accel)
        msg.block = bool(block)
        self._lc.publish(CHANNEL_JOINT, msg.encode())
        if not block:
            return True
        fb = self._wait_feedback(cmd_id, timeout or self._default_timeout)
        if not fb.success:
            raise RuntimeError(f"joint 指令 cmd_id={cmd_id} 执行失败: {fb.error}")
        return True

    def move_hand(self, positions, timeout: Optional[float] = None) -> bool:
        """发 hand 运动指令并阻塞等完成。positions: list[int] 0..255。"""
        cmd_id = self._alloc_cmd_id()
        msg = hand_command.hand_command()
        msg.cmd_id = cmd_id
        msg.n_dof = len(positions)
        msg.positions = [int(v) for v in positions]
        self._lc.publish(CHANNEL_HAND, msg.encode())
        fb = self._wait_feedback(cmd_id, timeout or self._default_timeout)
        if not fb.success:
            raise RuntimeError(f"hand 指令 cmd_id={cmd_id} 执行失败: {fb.error}")
        return True

    def close(self):
        self._stop_event.set()
        self._pump_thread.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
