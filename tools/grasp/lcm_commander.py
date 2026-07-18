#!/usr/bin/env python3
"""LCM 指令收发封装 (主控侧)。

三个独立 channel (无嵌套 struct):
  arm_command          主控 -> 接收端 (机械臂运动)
  hand_command         主控 -> 接收端 (灵巧手运动)
  execution_feedback   接收端 -> 主控 (执行反馈)

阻塞协议: 每条指令带自增 cmd_id; 接收端执行完后用相同 cmd_id 回 execution_feedback。
LcmCommander.move_arm()/move_hand() 发完指令后阻塞等待对应 cmd_id 的 feedback,
超时则报错。这样主控可以写顺序的阻塞式流程代码。

用法:
    with LcmCommander() as cmd:
        cmd.move_hand([255,10,255,255,255,255])   # 张开手, 阻塞到接收端反馈完成
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
import lcm                  # noqa: E402

CHANNEL_ARM = "arm_command"
CHANNEL_HAND = "hand_command"
CHANNEL_FEEDBACK = "execution_feedback"


class LcmCommander:
    """主控侧的 LCM 指令收发器。发 arm/hand 指令并阻塞等 execution_feedback。"""

    def __init__(self, lcm_url: str = "udpm://224.0.0.1?ttl=0",
                 default_timeout: float = 30.0):
        self._lc = lcm.LCM(lcm_url)
        self._default_timeout = default_timeout
        self._next_cmd_id = 1
        self._lock = threading.Lock()
        # cmd_id -> execution_feedback 对象; 收到反馈时填入。
        self._feedbacks = {}
        self._feedback_event = threading.Event()
        self._sub = self._lc.subscribe(CHANNEL_FEEDBACK, self._on_feedback)

    def _on_feedback(self, _channel, data):
        fb = execution_feedback.execution_feedback.decode(data)
        with self._lock:
            self._feedbacks[fb.cmd_id] = fb
        self._feedback_event.set()

    def _alloc_cmd_id(self) -> int:
        with self._lock:
            cid = self._next_cmd_id
            self._next_cmd_id += 1
            return cid

    def _wait_feedback(self, cmd_id: int, timeout: float) -> execution_feedback:
        """阻塞等指定 cmd_id 的 feedback。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # pump 收消息 (主控侧没有独立 spin 线程, 在等待循环里 handle)
            self._lc.handle_timeout(max(1, int((deadline - time.monotonic()) * 1000)))
            with self._lock:
                if cmd_id in self._feedbacks:
                    return self._feedbacks.pop(cmd_id)
        raise TimeoutError(f"cmd_id={cmd_id} 等待 feedback 超时 ({timeout}s)")

    # ---- 公开接口: 阻塞式指令 ----
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
        self._lc.unsubscribe(self._sub)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
