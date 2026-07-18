#!/usr/bin/env python3
"""
eye-in-hand 手眼标定 主程序 (重构后)。

依赖: cv2, numpy + 当前组合的 Arm SDK + Camera SDK, 不引入 ROS。
换硬件组合只需改下方两行 import (ARM_CLS / CAMERA_CLS) 及对应构造参数。

两种标定模式:
  - interactive (默认): 人通过 guiding/Xbox/手扶 移动臂, 按键采集 -> 求解。
        python main.py --mode interactive
  - sequence:           给定 joints 或 pose 序列 (JSON), 程序自动移动 -> 采集 -> 求解。
        python main.py --mode sequence --sequence poses.json

操作 (interactive): 实时预览窗口聚焦时按键生效
    c=采集当前位姿  s=求解保存  q=退出
"""
import argparse
import json
import os
import time
from typing import List

import cv2
import numpy as np

# ====== 换硬件组合: 仅改这两行 import + 构造参数 ======
from arm.franka import Franka        as ArmImpl      # noqa: E402
from camera.realsense import RealSenseCamera as CameraImpl  # noqa: E402
# 如要切到 lbot + orbbec:
#   from arm.lbot import LBot as ArmImpl
#   from camera.orbbec import OrbbecCamera as CameraImpl
# ===================================================

import calibration as calib                # noqa: E402
from arm.arm import Arm                    # noqa: E402
from camera.camera import Camera           # noqa: E402

# ---------------- 配置 ----------------
OUT_DIR = "./output"
WIN_NAME = "hand_eye (c=collect s=solve q=quit)"
HAND_EYE_NPZ = os.path.join(OUT_DIR, "hand_eye.npz")


# ---------------- 采集 (硬件无关) ----------------
def collect_sample(arm: Arm, camera: Camera, samples: List[calib.Sample], idx: int) -> int:
    """在当前位姿采集一个样本: 读位姿 -> 取帧 -> 检测棋盘 -> 推入 samples。
    成功返回 idx+1, 失败也返回 idx+1。"""
    T_g2b = arm.T_ee2base.copy()
    try:
        joints = arm.joints.copy()
    except NotImplementedError:
        joints = None
    rpy_show = calib.rpy_deg_from_T(T_g2b)

    bgr = camera.get_frame()
    if bgr is None:
        print("  取帧失败"); return idx + 1
    corners, pattern_used = calib.detect_chessboard(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if corners is None:
        cv2.imwrite(f"{OUT_DIR}/fail_{idx}.png", bgr)
        print(f"  未检测到棋盘 -> {OUT_DIR}/fail_{idx}.png  rpy(xyz)={rpy_show.round(0)}")
        return idx + 1
    samples.append(calib.Sample(
        T_gripper2base=T_g2b, corners=corners, pattern=pattern_used, joints=joints))
    vis = bgr.copy(); cv2.drawChessboardCorners(vis, pattern_used, corners, True)
    cv2.imwrite(f"{OUT_DIR}/sample_{idx:03d}.png", vis)
    print(f"  OK #{len(samples)} rpy_deg(xyz)={rpy_show.round(0)} board={pattern_used}")
    return idx + 1


def solve_and_save(samples: List[calib.Sample], K: np.ndarray, dist: np.ndarray) -> None:
    """求解 + 打印 + 存盘。"""
    if len(samples) < calib.MIN_SAMPLES:
        print(f"  至少 {calib.MIN_SAMPLES} 组, 当前 {len(samples)}"); return
    res = calib.solve_hand_eye(samples, K, dist)
    if res is None:
        print("  PnP 全失败或样本不足, 无法求解"); return
    print("\nX = T_cam2gripper (m) =\n",
          np.array2string(res.X, precision=5, suppress_small=True))
    print(f"残差({res.n_frames}帧): 旋转 {res.rot_mean_deg:.3f}/{res.rot_max_deg:.3f}°(均/最大)"
          f"  平移 {res.trans_mean_m*1000:.2f}/{res.trans_max_m*1000:.2f}mm(均/最大)")
    os.makedirs(OUT_DIR, exist_ok=True)
    T_g2b_all = np.stack([s.T_gripper2base for s in samples])
    joints_all = None
    if all(s.joints is not None for s in samples):
        joints_all = np.stack([s.joints for s in samples])
    savez = dict(X=res.X, K=K, dist=dist, T_gripper2base_all=T_g2b_all,
                 units="translation in m, joints in rad")
    if joints_all is not None:
        savez["joints_all"] = joints_all
    np.savez(HAND_EYE_NPZ, **savez)
    print(f"  -> {HAND_EYE_NPZ}")


# ---------------- 模式: interactive ----------------
def run_interactive(arm: Arm, camera: Camera) -> None:
    samples: List[calib.Sample] = []
    idx = 0
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    print("实时预览已打开 (窗口聚焦时按键生效): c=采集 s=求解 q=退出")
    try:
        while True:
            bgr = camera.get_frame()
            if bgr is None:
                continue
            disp = bgr.copy()
            cv2.putText(disp, f"samples={len(samples)}  [c]collect [s]solve [q]quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(WIN_NAME, disp)
            k = cv2.waitKey(1) & 0xFF
            ch = chr(k).lower()
            if ch == "q":
                break
            elif ch == "c":
                idx = collect_sample(arm, camera, samples, idx)
            elif ch == "s":
                solve_and_save(samples, camera.K, camera.dist)
    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        cv2.destroyAllWindows()


# ---------------- 模式: sequence ----------------
def load_sequence(path: str):
    """加载序列 JSON。每项形如:
        {"joints": [q0,...,q6]}        -> 用 move2joints
        {"pose": [[...4x4...]]}        -> 用 move2pose (4x4 平铺成 16 的 list 亦可)
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "sequence" in data:
        data = data["sequence"]
    return data


def run_sequence(arm: Arm, camera: Camera, path: str, settle_s: float = 0.8) -> None:
    seq = load_sequence(path)
    samples: List[calib.Sample] = []
    idx = 0
    print(f"序列模式: 共 {len(seq)} 个位姿, 每个到位后自动采集")
    for i, item in enumerate(seq):
        if "joints" in item:
            print(f"[{i+1}/{len(seq)}] move2joints ...")
            arm.move2joints(item["joints"])
        elif "pose" in item:
            T = np.array(item["pose"], dtype=np.float64).reshape(4, 4)
            print(f"[{i+1}/{len(seq)}] move2pose ...")
            arm.move2pose(T)
        else:
            print(f"[{i+1}] 跳过: 既无 joints 也无 pose"); continue
        time.sleep(settle_s)   # 等停稳 + 位姿刷新
        idx = collect_sample(arm, camera, samples, idx)
    solve_and_save(samples, camera.K, camera.dist)


# ---------------- 入口 ----------------
def main():
    ap = argparse.ArgumentParser(description="eye-in-hand 手眼标定")
    ap.add_argument("--mode", choices=["interactive", "sequence"], default="interactive")
    ap.add_argument("--sequence", default=None, help="sequence 模式的 JSON 文件路径")
    ap.add_argument("--robot-ip", default="172.16.0.8", help="Franka FCI IP")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== eye-in-hand 手眼标定 ===")
    arm = ArmImpl(robot_ip=args.robot_ip)
    camera = CameraImpl()
    print("K =\n", np.array2string(camera.K, precision=2))

    try:
        if args.mode == "sequence":
            if not args.sequence:
                print("sequence 模式需要 --sequence PATH"); return
            run_sequence(arm, camera, args.sequence)
        else:
            run_interactive(arm, camera)
    finally:
        camera.release()


if __name__ == "__main__":
    main()
