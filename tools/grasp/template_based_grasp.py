#!/usr/bin/env python3
"""
模板匹配抓取 tool —— 多视角扫描拼合点云 -> ICP 模板匹配 -> 经 LCM 发抓取指令。

进程边界:
    本 tool (toolbox 内, 纯净, 无 ROS)
      对 SCAN_POSES 每个扫描位姿: move_arm -> 取点云 + 实时 ee_pose
      -> 多帧点云转 base 系拼合 + voxel 下采样 + 分割
      -> ICP 模板匹配 -> 工件中心 (base 系)
      -> LcmCommander 发 arm_command / hand_command (阻塞等 execution_feedback)
    桥 (toolbox 外, ROS): 订阅 LCM -> lx_useful 控硬件 -> 回 feedback + 发 ee_pose

运行:
    python main.py grasp --workpiece doc/hex_hole.ply
"""
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .._hardware import get_camera_class
from . import pointcloud as pc
from .lcm_commander import LcmCommander


# 写死的手眼标定 T_cam2gripper (来自 tools/hand_eye 标定结果; 参考 demo.py 的 T_CAM2GRIPPER)。
# 后续换相机/机械臂需重新标定并改这里。平移单位米。
T_CAM2GRIPPER = np.array(
    [
        [0.0008, -0.9982, -0.0603, 0.071329],
        [-0.9995, 0.0011, -0.0317, 0.0092383],
        [0.0317, 0.0603, -0.9977, -0.0312081],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# ---- 扫描位姿 (4x4 T_ee2base, base 系): 多视角取点云拼合, 克服单帧遮挡 ----
# 第一个位姿沿用 xyz_bak client 的拍照位 (camera_graspnet_rpc_client.py:451-452, 已验证可达)。
# 多视角扫描需根据机器人工作空间 + 工件位置补充更多位姿。
def _scan_pose(xyz, rpy_deg):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rot.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = xyz
    return T

SCAN_POSES = [
    #_scan_pose([0.394, -0.292, -0.223], [-2.4, 2.4, 90.7]), # real example, dont delete
    _scan_pose([0.394, -0.292, -0.223], [0, 0, 90]),
    _scan_pose([0.350, -0.292, -0.223], [0, 0, 90]),
    # 可继续添加更多视角...
]
VOXEL_SIZE_M = 0.002          # 多帧拼合后体素降采样粒度

# ---- WARMUP/COOLDOWN 关节序列 (复用 xyz_bak client move2capture 验证过的 8 关节轨迹) ----
# 关节空间运动, 避开笛卡尔插值可能触发的奇异/碰撞。
# WARMUP: 零位 -> 扫描位附近 (正序); COOLDOWN: 扫描位 -> 零位 (倒序)。
_WARMUP_JOINTS = [
    [0,         0,        0,        0,        0, 0, 0],
    [0,        -np.pi/2,  0,        0,        0, 0, 0],
    [0,        -np.pi/2,  0,       -np.pi/2,  0, 0, 0],
    [0,        -np.pi/2,  0,       -np.pi/2,  0, 0,  np.pi/2],
    [np.pi/4,  -np.pi/4,  0,       -np.pi/2,  0, 0,  np.pi/2],
    [np.pi/4,  -np.pi/4, -np.pi/4, -np.pi/2,  0, 0,  np.pi/2],
    [0.740,    -0.556,   -0.774,   -1.525,   -0.880, 0.593, 1.747],
    [0.714,    -0.678,   -0.896,   -1.439,   -0.654, 0.535, 1.509],
]
WARMUP_JOINTS = _WARMUP_JOINTS                    # scan 前: 零位 -> 扫描位附近
COOLDOWN_JOINTS = list(reversed(_WARMUP_JOINTS))  # 抓取后: 扫描位 -> 零位

# ---- 抓取位姿偏置 (参考 xyz_bak client 的 execute_control_pose, base 系) ----
APPROACH_OFFSET_M = np.array([-0.07, 0.022, 0.23], dtype=np.float64)
APPROACH_RPY_RAD = np.array([-55.0, 0.0, 90.0]) * np.pi / 180.0
HAND_READY = [255, 10, 255, 255, 255, 255]      # 张开待命
HAND_GRASP = [0, 5, 70, 80, 80, 70]             # 抓取
ARM_SPEED, ARM_ACCEL = 0.8, 0.8
ARM_TIMEOUT, HAND_TIMEOUT = 60.0, 3.0


def scan_point_clouds(cmd, cam, scan_poses):
    """对 scan_poses 每个扫描位姿: move_arm 到位 -> 取点云 + 实时 ee_pose。
    返回 frames: list of (points_camera, colors_rgb, T_gripper2base)。
    多帧拼合在 base 系进行, 克服单帧遮挡, 物体点云更完整。"""
    frames = []
    for i, T in enumerate(scan_poses, 1):
        print(f"  scan [{i}/{len(scan_poses)}] move_arm -> xyz={np.round(T[:3,3],3).tolist()}")
        cmd.move_arm(T, speed=ARM_SPEED, accel=ARM_ACCEL, timeout=ARM_TIMEOUT)
        # 到位后取实时末端位姿 (拍照时刻的 T_gripper2base) + 点云
        T_gripper2base = cmd.get_ee_pose(timeout=3.0)
        points_cam, colors_cam = cam.get_point_cloud()
        print(f"    取到 {len(points_cam)} 点, ee xyz={np.round(T_gripper2base[:3,3],3).tolist()}")
        if len(points_cam) > 0:
            frames.append((points_cam, colors_cam, T_gripper2base))
    return frames


def run_joint_sequence(cmd, joints_list, label):
    """依次 move_joints 到关节序列的每个点 (阻塞)。用于 WARMUP/COOLDOWN。"""
    for i, joints in enumerate(joints_list, 1):
        print(f"  {label} [{i}/{len(joints_list)}] joints={np.round(joints,3).tolist()}")
        cmd.move_joints(joints, speed=ARM_SPEED, accel=ARM_ACCEL, timeout=ARM_TIMEOUT)


def main(argv=None):
    ap = argparse.ArgumentParser(description="模板匹配抓取 (多视角扫描 + ICP + LCM)")
    ap.add_argument("--workpiece", required=True, help="工件模板点云 .ply 路径 (ICP source)")
    ap.add_argument("--lcm-url", default="udpm://239.255.76.67:7667?ttl=0",
                    help="LCM 地址 (ttl=0 本机; 跨机用 ttl=1+)")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="跳过匹配后的人工确认, 直接执行 (自动化场景)")
    args = ap.parse_args(argv)

    CameraCls = get_camera_class()
    if CameraCls is None:
        raise RuntimeError("config.yaml 未配置 camera, 无法取点云")
    print(f"=== 模板匹配抓取 (camera={CameraCls.__name__}, {len(SCAN_POSES)} 个扫描位姿) ===")

    cam = CameraCls()
    try:
        with LcmCommander(lcm_url=args.lcm_url) as cmd:
            # 0. WARMUP: 关节空间从零位移动到扫描位附近 (避开奇异/碰撞)。
            print("=== WARMUP (零位 -> 扫描位附近) ===")
            run_joint_sequence(cmd, WARMUP_JOINTS, "warmup")

            # 1. 多视角扫描: 每个位姿取点云 + ee_pose。
            frames = scan_point_clouds(cmd, cam, SCAN_POSES)
            if not frames:
                raise RuntimeError("所有扫描位姿都没取到点云")

            # 2. 多帧拼合到 base 系 + voxel 下采样 + 分割 (去桌面)。
            pts_base, cols_base = pc.merge_frames_to_base(frames, T_CAM2GRIPPER)
            print(f"多帧拼合 (base 系): {len(pts_base)} 点")
            pts_base, cols_base = pc.voxel_downsample(pts_base, cols_base, VOXEL_SIZE_M)
            print(f"voxel 下采样 ({VOXEL_SIZE_M*1000}mm): {len(pts_base)} 点")
            pts_seg, cols_seg = pc.segment_workpiece_from_base(pts_base, cols_base)
            print(f"分割后: {len(pts_seg)} 点")
            if len(pts_seg) < 50:
                raise RuntimeError(f"分割后点太少 ({len(pts_seg)}), 无法做模板匹配")

            # 3. ICP 模板匹配 (场景点已在 base 系, 模板配准到 base 系)。
            match = pc.match_workpiece_point_cloud(
                pts_seg, args.workpiece, max_correspondence_distance=0.012, icp_iteration=80)
            center_base = match.aligned_points.mean(axis=0)
            print(f"工件中心 (base 系): {np.round(center_base, 4).tolist()}")

            # 4. 算 arm 目标位姿 (工件中心 + 固定偏置)。
            arm_xyz = center_base + APPROACH_OFFSET_M
            arm_T = np.eye(4, dtype=np.float64)
            arm_T[:3, :3] = Rot.from_euler("xyz", APPROACH_RPY_RAD).as_matrix()
            arm_T[:3, 3] = arm_xyz

            # 5. 人工确认: 点云可视化 + 打印匹配结果, 确认后再执行 (安全)。
            if not args.yes:
                pc.show_match(pts_seg, cols_seg, match.aligned_points)
            info = match.info
            print("\n========== 模板匹配结果 ==========")
            print(f"  ICP fitness = {info.get('fitness', '?'):.4f}   "
                  f"rmse = {info.get('rmse', '?'):.5f} m")
            print(f"  场景点 {info.get('segmented_points', '?')} <- 模板点 {info.get('model_points', '?')}")
            print(f"  工件中心 (base 系): {np.round(center_base, 4).tolist()} m")
            print(f"  arm 目标位姿 (base 系):")
            print(f"    xyz     = {np.round(arm_xyz, 4).tolist()} m")
            print(f"    rpy_deg = {np.round(np.degrees(APPROACH_RPY_RAD), 1).tolist()}")
            print("==================================")
            if not args.yes:
                ans = input("确认执行抓取? [y/N]: ").strip().lower()
                if ans not in ("y", "yes"):
                    print("已取消, 跳过抓取 (机械臂停在扫描位)。")
                    return

            # 6. 发抓取指令: 张手 -> 移臂 -> 抓取 (每步阻塞等桥 feedback)。
            print("=== 通过 LCM 发送运动指令 (阻塞等桥 feedback) ===")
            cmd.move_hand(HAND_READY, timeout=HAND_TIMEOUT)
            print("  [1/3] hand 张开 OK")
            cmd.move_arm(arm_T, speed=ARM_SPEED, accel=ARM_ACCEL, timeout=ARM_TIMEOUT)
            print("  [2/3] arm 到位 OK")
            cmd.move_hand(HAND_GRASP, timeout=HAND_TIMEOUT)
            print("  [3/3] hand 抓取 OK")

            # 7. COOLDOWN: 关节空间从扫描位回到零位 (WARMUP 倒序)。
            print("=== COOLDOWN (扫描位 -> 零位) ===")
            run_joint_sequence(cmd, COOLDOWN_JOINTS, "cooldown")
            print("=== 抓取流程完成 ===")
    finally:
        cam.release()


if __name__ == "__main__":
    main()
