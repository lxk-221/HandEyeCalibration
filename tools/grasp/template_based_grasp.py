#!/usr/bin/env python3
"""
模板匹配抓取 tool —— 多视角扫描拼合点云 -> ICP 模板匹配 -> 经 LCM 发抓取/放置指令。

流程 (main): WARMUP -> 对 WORKPIECES 每个工件 [扫描+拼合 -> grasp_by_template -> place] -> COOLDOWN
可复用动作见 actions.py:
  grasp_by_template: 点云 -> ICP匹配 -> 算位姿 -> [可视化确认] -> 张手/移臂/抓取
  place:             抓取位姿 -> 抬起 -> 放置位 -> 松手 -> 抬回

进程边界:
    本 tool (toolbox 内, 纯净, 无 ROS) 经 LCM 发指令
    桥 (toolbox 外, ROS): 订阅 LCM -> lx_useful 控硬件 -> 回 feedback + 发 ee_pose

运行:
    python main.py grasp
"""
import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .._hardware import get_camera_class
from . import pointcloud as pc
from . import actions
from .lcm_commander import LcmCommander

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


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
    _scan_pose([0.400, -0.300, -0.223], [0, 0, 90]),
    _scan_pose([0.350, -0.300, -0.223], [0, 0, 90]),
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

# ---- 抓取/手位/速度常量 ----
HAND_READY = [255, 10, 255, 255, 255, 255]      # 张开待命
HAND_GRASP = [0, 5, 70, 80, 80, 70]             # 抓取
HAND_RELEASE = [255, 5, 255, 255, 255, 255]     # 松手放置
ARM_SPEED, ARM_ACCEL = 0.8, 0.8
ARM_TIMEOUT, HAND_TIMEOUT = 60.0, 3.0

# ---- 工件真实厚度 (模板只有 5mm 薄层, 但真实厚度用于匹配后下降到工件中心抓取) ----
NUT_THICKNESS_M = {
    "hex_hole_40mm_45mm_35mm": 0.035,   # 大螺母, 真实厚度 35mm
    "hex_hole_30mm_35mm":      0.028,   # 中螺母, 真实厚度 28mm
    "hex_hole_24mm_27mm_M27":  0.022,   # 小螺母 M27, 真实厚度 22mm
}

# ---- 放置参数 (place() 用) ----
PLACE_ABOVE_XYZ = np.array([0.262, -0.09, 0.0], dtype=np.float64)   # 放置位 xy + 占位 z
PLACE_ABOVE_RPY_DEG = np.array([-60.0, 0.0, 145.0])

# ---- 工件配置 (依次抓放; 每件: 模板/真实厚度/ee偏置/ee姿态/放置位) ----
# ee_offset_m: ee 相对【模板厚度中心】的 base 系偏移 (米)
# ee_rpy_deg:  ee 绝对姿态 (xyz 序, 度)
# place_above_xyz/rpy_deg/descend_m/raise_m: 可选, 缺省用上面的 PLACE_ABOVE_*
# scan_poses: 可选, 缺省用 SCAN_POSES (工件都在相机视野内时共用)
APPROACH_EE_OFFSET_M = np.array([-0.07, 0.022, 0.23], dtype=np.float64)
APPROACH_EE_RPY_DEG = np.array([-55.0, 0.0, 90.0])
WORKPIECES = [
    {
        "name": "大螺母 (40mm)",
        "template": "hex_hole_40mm_45mm_35mm.ply",
        "thickness_m": NUT_THICKNESS_M["hex_hole_40mm_45mm_35mm"],
        "ee_offset_m": APPROACH_EE_OFFSET_M,
        "ee_rpy_deg": APPROACH_EE_RPY_DEG,
    },
    {
        "name": "中螺母 (30mm)",
        "template": "hex_hole_30mm_35mm.ply",
        "thickness_m": NUT_THICKNESS_M["hex_hole_30mm_35mm"],
        "ee_offset_m": APPROACH_EE_OFFSET_M,
        "ee_rpy_deg": APPROACH_EE_RPY_DEG,
    },
    {
        "name": "小螺母 M27 (24mm)",
        "template": "hex_hole_24mm_27mm_M27.ply",
        "thickness_m": NUT_THICKNESS_M["hex_hole_24mm_27mm_M27"],
        "ee_offset_m": APPROACH_EE_OFFSET_M,
        "ee_rpy_deg": APPROACH_EE_RPY_DEG,
    },
]


def scan_point_clouds(cmd, cam, scan_poses):
    """对 scan_poses 每个扫描位姿: move_arm 到位 -> 取点云 + 实时 ee_pose。
    返回 frames: list of (points_camera, colors_rgb, T_gripper2base)。"""
    frames = []
    for i, T in enumerate(scan_poses, 1):
        print(f"  scan [{i}/{len(scan_poses)}] move_arm -> xyz={np.round(T[:3,3],3).tolist()}")
        cmd.move_arm(T, speed=ARM_SPEED, accel=ARM_ACCEL, timeout=ARM_TIMEOUT)
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


def scan_and_merge(cmd, cam, scan_poses, voxel_size_m, visualize=False):
    """扫描多帧 -> 拼合到 base 系 -> voxel 下采样。返回 (points_base, colors_base)。
    可视化每步 (调参用)。"""
    frames = scan_point_clouds(cmd, cam, scan_poses)
    if not frames:
        raise RuntimeError("所有扫描位姿都没取到点云")
    pts, cols = pc.merge_frames_to_base(frames, T_CAM2GRIPPER)
    print(f"多帧拼合 (base 系): {len(pts)} 点")
    if visualize:
        pc.show_pointcloud(pts, cols, title="[1] 拼合后 (base系)")
    pts, cols = pc.voxel_downsample(pts, cols, voxel_size_m)
    print(f"voxel 下采样 ({voxel_size_m*1000}mm): {len(pts)} 点")
    if visualize:
        pc.show_pointcloud(pts, cols, title="[2] voxel 下采样后")
    return pts, cols


def main(argv=None):
    ap = argparse.ArgumentParser(description="模板匹配抓取 (多视角扫描 + ICP + LCM)")
    ap.add_argument("--lcm-url", default="udpm://239.255.76.67:7667?ttl=0",
                    help="LCM 地址 (ttl=0 本机; 跨机用 ttl=1+)")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="跳过匹配后的人工确认/可视化, 直接执行 (自动化场景)")
    args = ap.parse_args(argv)

    CameraCls = get_camera_class()
    if CameraCls is None:
        raise RuntimeError("config.yaml 未配置 camera, 无法取点云")
    print(f"=== 模板匹配抓取 (camera={CameraCls.__name__}, {len(WORKPIECES)} 个工件) ===")

    cam = CameraCls()
    try:
        with LcmCommander(lcm_url=args.lcm_url) as cmd:
            # 0. WARMUP: 关节空间从零位移动到扫描位附近 (一次性)。
            print("=== WARMUP (零位 -> 扫描位附近) ===")
            run_joint_sequence(cmd, WARMUP_JOINTS, "warmup")

            # 1. 依次抓取 + 放置每个工件。
            for idx, wp in enumerate(WORKPIECES, 1):
                print(f"\n========== 工件 [{idx}/{len(WORKPIECES)}] {wp['name']} ==========")
                # 1a. 扫描 + 拼合 (用该工件的 scan_poses, 默认共用 SCAN_POSES)。
                pts, cols = scan_and_merge(
                    cmd, cam, wp.get("scan_poses", SCAN_POSES), VOXEL_SIZE_M,
                    visualize=(not args.yes))
                # 1b. 模板匹配抓取 (含 ICP + 可视化确认 + 运动)。
                grasp_T = actions.grasp_by_template(
                    cmd, cam, pts, cols,
                    workpiece_path=os.path.join(TEMPLATES_DIR, wp["template"]),
                    T_cam2gripper=T_CAM2GRIPPER,
                    ee_offset_m=wp["ee_offset_m"],
                    ee_rpy_deg=wp["ee_rpy_deg"],
                    real_thickness_m=wp["thickness_m"],
                    confirm=(not args.yes), visualize=(not args.yes),
                    arm_speed=ARM_SPEED, arm_accel=ARM_ACCEL,
                    arm_timeout=ARM_TIMEOUT, hand_timeout=HAND_TIMEOUT)
                # 1c. 放置 (用该工件的放置位, 默认共用 PLACE_ABOVE_XYZ)。
                place_above = wp.get("place_above_xyz", PLACE_ABOVE_XYZ)
                place_rpy = wp.get("place_above_rpy_deg", PLACE_ABOVE_RPY_DEG)
                actions.place(
                    cmd, grasp_T, place_above, place_rpy,
                    descend_m=wp.get("descend_m", 0.14), raise_m=wp.get("raise_m", 0.15),
                    arm_speed=ARM_SPEED, arm_accel=ARM_ACCEL,
                    arm_timeout=ARM_TIMEOUT, hand_timeout=HAND_TIMEOUT)

            # 2. COOLDOWN: 回零位 (一次性)。
            print("\n=== COOLDOWN (-> 零位) ===")
            run_joint_sequence(cmd, COOLDOWN_JOINTS, "cooldown")
            print("=== 全部工件抓放完成 ===")
    finally:
        cam.release()


if __name__ == "__main__":
    main()
