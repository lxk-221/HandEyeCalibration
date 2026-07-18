#!/usr/bin/env python3
"""
模板匹配抓取 tool —— 从 RGB-D 出发做 ICP 模板匹配, 算出 arm/hand 目标位姿,
经 LCM 发布 (不直接控硬件)。

进程边界:
    本 tool (toolbox 内, 纯净, 无 ROS)
      OrbbecCamera.get_point_cloud() 取相机系点云
      -> pointcloud 模块: 距离过滤 + RANSAC 平面分割 + ICP 模板匹配
      -> 工件中心 (相机系) 经 T_cam2gripper + T_gripper2base 变到 base 系
      -> 算 arm 目标位姿 (工件中心 + 固定偏置) + hand 位姿
      -> LcmCommander 依次发 arm_command / hand_command (阻塞等 execution_feedback)
    另一个程序 (toolbox 外, 可依赖 ROS): 订阅 LCM -> lx_useful 控硬件 -> 回 feedback

运行:
    python main.py grasp --workpiece doc/hex_hole.ply --t-gripper2base T.npy
"""
import argparse
import os
import time

import numpy as np

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

# 抓取位姿偏置 (参考 xyz_bak client 的 execute_control_pose, base 系):
#   arm 目标 = 工件中心 + 这组偏置; rpy 固定 (从上方斜抓)。
APPROACH_OFFSET_M = np.array([-0.07, 0.022, 0.23], dtype=np.float64)
APPROACH_RPY_RAD = np.array([-55.0, 0.0, 90.0]) * np.pi / 180.0
HAND_DOF = 6
HAND_READY = [255, 10, 255, 255, 255, 255]      # 张开待命
HAND_GRASP = [0, 5, 70, 80, 80, 70]             # 抓取


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="模板匹配抓取 (ICP) + LCM 发布")
    ap.add_argument("--workpiece", required=True,
                    help="工件模板点云 .ply 路径 (ICP source)")
    ap.add_argument("--t-gripper2base", default=None,
                    help="调试覆盖: 手动指定 T_gripper2base (.npy 或 16 数逗号串)。"
                         "缺省从桥实时发布的 ee_pose 取 (推荐)")
    ap.add_argument("--icp-distance", type=float, default=0.012,
                    help="ICP 最大对应距离 (m)")
    ap.add_argument("--icp-iter", type=int, default=80, help="ICP 最大迭代")
    ap.add_argument("--depth-min", type=float, default=0.3, help="距离过滤最小 (m)")
    ap.add_argument("--depth-max", type=float, default=1.2, help="距离过滤最大 (m)")
    ap.add_argument("--plane-thresh", type=float, default=0.004, help="RANSAC 平面阈值 (m)")
    ap.add_argument("--lcm-url", default="udpm://224.0.0.1?ttl=0",
                    help="LCM 地址 (ttl=0 仅本机; 跨机用 ttl=1+ 多播)")
    ap.add_argument("--arm-timeout", type=float, default=60.0,
                    help="单条 arm 指令等 feedback 超时 (s)")
    ap.add_argument("--hand-timeout", type=float, default=3.0,
                    help="单条 hand 指令等 feedback 超时 (s)")
    ap.add_argument("--no-send", action="store_true", help="只算不发 LCM (调试)")
    return ap.parse_args(argv)


def _load_T_gripper2base_override(arg):
    """--t-gripper2base 调试覆盖: .npy 或 16 数逗号串 -> 4x4。None 返回 None (用实时 ee_pose)。"""
    if arg is None:
        return None
    if os.path.exists(arg):
        return np.asarray(np.load(arg), dtype=np.float64).reshape(4, 4)
    vals = [float(v) for v in arg.split(",")]
    return np.array(vals, dtype=np.float64).reshape(4, 4)


def run_pipeline(args):
    """主流程: 连桥 -> 取实时位姿+点云 -> ICP -> 发 LCM 抓取指令 (全程阻塞)。"""
    override = _load_T_gripper2base_override(args.t_gripper2base)

    CameraCls = get_camera_class()
    if CameraCls is None:
        raise RuntimeError("config.yaml 未配置 camera, 无法取点云")
    print(f"=== 模板匹配抓取 (camera={CameraCls.__name__}) ===")

    # 用 commander 贯穿全程: 拍照后取实时 ee_pose, 末尾发 arm/hand 指令。
    # --no-send 调试模式: 不连桥 (不需要 commander, 但需 override 或无法算 base 系)。
    cam = CameraCls()
    try:
        with (LcmCommander(lcm_url=args.lcm_url) if not args.no_send else _NullCtx()) as cmd:
            # 1. 取点云 (相机系)。
            points_cam, colors_cam = cam.get_point_cloud()
            print(f"取到点云: {len(points_cam)} 点 (相机系, m)")
            if len(points_cam) == 0:
                raise RuntimeError("点云为空")

            # 2. 拍照时刻末端位姿 T_gripper2base:
            #    默认从桥实时发布的 ee_pose 取 (拍照瞬间最新值); --t-gripper2base 可覆盖。
            if override is not None:
                T_gripper2base = override
                print(f"T_gripper2base (手动覆盖): xyz={np.round(T_gripper2base[:3,3],4).tolist()}")
            elif args.no_send:
                raise RuntimeError("--no-send 模式无桥, 取不到 ee_pose; 需配合 --t-gripper2base 覆盖")
            else:
                T_gripper2base = cmd.get_ee_pose(timeout=3.0)
                print(f"T_gripper2base (桥实时): xyz={np.round(T_gripper2base[:3,3],4).tolist()}")
            T_camera2base = pc.camera_to_base_transform(T_gripper2base, T_CAM2GRIPPER)

            # 3. 预处理: 距离过滤 + RANSAC 去平面。
            keep = pc.distance_filter_point_cloud(points_cam, args.depth_min, args.depth_max, mode="z")
            pts = points_cam[keep]
            print(f"距离过滤后: {len(pts)} 点")
            pts, _cols, _keep2, _pm, _inlier, _greater = \
                pc.ransac_remove_plane_and_greater_z(pts, pts, args.plane_thresh)
            print(f"RANSAC 去平面后: {len(pts)} 点")
            if len(pts) < 50:
                raise RuntimeError(f"分割后点太少 ({len(pts)}), 无法做模板匹配")

            # 4. ICP 模板匹配 -> 工件中心 (相机系) -> base 系。
            match = pc.match_workpiece_point_cloud(
                pts, args.workpiece, args.icp_distance, args.icp_iter)
            center_cam = match.aligned_points.mean(axis=0)
            center_base = pc.transform_points_to_base(center_cam[None, :], T_camera2base)[0]
            print(f"工件中心: 相机系 {np.round(center_cam,4).tolist()} | base 系 {np.round(center_base,4).tolist()}")

            # 5. 算 arm 目标位姿 (工件中心 + 固定偏置)。
            arm_xyz = center_base + APPROACH_OFFSET_M
            from scipy.spatial.transform import Rotation as Rot
            arm_T = np.eye(4, dtype=np.float64)
            arm_T[:3, :3] = Rot.from_euler("xyz", APPROACH_RPY_RAD).as_matrix()
            arm_T[:3, 3] = arm_xyz
            print(f"arm 目标位姿 (base 系): xyz={np.round(arm_xyz,4).tolist()} "
                  f"rpy_deg={np.round(np.degrees(APPROACH_RPY_RAD),1).tolist()}")

            # 6. 发指令 (仅非 --no-send): 张手 -> 移臂 -> 抓取, 每步阻塞等桥 feedback。
            if args.no_send:
                print("[--no-send] 不发 LCM")
                return
            print("=== 通过 LCM 发送运动指令 (阻塞等桥 feedback) ===")
            cmd.move_hand(HAND_READY, timeout=args.hand_timeout)
            print("  [1/3] hand 张开 OK")
            cmd.move_arm(arm_T, speed=0.8, accel=0.8, timeout=args.arm_timeout)
            print("  [2/3] arm 到位 OK")
            cmd.move_hand(HAND_GRASP, timeout=args.hand_timeout)
            print("  [3/3] hand 抓取 OK")
            print("=== 抓取流程完成 ===")
    finally:
        cam.release()


class _NullCtx:
    """--no-send 模式的占位 context manager (无桥)。"""
    def __enter__(self): return None
    def __exit__(self, *exc): return False


def main(argv=None):
    args = parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
