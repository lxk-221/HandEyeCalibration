"""可复用的抓取/放置动作 (基于模板匹配)。

把 grasp_by_template 和 place 从主流程抽象出来, 便于在多工件依次抓放等场景复用。
两者都通过 LcmCommander 发指令 (阻塞等桥 feedback), 不直接控硬件。

grasp_by_template: 点云(已 merge+降采样) -> ICP匹配 -> 算位姿 -> [可视化确认] -> 抓取运动
place:            抓取位姿 -> 抬起 -> 放置位 -> 松手 -> 抬回
"""
import os
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import pointcloud as pc


# 默认手位 (与主程序常量一致, 抽象动作的默认参数)
HAND_READY = [255, 10, 255, 255, 255, 255]
HAND_GRASP = [0, 5, 70, 80, 80, 70]
HAND_RELEASE = [255, 5, 255, 255, 255, 255]


def grasp_by_template(cmd, cam, scene_points, scene_colors, workpiece_path,
                      T_cam2gripper, ee_offset_m, ee_rpy_deg,
                      voxel_size_m=0.002, real_thickness_m=0.005,
                      confirm=True, visualize=True,
                      arm_speed=0.8, arm_accel=0.8, arm_timeout=60.0, hand_timeout=3.0,
                      icp_distance=0.012, icp_iter=80):
    """模板匹配抓取: 点云 -> ICP -> 算位姿 -> [可视化确认] -> 张手/移臂/抓取。

    输入:
      cmd: LcmCommander (已连桥)
      cam: Camera (取点云用, 若 scene_points 已给则不直接用; 此处保留接口对称)
      scene_points/scene_colors: 已 merge+降采样的场景点云 (base 系, 米)
      workpiece_path: 工件模板 .ply
      T_cam2gripper: 手眼标定 4x4
      ee_offset_m: ee 相对【模板中心(已按厚度调整)】的位置偏移 (base 系 xyz, 米)
      ee_rpy_deg: ee 绝对姿态 rpy (xyz 序, 度)
      real_thickness_m: 工件真实厚度 (匹配后下降到厚度中心; 默认 5mm = 模板薄层)
      confirm/visualize: 是否人工确认/可视化 (False 时直接执行)
    返回: 抓取位姿 arm_T (4x4 base 系), 供 place 使用。
    异常: ICP/确认取消/桥失败 -> raise, 主控退出 (安全)。"""
    # 1. 分割 (去桌面) + ICP 模板匹配。
    pts_seg, cols_seg = pc.segment_workpiece_from_base(scene_points, scene_colors)
    if len(pts_seg) < 50:
        raise RuntimeError(f"分割后点太少 ({len(pts_seg)}), 无法做模板匹配")
    match = pc.match_workpiece_point_cloud(
        pts_seg, workpiece_path, max_correspondence_distance=icp_distance, icp_iteration=icp_iter)
    center_base = match.aligned_points.mean(axis=0)

    # 2. 按真实厚度把模板中心(上表面薄层)修正到工件厚度中心。
    center_real = center_base.copy()
    center_real[2] -= max(0.0, real_thickness_m / 2 - 0.0025)   # 模板薄层半厚 2.5mm

    # 3. 算 arm 目标位姿 = 工件厚度中心 + ee_offset, 姿态用绝对 rpy。
    arm_T = np.eye(4, dtype=np.float64)
    arm_T[:3, :3] = Rot.from_euler("xyz", ee_rpy_deg, degrees=True).as_matrix()
    arm_T[:3, 3] = center_real + np.asarray(ee_offset_m, dtype=np.float64)

    # 4. 可视化 + 人工确认 (安全闸门)。
    if visualize:
        pc.show_match(pts_seg, cols_seg, match.aligned_points, center=center_base)
    info = match.info
    print("\n========== 模板匹配结果 ==========")
    print(f"  模板: {os.path.basename(workpiece_path)}  "
          f"真实厚度 {real_thickness_m*1000:.0f}mm")
    print(f"  ICP fitness = {info.get('fitness', 0):.4f}   "
          f"rmse = {info.get('rmse', 0):.5f} m")
    print(f"  工件中心 (base 系): {np.round(center_base, 4).tolist()} m")
    print(f"  arm 目标 (base 系): xyz={np.round(arm_T[:3,3],4).tolist()} "
          f"rpy_deg={np.round(ee_rpy_deg,1).tolist()}")
    print("==================================")
    if confirm:
        ans = input("确认执行抓取? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            raise RuntimeError("用户取消抓取")

    # 5. 抓取运动: 张手 -> 移臂到位 -> 抓取 (每步阻塞等桥 feedback)。
    print("=== grasp_by_template: 张手 -> 移臂 -> 抓取 ===")
    cmd.move_hand(HAND_READY, timeout=hand_timeout)
    print("  [1/3] hand 张开 OK")
    cmd.move_arm(arm_T, speed=arm_speed, accel=arm_accel, timeout=arm_timeout)
    print("  [2/3] arm 到位 OK")
    cmd.move_hand(HAND_GRASP, timeout=hand_timeout)
    print("  [3/3] hand 抓取 OK")
    return arm_T


def place(cmd, grasp_T,
          place_above_xyz, place_above_rpy_deg,
          descend_m=0.14, raise_m=0.15,
          arm_speed=0.8, arm_accel=0.8, arm_timeout=60.0, hand_timeout=3.0):
    """放置: 抓取后抬起 -> 移到放置位上方 -> 下降 -> 松手 -> 抬回上方。
    抓取位姿用于确定抬起后的 z (从 grasp_T 的 z + raise_m)。

    输入:
      grasp_T: 抓取位姿 4x4 (base 系), 来自 grasp_by_template
      place_above_xyz: 放置位上方 xy + 占位 z (base 系, 米); 实际 z 取 raise 后高度
      place_above_rpy_deg: 放置姿态 rpy (xyz, 度)
      descend_m: 放置时从上方下降高度 (米)
      raise_m: 抓取后抬起 / 松手后抬回的高度 (米)
    """
    lift_z = grasp_T[2, 3] + raise_m
    place_above_T = np.eye(4, dtype=np.float64)
    place_above_T[:3, :3] = Rot.from_euler("xyz", place_above_rpy_deg, degrees=True).as_matrix()
    place_above_T[:3, 3] = [place_above_xyz[0], place_above_xyz[1], lift_z]
    place_T = place_above_T.copy()
    place_T[2, 3] -= descend_m

    print("=== place: 抬起 -> 放置位 -> 松手 -> 抬回 ===")
    # 1. 抓取后抬起 (保持抓取 rpy, 只升 z)
    lift_T = grasp_T.copy()
    lift_T[2, 3] = lift_z
    cmd.move_arm(lift_T, speed=arm_speed, accel=arm_accel, timeout=arm_timeout)
    print("  [1/5] 抬起 OK")
    # 2. 移到放置位上方
    cmd.move_arm(place_above_T, speed=arm_speed, accel=arm_accel, timeout=arm_timeout)
    print("  [2/5] 移到放置位上方 OK")
    # 3. 下降到放置点
    cmd.move_arm(place_T, speed=arm_speed, accel=arm_accel, timeout=arm_timeout)
    print("  [3/5] 下降到放置点 OK")
    # 4. 松手
    cmd.move_hand(HAND_RELEASE, timeout=hand_timeout)
    print("  [4/5] hand 松手 OK")
    # 5. 抬回放置位上方
    cmd.move_arm(place_above_T, speed=arm_speed, accel=arm_accel, timeout=arm_timeout)
    print("  [5/5] 抬回 OK")
