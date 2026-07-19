#!/usr/bin/env python3
"""模板匹配抓取 tool (main 编排)。

具体抓取/扫描/匹配/放置逻辑封装在 grasp.py 的 GraspTemplateBased 类。
本文件只做: 构造 arm/hand/camera -> GraspTemplateBased -> 编排 (warm_up -> 遍历工件 scan+grasp+place -> cool_down)。

运行: python main.py grasp
"""
import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .._hardware import get_camera_class
from .grasp import GraspTemplateBased
from .lcm_commander import LcmCommander

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# 手眼标定 (写死; 换相机/臂需重标定后改)。平移单位米。
T_CAM2GRIPPER = np.array(
    [
        [0.0008, -0.9982, -0.0603, 0.071329],
        [-0.9995, 0.0011, -0.0317, 0.0092383],
        [0.0317, 0.0603, -0.9977, -0.0312081],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# ---- 扫描位姿 (4x4 T_ee2base, base 系) ----
def _pose(xyz, rpy_deg):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rot.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = xyz
    return T

SCAN_POSES = [
    # _pose([0.394, -0.292, -0.223], [-2.4, 2.4, 90.7]),  # real example, dont delete
    _pose([0.400, -0.300, -0.223], [0, 0, 90]),
    _pose([0.350, -0.300, -0.223], [0, 0, 90]),
]

# ---- ee 相对物体中心的位姿 (含厚度修正 + ee 偏置/姿态), 全部内化在此 4x4 ----
# 例: 上方斜抓, z = 0.23 + thickness/2 (抓厚度中心), rpy=(-55,0,90)
EE_OFFSET = np.array([-0.07, 0.022], dtype=np.float64)
EE_Z_BASE = 0.23
EE_RPY_DEG = np.array([-55.0, 0.0, 90.0])


def make_T_ee2object(thickness_m):
    """根据工件厚度构造 T_ee2object (ee 相对物体中心的位姿)。
    厚度内化: z 抬高 thickness/2 (抓厚度中心, 因 ICP 匹配的是上表面)。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rot.from_euler("xyz", EE_RPY_DEG, degrees=True).as_matrix()
    T[:3, 3] = [EE_OFFSET[0], EE_OFFSET[1], EE_Z_BASE + thickness_m / 2]
    return T


# ---- 工件配置 (遍历 TEMPLATES_DIR 的 .ply; 厚度按文件名查表) ----
NUT_THICKNESS_M = {
    "hex_hole_40mm_45mm_35mm": 0.035,
    "hex_hole_30mm_35mm":      0.028,
    "hex_hole_24mm_27mm_M27":  0.022,
}


# ---- 放置位 (base 系) ----
PLACE_RPY_DEG = np.array([-60.0, 0.0, 145.0])
PLACE_XYZ = np.array([0.262, -0.09, -0.22], dtype=np.float64)


def make_place_pose():
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rot.from_euler("xyz", PLACE_RPY_DEG, degrees=True).as_matrix()
    T[:3, 3] = PLACE_XYZ
    return T


def main(argv=None):
    ap = argparse.ArgumentParser(description="模板匹配抓取 (GraspTemplateBased)")
    ap.add_argument("--lcm-url", default="udpm://239.255.76.67:7667?ttl=0",
                    help="LCM 地址 (ttl=0 本机; 跨机用 ttl=1+)")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="跳过人工确认/可视化 (自动化)")
    args = ap.parse_args(argv)

    CameraCls = get_camera_class()
    if CameraCls is None:
        raise RuntimeError("config.yaml 未配置 camera")
    print(f"=== 模板匹配抓取 (camera={CameraCls.__name__}) ===")

    cam = CameraCls()
    try:
        with LcmCommander(lcm_url=args.lcm_url) as cmd:
            g = GraspTemplateBased(arm=cmd, hand=cmd, T_CAM2GRIPPER=T_CAM2GRIPPER, camera=cam)
            g.warm_up()
            pointcloud = g.scan(SCAN_POSES)

            # 遍历模板 (假设每个模板在场景中都能匹配到一个工件)
            templates = sorted(f for f in os.listdir(TEMPLATES_DIR) if f.endswith(".ply"))
            for tpl in templates:
                key = os.path.splitext(tpl)[0]
                thickness = NUT_THICKNESS_M.get(key, 0.005)
                print(f"\n--- 工件 {tpl} (厚度 {thickness*1000:.0f}mm) ---")
                grasp_pose = g.get_grasp_pose(
                    pointcloud, os.path.join(TEMPLATES_DIR, tpl), make_T_ee2object(thickness))
                print(f"  grasp_pose xyz={np.round(grasp_pose[:3,3],4).tolist()}")
                if not args.yes and input("确认抓取? [y/N]: ").strip().lower() not in ("y", "yes"):
                    print("  跳过")
                    continue
                g.approach(grasp_pose)
                g.grasp(grasp_pose)
                g.place(make_place_pose())

            g.cool_down()
            print("=== 全部完成 ===")
    finally:
        cam.release()


if __name__ == "__main__":
    main()
