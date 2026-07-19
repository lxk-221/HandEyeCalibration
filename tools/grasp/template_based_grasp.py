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
    _pose([0.350, -0.250, -0.223], [0, 0, 90]),
    _pose([0.400, -0.250, -0.223], [0, 0, 90]),
]

# ---- 工件配置 (遍历 TEMPLATES_DIR 的 .ply; 厚度按文件名查表) ----
NUT_THICKNESS_M = {
    "hex_hole_40mm_45mm_35mm": 0.035,
    "hex_hole_30mm_35mm":      0.028,
    "hex_hole_24mm_27mm_M27":  0.022,
}


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

            # 按大->中->小 顺序抓取 (显式指定, 不依赖文件名排序)
            # 每个工件都重新 scan -> match -> grasp -> place
            # (抓取可能碰动其他工件, 必须每次抓取前重新扫描获取最新状态)
            templates = [
                "hex_hole_40mm_45mm_35mm.ply",   # 大
                "hex_hole_30mm_35mm.ply",         # 中
                "hex_hole_24mm_27mm_M27.ply",     # 小 (M27)
            ]
            for tpl in templates:
                key = os.path.splitext(tpl)[0]
                thickness = NUT_THICKNESS_M.get(key, 0.005)
                print(f"\n--- 工件 {tpl} (厚度 {thickness*1000:.0f}mm) ---")
                pointcloud = g.scan(SCAN_POSES)          # 每次抓取前重新扫描
                grasp_pose = g.get_grasp_pose(
                    pointcloud, os.path.join(TEMPLATES_DIR, tpl), thickness=thickness)
                print(f"  grasp_pose xyz={np.round(grasp_pose[:3,3],4).tolist()}")
                if not args.yes and input("确认抓取? [y/N]: ").strip().lower() not in ("y", "yes"):
                    print("  跳过")
                    continue
                g.approach(grasp_pose)
                g.grasp(grasp_pose)
                g.place()

            g.cool_down()
            print("=== 全部完成 ===")
    finally:
        cam.release()


if __name__ == "__main__":
    main()
