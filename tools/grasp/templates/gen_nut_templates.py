#!/usr/bin/env python3
"""生成正六边形螺母(外六边形 + 内圆孔)的模板点云 .ply。

几何: 正六边形外缘 (边长 = 外接圆半径 R) + 中心圆孔 (半径 r), 厚度薄层。
坐标原点在螺母中心, z=0 在上表面 (z 厚度方向 0~thickness)。
用法: python gen_nut_templates.py  (生成全部三套到当前目录)

⚠️ 相机俯视只能扫上表面, 故厚度统一 5mm 薄层;
   真实厚度(用于抓取时下降到中心)在主程序里各自记录。
"""
import numpy as np
import open3d as o3d

# 三套螺母参数 (单位 m):
#   edge_mm:  正六边形边长 (= 外接圆半径)
#   hole_mm:  内螺纹直径 (半径 = hole_mm/2)
#   name:     模板文件名
NUTS = [
    {"name": "hex_hole_40mm_45mm_35mm",  "edge_mm": 40, "hole_mm": 45},  # 大 (已有, 重新生成作对照)
    {"name": "hex_hole_30mm_35mm",       "edge_mm": 30, "hole_mm": 35},  # 中
    {"name": "hex_hole_24mm_27mm_M27",   "edge_mm": 24, "hole_mm": 27},  # 小 M27
]
TEMPLATE_THICKNESS_MM = 5.0      # 模板厚度 (薄层, 相机俯视只能扫上表面)
POINTS_PER_EDGE = 200           # 每条边采样点数 (外缘密度)
POINTS_PER_RING = 200           # 内孔圆采样点数
TOP_DENSITY = 1.0 / 0.0015      # 上表面采样密度 (点/m), ~1.5mm 间距
N_Z_LAYERS = 2                  # z 方向层数 (上表面 + 下表面, 薄层)


def hex_outer_points(R, n_per_edge):
    """正六边形外缘点 (顶点在角度 0,60,...,300度, 边长=R=外接圆半径)。
    返回 (N,2) xy 坐标。"""
    verts = [(R * np.cos(a), R * np.sin(a))
             for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    pts = []
    for i in range(6):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % 6]
        for t in np.linspace(0, 1, n_per_edge, endpoint=False):
            pts.append((x0 * (1 - t) + x1 * t, y0 * (1 - t) + y1 * t))
    return np.array(pts)


def generate_nut_pcd(edge_mm, hole_mm, thickness_mm):
    """生成单个螺母模板点云 (open3d), 含外缘/内孔/上表面填充。"""
    R = edge_mm / 1000.0
    r = hole_mm / 2000.0
    thick = thickness_mm / 1000.0
    xy_all = []

    # 1. 外缘六边形 (多层 z)
    outer = hex_outer_points(R, POINTS_PER_EDGE)
    # 2. 内孔圆 (多层 z)
    ring = np.array([(r * np.cos(a), r * np.sin(a))
                     for a in np.linspace(0, 2 * np.pi, POINTS_PER_RING, endpoint=False)])
    edge_pts = np.vstack([outer, ring])

    for z in np.linspace(0, thick, N_Z_LAYERS, endpoint=True):
        xy_all.append(np.hstack([edge_pts, np.full((len(edge_pts), 1), z)]))

    # 3. 上表面填充: 在六边形内、孔外的区域均匀撒点
    n_top = int(np.pi * (R ** 2 - r ** 2) * TOP_DENSITY)
    for _ in range(n_top * 3):
        # 极坐标采样: 半径在 [r, R] 内均匀, 角度随机
        rho = np.sqrt(np.random.uniform(r ** 2, R ** 2))
        theta = np.random.uniform(0, 2 * np.pi)
        x, y = rho * np.cos(theta), rho * np.sin(theta)
        # 检查是否在正六边形内 (六边形: |angle| 到最近 60° 边的距离)
        # 简化: 用六边形不等式 - 对正六边形顶点在0°时, 内点满足 max|投影| <= R*cos(30°)
        # 更稳: 检查点到6条边的距离
        if _point_in_hexagon(x, y, R):
            xy_all.append(np.array([[x, y, thick]]))

    pts = np.vstack(xy_all)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.paint_uniform_color([0.7, 0.7, 0.68])   # 灰色, 与原模板一致
    return pcd


def _point_in_hexagon(x, y, R):
    """判断 (x,y) 是否在边长 R 的正六边形内 (顶点在 0°,60°,...)。"""
    # 正六边形 (顶点在 0°) 的 6 条边方程: 法向角度 30°,90°,150°,...
    # 内点满足: 对每个 k, x*cos(a)+y*sin(a) <= R*cos(30°), a = 30+60*k 度
    R_in = R * np.cos(np.pi / 6)   # 内切圆半径 = R*cos(30°)
    for k in range(6):
        a = np.deg2rad(30 + 60 * k)
        if x * np.cos(a) + y * np.sin(a) > R_in + 1e-9:
            return False
    return True


def main():
    np.random.seed(0)   # 可复现
    for nut in NUTS:
        pcd = generate_nut_pcd(nut["edge_mm"], nut["hole_mm"], TEMPLATE_THICKNESS_MM)
        path = f"{nut['name']}.ply"
        o3d.io.write_point_cloud(path, pcd)
        pts = np.asarray(pcd.points)
        print(f"{path}: {len(pts)} 点, 边长={nut['edge_mm']}mm 孔径={nut['hole_mm']}mm "
              f"厚度={TEMPLATE_THICKNESS_MM}mm, bbox xyz="
              f"{np.round(pts.max(0)-pts.min(0),4).tolist()}")


if __name__ == "__main__":
    main()
