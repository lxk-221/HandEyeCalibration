"""
点云处理核心模块 (5 个单一职责函数 + 坐标变换 + 可视化)。

坐标系: base 系 (米)。scan 后的点云已转到 base 系, 后续处理都在 base 系。
依赖: open3d + numpy。无 graspnet / 无 cv2 (除可视化外不需要)。

5 个核心职责 (每个是原子函数, 便于在调用方分步可视化):
  1. merge(frames, T_cam2gripper) -> base 系拼合点云
  2. range_filter(points, x/y/z min/max) -> 范围过滤 (缺省 inf = 不过滤)
  3. ransac_filter_plane(points, ...) -> RANSAC 去平面 (可选去/留平面之上的点)
  4. voxel_downsample(points, voxel_size) -> 体素降采样
  5. icp_match(scene, template) -> ICP 模板配准

可视化: show_pointcloud / show_match (shift+左键拾取点坐标)。
"""
import glob
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


# ============================================================================
# 1. 多帧拼合 (相机系 -> base 系)
# ============================================================================
def merge(frames, T_cam2gripper):
    """把多帧 (相机系点云 + 拍照时刻 T_gripper2base) 全部转到 base 系并拼接。
    frames: list of (points_camera Nx3 米, colors Nx3 uint8, T_gripper2base 4x4)。
    T_cam2gripper: 手眼标定 4x4 (平移米)。T_camera2base = T_gripper2base @ T_cam2gripper。
    返回 (points_base Nx3 float64, colors Nx3 uint8)。"""
    T_c2g = np.asarray(T_cam2gripper, dtype=np.float64).reshape(4, 4)
    all_pts, all_cols = [], []
    for points_cam, colors, T_g2b in frames:
        T_c2b = np.asarray(T_g2b, dtype=np.float64).reshape(4, 4) @ T_c2g
        p = np.asarray(points_cam, dtype=np.float64)
        pts_base = (T_c2b[:3, :3] @ p.T + T_c2b[:3, 3:4]).T
        all_pts.append(pts_base)
        all_cols.append(np.asarray(colors, dtype=np.uint8))
    if not all_pts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(all_pts, 0), np.concatenate(all_cols, 0)


# ============================================================================
# 2. 范围过滤 (6 轴 min/max, None = 不限)
# ============================================================================
def range_filter(points, colors=None,
                 x_min=None, x_max=None,
                 y_min=None, y_max=None,
                 z_min=None, z_max=None):
    """保留 [x_min,x_max]×[y_min,y_max]×[z_min,z_max] 内的点 (None = 该轴不过滤)。
    返回 (points_keep, colors_keep)。"""
    pts = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(pts), dtype=bool)
    for ax, lo, hi in [(0, x_min, x_max), (1, y_min, y_max), (2, z_min, z_max)]:
        if lo is not None:
            keep &= pts[:, ax] >= lo
        if hi is not None:
            keep &= pts[:, ax] <= hi
    out_pts = pts[keep]
    out_cols = np.asarray(colors)[keep] if colors is not None and len(colors) == len(pts) else None
    return out_pts, out_cols


# ============================================================================
# 3. RANSAC 平面分割 (可选去平面 + 平面之上的点)
# ============================================================================
def ransac_filter_plane(points, colors=None,
                        distance_thresh=0.01, ransac_n=3, ransac_iter=2000,
                        min_inlier_ratio=0.05, remove_below=True, max_above_m=0.04):
    """RANSAC 拟合一个平面 (通常是桌面), 去掉平面内点;
    remove_below=True 时还去掉平面【下方】的点;
    max_above_m>0 时还去掉平面之上【过高】的点 (只保留紧贴桌面的物体, 如螺母上表面)。
    -> 最终保留平面以上、max_above_m 以内的点 (桌面上的薄物体)。
    inlier_ratio < min_inlier_ratio 时认为没找到有效平面, 不过滤 (原样返回)。
    返回 (points_keep, colors_keep, plane_model)。plane_model=[a,b,c,d], ax+by+cz+d=0。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < ransac_n:
        return pts, (np.asarray(colors) if colors is not None else None), None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_thresh, ransac_n=ransac_n, num_iterations=ransac_iter)
    inliers = np.asarray(inliers, dtype=np.int64)
    if len(inliers) / float(len(pts)) < min_inlier_ratio:
        return pts, (np.asarray(colors) if colors is not None else None), plane_model

    plane_inlier_mask = np.zeros(len(pts), dtype=bool)
    plane_inlier_mask[inliers] = True
    a, b, c, d = [float(v) for v in plane_model]
    if abs(c) > 1e-9:
        z_plane = -(a * pts[:, 0] + b * pts[:, 1] + d) / c
        height = pts[:, 2] - z_plane   # 各点相对平面的高度 (>0 在上方)
    else:
        height = np.zeros(len(pts))
    # below: 平面及下方; above_max: 平面之上过高 (超出 max_above_m)
    below_mask = (height <= 0) if remove_below else np.zeros(len(pts), dtype=bool)
    above_max_mask = (height > max_above_m) if max_above_m and max_above_m > 0 else np.zeros(len(pts), dtype=bool)
    keep = ~(plane_inlier_mask | below_mask | above_max_mask)
    out_cols = np.asarray(colors)[keep] if colors is not None and len(colors) == len(pts) else None
    print(f"  RANSAC 平面过滤: 去桌面 {plane_inlier_mask.sum()} 点, "
          f"去下方 {below_mask.sum()} 点, 去平面之上>{max_above_m*1000:.0f}mm 高 {above_max_mask.sum()} 点, "
          f"保留 {keep.sum()}/{len(pts)}")
    return pts[keep], out_cols, plane_model


# ============================================================================
# 4. 体素降采样
# ============================================================================
def voxel_downsample(points, colors=None, voxel_size=0.002):
    """体素降采样 (open3d)。voxel_size 米。返回 (points_down, colors_down)。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0 or voxel_size <= 0:
        return pts, (np.asarray(colors) if colors is not None else None)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None and len(colors) == len(pts):
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(colors, dtype=np.float64) / 255.0, 0.0, 1.0))
    pcd = pcd.voxel_down_sample(voxel_size)
    out_pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64)
    if len(cols) == len(out_pts):
        out_cols = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
    else:
        out_cols = np.tile(np.array([[180, 180, 173]], dtype=np.uint8), (len(out_pts), 1))
    return out_pts, out_cols


# ============================================================================
# 5. 欧式聚类 (DBSCAN) — 把空间分离的多个物体分成独立簇
# ============================================================================
def cluster(points, colors=None, eps=0.020, min_points=10,
            max_z_spread=0.030):
    """DBSCAN 欧式聚类 -> 多个物体簇。
    eps: 邻域半径 (米), 同一簇内点间距 < eps。
    min_points: 核心点邻域最少点数, 过滤小噪声簇。
    max_z_spread: 丢弃 z 跨度过大的簇 (螺母上表面 ~5mm 薄层, 过大的簇是噪声/桌沿混入)。
    返回 list of (points, colors), 每个是一个独立簇, 按点数降序。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return []
    cols = np.asarray(colors) if colors is not None and len(colors) == len(pts) else None
    labels = np.array(pcd_dbscan_labels(pts, eps, min_points))
    clusters = []
    for label in sorted(set(labels) - {-1}):   # -1 = 噪声, 丢弃
        mask = labels == label
        c_pts = pts[mask]
        if len(c_pts) < min_points:
            continue
        z_spread = c_pts[:, 2].max() - c_pts[:, 2].min()
        if z_spread > max_z_spread:   # z 跨度过大, 不是扁平螺母上表面
            continue
        c_cols = cols[mask] if cols is not None else None
        clusters.append((c_pts, c_cols))
    clusters.sort(key=lambda c: len(c[0]), reverse=True)   # 点数降序
    print(f"  聚类: eps={eps*1000:.0f}mm min_points={min_points} -> {len(clusters)} 簇 "
          f"(各簇点数: {[len(c[0]) for c in clusters]})")
    return clusters


def pcd_dbscan_labels(points, eps, min_points):
    """封装 open3d DBSCAN, 返回 labels 数组 (-1=噪声)。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    labels = pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    return list(labels)


# ============================================================================
# 6. ICP 模板匹配
# ============================================================================
@dataclass
class MatchResult:
    aligned_points: np.ndarray       # (N,3) 模板配准后 (与 scene 同坐标系), 米
    aligned_colors: np.ndarray       # (N,3) float 0..1
    transform: np.ndarray            # 4x4 source(template)->target(scene)
    info: dict = field(default_factory=dict)


def _downsample_and_fpfh(pcd, voxel_size):
    """降采样 + 估法向 + 算 FPFH 特征。返回 (downsampled_pcd, fpfh_feature)。
    FPFH 需要法向; 降采样让特征计算更快更稳。"""
    pcd_ds = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2.0
    pcd_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_ds, fpfh


def icp_match(scene_points, template_path,
              max_correspondence_distance=0.012, icp_iteration=80,
              voxel_feature=None, ransac_n=4000000):
    """FPFH 全局配准 + ICP 精配 (解决多物体场景下质心初始化跑偏的问题)。

    流程:
      1. 模板/场景降采样 + 算 FPFH 特征 (描述局部几何形状)
      2. RANSAC 全局配准: 用 FPFH 特征匹配找模板在场景的大致位置 (不需初值)
      3. ICP 精配: 用全局结果作初值, 精细对齐

    scene_points: (N,3) 米。template_path: .ply 模板。
    voxel_feature: FPFH 降采样粒度 (米), None=模板直径的 1/5。
    ransac_n: RANSAC 采样次数 (越大越稳越慢)。
    返回 MatchResult。"""
    pts = np.asarray(scene_points, dtype=np.float64)
    if len(pts) == 0:
        raise ValueError("场景点云为空, 无法匹配")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"模板不存在: {template_path}")
    source = o3d.io.read_point_cloud(template_path)
    if len(source.points) == 0:
        raise ValueError(f"模板点云为空: {template_path}")
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pts)

    # FPFH 降采样粒度: 默认 5mm (工件多在 5~8cm 量级, 5mm 能保留足够点算特征)。
    # 太大会把稀疏模板降到几十点, FPFH 不稳定; 太小则场景特征计算慢。
    voxel_size = float(voxel_feature) if voxel_feature else 0.005
    print(f"  FPFH 全局配准: voxel={voxel_size*1000:.1f}mm")

    # 1. 降采样 + FPFH 特征
    source_ds, source_fpfh = _downsample_and_fpfh(source, voxel_size)
    target_ds, target_fpfh = _downsample_and_fpfh(target, voxel_size)
    print(f"    模板 {len(source_ds.points)} 点 (降采样后), 场景 {len(target_ds.points)} 点")

    # 2. RANSAC 全局配准 (FPFH 特征匹配, 找模板在场景的大致位置)
    distance_threshold_global = voxel_size * 1.5
    result_global = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_ds, target_ds, source_fpfh, target_fpfh, True,
        distance_threshold_global,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold_global)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_n, 0.999))
    print(f"    全局 RANSAC: fitness={result_global.fitness:.4f} rmse={result_global.inlier_rmse:.5f}")

    # 3. ICP 精配 (用全局结果作初值)
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, result_global.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(icp_iteration)))

    source_aligned = o3d.geometry.PointCloud(source)
    source_aligned.transform(result.transformation)
    aligned_points = np.asarray(source_aligned.points, dtype=np.float32)
    aligned_colors = np.asarray(source_aligned.colors, dtype=np.float32)
    if len(aligned_colors) != len(aligned_points):
        aligned_colors = np.tile(np.array([[0.72, 0.72, 0.68]], dtype=np.float32),
                                 (len(aligned_points), 1))
    info = {
        "fitness": float(result.fitness),
        "rmse": float(result.inlier_rmse),
        "global_fitness": float(result_global.fitness),
        "scene_points": int(len(pts)),
        "model_points": int(len(aligned_points)),
    }
    print(f"  ICP 精配: {len(pts)} 场景点 <- {len(aligned_points)} 模板点, "
          f"fitness={info['fitness']:.4f} rmse={info['rmse']:.5f}")
    # 诊断: 配准后模板 + 场景的 bbox, 看是否重合
    print(f"    场景 bbox: [{np.round(pts.min(0),3).tolist()} ~ {np.round(pts.max(0),3).tolist()}]")
    print(f"    模板 bbox: [{np.round(aligned_points.min(0),3).tolist()} ~ {np.round(aligned_points.max(0),3).tolist()}]")
    return MatchResult(aligned_points, aligned_colors,
                       np.asarray(result.transformation, dtype=np.float64), info)


# ============================================================================
# 可视化 (shift+左键拾取点 -> 终端打印坐标)
# ============================================================================
def _to_pcd(points, colors=None, default_color=(0.7, 0.7, 0.7)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(colors, dtype=np.float64) / 255.0, 0.0, 1.0))
    else:
        pcd.paint_uniform_color(default_color)
    return pcd


def _show_with_picking(geometries, title, point_size=5.0):
    """弹窗显示点云。关闭窗口后返回 (阻塞)。
    用 draw_geometries (与原 client visualize_rpc_debug.py 一致, 渲染最可靠)。
    可视化时按 n 打开全局设置面板可调点大小; shift+右键拖动旋转, 滚轮缩放。"""
    o3d.visualization.draw_geometries(geometries, window_name=title)


def show_pointcloud(points, colors=None, title="point cloud", center=None, frame_size=0.5):
    """显示单一点云 + base 坐标轴 (红x 绿y 蓝z, 原点=base)。center: 白球标注点。"""
    geos = [_to_pcd(points, colors),
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)]
    if center is not None:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sph.translate(np.asarray(center, dtype=np.float64))
        sph.paint_uniform_color([1.0, 1.0, 1.0])
        geos.append(sph)
    print(f"[{title}] 红x 绿y 蓝z=base坐标系; shift+左键拾取点; 关闭窗口继续")
    _show_with_picking(geos, title)


def show_match(scene_points, scene_colors, workpiece_points,
               workpiece_color=(1.0, 0.1, 0.1), center=None, title="ICP match"):
    """场景(原色/灰) + 工件模板(亮红) + base 坐标轴 + 工件中心黄球。"""
    scene_points = np.asarray(scene_points, dtype=np.float64)
    workpiece_points = np.asarray(workpiece_points, dtype=np.float64)
    # 诊断: 打印两者 bbox, 不重合 = ICP 失败/模板飞走 (看不到红点的常见原因)
    if len(scene_points) and len(workpiece_points):
        print(f"  [诊断] 场景 bbox: {np.round(scene_points.min(0),3).tolist()} ~ {np.round(scene_points.max(0),3).tolist()} ({len(scene_points)} 点)")
        print(f"  [诊断] 模板 bbox: {np.round(workpiece_points.min(0),3).tolist()} ~ {np.round(workpiece_points.max(0),3).tolist()} ({len(workpiece_points)} 点)")
    wp_pcd = _to_pcd(workpiece_points, default_color=workpiece_color)
    geos = [_to_pcd(scene_points, scene_colors),
            wp_pcd,
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)]
    if center is not None:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sph.translate(np.asarray(center, dtype=np.float64))
        sph.paint_uniform_color([1.0, 1.0, 0.0])
        geos.append(sph)
    print(f"[{title}] 亮红=模板 灰=场景 黄球=工件中心 红x绿y蓝z=base; shift+左键拾取点; 关闭继续")
    _show_with_picking(geos, f"{title} (红=模板 灰=场景)")

if __name__ == "__main__":
    # 在同一窗口并排显示三个模板 (各自加 x 方向 offset 平移开), 便于一眼对比尺寸。
    template_dir = "/home/grasp/xukun/grasp/BasicToolBox/tools/grasp/templates"
    template_files = sorted(glob.glob(os.path.join(template_dir, "*.ply")))
    colors = [(1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.1, 0.1, 1.0)]   # 红/绿/蓝
    X_STEP = 0.12   # 各模板 x 方向间隔 (米), 大于最大模板直径 0.08 即可分开
    geos = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)]
    for i, tpl_path in enumerate(template_files):
        tpl = o3d.io.read_point_cloud(tpl_path)
        tpl.translate([i * X_STEP, 0, 0])      # 各自 offset 平移
        tpl.paint_uniform_color(colors[i % len(colors)])
        pts = np.asarray(tpl.points)
        print(f"{os.path.basename(tpl_path)}: {len(pts)} 点, "
              f"边长≈{(pts[:,0].max()-pts[:,0].min())*1000:.0f}mm "
              f"(offset x={i*X_STEP:.2f}m, 色={colors[i%len(colors)]})")
        geos.append(tpl)
    o3d.visualization.draw_geometries(geos, window_name="3 templates size compare (红/绿/蓝)")