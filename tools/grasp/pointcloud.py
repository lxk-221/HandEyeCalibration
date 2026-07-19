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
                        distance_thresh=0.004, ransac_n=3, ransac_iter=1000,
                        min_inlier_ratio=0.05, remove_below=True):
    """RANSAC 拟合一个平面 (通常是桌面), 去掉平面内点;
    remove_below=True 时还去掉平面【下方】的点 (保留桌面之上的物体)。
    base 系下桌面是大平面, 物体在桌面之上 -> 默认 remove_below=True 保留物体。
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
    if remove_below and abs(c) > 1e-9:
        z_plane = -(a * pts[:, 0] + b * pts[:, 1] + d) / c
        below_mask = pts[:, 2] <= z_plane   # 平面及下方
    else:
        below_mask = np.zeros(len(pts), dtype=bool)
    keep = ~(plane_inlier_mask | below_mask)
    out_cols = np.asarray(colors)[keep] if colors is not None and len(colors) == len(pts) else None
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
# 5. ICP 模板匹配
# ============================================================================
@dataclass
class MatchResult:
    aligned_points: np.ndarray       # (N,3) 模板配准后 (与 scene 同坐标系), 米
    aligned_colors: np.ndarray       # (N,3) float 0..1
    transform: np.ndarray            # 4x4 source(template)->target(scene)
    info: dict = field(default_factory=dict)


def icp_match(scene_points, template_path,
              max_correspondence_distance=0.012, icp_iteration=80):
    """把完整模板点云 (source) ICP 配准到场景物体点 (target)。
    scene_points: (N,3) 米 (与输出同坐标系)。template_path: .ply 模板。
    返回 MatchResult。aligned_points 中心 = 物体中心 (在该坐标系下)。"""
    pts = np.asarray(scene_points, dtype=np.float64)
    if len(pts) == 0:
        raise ValueError("场景点云为空, 无法 ICP 匹配")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"模板不存在: {template_path}")
    source = o3d.io.read_point_cloud(template_path)
    if len(source.points) == 0:
        raise ValueError(f"模板点云为空: {template_path}")

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pts)
    # 初始变换: 质心对齐
    init = np.eye(4, dtype=np.float64)
    init[:3, 3] = pts.mean(0) - np.asarray(source.points).mean(0)
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, init,
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
        "scene_points": int(len(pts)),
        "model_points": int(len(aligned_points)),
    }
    print(f"  ICP: {len(pts)} 场景点 <- {len(aligned_points)} 模板点, "
          f"fitness={info['fitness']:.4f} rmse={info['rmse']:.5f}")
    # 诊断: 配准后模板 + 场景的 bbox, 看是否重合 (不重合 = ICP 失败, 模板飞走)
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


def _show_with_picking(geometries, title):
    """弹窗显示, shift+左键选点 -> 终端打印本次新选点坐标 (米)。关闭窗口继续。"""
    viz = o3d.visualization.VisualizerWithVertexSelection()
    viz.create_window(window_name=title)
    for g in geometries:
        viz.add_geometry(g)
    _last_count = [0]

    def _on_changed():
        picked = viz.get_picked_points()
        for p in picked[_last_count[0]:]:
            print(f"  拾取点 (m): [{p.coord[0]:.4f}, {p.coord[1]:.4f}, {p.coord[2]:.4f}]")
        _last_count[0] = len(picked)

    viz.register_selection_changed_callback(_on_changed)
    viz.run()
    viz.destroy_window()


def show_pointcloud(points, colors=None, title="point cloud", center=None, frame_size=0.2):
    """显示单一点云 + base 坐标轴 (红x 绿y 蓝z)。center: 白球标注点。"""
    geos = [_to_pcd(points, colors),
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)]
    if center is not None:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sph.translate(np.asarray(center, dtype=np.float64))
        sph.paint_uniform_color([1.0, 1.0, 1.0])
        geos.append(sph)
    print(f"[{title}] shift+左键拾取点; 关闭窗口继续")
    _show_with_picking(geos, title)


def show_match(scene_points, scene_colors, workpiece_points,
               workpiece_color=(0.05, 0.85, 0.20), center=None, title="ICP match"):
    """场景(原色/灰) + 工件模板(绿) + 坐标轴 + 工件中心黄球。"""
    geos = [_to_pcd(scene_points, scene_colors),
            _to_pcd(workpiece_points, default_color=workpiece_color),
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)]
    if center is not None:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sph.translate(np.asarray(center, dtype=np.float64))
        sph.paint_uniform_color([1.0, 1.0, 0.0])
        geos.append(sph)
    print(f"[{title}] 绿=模板 灰=场景 黄球=工件中心; shift+左键拾取点; 关闭继续")
    _show_with_picking(geos, f"{title} (绿=模板 灰=场景 黄球=工件中心)")
