"""
点云处理 + 模板匹配 (ICP) 核心模块。

从 indust_grasp/graspnet-baseline/demo.py 提取, 剥离所有 graspnet 推理依赖,
全部改成显式函数参数 (去掉 demo.py 的 cfgs 全局对象)。

依赖: 仅 open3d + cv2 + numpy。

坐标系约定: 所有点云都在**相机坐标系**下 (z 朝远、x 向右、y 向下, OpenCV/RGBD 约定),
单位米。T_gripper2base 用于把相机系结果变到 base 系 (在 template_based_grasp 里做)。

主要流程 (process_scene):
    RGB-D/点云
      -> (可选) 去畸变 + 深度平滑
      -> 距离过滤        distance_filter_point_cloud
      -> RANSAC 平面分割  ransac_remove_plane_and_greater_z  (去掉桌面及桌面上方)
      -> ICP 模板匹配     match_workpiece_point_cloud        (完整模板对齐到场景物体)
      -> 输出: 模板点云(base系) + 工件中心(base系) + 配准信息
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d


# ---------------- 相机模型 (内联自 data_utils, 仅 numpy) ----------------
@dataclass
class CameraInfo:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    scale: float = 1.0


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """depth (H,W) + CameraInfo -> 点云。organized=True 保留 (H,W,3), 否则展平 (N,3)。
    复刻自 data_utils.create_point_cloud_from_depth_image。"""
    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
    z = depth.astype(np.float32) * camera.scale
    x_axis = np.arange(camera.width, dtype=np.float32)   # 沿 W
    y_axis = np.arange(camera.height, dtype=np.float32)  # 沿 H
    x = (x_axis[None, :] - cx) * z / fx   # (H,W) 广播
    y = (y_axis[:, None] - cy) * z / fy
    cloud = np.stack([x, y, z], axis=-1)  # (H,W,3)
    return cloud if organized else cloud.reshape(-1, 3)


# ---------------- 深度预处理 ----------------
def _odd_kernel(value):
    value = int(value)
    if value <= 0:
        return 0
    return value if value % 2 == 1 else value + 1


def smooth_depth(depth_uint16, factor_depth,
                 median_kernel=0, bilateral_d=0,
                 bilateral_sigma_color=0.0, bilateral_sigma_space=0.0):
    """深度平滑: 中值滤波 + 双边滤波 (只在有效像素上)。复刻自 demo.smooth_depth。
    depth_uint16: (H,W) uint16 mm; factor_depth: mm->m 的除数 (1000)。"""
    depth = depth_uint16.astype(np.uint16, copy=True)
    median_kernel = _odd_kernel(median_kernel)
    if median_kernel > 1:
        depth = cv2.medianBlur(depth, median_kernel)
    if bilateral_d > 0:
        valid = depth > 0
        depth_m = depth.astype(np.float32) / factor_depth
        filtered = cv2.bilateralFilter(depth_m, d=bilateral_d,
                                       sigmaColor=bilateral_sigma_color,
                                       sigmaSpace=bilateral_sigma_space)
        depth_m[valid] = filtered[valid]
        depth_m[~valid] = 0.0
        depth = np.clip(depth_m * factor_depth, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return depth


def undistort_rgbd(color, depth, workspace_mask, intrinsic_matrix, dist_coeffs):
    """RGB-D 去畸变 (用 initUndistortRectifyMap + remap)。
    复刻自 demo.undistort_rgbd, 但把写死的 CALI_DIST 改成显式 dist_coeffs 参数。
    workspace_mask: bool 数组; 返回 (color, depth_uint16, mask_bool)。"""
    h, w = depth.shape
    if color.shape[:2] != (h, w):
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)
    intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float32).reshape(3, 3)
    map_x, map_y = cv2.initUndistortRectifyMap(
        intrinsic_matrix, np.asarray(dist_coeffs, dtype=np.float32),
        None, intrinsic_matrix, (w, h), cv2.CV_32FC1)
    color = cv2.remap(color, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT)
    depth = cv2.remap(depth, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT)
    workspace_mask = cv2.remap(
        (np.asarray(workspace_mask) > 0).astype(np.uint8) * 255, map_x, map_y,
        interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    return color, depth.astype(np.uint16), workspace_mask


# ---------------- 点云过滤 / 分割 ----------------
def distance_filter_point_cloud(points, min_distance_m, max_distance_m, mode='z'):
    """保留 [min, max] 距离内的点。mode='z' 用相机 z; 'radius' 用欧氏距离。
    返回 bool mask (len(points),)。"""
    if mode == 'radius':
        distance = np.linalg.norm(points, axis=1)
    else:
        distance = points[:, 2]
    return (distance >= min_distance_m) & (distance <= max_distance_m)


def ransac_remove_plane_and_greater_z(points, colors, distance_thresh,
                                      ransac_n=3, ransac_iter=1000,
                                      min_inlier_ratio=0.0):
    """RANSAC 拟合平面, 去掉平面内点 + z 大于平面的点 (即桌面及桌面上方)。
    返回 (points_kept, colors_kept, keep_mask, plane_model, plane_inlier_mask, greater_plane_mask)。
    inlier_ratio < min_inlier_ratio 时跳过删除 (认为没找到有效平面)。"""
    if len(points) < ransac_n:
        empty = np.zeros(len(points), dtype=bool)
        return points, colors, np.ones(len(points), dtype=bool), None, empty, empty

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_thresh, ransac_n=ransac_n, num_iterations=ransac_iter)
    inliers = np.asarray(inliers, dtype=np.int64)
    inlier_ratio = len(inliers) / float(len(points))
    if inlier_ratio < min_inlier_ratio:
        empty = np.zeros(len(points), dtype=bool)
        return points, colors, np.ones(len(points), dtype=bool), plane_model, empty, empty

    plane_inlier_mask = np.zeros(len(points), dtype=bool)
    plane_inlier_mask[inliers] = True
    a, b, c, d = [float(v) for v in plane_model]
    if abs(c) < 1e-9:
        greater_plane_mask = np.zeros(len(points), dtype=bool)
    else:
        z_plane = -(a * points[:, 0] + b * points[:, 1] + d) / c
        greater_plane_mask = points[:, 2] > z_plane

    keep_mask = ~(plane_inlier_mask | greater_plane_mask)
    return (points[keep_mask], colors[keep_mask], keep_mask,
            plane_model, plane_inlier_mask, greater_plane_mask)


# ---------------- ICP 模板匹配 ----------------
def _centroid_initial_transform(source_points, target_points):
    """初始变换: 用质心差做平移 (ICP 初值)。"""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = target_points.mean(axis=0) - source_points.mean(axis=0)
    return transform


def match_workpiece_point_cloud(target_points, workpiece_cloud_path,
                                max_correspondence_distance, icp_iteration):
    """把完整模板点云 (source) ICP 配准到场景分割出的物体点 (target)。
    target_points: (N,3) 相机系米; workpiece_cloud_path: .ply 模板。
    返回 MatchResult (aligned_points 相机系米, transform 4x4 source->target, info dict)。"""
    if len(target_points) == 0:
        raise ValueError('No target object points for workpiece registration.')
    if not os.path.exists(workpiece_cloud_path):
        raise FileNotFoundError('Workpiece point cloud not found: %s' % workpiece_cloud_path)

    source = o3d.io.read_point_cloud(workpiece_cloud_path)
    if len(source.points) == 0:
        raise ValueError('Workpiece point cloud is empty: %s' % workpiece_cloud_path)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points.astype(np.float64))

    source_points = np.asarray(source.points, dtype=np.float64)
    init = _centroid_initial_transform(source_points, target_points.astype(np.float64))
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
        'method': 'centroid_init_point_to_point_icp',
        'segmented_points': int(len(target_points)),
        'model_points': int(len(aligned_points)),
        'fitness': float(result.fitness),
        'rmse': float(result.inlier_rmse),
        'model_center_m': source_points.mean(axis=0).astype(float).tolist(),
        'aligned_center_camera_m': aligned_points.mean(axis=0).astype(float).tolist(),
    }
    print('-> workpiece ICP: %d scene pts <- %d model pts, fitness=%.4f rmse=%.5f'
          % (len(target_points), len(aligned_points), result.fitness, result.inlier_rmse))
    return MatchResult(
        aligned_points=aligned_points, aligned_colors=aligned_colors,
        transform=np.asarray(result.transformation, dtype=np.float64), info=info)


@dataclass
class MatchResult:
    aligned_points: np.ndarray       # (N,3) 模板配准后, 相机系米
    aligned_colors: np.ndarray       # (N,3) float 0..1
    transform: np.ndarray            # 4x4 source->target
    info: dict = field(default_factory=dict)


# ---------------- 坐标变换 ----------------
def camera_to_base_transform(T_gripper2base, T_cam2gripper):
    """T_camera2base = T_gripper2base @ T_cam2gripper。
    T_cam2gripper: 手眼标定结果 (X = T_cam2gripper), 4x4, 平移米。"""
    T_g2b = np.asarray(T_gripper2base, dtype=np.float64).reshape(4, 4)
    T_c2g = np.asarray(T_cam2gripper, dtype=np.float64).reshape(4, 4)
    return T_g2b @ T_c2g


def transform_points_to_base(points_camera, T_camera2base):
    """把相机系点云 (N,3) 米变到 base 系 (N,3) 米。"""
    T = np.asarray(T_camera2base, dtype=np.float64).reshape(4, 4)
    p = np.asarray(points_camera, dtype=np.float64)
    return (T[:3, :3] @ p.T + T[:3, 3:4]).T


def transform_pose_to_base(rotation_camera, translation_camera, T_camera2base):
    """把相机系下一位姿 (R 3x3, t 3) 变到 base 系, 返回 4x4。"""
    T = np.asarray(T_camera2base, dtype=np.float64).reshape(4, 4)
    T_cam_grasp = np.eye(4, dtype=np.float64)
    T_cam_grasp[:3, :3] = np.asarray(rotation_camera, dtype=np.float64)
    T_cam_grasp[:3, 3] = np.asarray(translation_camera, dtype=np.float64)
    return T @ T_cam_grasp


# ---------------- 多帧拼合 (scan) ----------------
def merge_frames_to_base(frames, T_cam2gripper):
    """把多帧 (相机系点云 + 拍照时刻 T_gripper2base) 全部转到 base 系并拼接。
    frames: list of (points_camera Nx3, colors_rgb Nx3 uint8, T_gripper2base 4x4)。
    返回 (points_base Nx3 float64, colors_rgb Nx3 uint8)。"""
    all_pts, all_cols = [], []
    for points_cam, colors, T_g2b in frames:
        T_c2b = camera_to_base_transform(T_g2b, T_cam2gripper)
        pts_base = transform_points_to_base(points_cam, T_c2b)
        all_pts.append(pts_base)
        all_cols.append(np.asarray(colors, dtype=np.uint8))
    if not all_pts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(all_pts, 0), np.concatenate(all_cols, 0)


def voxel_downsample(points, colors, voxel_size):
    """体素降采样 (open3d)。点云密度均匀化, 是多帧拼合/ICP 前的标准处理。
    返回 (points_down Nx3 float64, colors_down Nx3 uint8)。voxel_size 单位米。"""
    if len(points) == 0 or voxel_size <= 0:
        return np.asarray(points), np.asarray(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(colors, dtype=np.float64) / 255.0, 0.0, 1.0))
    pcd = pcd.voxel_down_sample(voxel_size)
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64)
    cols = np.clip(cols * 255.0, 0, 255).astype(np.uint8) if len(cols) == len(pts) else \
        np.tile(np.array([[180, 180, 173]], dtype=np.uint8), (len(pts), 1))
    return pts, cols


def segment_workpiece_from_base(points_base, colors_base,
                                distance_min_m=0.3, distance_max_m=1.2,
                                plane_thresh_m=0.004):
    """base 系点云的分割: 距离过滤 (到 base 原点的半径) + RANSAC 去平面。
    把点云处理的默认参数集中在此 (避免散落到主程序的 argparse)。
    返回 (points_seg, colors_seg)。"""
    points_base = np.asarray(points_base, dtype=np.float64)
    # 距离过滤: base 系下用到原点的半径 (xy 主要分布范围)
    radius = np.linalg.norm(points_base[:, :2], axis=1)
    keep = (radius >= distance_min_m) & (radius <= distance_max_m)
    pts = points_base[keep]
    cols = np.asarray(colors_base)[keep] if colors_base is not None else None
    # RANSAC 去平面 (桌面): base 系下桌面近似 z=const, segment_plane 仍适用
    pts, cols, _, _, _, _ = ransac_remove_plane_and_greater_z(
        pts, pts, plane_thresh_m, ransac_n=3, ransac_iter=1000, min_inlier_ratio=0.05)
    return pts, cols


# ---------------- 可视化 ----------------
def show_match(scene_points, scene_colors, workpiece_points,
               workpiece_color=(0.05, 0.85, 0.20)):
    """弹窗显示: 场景点云 (原色/灰) + 工件模板点云 (绿色) 叠加, 直观核对 ICP 匹配。
    关闭窗口后返回 (阻塞)。
    - scene_points: (N,3) base 系米; scene_colors: (N,3) uint8 RGB 或 None
    - workpiece_points: (M,3) base 系米 (已配准); workpiece_color: float (r,g,b) 0..1"""
    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(np.asarray(scene_points, dtype=np.float64))
    if scene_colors is not None and len(scene_colors) == len(scene_points):
        scene.colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(scene_colors, dtype=np.float64) / 255.0, 0.0, 1.0))
    else:
        scene.paint_uniform_color([0.7, 0.7, 0.7])

    wp = o3d.geometry.PointCloud()
    wp.points = o3d.utility.Vector3dVector(np.asarray(workpiece_points, dtype=np.float64))
    wp.paint_uniform_color(workpiece_color)

    o3d.visualization.draw_geometries([scene, wp],
                                      window_name="ICP match (绿=工件模板, 灰/原色=场景; 关闭继续)")
