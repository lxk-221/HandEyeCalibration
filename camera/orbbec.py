"""
Orbbec 相机实现 (pyorbbecsdk, 无 ROS)。

只采集 COLOR 流喂给手眼标定。内参 K / dist 默认从 SDK 出厂值读取;
若构造时传入标定过的 K / dist 则覆盖出厂值 (精度更高)。

实现细节对齐 xyz_bak/hand_eye_calibration.py 中实战验证过的
init_camera() / grab_color(): 代理清理、默认流配置、enable_frame_sync、
get_camera_param 读内参、取帧循环。
"""
import os

# 防代理干扰 Orbbec USB 通信 (实战验证: ALL_PROXY 会导致取帧异常)。
for _p in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_p, None)

import cv2
import numpy as np
from pyorbbecsdk import (  # type: ignore
    Config,
    Context,
    OBFormat,
    OBSensorType,
    Pipeline,
)

from .camera import Camera


class OrbbecCamera(Camera):
    def __init__(self, K: np.ndarray = None, dist: np.ndarray = None,
                 warmup_frames: int = 20):
        """
        K / dist: 可选, 标定过的内参/畸变 (覆盖出厂值, 推荐用于手眼标定)。
                  不传则用 pyorbbecsdk 读出的出厂内参。
        warmup_frames: 启动后丢弃的帧数, 让 SDK 稳定 + 解析内参 (实战用 20)。
        """
        if Context().query_devices().get_count() == 0:
            raise RuntimeError("未发现 Orbbec 相机")

        self._pipe = Pipeline()
        cfg = Config()
        cfg.enable_stream(self._pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                          .get_default_video_stream_profile())
        cfg.enable_stream(self._pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                          .get_default_video_stream_profile())
        self._pipe.enable_frame_sync()
        self._pipe.start(cfg)

        # 取 warmup 帧让 SDK 解析内参 (出厂值), 取不到则抛错。
        factory_K = factory_dist = None
        for _ in range(max(1, warmup_frames + 1)):
            fs = self._pipe.wait_for_frames(1000)
            if fs:
                cp = self._pipe.get_camera_param()
                # color 内参 -> K (手眼标定用 get_frame 的 color 图)
                ri = cp.rgb_intrinsic
                factory_K = np.array([[ri.fx, 0.0,    ri.cx],
                                      [0.0,    ri.fy, ri.cy],
                                      [0.0,    0.0,   1.0   ]], dtype=np.float64)
                factory_dist = np.array([cp.rgb_distortion.k1, cp.rgb_distortion.k2,
                                         cp.rgb_distortion.p1, cp.rgb_distortion.p2,
                                         cp.rgb_distortion.k3], dtype=np.float64)
                # depth 内参 -> depth_K (点云用 get_rgbd 的 depth 图, 分辨率与 color 不同)
                di = cp.depth_intrinsic
                self._factory_depth_K = np.array([[di.fx, 0.0,    di.cx],
                                                  [0.0,    di.fy, di.cy],
                                                  [0.0,    0.0,   1.0   ]], dtype=np.float64)
                break
        if factory_K is None:
            self.release()
            raise RuntimeError("取不到 Orbbec 出厂内参")

        # 构造传入的标定值优先 (精度更高); 否则用出厂值。
        self.K = np.asarray(K, dtype=np.float64) if K is not None else factory_K
        self.dist = np.asarray(dist, dtype=np.float64) if dist is not None else factory_dist
        self.depth_K = self._factory_depth_K   # depth 流内参 (点云反投影用)
        super().__init__()

    def get_point_cloud(self, stride: int = 1):
        """覆写: 用 depth 流内参 depth_K 反投影 (基类默认用 self.K=rgb 内参, 分辨率不匹配)。"""
        bgr, depth_mm = self.get_rgbd()
        fx, fy = self.depth_K[0, 0], self.depth_K[1, 1]
        cx, cy = self.depth_K[0, 2], self.depth_K[1, 2]
        depth_m = depth_mm.astype(np.float32) * 0.001
        h, w = depth_m.shape
        vv, uu = np.indices((h, w), dtype=np.float32)
        valid = depth_m > 0
        if stride > 1:
            sm = np.zeros_like(valid)
            sm[::stride, ::stride] = True
            valid &= sm
        z = depth_m[valid]
        x = (uu[valid] - cx) * z / fx
        y = (vv[valid] - cy) * z / fy
        points = np.column_stack((x, y, z)).astype(np.float32)
        colors_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[valid].astype(np.uint8)
        return points, colors_rgb

    def get_frame(self) -> np.ndarray:
        """返回一帧 BGR uint8 (H, W, 3)。取不到帧返回 None。"""
        for _ in range(30):
            fs = self._pipe.wait_for_frames(1000)
            if fs and fs.get_color_frame() is not None:
                bgr = _frame_to_bgr(fs.get_color_frame())
                if bgr is not None:
                    return bgr
        return None

    def get_rgbd(self):
        """返回 (bgr uint8 HxWx3, depth_uint16_mm HxW)。
        color resize 到 depth 分辨率 (depth 流分辨率可能不同), 与基类 get_point_cloud 的 K 对齐。"""
        for _ in range(30):
            fs = self._pipe.wait_for_frames(1000)
            if not fs:
                continue
            color_frame = fs.get_color_frame()
            depth_frame = fs.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue
            bgr = _frame_to_bgr(color_frame)
            if bgr is None:
                continue
            depth_mm = _frame_to_depth_mm(depth_frame)
            if depth_mm is None:
                continue
            # color 对齐到 depth 尺寸 (K 来自 depth 流内参)。
            if bgr.shape[:2] != depth_mm.shape:
                bgr = cv2.resize(bgr, (depth_mm.shape[1], depth_mm.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
            return bgr, depth_mm
        return None

    def release(self) -> None:
        try:
            self._pipe.stop()
        except Exception:
            pass


def _frame_to_bgr(frame) -> np.ndarray:
    """Orbbec VideoFrame -> BGR uint8 (H, W, 3)。不支持的格式返回 None。
    覆盖常见格式: RGB / BGR / MJPG / YUYV。自包含, 不依赖 xyz_bak/utils。"""
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())
    if fmt == OBFormat.RGB:
        img = np.resize(data, (height, width, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.YUYV:
        img = np.resize(data, (height, width, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    return None


def _frame_to_depth_mm(frame) -> np.ndarray:
    """Orbbec depth VideoFrame -> uint16 mm (H, W)。
    原始数据乘 get_depth_scale() 得到 mm, NaN/Inf->0, clip 到 uint16。"""
    try:
        width = frame.get_width()
        height = frame.get_height()
        scale = float(frame.get_depth_scale())
    except Exception:
        return None
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((height, width))
    depth_mm = np.nan_to_num(raw.astype(np.float32) * scale, nan=0.0,
                             posinf=0.0, neginf=0.0)
    depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max)
    return depth_mm.astype(np.uint16)
