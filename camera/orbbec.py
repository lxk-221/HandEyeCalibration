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

from camera.camera import Camera


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
                intr = self._pipe.get_camera_param().rgb_intrinsic
                d = self._pipe.get_camera_param().rgb_distortion
                factory_K = np.array([[intr.fx, 0.0,       intr.cx],
                                      [0.0,       intr.fy, intr.cy],
                                      [0.0,       0.0,     1.0    ]], dtype=np.float64)
                factory_dist = np.array([d.k1, d.k2, d.p1, d.p2, d.k3], dtype=np.float64)
                break
        if factory_K is None:
            self.release()
            raise RuntimeError("取不到 Orbbec 出厂内参")

        # 构造传入的标定值优先 (精度更高); 否则用出厂值。
        self.K = np.asarray(K, dtype=np.float64) if K is not None else factory_K
        self.dist = np.asarray(dist, dtype=np.float64) if dist is not None else factory_dist
        super().__init__()

    def get_frame(self) -> np.ndarray:
        """返回一帧 BGR uint8 (H, W, 3)。取不到帧返回 None。"""
        for _ in range(30):
            fs = self._pipe.wait_for_frames(1000)
            if fs and fs.get_color_frame() is not None:
                bgr = _frame_to_bgr(fs.get_color_frame())
                if bgr is not None:
                    return bgr
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
