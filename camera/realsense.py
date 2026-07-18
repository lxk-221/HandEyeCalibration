"""
Intel RealSense 相机实现 (pyrealsense2, 无 ROS)。

只采集 COLOR 流, 请求 BGR8 格式, 这样拿到的帧可直接 reshape 成 BGR 图喂给 cv2,
无需 RGB<->BGR 转换。内参 K / dist 从启动后第一帧的 stream profile 读取。
"""
import numpy as np
import pyrealsense2 as rs

from camera.camera import Camera


class RealSenseCamera(Camera):
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30,
                 align_to_color: bool = True):
        """
        width/height/fps: COLOR 流分辨率与帧率 (D435I 支持 1280x720@30)。
        align_to_color: 若同时开 depth, 对齐到 color; 这里只取 color, 仍保留以便扩展。
        """
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        # depth 不参与手眼标定, 但保持可选 (默认不开, 减少依赖/带宽)
        self._align = rs.align(rs.stream.color) if align_to_color else None
        self._profile = self._pipe.start(cfg)
        self._read_intrinsics()
        super().__init__()

    def _read_intrinsics(self):
        """从 color stream profile 读取出厂内参 -> K, dist。"""
        sp = self._profile.get_stream(rs.stream.color)
        intr = sp.as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0.0,       intr.ppx],
                           [0.0,       intr.fy, intr.ppy],
                           [0.0,       0.0,     1.0      ]], dtype=np.float64)
        self.dist = np.array(intr.coeffs[:5], dtype=np.float64)  # k1,k2,p1,p2,k3

    def get_frame(self) -> np.ndarray:
        """返回一帧 BGR uint8 (H, W, 3)。取不到帧返回 None。"""
        for _ in range(30):
            fs = self._pipe.wait_for_frames(timeout_ms=1000)
            if self._align is not None:
                fs = self._align.process(fs)
            f = fs.get_color_frame()
            if f:
                img = np.asanyarray(f.get_data())  # 已是 BGR8
                if img.size > 0:
                    return img
        return None

    def release(self) -> None:
        try:
            self._pipe.stop()
        except Exception:
            pass
