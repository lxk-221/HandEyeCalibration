"""
Camera 抽象基类。

硬件初始化在子类 __init__ 中完成; K / dist 作为属性暴露, 接口里不出现实现 handle。

约定:
  - `get_frame()` 始终返回 BGR uint8 HxWx3 的 numpy 数组 (喂给 cv2 检测)。
  - `get_rgbd()` 返回 (BGR, depth_uint16_mm): RGBD/双目相机子类实现, 其余 raise。
  - `get_point_cloud()`: 返回 (points_m Nx3, colors_rgb Nx3)。默认基于 get_rgbd + K 反投影;
    若相机不支持深度则 raise NotImplementedError。
  - `K`   : 3x3 相机内参矩阵 (与 depth 流分辨率一致)。
  - `dist`: 畸变系数 (OpenCV 约定, k1,k2,p1,p2,k3[,...])。
"""
import numpy as np


class Camera:
    K: np.ndarray
    dist: np.ndarray

    def __init__(self):
        pass

    def get_frame(self) -> np.ndarray:
        """返回一帧 BGR uint8 (H, W, 3) 图像。失败返回 None。"""
        raise NotImplementedError

    def get_rgbd(self):
        """返回 (bgr uint8 HxWx3, depth_uint16_mm HxW)。RGBD/双目相机子类实现。
        不支持的相机 raise NotImplementedError。"""
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 get_rgbd, 需 RGBD/双目相机子类实现"
        )

    def get_point_cloud(self, stride: int = 1):
        """返回 (points_m Nx3 float32, colors_rgb Nx3 uint8), 单位米。
        默认实现: get_rgbd() 取帧 -> 用 K 反投影。子类有更优实现可覆写。
        点云在相机坐标系下 (z 朝远、x 向右、y 向下, OpenCV/RGBD 约定)。"""
        bgr, depth_mm = self.get_rgbd()
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
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
        colors = bgr[valid] if bgr.shape[:2] == depth_mm.shape else \
            bgr[np.indices((h, w))[0][valid], np.indices((h, w))[1][valid]]
        colors_rgb = colors[..., ::-1].astype(np.uint8)  # BGR->RGB
        return points, colors_rgb

    def release(self) -> None:
        """释放硬件资源。"""
        raise NotImplementedError

