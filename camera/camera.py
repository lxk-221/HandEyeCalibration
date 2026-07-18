"""
Camera 抽象基类。

手眼标定只需要: 拿一帧 BGR 彩色图 + 相机内参 (K, dist)。
硬件初始化在子类 __init__ 中完成; K / dist 作为属性暴露, 接口里不出现实现 handle。

约定:
  - `get_frame()` 始终返回 BGR uint8 HxWx3 的 numpy 数组 (喂给 cv2 检测)。
  - `K`   : 3x3 相机内参矩阵。
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

    def release(self) -> None:
        """释放硬件资源。"""
        raise NotImplementedError
