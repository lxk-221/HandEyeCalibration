"""
eye-in-hand 手眼标定 —— 纯数学模块 (无硬件代码)。

命名遵循 OpenCV 手眼标定约定:
    T_A2B 表示把 A 系的点变到 B 系, 即 p_B = T_A2B @ p_A。
        FK        -> T_gripper2base  (每帧不同)
        solvePnP  -> T_target2cam    (每帧不同)
        待求 X    =  T_cam2gripper   (相机相对末端的固定位姿)
    标定板固定于 base, 恒等式:
        p_base = T_gripper2base @ T_cam2gripper @ T_target2cam @ p_target
        => T_gripper2base @ X @ T_target2cam = T_target2base = 常量  (残差据此验证)

单位: 全程米 (m) —— 标定板 objectPoints 用米, 位姿平移用米, 输出 X 平移用米。

依赖: 仅 cv2, numpy, scipy.spatial.transform.Rotation。
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot


# ---------------- 配置常量 ----------------
PATTERN = (7, 10)          # 棋盘内角点 (列, 行); calib.io 8x11 板 = 7x10
SQUARE_M = 0.015           # 方格边长 m (统一米制)
MIN_SAMPLES = 10
DEFAULT_METHOD = cv2.CALIB_HAND_EYE_TSAI


@dataclass
class Sample:
    """单帧采集结果。所有平移单位 m。"""
    T_gripper2base: np.ndarray   # 4x4
    corners: np.ndarray          # N×1×2 像素角点
    pattern: Tuple[int, int]     # 检测时实际用的 (列, 行)
    joints: Optional[np.ndarray] = None   # 关节角 rad (复现用, 可选)


# ---------------- 标定板 / 检测 ----------------
def make_obj_points(pattern: Tuple[int, int] = PATTERN) -> np.ndarray:
    """标定板角点在 target 系下的 3D 坐标 (m, Z=0)。"""
    nx, ny = pattern
    p = np.zeros((nx * ny, 3), np.float32)
    p[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    return p * SQUARE_M


def detect_chessboard(gray: np.ndarray,
                      pattern: Tuple[int, int] = PATTERN
                      ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    """检测棋盘角点 (自动试横向/纵向)。
    成功返回 (corners, pattern_used), 否则 (None, None)。
    返回实际检测用的 pattern —— objectPoints 的角点排列顺序依赖它。
    """
    for ps in (pattern, (pattern[1], pattern[0])):
        ok, c = cv2.findChessboardCornersSB(gray, ps, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok and len(c) == ps[0] * ps[1]:
            c = cv2.cornerSubPix(
                gray, c, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
            return c, ps
    return None, None


# ---------------- 单帧 PnP ----------------
def solve_pnp(corners: np.ndarray,
              pattern: Tuple[int, int],
              K: np.ndarray,
              dist: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """对一帧像素角点求 T_target2cam。返回 (R_3x3, t_3) 或 None。
    R, t 满足 p_cam = R @ p_target + t (单位 m)。"""
    objp = make_obj_points(pattern)
    ok, rv, tv = cv2.solvePnP(objp, corners, K, dist,
                              flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rv)
    return R, tv.reshape(3)


# ---------------- 核心: AX=XB 手眼标定 ----------------
@dataclass
class HandEyeResult:
    X: np.ndarray              # T_cam2gripper, 4x4, 平移 m
    n_frames: int
    rot_mean_deg: float
    rot_max_deg: float
    trans_mean_m: float
    trans_max_m: float


def solve_hand_eye(samples: List[Sample],
                   K: np.ndarray,
                   dist: np.ndarray,
                   method: int = DEFAULT_METHOD) -> Optional[HandEyeResult]:
    """eye-in-hand AX=XB 手眼标定。
    返回 X = T_cam2gripper (4x4, 平移 m) 及残差统计; 样本不足/PnP 全失败返回 None。

    残差: 各帧 T_target2base = T_gripper2base @ X @ T_target2cam 应一致。
    """
    R_g2b, t_g2b = [], []      # T_gripper2base -> R, t
    R_t2c, t_t2c = [], []      # T_target2cam   -> R, t
    triples = []               # (T_g2b_4x4, R_t2c, t_t2c) 残差用
    for s in samples:
        pnp = solve_pnp(s.corners, s.pattern, K, dist)
        if pnp is None:
            continue
        Rc, tc = pnp
        Tg2b = np.asarray(s.T_gripper2base, dtype=np.float64).copy()
        R_g2b.append(Tg2b[:3, :3]); t_g2b.append(Tg2b[:3, 3])
        R_t2c.append(Rc);          t_t2c.append(tc)
        triples.append((Tg2b, Rc, tc))

    if len(triples) < MIN_SAMPLES:
        return None

    R_x, t_x = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    X = _se3(R_x, t_x.reshape(3))

    # ---- 残差: 各帧 T_target2base 应一致 ----
    Ttb = [Tg2b @ X @ _se3(Rc, tc) for (Tg2b, Rc, tc) in triples]
    Ttb = np.stack(Ttb)
    Rm = Rot.from_matrix(Ttb[:, :3, :3]).mean().as_matrix()
    tm = Ttb[:, :3, 3].mean(0)
    rot_deg = np.array([np.degrees(np.arccos(np.clip(
        (np.trace(Rm.T @ Ttb[i, :3, :3]) - 1) / 2, -1, 1)))
        for i in range(len(Ttb))])
    trans_m = np.linalg.norm(Ttb[:, :3, 3] - tm, axis=1)
    return HandEyeResult(
        X=X, n_frames=len(triples),
        rot_mean_deg=float(rot_deg.mean()), rot_max_deg=float(rot_deg.max()),
        trans_mean_m=float(trans_m.mean()), trans_max_m=float(trans_m.max()))


# ---------------- 小工具 ----------------
def _se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rpy_deg_from_T(T: np.ndarray) -> np.ndarray:
    """从 4x4 旋转矩阵算 RPY (xyz, 度), 仅用于显示 (约定无关的稳定显示)。"""
    return np.degrees(Rot.from_matrix(T[:3, :3]).as_euler("xyz"))
