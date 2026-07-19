"""抓取抽象 + 模板匹配实现。

Grasp 基类定义抓取生命周期: warm_up -> scan -> get_grasp_pose -> approach/grasp/place -> cool_down。
arm/hand 在 __init__ 注入一次, 后续方法共享, 遵循参数克制原则。
子类按"如何决定抓取位姿"区分:
  GraspTemplateBased: 模板 ICP 匹配 (本实现)
  GraspModelBased:    未来用 graspnet 等模型

arm/hand 接口约定 (满足即可, 如 LcmCommander):
  arm.get_ee_pose() / arm.move_arm(T) / arm.move_joints(joints)
  hand.move_hand(positions)
"""
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import pointcloud as pc


def _rotmat_to_quat(R):
    """3x3 旋转矩阵 -> xyzw 四元数 (打印调试用)。"""
    return Rot.from_matrix(R).as_quat()


class Grasp():
    def __init__(self, arm, hand) -> None:
        """Grasp class need arm and hand control class."""
        self.arm = arm
        self.hand = hand

    def get_grasp_pose(self):
        """Get Grasp Pose in World Frame."""
        raise NotImplementedError

    def approach(self):
        """Approaching to Grasp Pose."""
        raise NotImplementedError

    def grasp(self):
        """Grasping."""
        raise NotImplementedError

    def place(self):
        """Placing Object (For most grasp operation, place will be a simple following operation)."""
        raise NotImplementedError


class GraspTemplateBased(Grasp):
    """模板匹配抓取。eye-in-hand (相机装末端), T_cam2gripper 为手眼标定结果。

    用法 (见文件末尾 __main__):
        g = GraspTemplateBased(arm, hand, T_CAM2GRIPPER)
        g.warm_up()
        pc = g.scan(scan_poses)
        grasp_pose = g.get_grasp_pose(pc, template_path, T_ee2object)
        g.approach(grasp_pose); g.grasp(grasp_pose); g.place(place_pose)
        g.cool_down()
    """

    # ---- 写死的默认配置 (场景相关, 后续可加 setter 覆盖) ----
    VOXEL_SIZE_M = 0.002
    HAND_READY = [255, 10, 255, 255, 255, 255]
    HAND_GRASP = [0, 5, 70, 80, 80, 70]
    HAND_RELEASE = [255, 5, 255, 255, 255, 255]
    HAND_INIT = [255, 255, 255, 255, 255, 255]
    ARM_SPEED, ARM_ACCEL = 0.8, 0.8
    ARM_TIMEOUT, HAND_TIMEOUT = 60.0, 3.0
    # approach 预置高度 (相对 grasp_pose 上方), approach 时先到这里再下降
    APPROACH_PRE_Z_M = 0.10
    # place 抬起 / 下降高度
    PLACE_RAISE_M = 0.15
    PLACE_DESCEND_M = 0.14
    # WARMUP/COOLDOWN 关节序列 (复用 xyz_bak client 验证过的 8 关节轨迹, 避奇异/碰撞)
    _WARMUP_JOINTS = [
        [0,         0,        0,        0,        0, 0, 0],
        [0,        -np.pi/2,  0,        0,        0, 0, 0],
        [0,        -np.pi/2,  0,       -np.pi/2,  0, 0, 0],
        [0,        -np.pi/2,  0,       -np.pi/2,  0, 0,  np.pi/2],
        [np.pi/4,  -np.pi/4,  0,       -np.pi/2,  0, 0,  np.pi/2],
        [np.pi/4,  -np.pi/4, -np.pi/4, -np.pi/2,  0, 0,  np.pi/2],
        [0.740,    -0.556,   -0.774,   -1.525,   -0.880, 0.593, 1.747],
        [0.714,    -0.678,   -0.896,   -1.439,   -0.654, 0.535, 1.509],
    ]

    def __init__(self, arm, hand, T_CAM2GRIPPER=None, camera=None) -> None:
        """Temporary only support optional T_CAM2GRIPPER (eye on hand).
        camera: 取点云用 (需有 get_point_cloud), scan() 时必需。"""
        super().__init__(arm, hand)
        self.T_cam2gripper = T_CAM2GRIPPER if T_CAM2GRIPPER is not None else np.eye(4)
        self.camera = camera
        self.warmup_joints = list(self._WARMUP_JOINTS)
        self.cooldown_joints = list(reversed(self._WARMUP_JOINTS))

    # ---------------- 生命周期 ----------------
    def warm_up(self):
        """手复位 -> 关节空间从零位移动到扫描位附近 (避奇异/碰撞)。"""
        print("=== warm_up ===")
        print(f"  hand INIT {self.HAND_INIT}")
        self.hand.move_hand(self.HAND_INIT, timeout=self.HAND_TIMEOUT)
        for i, j in enumerate(self.warmup_joints, 1):
            print(f"  [{i}/{len(self.warmup_joints)}] {np.round(j,3).tolist()}")
            self.arm.move_joints(j, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                                 timeout=self.ARM_TIMEOUT)

    def cool_down(self):
        """手复位 -> 关节空间回零位 (warm_up 倒序)。"""
        print("=== cool_down ===")
        print(f"  hand INIT {self.HAND_INIT}")
        self.hand.move_hand(self.HAND_INIT, timeout=self.HAND_TIMEOUT)
        for i, j in enumerate(self.cooldown_joints, 1):
            print(f"  [{i}/{len(self.cooldown_joints)}] {np.round(j,3).tolist()}")
            self.arm.move_joints(j, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                                 timeout=self.ARM_TIMEOUT)

    def scan(self, scan_poses):
        """按 scan_poses 多视角取点云 -> 转到 base 系拼合 -> voxel 下采样。
        返回 (points_base Nx3, colors_base Nx3) base 系米。"""
        if self.camera is None:
            raise RuntimeError("scan 需要 camera (构造时传入)")
        frames = []
        for i, T in enumerate(scan_poses, 1):
            print(f"  scan [{i}/{len(scan_poses)}] 目标 xyz={np.round(T[:3,3],3).tolist()}")
            self.arm.move_arm(T, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                              timeout=self.ARM_TIMEOUT)
            # 到位后等一下让位姿稳定 + ee_pose 刷新到最新 (桥 20Hz 发布有滞后)
            time.sleep(1)
            T_gripper2base = self.arm.get_ee_pose(timeout=3.0)
            print(f"    ee2base xyz={np.round(T_gripper2base[:3,3],4).tolist()} "
                  f"quat={np.round(_rotmat_to_quat(T_gripper2base[:3,:3]),4).tolist()}")
            pts_cam, cols_cam = self.camera.get_point_cloud()
            print(f"    点云 {len(pts_cam)} 点")
            if len(pts_cam) > 0:
                frames.append((pts_cam, cols_cam, T_gripper2base))
        if not frames:
            raise RuntimeError("扫描未取到任何点云")
        pts, cols = pc.merge(frames, self.T_cam2gripper)
        print(f"  拼合 {len(pts)} 点")
        pc.show_pointcloud(pts, cols, title="[1] 拼合后 (base系)")
        pts, cols = pc.voxel_downsample(pts, cols, self.VOXEL_SIZE_M)
        print(f"  下采样 {len(pts)} 点")
        pc.show_pointcloud(pts, cols, title="[2] voxel 下采样后")
        return pts, cols

    # 各动作高度参数 (复刻 xyz_bak client execute_control_pose 的数值, 保证一致):
    #   grasp_pose = center + GRASP_OFFSET, rpy=GRASP_RPY (即原 control_xyz/control_rpy)
    #   approach 先到 grasp_pose 上方 DESCEND_M, 再下降 DESCEND_M 抓取 (原程序 step2->step3)
    #   place: 抬起 RAISE_M -> 放置位上方 -> 下降 PLACE_DESCEND_M -> 松手 -> 抬回
    GRASP_OFFSET_M = np.array([-0.07, 0.022, 0.23], dtype=np.float64)
    GRASP_RPY_RAD = np.array([-55.0, 0.0, 90.0]) * np.pi / 180.0
    DESCEND_M = 0.13           # approach: 先到 grasp_pose 上方 0.13, 再下降 0.13 (原程序 step2->3)
    RAISE_M = 0.15             # 抓后抬起 (原程序 step5)
    PLACE_DESCEND_M = 0.14     # 放置: 从上方下降 (原程序 step7)
    PLACE_RPY_RAD = np.array([-60.0, 0.0, 145.0]) * np.pi / 180.0
    PLACE_XYZ = np.array([0.262, -0.09, 0.0], dtype=np.float64)   # 放置 xy (z 动态取抬起高度)

    def get_grasp_pose(self, pointcloud, template, T_ee2object=None):
        """模板 ICP 匹配物体 -> 直接按写死的 offset/rpy 算 grasp pose (复刻原程序)。

        pointcloud: scan() 返回的 (points, colors) 或仅 points (base 系)
        template:   工件模板 .ply 路径
        T_ee2object: 暂保留参数 (未使用)。后续若要统一偏置语义再启用。
        返回: grasp_pose 4x4 (base 系)。xyz = 物体中心 + GRASP_OFFSET_M, rpy = GRASP_RPY_RAD。
        """
        pts = pointcloud[0] if isinstance(pointcloud, tuple) else pointcloud
        cols = pointcloud[1] if isinstance(pointcloud, tuple) else None

        # 分割 = 范围过滤 + RANSAC 去平面 (两步原子, 中间可视化便于调参)
        pts, cols = pc.range_filter(pts, cols, x_min=0.2, z_min=-0.55)
        print(f"  范围过滤后 {len(pts)} 点")
        pc.show_pointcloud(pts, cols, title="[3] 范围过滤后")
        pts, cols, _plane = pc.ransac_filter_plane(pts, cols)
        print(f"  RANSAC 去平面后 {len(pts)} 点")
        pc.show_pointcloud(pts, cols, title="[4] RANSAC 去平面后")

        if len(pts) < 50:
            raise RuntimeError(f"分割后点太少 ({len(pts)})")

        # 聚类: 把多个物体分成独立簇 (解决多物体场景 ICP 匹配歧义)
        clusters = pc.cluster(pts, cols, eps=0.015, min_points=20, max_z_spread=0.020)
        if not clusters:
            raise RuntimeError("聚类未得到任何簇 (调 eps/min_points 或前面的过滤参数)")
        # 可视化: 每簇一种颜色 (随机色), 看聚类是否分对了
        rng = np.random.default_rng(42)
        all_pts, all_cols = [], []
        for c_pts, c_cols in clusters:
            color = (rng.random(3) * 255).astype(np.uint8)
            all_pts.append(c_pts)
            all_cols.append(np.tile(color, (len(c_pts), 1)))
        pc.show_pointcloud(np.vstack(all_pts), np.vstack(all_cols),
                           title="[5] 聚类后 (每簇一色)")

        # 逐簇 ICP 匹配该模板, 取 fitness 最高 (rmse 次之) 的簇作为目标物体。
        # size 天然区分: 大模板匹配大簇 fitness 高, 匹配中小簇 fitness 低。
        best_match, best_pts, best_cols, best_center, best_score = None, None, None, None, -1
        for ci, (c_pts, c_cols) in enumerate(clusters):
            print(f"  --- 簇 {ci+1}/{len(clusters)} ({len(c_pts)} 点) 匹配 ---")
            try:
                m = pc.icp_match(c_pts, template,
                                 max_correspondence_distance=0.012, icp_iteration=80)
            except Exception as e:
                print(f"    跳过: {e}")
                continue
            score = m.info.get("fitness", 0)
            center = m.aligned_points.mean(axis=0)
            print(f"    score(fitness)={score:.4f} center={np.round(center,4).tolist()}")
            if score > best_score:
                best_match, best_pts, best_cols, best_center, best_score = \
                    m, c_pts, c_cols, center, score
        if best_match is None:
            raise RuntimeError("所有簇都匹配失败")
        print(f"  最优簇: fitness={best_score:.4f}, 物体中心 {np.round(best_center,4).tolist()}")
        pc.show_match(best_pts, best_cols, best_match.aligned_points, center=best_center,
                      title="[6] ICP match (最优簇)")

        # grasp_pose = 物体中心 + 写死偏置, 姿态写死 (复刻原程序 control_xyz/control_rpy)
        grasp_pose = np.eye(4, dtype=np.float64)
        grasp_pose[:3, :3] = Rot.from_euler("xyz", self.GRASP_RPY_RAD).as_matrix()
        grasp_pose[:3, 3] = best_center + self.GRASP_OFFSET_M
        print(f"  grasp_pose xyz={np.round(grasp_pose[:3,3],4).tolist()} "
              f"rpy_deg={np.round(np.degrees(self.GRASP_RPY_RAD),1).tolist()}")
        return grasp_pose

    def approach(self, grasp_pose):
        """接近抓取位 (复刻原程序 step1 张手 + step2 到 grasp_pose):
          step1: 张手 READY
          step2: 移到 grasp_pose (即原 control_xyz, 已是抬高的待下降位)。"""
        print("=== approach ===")
        print(f"  step1 hand READY {self.HAND_READY}")
        self.hand.move_hand(self.HAND_READY, timeout=self.HAND_TIMEOUT)
        print(f"  step2 move to grasp_pose xyz={np.round(grasp_pose[:3,3],4).tolist()} "
              f"rpy_deg={np.round(np.degrees(self.GRASP_RPY_RAD),1).tolist()}")
        self.arm.move_arm(grasp_pose, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                          timeout=self.ARM_TIMEOUT)

    def grasp(self, grasp_pose):
        """抓取 (复刻原程序 step3 下降 + step4 抓):
          step3: 从 grasp_pose 下降 DESCEND_M 到真正抓取点
          step4: 抓手 GRASP。"""
        grasp_down = np.asarray(grasp_pose, dtype=np.float64).copy()
        grasp_down[2, 3] -= self.DESCEND_M
        self._last_grasp_down_T = grasp_down   # 记录给 place() 用
        print("=== grasp ===")
        print(f"  step3 descend {self.DESCEND_M*1000:.0f}mm to xyz={np.round(grasp_down[:3,3],4).tolist()}")
        self.arm.move_arm(grasp_down, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                          timeout=self.ARM_TIMEOUT)
        print(f"  step4 hand GRASP {self.HAND_GRASP}")
        self.hand.move_hand(self.HAND_GRASP, timeout=self.HAND_TIMEOUT)

    def place(self):
        """放置 (复刻原程序 step5~9, 放置位写死 PLACE_XYZ/PLACE_RPY):
          step5: 从抓取位抬起 RAISE_M
          step6: 移到放置位上方 (固定 xy + 抬起 z, PLACE_RPY)
          step7: 下降 PLACE_DESCEND_M 到放置点
          step8: 松手 RELEASE
          step9: 抬回放置位上方。"""
        if not hasattr(self, "_last_grasp_down_T"):
            raise RuntimeError("place 需先调 grasp() 记录抓取位")
        grasp_down = self._last_grasp_down_T
        print("=== place ===")
        # step5 抬起 (从抓取位 + RAISE_M, 保持抓取 xy/rpy)
        raise_T = grasp_down.copy()
        raise_T[2, 3] += self.RAISE_M
        print(f"  step5 raise {self.RAISE_M*1000:.0f}mm to z={raise_T[2,3]:.4f}")
        self.arm.move_arm(raise_T, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                          timeout=self.ARM_TIMEOUT)
        # step6 移到放置位上方 (固定 xy, z=抬起高度, 固定放置 rpy)
        place_above_T = np.eye(4, dtype=np.float64)
        place_above_T[:3, :3] = Rot.from_euler("xyz", self.PLACE_RPY_RAD).as_matrix()
        place_above_T[:3, 3] = [self.PLACE_XYZ[0], self.PLACE_XYZ[1], raise_T[2, 3]]
        print(f"  step6 place_above xyz={np.round(place_above_T[:3,3],4).tolist()} "
              f"rpy_deg={np.round(np.degrees(self.PLACE_RPY_RAD),1).tolist()}")
        self.arm.move_arm(place_above_T, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                          timeout=self.ARM_TIMEOUT)
        # step7 下降到放置点
        place_T = place_above_T.copy()
        place_T[2, 3] -= self.PLACE_DESCEND_M
        print(f"  step7 place descend {self.PLACE_DESCEND_M*1000:.0f}mm to z={place_T[2,3]:.4f}")
        self.arm.move_arm(place_T, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                          timeout=self.ARM_TIMEOUT)
        # step8 松手
        print(f"  step8 hand RELEASE {self.HAND_RELEASE}")
        self.hand.move_hand(self.HAND_RELEASE, timeout=self.HAND_TIMEOUT)
        # step9 抬回放置位上方
        print(f"  step9 raise back to place_above")
        self.arm.move_arm(place_above_T, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                          timeout=self.ARM_TIMEOUT)


class GraspModelBased(Grasp):
    """in the future we can implement this class by utilizing graspnet or other model based method."""
    def __init__(self, arm, hand, model) -> None:
        super().__init__(arm, hand)
        self.model = model

    def get_grasp_pose(self, pointcloud, T_ee2object):
        """Get grasp pose in robot base frame. (use model to predict grasp pose)"""
        raise NotImplementedError


if __name__ == "__main__":
    # example for using GraspTemplateBased
    import os
    from scipy.spatial.transform import Rotation as Rot
    from .lcm_commander import LcmCommander
    from .._hardware import get_camera_class

    TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    # T_ee2object: ee 相对物体中心的位姿 (含 z 方向厚度修正 + ee 偏置/姿态)
    # 例: 上方斜抓, z 抬高 0.23+thickness/2, rpy=(-55,0,90)
    def _ee2object(thickness_m):
        T = np.eye(4)
        T[:3, :3] = Rot.from_euler("xyz", [-55, 0, 90], degrees=True).as_matrix()
        T[:3, 3] = [-0.07, 0.022, 0.23 + thickness_m / 2]
        return T

    # create arm and hand control (LCM commander 兼任 arm/hand)
    with LcmCommander() as cmd:
        CameraCls = get_camera_class()
        cam = CameraCls()
        # create grasp class
        g = GraspTemplateBased(arm=cmd, hand=cmd, T_CAM2GRIPPER=None, camera=cam)
        # warm up
        g.warm_up()
        # scan object point cloud
        scan_poses = []   # 填入实际扫描位姿 (4x4)
        pointcloud = g.scan(scan_poses)
        # 遍历模板依次抓放
        for tpl in sorted(os.listdir(TEMPLATES_DIR)):
            if not tpl.endswith(".ply"):
                continue
            thickness = 0.035   # 按 tpl 文件名查表得真实厚度
            grasp_pose = g.get_grasp_pose(pointcloud, os.path.join(TEMPLATES_DIR, tpl),
                                          _ee2object(thickness))
            g.approach(grasp_pose)
            g.grasp(grasp_pose)
            g.place()
        # cool down
        g.cool_down()
        cam.release()
