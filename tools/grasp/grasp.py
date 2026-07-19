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
        """关节空间从零位移动到扫描位附近 (避奇异/碰撞)。"""
        print("=== warm_up ===")
        for i, j in enumerate(self.warmup_joints, 1):
            print(f"  [{i}/{len(self.warmup_joints)}] {np.round(j,3).tolist()}")
            self.arm.move_joints(j, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                                 timeout=self.ARM_TIMEOUT)

    def cool_down(self):
        """关节空间回零位 (warm_up 倒序)。"""
        print("=== cool_down ===")
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

    # grasp 偏置与姿态 (写死, 复刻 xyz_bak camera_graspnet_rpc_client.py 的 control_xyz/rpy)。
    # 数值与原程序完全一致, 便于验证 ICP 与原 GraspNet 工件匹配结果等价。
    GRASP_OFFSET_M = np.array([-0.07, 0.022, 0.23], dtype=np.float64)
    GRASP_RPY_RAD = np.array([-55.0, 0.0, 90.0]) * np.pi / 180.0

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
        pts, cols = pc.range_filter(pts, cols, x_min=0.4, z_min=-0.55)
        print(f"  范围过滤后 {len(pts)} 点")
        pc.show_pointcloud(pts, cols, title="[3] 范围过滤后")
        pts, cols, _plane = pc.ransac_filter_plane(pts, cols)
        print(f"  RANSAC 去平面后 {len(pts)} 点")
        pc.show_pointcloud(pts, cols, title="[4] RANSAC 去平面后")

        if len(pts) < 50:
            raise RuntimeError(f"分割后点太少 ({len(pts)})")
        match = pc.icp_match(pts, template,
                             max_correspondence_distance=0.012, icp_iteration=80)
        center = match.aligned_points.mean(axis=0)
        print(f"  物体中心 (base) {np.round(center,4).tolist()}")
        pc.show_match(pts, cols, match.aligned_points, center=center)

        # grasp_pose = 物体中心 + 写死偏置, 姿态写死 (复刻原程序 control_xyz/control_rpy)
        grasp_pose = np.eye(4, dtype=np.float64)
        grasp_pose[:3, :3] = Rot.from_euler("xyz", self.GRASP_RPY_RAD).as_matrix()
        grasp_pose[:3, 3] = center + self.GRASP_OFFSET_M
        print(f"  grasp_pose xyz={np.round(grasp_pose[:3,3],4).tolist()} "
              f"rpy_deg={np.round(np.degrees(self.GRASP_RPY_RAD),1).tolist()}")
        return grasp_pose

    def approach(self, grasp_pose):
        """接近抓取位: 张手 -> 移到 grasp_pose 上方 (APPROACH_PRE_Z_M)。"""
        above = np.asarray(grasp_pose, dtype=np.float64).copy()
        above[2, 3] += self.APPROACH_PRE_Z_M
        print(f"=== approach (上方 {self.APPROACH_PRE_Z_M*1000:.0f}mm) ===")
        self.hand.move_hand(self.HAND_READY, timeout=self.HAND_TIMEOUT)
        self.arm.move_arm(above, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                          timeout=self.ARM_TIMEOUT)

    def grasp(self, grasp_pose):
        """抓取: 下降到 grasp_pose -> 抓手。"""
        print("=== grasp (下降+抓) ===")
        self.arm.move_arm(grasp_pose, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                          timeout=self.ARM_TIMEOUT)
        self.hand.move_hand(self.HAND_GRASP, timeout=self.HAND_TIMEOUT)

    def place(self, place_pose):
        """放置: 抬起 -> 移到 place_pose 上方 -> 下降 -> 松手 -> 抬回上方。
        place_pose: 放置点 4x4 (base 系); 内部自动加抬起/下降高度。"""
        above = np.asarray(place_pose, dtype=np.float64).copy()
        above[2, 3] += self.PLACE_DESCEND_M   # 上方 = 放置点 + 下降量
        print("=== place (抬->放置位->松手->抬回) ===")
        # 抬起 (保持当前位姿只升 z) - 用 place 上方作为安全高度
        self.arm.move_arm(above, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                          timeout=self.ARM_TIMEOUT)
        self.arm.move_arm(place_pose, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
                          timeout=self.ARM_TIMEOUT)
        self.hand.move_hand(self.HAND_RELEASE, timeout=self.HAND_TIMEOUT)
        self.arm.move_arm(above, speed=self.ARM_SPEED, accel=self.ARM_ACCEL,
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
            # place_pose: 放置位 (base 系)
            place_pose = np.eye(4)
            place_pose[:3, :3] = Rot.from_euler("xyz", [-60, 0, 145], degrees=True).as_matrix()
            place_pose[:3, 3] = [0.262, -0.09, -0.22]
            g.place(place_pose)
        # cool down
        g.cool_down()
        cam.release()
