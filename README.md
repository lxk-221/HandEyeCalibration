# Personal HandEye Calibration Repo

eye-in-hand 手眼标定 (相机装在末端, 标定板固定)。重构后只依赖 `cv2 + numpy` + 当前
组合的 Arm SDK + Camera SDK, **不引入 ROS**。换硬件组合只需实现新的 Arm/Camera 子类。

## 架构

```
HandEyeCalibration/
├── arm/
│   ├── arm.py        # 基类: joints, T_ee2base (必读) + move2joints/move2pose (可选)
│   └── franka.py     # Franka FR3 实现 (franky SDK, TCP 直连, 无 ROS)
├── camera/
│   ├── camera.py     # 基类: K/dist 属性 + get_frame()->BGR + release()
│   └── realsense.py  # Intel RealSense 实现 (pyrealsense2)
├── calibration.py    # 纯数学: 棋盘检测 + PnP + cv2.calibrateHandEye + 残差 (全程米制)
├── main.py           # 入口: interactive / sequence 两种模式
└── poses_example.json
```

## 抽象约定

**Arm** (`arm/arm.py`):
- `joints` (property, 必须): 当前关节角 rad。
- `T_ee2base` (property, 必须): `T_gripper2base`, 4x4, 平移 **米**。遵循 OpenCV 手眼标定约定。
- `move2joints` / `move2pose` (可选): 按硬件便利二选一或都实现; 未实现抛 `NotImplementedError`。
  交互式标定不需要任何移动方法; sequence 模式按序列类型自动选调用哪个。

**Camera** (`camera/camera.py`):
- `K` / `dist` (属性): 子类 `__init__` 完成硬件初始化并填好。
- `get_frame() -> BGR uint8 (H,W,3)`: 始终返回 BGR 图喂给 cv2。
- `release()`.

## 用法

### 1. 交互式标定 (人移动臂, 按键采集)
```bash
cd ~/xukun/HandEyeCalibration
python3 main.py --mode interactive --robot-ip 172.16.0.8
# 预览窗口聚焦时: c=采集当前位姿  s=求解保存  q=退出
```
移动臂可通过 Franka Desk guiding 或 Xbox 遥操脚本 (另一仓库), 程序只负责读取位姿。

### 2. 序列式标定 (程序自动移动+采集+求解)
```bash
python3 main.py --mode sequence --sequence poses_example.json --robot-ip 172.16.0.8
```
序列 JSON 每项二选一: `{"joints": [7 个 rad]}` 或 `{"pose": 4x4 平移米}`。

### 3. 换硬件组合
打开 `main.py`, 改顶部两行 import:
```python
from arm.franka     import Franka          as ArmImpl
from camera.realsense import RealSenseCamera as CameraImpl
```
并实现对应的 `Arm` / `Camera` 子类即可 (4 个方法, 见上)。无需改动 `calibration.py` / `main.py`。

## Before Calibration
- 棋盘格: calib.io 8x11 板 = 内角点 7x10, 方格 15mm (在 `calibration.py` 顶部 `PATTERN` / `SQUARE_M` 调整)。
- 单位统一为 **米**。
- 相机内参从相机出厂/驱动读取 (RealSense 从 stream profile 读; 也可改用手动标定值)。
