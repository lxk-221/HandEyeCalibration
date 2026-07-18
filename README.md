# BasicToolBox

解耦的机器人基础工具箱。Arm / Camera 各自抽象 + 按硬件实现子类, 工具 (tools/) 复用
同一套抽象 —— 换硬件组合只需实现新的 Arm/Camera 子类, 工具代码零改动。
当前工具: 手眼标定、相机查看。核心只依赖 `cv2 + numpy`, **不引入 ROS**。

## 架构

```
BasicToolBox/
├── main.py             # 统一入口: python main.py <tool> [...]
├── config.yaml         # 当前硬件组合 (arm/camera 类名, 可为 null)
├── arm/
│   ├── arm.py          # 基类: joints, T_ee2base (必读) + move2joints/move2pose (可选)
│   └── franka.py       # Franka FR3 实现 (franky SDK, TCP 直连, 无 ROS)
├── camera/
│   ├── camera.py       # 基类: K/dist 属性 + get_frame()->BGR + release()
│   ├── realsense.py    # Intel RealSense 实现 (pyrealsense2)
│   └── orbbec.py       # Orbbec 实现 (pyorbbecsdk2)
└── tools/
    ├── _hardware.py          # 读 config.yaml + importlib 懒加载硬件子类
    ├── camera_viewer.py      # 相机实时查看 (RGB)
    ├── hand_eye/             # 手眼标定 (eye-in-hand)
    │   ├── main.py           # 入口: interactive / sequence 两种模式
    │   ├── calibration.py    # 纯数学: 棋盘检测 + PnP + cv2.calibrateHandEye + 残差
    │   └── poses_example.json
    └── grasp/                # 模板匹配抓取 (ICP) + LCM
        ├── template_based_grasp.py  # 编排: 取点云->ICP匹配->算位姿->发LCM指令
        ├── pointcloud.py     # 从 demo.py 提取的点云处理 + ICP (无 graspnet)
        ├── lcm_commander.py  # LCM 主控侧收发 (cmd_id 阻塞等 feedback)
        ├── lcm_types/        # lcm-gen 生成的消息 (arm/hand/feedback/ee_pose, 独立无嵌套)
        └── templates/        # 工件模板点云 (*.ply, 不纳入 git, 仅本地)
```

## Installation

### 1. 建环境 (推荐 Python 3.10, 机器人 SDK 适应性最强)

```bash
conda create -n btb python=3.10 -y
conda activate btb
```

核心数学依赖 (必装):

```bash
pip install opencv-python numpy scipy pyyaml
```

### 2. 按子类装硬件 SDK

各子类依赖相互独立 (解耦设计: 换硬件只改 import + 装对应库)。

#### Arm
- **Franka** (`arm/franka.py`): `pip install franky`  (TCP 直连, 无 ROS)
- **LBot** (`arm/lbot.py`): *实现暂缓* — 依赖 `rclpy` + `lbot_arm_interfaces` (ROS2 自定义 msg), 对纯 toolbox conda 环境较重。

#### Camera
- **RealSense** (`camera/realsense.py`): `pip install pyrealsense2`
- **Orbbec** (`camera/orbbec.py`): `pip install pyorbbecsdk2`  (import 名仍为 `pyorbbecsdk`)

#### Tools (按需)
- **grasp** (`tools/grasp/`): `pip install open3d lcm`
  - 点云 ICP (open3d) + LCM 通信 (lcm)
  - `lcm-gen` 命令随 `lcm` 包安装, 用于从 `*.lcm` 生成 Python 消息类型
  - LCM 用三个**独立 struct (无嵌套)**: `arm_command` / `hand_command` / `execution_feedback`。
    避免嵌套 (lcm-gen 对无 package 嵌套 struct 生成的代码有已知缺陷)。
    接收端执行完指令后回 `execution_feedback` (cmd_id 配对), 主控用它做阻塞式流程。

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

统一入口 `python main.py <tool> [tool 参数 ...]`, 也可用 `python -m tools.xxx` 单独运行某个 tool。

### 0. 相机查看 (验证相机连通)
```bash
python main.py camera
# q / ESC 退出
```

### 1. 手眼标定 - 交互式 (人移动臂, 按键采集)
```bash
python main.py hand_eye --mode interactive --robot-ip 172.16.0.8
# 预览窗口聚焦时: c=采集当前位姿  s=求解保存  q=退出
```
移动臂可通过 Franka Desk guiding 或 Xbox 遥操脚本 (另一仓库), 程序只负责读取位姿。

### 2. 手眼标定 - 序列式 (程序自动移动+采集+求解)
```bash
python main.py hand_eye --mode sequence --sequence tools/hand_eye/poses_example.json --robot-ip 172.16.0.8
```
序列 JSON 每项二选一: `{"joints": [7 个 rad]}` 或 `{"pose": 4x4 平移米}`。

### 3. 模板匹配抓取 (ICP + LCM)
```bash
python main.py grasp --workpiece doc/hex_hole.ply --t-gripper2base T.npy
```
流程: OrbbecCamera 取点云 -> 距离过滤 + RANSAC 去平面 -> ICP 模板匹配 -> 工件中心变到 base 系 -> 算 arm/hand 位姿 -> 经 LCM 发指令 (张手→移臂→抓取, 每步阻塞等接收端反馈)。`--no-send` 只算不发。
> 硬件控制不在本仓库: 另起一个依赖 ROS 的程序订阅 LCM (`arm_command`/`hand_command`), 用 lx_useful 控硬件, 执行完发 `execution_feedback` (LCM 作为 toolbox 与 ROS 的隔离层)。
> 手眼标定 `T_cam2gripper` 当前写死在 `tools/grasp/template_based_grasp.py`, 换相机/臂需重新标定后改它。
> LCM 消息改了定义: 见 `tools/grasp/lcm_types/__init__.py` 顶部注释的重新生成命令。

### 4. 换硬件组合
改 `config.yaml` 的 `arm` / `camera` 字段即可, 无需改任何代码 (懒加载, 见 `tools/_hardware.py`):
```yaml
arm: Franka          # Franka / LBot / null
camera: OrbbecCamera # OrbbecCamera / RealSenseCamera / null
```
留 `null` 表示该类硬件不使用 —— 对应模块不会被加载, 因此也不需要装其 SDK。
新增硬件时, 实现对应的 `Arm` / `Camera` 子类, 并在 `tools/_hardware.py` 的映射表加一行。

## Before Calibration
- 棋盘格: calib.io 8x11 板 = 内角点 7x10, 方格 15mm (在 `tools/hand_eye/calibration.py` 顶部 `PATTERN` / `SQUARE_M` 调整)。
- 单位统一为 **米**。
- 相机内参从相机出厂/驱动读取 (RealSense 从 stream profile 读; Orbbec 构造时可传入标定值覆盖出厂值)。
