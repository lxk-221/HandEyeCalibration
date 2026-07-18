"""
硬件懒加载器: 读 config.yaml -> 按 class 名动态 import 对应子类。

职责分离:
  - config.yaml  选硬件 (class 名字符串)
  - importlib    懒加载类 (只在调用时才真正 import 该模块)
  - tool 脚本    用 argparse 参数实例化

真懒加载: 配置为 null 时连 importlib 都不调用 —— 未使用的硬件 SDK 不必安装。
"""
import importlib
import os
from typing import Optional

import yaml

# 项目根 (本文件上两级 BasicToolBox/)。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")

# class 名 -> 模块名 的显式映射 (新增子类时加一行, 比字符串推断稳)。
_ARM_MODULES = {"Franka": "franka", "LBot": "lbot"}
_CAMERA_MODULES = {"OrbbecCamera": "orbbec", "RealSenseCamera": "realsense"}


def load_config() -> dict:
    """读取项目根的 config.yaml。文件缺失或为空返回空 dict。"""
    if not os.path.exists(_CONFIG_PATH):
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _load_class(kind: str, mapping: dict, package: str) -> Optional[type]:
    """通用加载: 读 yaml 的 kind 字段 -> 按 mapping 找模块名 -> importlib 取类。
    null / 缺省 -> None; 名字不在映射表 -> ValueError; import 失败 -> RuntimeError(带提示)。"""
    cfg = load_config()
    name = cfg.get(kind)
    if name is None:
        return None
    module_name = mapping.get(name)
    if module_name is None:
        raise ValueError(
            f"config.yaml 的 {kind}: '{name}' 不在可选值 {list(mapping)} 中"
        )
    try:
        mod = importlib.import_module(f"{package}.{module_name}")
    except Exception as exc:
        raise RuntimeError(
            f"加载 {kind}='{name}' 失败 ({type(exc).__name__}: {exc})。"
            f"请确认已安装对应 SDK, 或改 config.yaml 换一个已装的硬件。"
        ) from exc
    return getattr(mod, name)


def get_arm_class() -> Optional[type]:
    """返回 config.yaml 配置的 Arm 子类; 未配置返回 None。"""
    return _load_class("arm", _ARM_MODULES, "arm")


def get_camera_class() -> Optional[type]:
    """返回 config.yaml 配置的 Camera 子类; 未配置返回 None。"""
    return _load_class("camera", _CAMERA_MODULES, "camera")
