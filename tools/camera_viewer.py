#!/usr/bin/env python3
"""
相机查看 tool —— 实时显示相机画面。

用哪个相机由 config.yaml 的 camera 字段决定 (懒加载, 见 tools/_hardware.py)。
与手眼标定共用同一套 camera 抽象 (get_frame() -> BGR)。当前显示 RGB 彩色;
后续可扩展 depth/点云 (按 "有什么显示什么" 的原则, 由 camera 子类暴露的能力决定)。

运行:
    python main.py camera          # 经顶层入口
    python -m tools.camera_viewer  # 或直接作为模块
按 q / ESC 退出。
"""
import argparse

import cv2

from ._hardware import get_camera_class


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="相机实时查看 (RGB)")
    ap.add_argument("--win", default="camera viewer (q/ESC to quit)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    CameraImpl = get_camera_class()
    if CameraImpl is None:
        raise RuntimeError("config.yaml 未配置 camera (或为 null), 无法启动相机查看")
    print(f"=== 相机查看 (RGB): {CameraImpl.__name__} ===")
    cam = CameraImpl()
    if hasattr(cam, "K"):
        print("K =\n", cam.K)
    cv2.namedWindow(args.win, cv2.WINDOW_NORMAL)
    try:
        while True:
            bgr = cam.get_frame()
            if bgr is None:
                continue
            cv2.imshow(args.win, bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q 或 ESC
                break
    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        cv2.destroyAllWindows()
        cam.release()


if __name__ == "__main__":
    main()


