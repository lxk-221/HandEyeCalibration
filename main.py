#!/usr/bin/env python3
"""
BasicToolBox 统一入口。

用法:
    python main.py <tool> [tool 的参数 ...]

可用 tool:
    camera      相机实时查看 (RGB)
    hand_eye    手眼标定 (eye-in-hand)
    grasp       模板匹配抓取 (ICP) + LCM 发布

各 tool 也可单独运行:
    python -m tools.camera_viewer
    python -m tools.hand_eye.main
"""
import argparse
import sys


def main():
    # 顶层只解析 tool 名, 剩余参数原样转发给对应 tool 的 main()。
    ap = argparse.ArgumentParser(
        prog="basictoolbox",
        description="BasicToolBox 统一入口",
        usage="python main.py <tool> [tool args ...]",
    )
    ap.add_argument("tool", choices=["camera", "hand_eye", "grasp"], help="要运行的 tool")
    args, rest = ap.parse_known_args()

    if args.tool == "camera":
        from tools.camera_viewer import main as run
        run(rest)
    elif args.tool == "hand_eye":
        from tools.hand_eye.main import main as run
        run(rest)
    elif args.tool == "grasp":
        from tools.grasp.template_based_grasp import main as run
        run(rest)


if __name__ == "__main__":
    main()
