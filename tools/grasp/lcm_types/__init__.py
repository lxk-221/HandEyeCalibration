"""LCM 生成的消息类型 (由 lcm-gen 从 *.lcm 生成)。

三个独立 struct (无嵌套):
  arm_command          主控 -> 接收端 (机械臂运动)
  hand_command         主控 -> 接收端 (灵巧手运动)
  execution_feedback   接收端 -> 主控 (执行完成反馈, 用 cmd_id 配对阻塞)

重新生成 (在本目录): lcm-gen -p arm_command.lcm hand_command.lcm execution_feedback.lcm

用法 (lcm_publisher 内部): sys.path 注入本目录 -> import arm_command -> arm_command.arm_command()
(模块名与类名同名, 用 模块.类 访问; 这是 Python 标准行为, 非 lcm bug)。
"""
