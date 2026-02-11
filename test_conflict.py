import sys
import os

# 读取环境变量来决定谁先加载
# 模式 "torch_first": 模拟现在的 bug (actor.py 的行为)
# 模式 "ros_first": 模拟期望的修复
mode = os.environ.get("TEST_ORDER", "torch_first")

print(f"\n========== 测试模式: {mode} ==========")

if mode == "ros_first":
    print(">>> [1] 正在加载 ROS (rclpy)...")
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    print(">>> [2] 正在加载 PyTorch...")
    import torch
else:
    print(">>> [1] 正在加载 PyTorch...")
    import torch
    print(">>> [2] 正在加载 ROS (rclpy)...")
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

print(f"--- 环境信息 ---")
print(f"Torch Version: {torch.__version__}")
print(f"RCLPY File: {rclpy.__file__}")

print(f"--- 开始 Executor 初始化测试 ---")
try:
    # 必须先 init，否则会报 Context 错误
    if not rclpy.ok():
        rclpy.init()
    
    # 关键时刻：看看能不能造出 Executor
    executor = SingleThreadedExecutor()
    
    if executor is None:
        print("❌❌❌ 测试失败：Executor 返回了 None！(ABI 冲突实锤)")
    else:
        print(f"✅✅✅ 测试成功：Executor 创建正常: {executor}")
        
    rclpy.shutdown()

except Exception as e:
    print(f"❌❌❌ 发生严重崩溃: {e}")

print("======================================\n")