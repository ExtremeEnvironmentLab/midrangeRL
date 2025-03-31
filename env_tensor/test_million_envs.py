# -*- coding: utf-8 -*-

import torch
import time
import os
import sys

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入优化后的环境管理器
from env_tensor.tensor_env_manager import TensorEnvManager

def test_million_envs(num_envs=1000000, steps=10, device='cuda'):
    """测试超大规模并行环境
    
    Args:
        num_envs: 并行环境数量，默认100万
        steps: 运行步数，默认10
        device: 计算设备，默认CUDA
    """
    print(f"===== 百万级环境测试 - 向量物理版 =====")
    print(f"并行环境数量: {num_envs}")
    print(f"测试步数: {steps}")
    print(f"设备: {device}")
    
    # 显示系统信息
    print("\n===== 系统信息 =====")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    
    # 预热GPU
    if device == 'cuda':
        print("\n预热GPU...")
        x = torch.randn(2000, 2000, device=device)
        y = torch.randn(2000, 2000, device=device)
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("GPU预热完成")
    
    # 测量初始化时间
    print("\n开始初始化环境...")
    start_time = time.time()
    env = TensorEnvManager(num_envs=num_envs, device=device)
    init_time = time.time() - start_time
    print(f"环境初始化完成，耗时: {init_time:.4f}秒")
    
    # 显示当前GPU内存使用
    if device == 'cuda':
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"GPU内存占用峰值: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    
    # 准备批量动作
    print("\n预生成所有动作...")
    all_actions = []
    
    for _ in range(steps):
        red_actions = {
            'throttle': torch.rand(num_envs, device=device),
            'rudder': torch.rand(num_envs, device=device) * 2 - 1,  # -1到1的范围
            'fire': torch.rand(num_envs, device=device) > 0.9  # 10%的概率发射导弹
        }
        
        blue_actions = {
            'throttle': torch.rand(num_envs, device=device),
            'rudder': torch.rand(num_envs, device=device) * 2 - 1,
            'fire': torch.rand(num_envs, device=device) > 0.9
        }
        
        actions = {'red': red_actions, 'blue': blue_actions}
        all_actions.append(actions)
    
    # 如果使用CUDA，同步以确保所有数据都已加载到GPU
    if device == 'cuda':
        torch.cuda.synchronize()
    
    print("动作生成完成")
    
    # 运行环境
    print("\n开始执行环境步骤...")
    step_times = []
    
    for step in range(steps):
        # 使用预先计算的动作
        actions = all_actions[step]
        
        # 测量步骤执行时间
        step_start = time.time()
        observations, rewards, dones, info = env.step(actions)
        
        # 如果使用CUDA，确保所有操作都已完成
        if device == 'cuda':
            torch.cuda.synchronize()
            
        step_end = time.time()
        step_time = step_end - step_start
        step_times.append(step_time)
        
        print(f"步骤 {step+1}/{steps}:")
        print(f"  耗时: {step_time:.4f}秒")
        print(f"  红方平均奖励: {rewards['red'].mean().item():.4f}")
        print(f"  蓝方平均奖励: {rewards['blue'].mean().item():.4f}")
        print(f"  已完成环境数: {dones.sum().item()}/{num_envs}")
        
        # 显示当前GPU内存使用
        if device == 'cuda':
            print(f"  当前GPU内存使用: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    
    # 计算性能指标
    total_time = sum(step_times)
    avg_step_time = total_time / steps
    steps_per_second = steps / total_time
    envs_steps_per_second = steps_per_second * num_envs
    
    print("\n===== 性能统计 =====")
    print(f"总运行时间: {total_time:.4f}秒")
    print(f"平均每步耗时: {avg_step_time:.4f}秒")
    print(f"每秒步数: {steps_per_second:.2f}")
    print(f"每秒环境步数（所有环境）: {envs_steps_per_second:.2f}")
    
    # 显示最终GPU内存使用
    if device == 'cuda':
        print(f"最终GPU内存使用: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    
    # 清理资源
    env.close()
    if device == 'cuda':
        torch.cuda.empty_cache()
        
    print("测试完成，资源已释放")

if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='百万级环境测试 - 向量物理版')
    parser.add_argument('--num_envs', type=int, default=1000000, 
                        help='并行环境数量，默认100万')
    parser.add_argument('--steps', type=int, default=10, 
                        help='运行步数，默认10')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda',
                        help='计算设备，默认CUDA')
    
    args = parser.parse_args()
    
    # 设置OpenMP环境变量，避免警告
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 运行测试
    test_million_envs(
        num_envs=args.num_envs,
        steps=args.steps,
        device=args.device
    ) 