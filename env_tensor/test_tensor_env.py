# -*- coding: utf-8 -*-

import torch
import time
import os
import sys

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入张量环境管理器
from env_tensor.tensor_env_manager import TensorEnvManager

def test_tensor_env(num_envs=100, steps=50, device='cuda'):
    """测试张量环境的基本功能
    
    Args:
        num_envs: 并行环境数量
        steps: 测试步数
        device: 计算设备
    """
    print(f"===== 张量环境基础功能测试 =====")
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
    
    # 测量初始化时间
    print("\n开始初始化环境...")
    start_time = time.time()
    env = TensorEnvManager(num_envs=num_envs, device=device)
    init_time = time.time() - start_time
    print(f"环境初始化完成，耗时: {init_time:.4f}秒")
    
    # 测试环境重置
    print("\n测试环境重置...")
    reset_start = time.time()
    observations = env.reset()
    reset_time = time.time() - reset_start
    print(f"环境重置完成，耗时: {reset_time:.4f}秒")
    
    # 检查观察空间格式和数值范围
    print("\n检查观察空间...")
    check_observations(observations, num_envs)
    
    # 运行环境并检查状态
    print("\n测试环境运行...")
    step_times = []
    rewards_history = []
    done_counts = []
    
    for step in range(steps):
        # 生成随机动作
        actions = generate_random_actions(num_envs, device)
        
        # 执行步骤
        step_start = time.time()
        observations, rewards, dones, info = env.step(actions)
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # 记录结果
        rewards_history.append((rewards['red'].mean().item(), rewards['blue'].mean().item()))
        done_counts.append(dones.sum().item())
        
        # 每10步打印一次状态
        if (step + 1) % 10 == 0 or step == 0:
            print(f"步骤 {step+1}/{steps}:")
            print(f"  耗时: {step_time:.4f}秒")
            print(f"  红方平均奖励: {rewards['red'].mean().item():.4f}")
            print(f"  蓝方平均奖励: {rewards['blue'].mean().item():.4f}")
            print(f"  已完成环境数: {dones.sum().item()}/{num_envs}")
            
            # 检查飞机状态
            check_aircraft_state(observations)
    
    # 计算性能指标
    total_step_time = sum(step_times)
    avg_step_time = total_step_time / steps
    steps_per_second = steps / total_step_time
    envs_steps_per_second = steps_per_second * num_envs
    
    print("\n===== 性能统计 =====")
    print(f"环境初始化时间: {init_time:.4f}秒")
    print(f"环境重置时间: {reset_time:.4f}秒")
    print(f"总运行时间（步骤）: {total_step_time:.4f}秒")
    print(f"平均每步耗时: {avg_step_time:.4f}秒")
    print(f"每秒步数: {steps_per_second:.2f}")
    print(f"每秒环境步数（所有环境）: {envs_steps_per_second:.2f}")
    
    # 显示GPU内存使用（如果适用）
    if device == 'cuda':
        print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"GPU内存占用峰值: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    
    # 打印结果摘要
    print("\n===== 结果摘要 =====")
    print(f"最终完成环境数: {done_counts[-1]}/{num_envs} ({done_counts[-1]/num_envs*100:.2f}%)")
    
    # 关闭环境
    env.close()
    print("\n测试完成")

def generate_random_actions(num_envs, device):
    """生成随机动作
    
    Args:
        num_envs: 环境数量
        device: 计算设备
        
    Returns:
        dict: 动作字典
    """
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
    
    return {'red': red_actions, 'blue': blue_actions}

def check_observations(observations, num_envs):
    """检查观察空间格式和数值范围
    
    Args:
        observations: 观察空间字典
        num_envs: 环境数量
    """
    # 检查格式
    if not isinstance(observations, dict):
        print("错误：观察空间不是字典类型")
        return
        
    if 'red' not in observations or 'blue' not in observations:
        print("错误：观察空间缺少红方或蓝方数据")
        return
    
    # 检查数值范围（红方）
    red_obs = observations['red']
    
    print(f"红方观察空间包含 {len(red_obs)} 个键")
    print(f"观察空间形状：")
    
    # 打印每个键的形状
    for key, value in red_obs.items():
        shape_ok = value.shape[0] == num_envs
        print(f"  {key}: {value.shape} {'✓' if shape_ok else '✗'}")
        
        # 检查值范围
        if torch.isnan(value).any():
            print(f"  警告：{key} 包含 NaN 值")
        if torch.isinf(value).any():
            print(f"  警告：{key} 包含无穷值")
    
    print("观察空间检查完成")

def check_aircraft_state(observations):
    """检查飞机状态的合理性
    
    Args:
        observations: 观察空间字典
    """
    # 检查速度范围
    red_speed = observations['red']['speed']
    blue_speed = observations['blue']['speed']
    
    if torch.any(red_speed < 0):
        print("警告：红方存在负速度")
    if torch.any(blue_speed < 0):
        print("警告：蓝方存在负速度")
    
    # 打印状态摘要
    print(f"  红方速度范围: {red_speed.min().item():.1f}-{red_speed.max().item():.1f} m/s")
    print(f"  蓝方速度范围: {blue_speed.min().item():.1f}-{blue_speed.max().item():.1f} m/s")
    
    # 检查位置范围
    red_x, red_y = observations['red']['x'], observations['red']['y']
    blue_x, blue_y = observations['blue']['x'], observations['blue']['y']
    
    battlefield_size = observations['red']['battlefield_size'][0].item()
    
    # 检查是否在战场内
    if torch.any((red_x < 0) | (red_x > battlefield_size) | (red_y < 0) | (red_y > battlefield_size)):
        print("警告：红方存在超出战场边界的情况")
    if torch.any((blue_x < 0) | (blue_x > battlefield_size) | (blue_y < 0) | (blue_y > battlefield_size)):
        print("警告：蓝方存在超出战场边界的情况")

if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='张量环境基础功能测试')
    parser.add_argument('--num_envs', type=int, default=100, 
                        help='并行环境数量，默认100')
    parser.add_argument('--steps', type=int, default=50, 
                        help='测试步数，默认50')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda',
                        help='计算设备，默认CUDA')
    
    args = parser.parse_args()
    
    # 设置OpenMP环境变量，避免警告
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 运行测试
    test_tensor_env(
        num_envs=args.num_envs,
        steps=args.steps,
        device=args.device
    ) 