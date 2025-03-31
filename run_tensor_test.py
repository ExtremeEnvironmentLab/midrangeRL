# -*- coding: utf-8 -*-

import argparse
import time
import os
import sys

# 设置OpenMP环境变量，避免警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def run_tensor_test(test_type, num_envs=1000, steps=10, device='cuda', num_trials=1):
    """运行不同类型的张量环境测试
    
    Args:
        test_type: 测试类型，可以是'basic', 'million', 'compare'
        num_envs: 并行环境数量
        steps: 测试步数
        device: 计算设备，'cuda'或'cpu'
        num_trials: 重复测试次数，计算平均性能
    """
    print(f"===== 运行 {test_type} 测试 =====")
    print(f"配置: {num_envs}个环境, {steps}步, 设备: {device}, {num_trials}次重复")
    
    if test_type == 'basic':
        # 基础功能测试
        print("\n运行基础功能测试...")
        from env_tensor.test_tensor_env import test_tensor_env
        test_tensor_env(num_envs=num_envs, steps=steps, device=device)
        
    elif test_type == 'million':
        # 百万级环境测试
        print("\n运行百万级环境测试...")
        from env_tensor.test_million_envs import test_million_envs
        
        # 运行多次测试并计算平均性能
        all_results = []
        for i in range(num_trials):
            print(f"\n试验 {i+1}/{num_trials}")
            start_time = time.time()
            test_million_envs(num_envs=num_envs, steps=steps, device=device)
            end_time = time.time()
            total_time = end_time - start_time
            all_results.append(total_time)
            
        # 打印平均性能
        if num_trials > 1:
            avg_time = sum(all_results) / num_trials
            print(f"\n平均总运行时间 ({num_trials}次): {avg_time:.4f}秒")
            print(f"每秒环境步数: {num_envs * steps / avg_time:.2f}")
            
    elif test_type == 'compare':
        # CPU和GPU性能比较测试
        print("\n运行CPU/GPU性能比较测试...")
        from env_tensor.test_million_envs import test_million_envs
        
        # CPU测试
        if 'cpu' in device.lower() or device.lower() == 'both':
            print("\n=== CPU测试 ===")
            cpu_start = time.time()
            test_million_envs(num_envs=num_envs, steps=steps, device='cpu')
            cpu_time = time.time() - cpu_start
            print(f"CPU总运行时间: {cpu_time:.4f}秒")
            
        # GPU测试
        if 'cuda' in device.lower() or device.lower() == 'both':
            print("\n=== GPU测试 ===")
            gpu_start = time.time()
            test_million_envs(num_envs=num_envs, steps=steps, device='cuda')
            gpu_time = time.time() - gpu_start
            print(f"GPU总运行时间: {gpu_time:.4f}秒")
            
        # 打印加速比
        if 'both' == device.lower():
            speedup = cpu_time / gpu_time
            print(f"\nGPU加速比: {speedup:.2f}x")
            
    else:
        print(f"未知测试类型: {test_type}")
        print("可用测试: basic, million, compare")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='向量环境性能测试')
    parser.add_argument('--test', type=str, default='basic', choices=['basic', 'million', 'compare'],
                        help='测试类型: basic, million, compare')
    parser.add_argument('--num_envs', type=int, default=1000,
                        help='并行环境数量，默认1000')
    parser.add_argument('--steps', type=int, default=10,
                        help='运行步数，默认10')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备，可选 cuda, cpu, both (仅用于compare测试)')
    parser.add_argument('--trials', type=int, default=1,
                        help='重复测试次数，计算平均性能')
    
    args = parser.parse_args()
    
    run_tensor_test(
        test_type=args.test,
        num_envs=args.num_envs,
        steps=args.steps,
        device=args.device,
        num_trials=args.trials
    ) 