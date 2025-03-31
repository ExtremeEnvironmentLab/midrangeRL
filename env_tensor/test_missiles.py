# -*- coding: utf-8 -*-

import torch
import time
import os
import sys

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入张量环境管理器
from env_tensor.tensor_env_manager import TensorEnvManager

def test_missiles(num_envs=50, steps=100, device='cuda'):
    """测试导弹发射和制导功能
    
    Args:
        num_envs: 并行环境数量
        steps: 测试步数
        device: 计算设备
    """
    print(f"===== 导弹功能测试 =====")
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
    
    # 强制红蓝双方初始位置较近
    print("设置红蓝双方初始位置更接近...")
    
    # 设置红方位置在战场中心附近
    center = env.battlefield_size / 2
    radius = env.battlefield_size / 10  # 战场大小的10%
    
    # 使用随机偏移确保随机性
    red_offset_x = torch.rand(num_envs, device=device) * radius - radius/2
    red_offset_y = torch.rand(num_envs, device=device) * radius - radius/2
    
    # 设置红方位置
    env.red_fighters['x'] = center + red_offset_x
    env.red_fighters['y'] = center + red_offset_y
    
    # 设置蓝方位置，与红方相距不远
    blue_offset_x = torch.rand(num_envs, device=device) * radius
    blue_offset_y = torch.rand(num_envs, device=device) * radius
    
    # 蓝方位置 = 红方位置 + 小偏移
    env.blue_fighters['x'] = env.red_fighters['x'] + blue_offset_x
    env.blue_fighters['y'] = env.red_fighters['y'] + blue_offset_y
    
    # 重新获取观察空间
    observations = env.get_observations()
    
    reset_time = time.time() - reset_start
    print(f"环境重置完成，耗时: {reset_time:.4f}秒")
    
    # 打印初始距离
    red_x = env.red_fighters['x'][0].item()
    red_y = env.red_fighters['y'][0].item()
    blue_x = env.blue_fighters['x'][0].item()
    blue_y = env.blue_fighters['y'][0].item()
    
    initial_distance = ((red_x - blue_x)**2 + (red_y - blue_y)**2)**0.5
    print(f"红蓝双方初始距离（示例）: {initial_distance:.2f}米")
    
    # 运行环境并检查状态
    print("\n测试导弹发射与制导...")
    step_times = []
    rewards_history = []
    done_counts = []
    
    # 记录发射的导弹和命中情况
    missiles_fired = 0
    missiles_hit = 0
    
    for step in range(steps):
        # 在第1步和第50步让所有环境发射导弹
        if step == 1 or step == 50:
            red_fire = torch.ones(num_envs, dtype=torch.bool, device=device)
            blue_fire = torch.ones(num_envs, dtype=torch.bool, device=device)
            print(f"步骤 {step+1}: 所有飞机发射导弹")
            missiles_fired += 2 * num_envs  # 红蓝双方各发射num_envs枚导弹
        else:
            # 随机发射导弹（10%概率）
            red_fire = torch.rand(num_envs, device=device) > 0.9
            blue_fire = torch.rand(num_envs, device=device) > 0.9
            
            # 统计发射的导弹数量
            missiles_fired += red_fire.sum().item() + blue_fire.sum().item()
        
        # 生成动作
        actions = {
            'red': {
                'throttle': torch.rand(num_envs, device=device),
                'rudder': torch.rand(num_envs, device=device) * 2 - 1,  # -1到1的范围
                'fire': red_fire
            },
            'blue': {
                'throttle': torch.rand(num_envs, device=device),
                'rudder': torch.rand(num_envs, device=device) * 2 - 1,
                'fire': blue_fire
            }
        }
        
        # 执行步骤
        step_start = time.time()
        observations, rewards, dones, info = env.step(actions)
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # 记录命中数据（通过奖励判断）
        if isinstance(rewards['red'], dict) and 'hit_reward' in rewards['red']:
            hits_this_step = (rewards['red']['hit_reward'] > 0).sum().item() + (rewards['blue']['hit_reward'] > 0).sum().item()
            missiles_hit += hits_this_step
            if hits_this_step > 0:
                print(f"步骤 {step+1}: 检测到 {hits_this_step} 次导弹命中")
        
        # 记录结果
        rewards_history.append((
            rewards['red'].mean().item() if isinstance(rewards['red'], torch.Tensor) else sum(r.mean().item() for r in rewards['red'].values()),
            rewards['blue'].mean().item() if isinstance(rewards['blue'], torch.Tensor) else sum(r.mean().item() for r in rewards['blue'].values())
        ))
        done_counts.append(dones.sum().item())
        
        # 每10步打印一次状态
        if (step + 1) % 10 == 0 or step == 0:
            print(f"步骤 {step+1}/{steps}:")
            print(f"  耗时: {step_time:.4f}秒")
            
            # 处理复合奖励（如果是字典形式）
            if isinstance(rewards['red'], dict):
                red_reward = sum(r.mean().item() for r in rewards['red'].values())
                blue_reward = sum(r.mean().item() for r in rewards['blue'].values())
            else:
                red_reward = rewards['red'].mean().item()
                blue_reward = rewards['blue'].mean().item()
                
            print(f"  红方平均奖励: {red_reward:.4f}")
            print(f"  蓝方平均奖励: {blue_reward:.4f}")
            print(f"  已完成环境数: {dones.sum().item()}/{num_envs}")
            
            # 获取导弹状态
            active_missiles = sum(1 for m in env.missiles if m is not None and m['active'].any())
            print(f"  活跃导弹数: {active_missiles}")
            
            # 添加调试信息
            if step % 20 == 0 and active_missiles > 0:
                # 打印导弹和飞机的距离
                for i, missile in enumerate(env.missiles):
                    if missile is not None and missile['active'].any():
                        # 获取第一个活跃导弹的索引
                        missile_idx = torch.where(missile['active'])[0][0].item()
                        target_type = "red" if missile['is_player1'][missile_idx] else "blue"  # 导弹的目标类型
                        targets = env.blue_fighters if target_type == "blue" else env.red_fighters
                        
                        # 计算导弹到所有目标的距离
                        dx = missile['x'][missile_idx] - targets['x']
                        dy = missile['y'][missile_idx] - targets['y']
                        distance = torch.sqrt(dx*dx + dy*dy)
                        
                        # 获取最近的目标
                        min_distance, min_idx = torch.min(distance, dim=0)
                        
                        print(f"  导弹 {i}, 位置: ({missile['x'][missile_idx]:.1f}, {missile['y'][missile_idx]:.1f}), " +
                              f"速度: {missile['speed'][missile_idx]:.1f}, " +
                              f"到最近{target_type}方目标距离: {min_distance.item():.1f}")
                        
                        # 检查是否有目标索引
                        if 'target_idx' in missile:
                            target_idx = missile['target_idx'][missile_idx].item()
                            if target_idx >= 0:
                                target_distance = distance[target_idx].item()
                                print(f"    指定目标 {target_idx} 距离: {target_distance:.1f}")
    
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
    
    # 打印导弹统计
    print("\n===== 导弹统计 =====")
    print(f"发射导弹总数: {missiles_fired}")
    print(f"命中导弹总数: {missiles_hit}")
    print(f"命中率: {missiles_hit / max(1, missiles_fired) * 100:.2f}%")
    
    # 打印结果摘要
    print("\n===== 结果摘要 =====")
    print(f"最终完成环境数: {done_counts[-1]}/{num_envs} ({done_counts[-1]/num_envs*100:.2f}%)")
    
    # 关闭环境
    env.close()
    print("\n测试完成")

if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='导弹功能测试')
    parser.add_argument('--num_envs', type=int, default=50, 
                        help='并行环境数量，默认50')
    parser.add_argument('--steps', type=int, default=100, 
                        help='测试步数，默认100')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda',
                        help='计算设备，默认CUDA')
    
    args = parser.parse_args()
    
    # 设置OpenMP环境变量，避免警告
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 运行测试
    test_missiles(
        num_envs=args.num_envs,
        steps=args.steps,
        device=args.device
    ) 