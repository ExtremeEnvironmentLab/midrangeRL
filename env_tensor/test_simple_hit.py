# -*- coding: utf-8 -*-

import torch
import time
import os
import sys

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入所需模块
from env_tensor.tensor_env_manager import TensorEnvManager
from env_tensor.game_events_tensor import GameEventsTensor

def test_simple_hit(device='cuda'):
    """
    简单导弹命中测试
    
    Args:
        device: 计算设备
    """
    print(f"===== 简单导弹命中测试 =====")
    print(f"设备: {device}")
    
    # 初始化游戏事件处理器
    events = GameEventsTensor(battlefield_size=50000, device=device)
    print(f"命中距离设置为: {events.hit_distance.item():.1f}米")
    
    # 创建手动测试案例
    print("\n创建测试数据...")
    
    # 创建一个活跃的导弹
    batch_size = 1
    missile = {
        'x': torch.tensor([25000.0], device=device),      # 导弹在场地中心
        'y': torch.tensor([25000.0], device=device),
        'vx': torch.tensor([10.0], device=device),
        'vy': torch.tensor([0.0], device=device),
        'active': torch.tensor([True], device=device, dtype=torch.bool),
        'target_idx': torch.tensor([0], device=device, dtype=torch.long)
    }
    
    # 创建一个靠近导弹的目标
    # 第一个目标在命中范围内，第二个和第三个在范围外
    target = {
        'x': torch.tensor([25100.0, 26000.0, 30000.0], device=device),  # 第一个目标距离导弹100米
        'y': torch.tensor([25000.0, 25000.0, 25000.0], device=device),
        'active': torch.tensor([True, True, True], device=device, dtype=torch.bool)
    }
    
    # 计算导弹到目标的距离
    for i in range(len(target['x'])):
        dx = missile['x'][0] - target['x'][i]
        dy = missile['y'][0] - target['y'][i]
        dist = torch.sqrt(dx*dx + dy*dy).item()
        print(f"导弹到目标 {i} 的距离: {dist:.1f}米")
    
    # 测试命中检测
    print("\n测试命中检测...")
    
    # 1. 使用目标索引进行命中检测
    hit_mask = events.check_missile_hit_batch(missile, target)
    print(f"命中掩码: {hit_mask}")
    
    # 2. 更改目标索引，测试不同目标
    print("\n更改目标索引，测试不同目标...")
    missile['target_idx'] = torch.tensor([1], device=device, dtype=torch.long)
    hit_mask = events.check_missile_hit_batch(missile, target)
    print(f"命中掩码 (目标索引=1): {hit_mask}")
    
    # 3. 设置无效目标索引，测试边界情况
    print("\n设置无效目标索引...")
    missile['target_idx'] = torch.tensor([-1], device=device, dtype=torch.long)
    hit_mask = events.check_missile_hit_batch(missile, target)
    print(f"命中掩码 (无效目标索引): {hit_mask}")
    
    # 4. 逐步缩小导弹和目标的距离，直到命中
    print("\n逐步缩小距离测试...")
    missile['target_idx'] = torch.tensor([0], device=device, dtype=torch.long)
    distances = [500, 400, 300, 200, 100, 50, 25, 10]
    
    for d in distances:
        # 设置导弹和目标的位置
        missile['x'] = torch.tensor([25000.0], device=device)
        missile['y'] = torch.tensor([25000.0], device=device)
        target['x'] = torch.tensor([25000.0 + d, 26000.0, 30000.0], device=device)
        
        # 打印当前距离
        dx = missile['x'][0] - target['x'][0]
        dy = missile['y'][0] - target['y'][0]
        dist = torch.sqrt(dx*dx + dy*dy).item()
        print(f"距离: {dist:.1f}米", end=" - ")
        
        # 检测命中
        hit_mask = events.check_missile_hit_batch(missile, target)
        print(f"命中: {hit_mask.item()}")
    
    # 5. 测试应用伤害
    print("\n测试应用伤害...")
    # 设置一个肯定命中的距离
    missile['x'] = torch.tensor([25000.0], device=device)
    missile['y'] = torch.tensor([25000.0], device=device)
    target['x'] = torch.tensor([25010.0, 26000.0, 30000.0], device=device)
    target['active'] = torch.tensor([True, True, True], device=device, dtype=torch.bool)
    
    # 检查命中并应用伤害
    hit_mask = events.check_missile_hit_batch(missile, target)
    print(f"命中掩码: {hit_mask}")
    
    if hit_mask.any():
        # 选择一个特定的目标索引
        missile['target_idx'] = torch.tensor([0], device=device, dtype=torch.long)
        print(f"导弹目标索引: {missile['target_idx'].item()}")
        
        print("应用伤害到目标...")
        # 在应用伤害前保存活跃状态
        before_active = target['active'].clone()
        
        # 获取索引为0的目标的伤害掩码
        target_damage_mask = torch.zeros_like(target['active'])
        target_damage_mask[0] = True  # 只标记第一个目标
        
        # 应用伤害
        events.apply_damage_batch(target, target_damage_mask)
        
        # 打印伤害应用结果
        print(f"目标活跃状态(前): {before_active}")
        print(f"伤害掩码: {target_damage_mask}")
        print(f"目标活跃状态(后): {target['active']}")
        
        # 验证只有索引为0的目标被标记为非活跃
        for i in range(len(target['active'])):
            status = "非活跃" if not target['active'][i] else "活跃"
            print(f"目标 {i}: {status}")
    else:
        print("无命中，不应用伤害")
    
    # 6. 测试在实际TensorEnvManager中的表现
    print("\n测试在TensorEnvManager中的表现...")
    try:
        # 创建一个小型环境
        env = TensorEnvManager(num_envs=1, device=device)
        print("环境初始化成功")
        
        # 重置环境
        env.reset()
        print("环境重置成功")
        
        # 强制设置红蓝双方在很近的位置
        env.red_fighters['x'] = torch.tensor([25000.0], device=device)
        env.red_fighters['y'] = torch.tensor([25000.0], device=device)
        env.blue_fighters['x'] = torch.tensor([26000.0], device=device)  # 设置距离为1000米，超出命中半径
        env.blue_fighters['y'] = torch.tensor([25000.0], device=device)
        
        # 打印初始距离
        dx = env.red_fighters['x'][0] - env.blue_fighters['x'][0]
        dy = env.red_fighters['y'][0] - env.blue_fighters['y'][0]
        distance = torch.sqrt(dx*dx + dy*dy).item()
        print(f"红蓝双方距离: {distance:.1f}米")
        
        # 让红方发射一枚导弹
        actions = {
            'red': {
                'throttle': torch.tensor([0.5], device=device),
                'rudder': torch.tensor([0.0], device=device),
                'fire': torch.tensor([True], device=device, dtype=torch.bool)
            },
            'blue': {
                'throttle': torch.tensor([0.5], device=device),
                'rudder': torch.tensor([0.0], device=device),
                'fire': torch.tensor([False], device=device, dtype=torch.bool)
            }
        }
        
        # 确保红方有导弹可发射
        print(f"红方导弹数量: {env.red_fighters['missiles'].item()}")
        env.red_fighters['missiles'] = torch.tensor([10], device=device)
        print(f"更新后红方导弹数量: {env.red_fighters['missiles'].item()}")
        
        print("红方发射导弹...")
        observations, rewards, dones, info = env.step(actions)
        
        # 检查导弹是否创建
        missile_count = sum(1 for m in env.missiles if m is not None and m['active'].any())
        print(f"活跃导弹数量: {missile_count}")
        
        if missile_count > 0:
            # 打印导弹信息
            for i, missile in enumerate(env.missiles):
                if missile is not None and missile['active'].any():
                    print(f"导弹 {i} 信息:")
                    print(f"  位置: ({missile['x'][0].item():.1f}, {missile['y'][0].item():.1f})")
                    print(f"  速度: {missile['speed'][0].item():.1f}")
                    print(f"  目标索引: {missile['target_idx'][0].item()}")
            
            # 运行几步，看导弹是否会命中
            print("运行10步，测试导弹命中...")
            for i in range(10):
                observations, rewards, dones, info = env.step(actions)
                missile_count = sum(1 for m in env.missiles if m is not None and m['active'].any())
                print(f"步骤 {i+1}: 活跃导弹数: {missile_count}, 蓝方活跃: {env.blue_fighters['active'].item()}")
                
                # 打印每个活跃导弹的信息
                for j, missile in enumerate(env.missiles):
                    if missile is not None and missile['active'].any():
                        # 计算到蓝方的距离
                        dx = missile['x'][0] - env.blue_fighters['x'][0]
                        dy = missile['y'][0] - env.blue_fighters['y'][0]
                        distance = torch.sqrt(dx*dx + dy*dy).item()
                        print(f"  导弹 {j}: 距离蓝方 {distance:.1f}米")
                
                # 如果导弹数量变化或蓝方被击中，退出循环
                if missile_count == 0 or not env.blue_fighters['active'].item():
                    print("检测到命中或导弹已消失!")
                    break
            
            # 检查最终状态
            if not env.blue_fighters['active'].item():
                print("测试成功: 蓝方被导弹命中!")
            else:
                print("测试失败: 蓝方未被命中")
        else:
            print("测试失败: 未能创建导弹")
            # 检查红方导弹数量是否已减少
            print(f"红方剩余导弹数量: {env.red_fighters['missiles'].item()}")
            print(f"红方fire动作: {actions['red']['fire'].item()}")
            
            # 尝试直接调用导弹创建函数
            print("\n尝试直接调用导弹创建函数...")
            missile = env.events.create_missiles_batch(
                env.red_fighters,
                target_indices=torch.tensor([0], device=device, dtype=torch.long),
                batch_size=1
            )
            
            if missile is not None:
                print("直接创建导弹成功!")
                print(f"导弹位置: ({missile['x'][0].item():.1f}, {missile['y'][0].item():.1f})")
                print(f"导弹活跃状态: {missile['active'][0].item()}")
            else:
                print("直接创建导弹失败!")
            
        # 关闭环境
        env.close()
    except Exception as e:
        print(f"测试环境时出错: {e}")
    
    print("\n测试完成")

if __name__ == "__main__":
    # 设置OpenMP环境变量，避免警告
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 运行测试
    test_simple_hit(device='cuda' if torch.cuda.is_available() else 'cpu') 