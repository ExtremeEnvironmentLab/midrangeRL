# -*- coding: utf-8 -*-

import torch
import time
import math

class TensorEnvManager:
    def __init__(self, 
                 num_envs=32, 
                 battlefield_size=50000, 
                 time_step=0.1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化张量环境管理器
        
        Args:
            num_envs: 并行环境数量
            battlefield_size: 战场大小（米）
            time_step: 模拟时间步长（秒）
            device: 计算设备
        """
        init_start = time.time()
        
        self.num_envs = num_envs
        self.battlefield_size = battlefield_size
        self.time_step = time_step
        self.device = device
        
        print(f"初始化环境管理器，并行环境数量: {num_envs}，设备: {device}")
        
        # 导入环境组件
        from env_tensor.aerodynamic_tensor import AerodynamicTensor
        from env_tensor.missile_guidance_tensor import MissileGuidanceTensor
        from env_tensor.game_events_tensor import GameEventsTensor
        
        # 创建环境组件
        self.aero = AerodynamicTensor(device=device)
        self.guidance = MissileGuidanceTensor(device=device)
        self.events = GameEventsTensor(battlefield_size=battlefield_size, device=device)
        
        # 环境状态
        self.current_time = torch.zeros(num_envs, device=device)
        
        # 预分配内存给常用张量
        self.done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.winner = torch.zeros(num_envs, dtype=torch.int8, device=device)
        
        # 导弹列表
        self.missiles = []
        
        # 使用批量初始化红蓝方飞机
        self._initialize_aircraft()
        
        # 重置制导系统历史
        self.guidance.reset_guidance_history(num_envs)
        
        init_end = time.time()
        print(f"环境初始化完成，耗时: {init_end - init_start:.4f}秒")
    
    def _initialize_aircraft(self):
        """批量初始化红蓝双方飞机 - 高效向量化版本"""
        # 创建基本飞机属性 - 红方
        self.red_fighters = self.aero.create_aircraft_batch(self.num_envs, is_missile=False)
        self.red_fighters['active'] = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.red_fighters['missiles'] = torch.full((self.num_envs,), 6, device=self.device)
        self.red_fighters['is_player1'] = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 创建基本飞机属性 - 蓝方
        self.blue_fighters = self.aero.create_aircraft_batch(self.num_envs, is_missile=False)
        self.blue_fighters['active'] = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.blue_fighters['missiles'] = torch.full((self.num_envs,), 6, device=self.device)
        self.blue_fighters['is_player1'] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 使用一次性向量化操作生成随机初始位置和状态
        # 红方位置（左侧）
        self.red_fighters['x'] = torch.rand(self.num_envs, device=self.device) * 10000 + 5000
        self.red_fighters['y'] = torch.rand(self.num_envs, device=self.device) * (self.battlefield_size - 10000) + 5000
        self.red_fighters['angle'] = torch.rand(self.num_envs, device=self.device) * 360
        self.red_fighters['speed'] = torch.rand(self.num_envs, device=self.device) * 100 + 250  # 250-350
        
        # 蓝方位置（右侧）
        self.blue_fighters['x'] = torch.rand(self.num_envs, device=self.device) * 10000 + (self.battlefield_size - 15000)
        self.blue_fighters['y'] = torch.rand(self.num_envs, device=self.device) * (self.battlefield_size - 10000) + 5000
        self.blue_fighters['angle'] = torch.rand(self.num_envs, device=self.device) * 360
        self.blue_fighters['speed'] = torch.rand(self.num_envs, device=self.device) * 100 + 250  # 250-350
        
        # 根据角度计算速度分量
        deg_to_rad = torch.tensor(math.pi / 180.0, device=self.device)
        red_angle_rad = self.red_fighters['angle'] * deg_to_rad
        blue_angle_rad = self.blue_fighters['angle'] * deg_to_rad
        
        self.red_fighters['vx'] = torch.cos(red_angle_rad) * self.red_fighters['speed']
        self.red_fighters['vy'] = torch.sin(red_angle_rad) * self.red_fighters['speed']
        self.blue_fighters['vx'] = torch.cos(blue_angle_rad) * self.blue_fighters['speed']
        self.blue_fighters['vy'] = torch.sin(blue_angle_rad) * self.blue_fighters['speed']
    
    def reset(self, env_mask=None):
        """重置环境状态
        
        Args:
            env_mask: 指定哪些环境需要重置的掩码，默认重置所有环境
            
        Returns:
            dict: 初始化的观察空间
        """
        # 如果未提供掩码，重置所有环境
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 重置时间
        self.current_time = torch.zeros(self.num_envs, device=self.device)
        
        # 使用广播重置红蓝方飞机的随机初始状态
        reset_envs = env_mask.nonzero().squeeze(-1)
        if reset_envs.numel() > 0:
            # 使用一次性向量化操作生成随机初始位置和状态
            # 红方位置（左侧）
            rand_x = torch.rand(reset_envs.shape[0], device=self.device) * 10000 + 5000
            rand_y = torch.rand(reset_envs.shape[0], device=self.device) * (self.battlefield_size - 10000) + 5000
            rand_angle = torch.rand(reset_envs.shape[0], device=self.device) * 360
            rand_speed = torch.rand(reset_envs.shape[0], device=self.device) * 100 + 250  # 250-350
            
            # 使用索引更新特定环境
            self.red_fighters['x'].index_copy_(0, reset_envs, rand_x)
            self.red_fighters['y'].index_copy_(0, reset_envs, rand_y)
            self.red_fighters['angle'].index_copy_(0, reset_envs, rand_angle)
            self.red_fighters['speed'].index_copy_(0, reset_envs, rand_speed)
            
            # 蓝方位置（右侧）
            rand_x = torch.rand(reset_envs.shape[0], device=self.device) * 10000 + (self.battlefield_size - 15000)
            rand_y = torch.rand(reset_envs.shape[0], device=self.device) * (self.battlefield_size - 10000) + 5000
            rand_angle = torch.rand(reset_envs.shape[0], device=self.device) * 360
            rand_speed = torch.rand(reset_envs.shape[0], device=self.device) * 100 + 250  # 250-350
            
            # 使用索引更新特定环境
            self.blue_fighters['x'].index_copy_(0, reset_envs, rand_x)
            self.blue_fighters['y'].index_copy_(0, reset_envs, rand_y)
            self.blue_fighters['angle'].index_copy_(0, reset_envs, rand_angle)
            self.blue_fighters['speed'].index_copy_(0, reset_envs, rand_speed)
            
            # 计算速度分量
            deg_to_rad = torch.tensor(math.pi / 180.0, device=self.device)
            
            # 红方
            red_angle_rad = self.red_fighters['angle'][reset_envs] * deg_to_rad
            red_vx = torch.cos(red_angle_rad) * self.red_fighters['speed'][reset_envs]
            red_vy = torch.sin(red_angle_rad) * self.red_fighters['speed'][reset_envs]
            self.red_fighters['vx'].index_copy_(0, reset_envs, red_vx)
            self.red_fighters['vy'].index_copy_(0, reset_envs, red_vy)
            
            # 蓝方
            blue_angle_rad = self.blue_fighters['angle'][reset_envs] * deg_to_rad
            blue_vx = torch.cos(blue_angle_rad) * self.blue_fighters['speed'][reset_envs]
            blue_vy = torch.sin(blue_angle_rad) * self.blue_fighters['speed'][reset_envs]
            self.blue_fighters['vx'].index_copy_(0, reset_envs, blue_vx)
            self.blue_fighters['vy'].index_copy_(0, reset_envs, blue_vy)
            
            # 重置活跃状态和导弹数量
            active_mask = torch.ones(reset_envs.shape[0], dtype=torch.bool, device=self.device)
            missiles_count = torch.full((reset_envs.shape[0],), 6, device=self.device)
            
            self.red_fighters['active'].index_copy_(0, reset_envs, active_mask)
            self.red_fighters['missiles'].index_copy_(0, reset_envs, missiles_count)
            
            self.blue_fighters['active'].index_copy_(0, reset_envs, active_mask)
            self.blue_fighters['missiles'].index_copy_(0, reset_envs, missiles_count)
        
        # 清空导弹列表
        self.missiles = []
        
        # 重置游戏结束状态
        self.done.fill_(False)
        self.winner.fill_(0)
        
        # 重置制导系统历史
        self.guidance.reset_guidance_history(self.num_envs)
        
        # 返回初始观察空间
        return self.get_observations()
    
    def step(self, actions):
        """执行环境步进
        
        Args:
            actions: 动作字典，包含红蓝双方的动作
                    {'red': {'throttle': tensor, 'rudder': tensor, 'fire': tensor},
                     'blue': {'throttle': tensor, 'rudder': tensor, 'fire': tensor}}
                     
        Returns:
            tuple: (observations, rewards, dones, info)
        """
        # 更新当前时间
        self.current_time += self.time_step
        
        # 应用红方动作
        if 'red' in actions:
            red_actions = actions['red']
            if 'throttle' in red_actions:
                self.red_fighters['throttle'] = torch.clamp(red_actions['throttle'], 0.0, 1.0)
            if 'rudder' in red_actions:
                self.red_fighters['rudder'] = torch.clamp(red_actions['rudder'], -1.0, 1.0)
        
        # 应用蓝方动作
        if 'blue' in actions:
            blue_actions = actions['blue']
            if 'throttle' in blue_actions:
                self.blue_fighters['throttle'] = torch.clamp(blue_actions['throttle'], 0.0, 1.0)
            if 'rudder' in blue_actions:
                self.blue_fighters['rudder'] = torch.clamp(blue_actions['rudder'], -1.0, 1.0)
        
        # 处理红方发射导弹动作
        if 'red' in actions and 'fire' in actions['red']:
            fire_mask = actions['red']['fire'].bool() & self.red_fighters['active'] & ~self.done
            if fire_mask.any():
                # 创建导弹，默认目标为蓝方飞机
                # 为每个导弹设置目标为索引0（蓝方飞机）
                target_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                
                # 标记发射位置为红方飞机
                launchers = self.red_fighters.copy()
                
                # 设置is_player1标记为True，表示红方导弹
                launchers['is_player1'] = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                
                # 添加活跃掩码，只为活跃的发射位置创建导弹
                new_missiles = self.events.create_missiles_batch(launchers, target_indices)
                
                if new_missiles is not None:
                    # 检查导弹创建是否成功
                    # print(f"已创建红方导弹，活跃状态: {new_missiles['active'].sum().item()}/{self.num_envs}")
                    self.missiles.append(new_missiles)
        
        # 处理蓝方发射导弹动作
        if 'blue' in actions and 'fire' in actions['blue']:
            fire_mask = actions['blue']['fire'].bool() & self.blue_fighters['active'] & ~self.done
            if fire_mask.any():
                # 创建导弹，默认目标为红方飞机
                # 为每个导弹设置目标为索引0（红方飞机）
                target_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                
                # 标记发射位置为蓝方飞机
                launchers = self.blue_fighters.copy()
                
                # 设置is_player1标记为False，表示蓝方导弹
                launchers['is_player1'] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                
                # 添加活跃掩码，只为活跃的发射位置创建导弹
                new_missiles = self.events.create_missiles_batch(launchers, target_indices)
                
                if new_missiles is not None:
                    # 检查导弹创建是否成功
                    # print(f"已创建蓝方导弹，活跃状态: {new_missiles['active'].sum().item()}/{self.num_envs}")
                    self.missiles.append(new_missiles)
        
        # 更新红蓝飞机的物理状态
        self.red_fighters = self.aero.calculate_physics_batch(self.red_fighters, self.time_step)
        self.blue_fighters = self.aero.calculate_physics_batch(self.blue_fighters, self.time_step)
        
        # 更新所有导弹的物理状态和制导
        for i, missile in enumerate(self.missiles):
            if missile is None or not missile['active'].any():
                continue
                
            # 确定目标（红方导弹瞄准蓝方，蓝方导弹瞄准红方）
            targets = self.blue_fighters if 'is_player1' in missile and missile['is_player1'].any() else self.red_fighters
            
            # 计算制导舵量
            guidance_rudder = self.guidance.calculate_guidance_batch(missile, targets, self.current_time)
            missile['rudder'] = guidance_rudder
            
            # 更新物理状态
            missile = self.aero.calculate_physics_batch(missile, self.time_step)
            self.missiles[i] = missile
            
            # 检查导弹命中
            hit_mask = self.events.check_missile_hit_batch(missile, targets)
            if hit_mask.any():
                # 应用伤害到目标
                self.events.apply_damage_batch(targets, hit_mask)
                # 将击中的导弹标记为非活跃
                missile['active'] = missile['active'] & ~hit_mask
            
            # 检查导弹自毁（速度过低）
            self_destruct_mask = self.events.check_missile_self_destruct_batch(missile)
            if self_destruct_mask.any():
                missile['active'] = missile['active'] & ~self_destruct_mask
        
        # 确保所有实体都在战场边界内
        self.red_fighters = self.events.limit_to_battlefield_batch(self.red_fighters)
        self.blue_fighters = self.events.limit_to_battlefield_batch(self.blue_fighters)
        for i, missile in enumerate(self.missiles):
            if missile is not None:
                self.missiles[i] = self.events.limit_to_battlefield_batch(missile)
        
        # 检查游戏是否结束
        game_over_mask, winner_mask = self.events.check_game_over_batch(self.red_fighters, self.blue_fighters, self.missiles)
        
        # 更新已完成的环境掩码
        new_done_mask = game_over_mask & ~self.done
        self.done |= game_over_mask
        
        # 当有新的环境完成时更新获胜者
        if new_done_mask.any():
            self.winner = torch.where(new_done_mask, winner_mask, self.winner)
        
        # 计算奖励
        rewards = self._compute_rewards(new_done_mask, winner_mask)
        
        # 获取观察空间
        observations = self.get_observations()
        
        # 构建信息字典
        info = {
            'time': self.current_time.clone(),
            'winner': self.winner.clone()
        }
        
        return observations, rewards, self.done.clone(), info
    
    def _compute_rewards(self, new_done_mask, winner_mask):
        """计算奖励
        
        Args:
            new_done_mask: 新完成环境掩码
            winner_mask: 获胜者掩码
            
        Returns:
            dict: 奖励字典
        """
        # 使用events模块计算奖励
        rewards = self.events.generate_reward_batch({
            'red_fighters': self.red_fighters,
            'blue_fighters': self.blue_fighters,
            'missiles': self.missiles
        }, winner_mask, new_done_mask)
        
        return rewards
    
    def get_observations(self):
        """获取观察空间
        
        Returns:
            dict: 观察空间字典
        """
        # 构建红蓝双方的观察空间
        red_obs = self._get_side_observations(is_red=True)
        blue_obs = self._get_side_observations(is_red=False)
        
        return {'red': red_obs, 'blue': blue_obs}
    
    def _get_side_observations(self, is_red=True):
        """获取一方的观察空间
        
        Args:
            is_red: 是否为红方
            
        Returns:
            dict: 观察空间字典
        """
        # 选择己方和敌方飞机
        self_fighters = self.red_fighters if is_red else self.blue_fighters
        enemy_fighters = self.blue_fighters if is_red else self.red_fighters
        
        # 计算相对位置
        dx = enemy_fighters['x'] - self_fighters['x']
        dy = enemy_fighters['y'] - self_fighters['y']
        
        # 计算两机距离
        distance = torch.sqrt(dx*dx + dy*dy)
        
        # 计算方位角
        bearing = torch.atan2(dy, dx) * (180.0 / math.pi)
        # 规范化到0-360度
        bearing = bearing % 360.0
        
        # 计算相对角度（敌机方位角相对于己方朝向）
        relative_angle = bearing - self_fighters['angle']
        # 规范化到-180到180度
        relative_angle = (relative_angle + 180.0) % 360.0 - 180.0
        
        # 构建观察空间
        obs = {
            # 己方状态
            'x': self_fighters['x'],
            'y': self_fighters['y'],
            'vx': self_fighters['vx'],
            'vy': self_fighters['vy'],
            'speed': self_fighters['speed'],
            'angle': self_fighters['angle'],
            'missiles': self_fighters['missiles'],
            
            # 敌方相对信息
            'enemy_distance': distance,
            'enemy_bearing': bearing,
            'enemy_relative_angle': relative_angle,
            'enemy_speed': enemy_fighters['speed'],
            'enemy_angle': enemy_fighters['angle'],
            
            # 全局信息
            'battlefield_size': torch.full_like(self_fighters['x'], self.battlefield_size),
            'time': self.current_time,
        }
        
        return obs
    
    def close(self):
        """关闭环境，释放资源"""
        # 清空导弹列表
        self.missiles = []
        
        # 在CUDA上清理缓存
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        print("环境已关闭，资源已释放") 