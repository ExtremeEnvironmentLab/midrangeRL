# -*- coding: utf-8 -*-

import torch
import math

class GameEventsTensor:
    def __init__(self, battlefield_size=50000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化游戏事件处理系统的Tensor版本
        
        Args:
            battlefield_size: 战场大小（米）
            device: 计算设备，默认使用CUDA（如果可用）
        """
        self.device = device
        
        # 转换为张量
        if not isinstance(battlefield_size, torch.Tensor):
            self.battlefield_size = torch.tensor(battlefield_size, dtype=torch.float32, device=device)
        else:
            self.battlefield_size = battlefield_size.to(device)
            
        # 导弹碰撞和自毁参数
        self.hit_distance = torch.tensor(500.0, device=device)  # 导弹命中半径（米）
        self.min_missile_speed = torch.tensor(200.0, device=device)  # 导弹最小速度（米/秒）
        
        # 数学常数
        self.PI = torch.tensor(math.pi, device=device)
        self.DEG_TO_RAD = torch.tensor(math.pi / 180.0, device=device)
        self.RAD_TO_DEG = torch.tensor(180.0 / math.pi, device=device)
        
        # 预先创建气动力学对象
        from env_tensor.aerodynamic_tensor import AerodynamicTensor
        self.aero = AerodynamicTensor(device=device)
    
    def create_missiles_batch(self, launchers, target_indices=None, batch_size=None):
        """从飞机批量创建导弹的张量表示
        
        Args:
            launchers: 发射导弹的飞机张量字典
            target_indices: 每个导弹的目标索引，形状为[batch_size]
            batch_size: 批次大小（如果未提供，将使用launchers的batch_size）
            
        Returns:
            dict: 导弹的张量字典
        """
        if batch_size is None:
            batch_size = launchers['x'].shape[0]
            
        # 检查是否还有导弹
        missiles_available = None
        if 'missiles' in launchers:
            missiles_left = launchers['missiles']
            
            # 创建掩码，标识哪些位置可以发射导弹
            missiles_available = missiles_left > 0
            
            # 如果没有可用导弹，返回None
            if not missiles_available.any():
                return None
                
            # 为有导弹的发射者减少导弹数量
            launchers['missiles'] = torch.where(
                missiles_available,
                missiles_left - 1,
                missiles_left
            )
        else:
            # 假设所有发射者都可以发射
            missiles_available = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        # 提取发射者位置和角度
        x = launchers['x']
        y = launchers['y']
        vx = launchers['vx']
        vy = launchers['vy']
        angle = launchers['angle']
        speed = launchers['speed']
        
        # 计算导弹初始位置（略微前移，以避免碰撞）- 使用向量化操作
        offset = torch.full((batch_size,), 20.0, device=self.device)  # 前移距离（米）
        angle_rad = angle * self.DEG_TO_RAD
        
        # 使用三角函数批量计算偏移
        cos_angle = torch.cos(angle_rad)
        sin_angle = torch.sin(angle_rad)
        missile_x = x + cos_angle * offset
        missile_y = y + sin_angle * offset
        
        # 创建导弹对象 - 使用预先创建的气动力学对象
        missile = self.aero.create_aircraft_batch(batch_size, is_missile=True)
        
        # 设置导弹的初始状态 - 一次性批量设置
        missile['x'] = missile_x
        missile['y'] = missile_y
        missile['vx'] = vx
        missile['vy'] = vy
        missile['angle'] = angle
        missile['speed'] = speed  # 导弹继承飞机的速度
        
        # 设置目标索引
        if target_indices is not None:
            missile['target_idx'] = target_indices
        else:
            # 默认目标为0
            missile['target_idx'] = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # 导弹来源标记（用于区分红/蓝方）
        if 'is_player1' in launchers:
            missile['is_player1'] = launchers['is_player1']
        
        # 为不可用的发射者（如没有导弹的）将导弹标记为非活跃
        if not missiles_available.all():
            missile['active'] = missiles_available
        
        return missile
    
    def check_missile_hit_batch(self, missiles, targets, target_indices=None):
        """检查导弹是否击中目标（批处理版本）
        
        Args:
            missiles: 导弹的张量字典批次
            targets: 目标飞机的张量字典批次
            target_indices: 每个导弹的目标索引，形状为[batch_size]
            
        Returns:
            torch.Tensor: 命中掩码，形状为[batch_size]，表示哪些导弹击中了目标
        """
        batch_size = missiles['x'].shape[0]
        
        # 初始化命中掩码
        hit_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # 获取导弹位置
        missile_x = missiles['x']
        missile_y = missiles['y']
        missile_active = missiles.get('active', torch.ones(batch_size, dtype=torch.bool, device=self.device))
        
        # 处理目标格式
        if isinstance(targets, dict):
            # 单一目标字典，提取目标位置
            target_x = targets['x']
            target_y = targets['y']
            target_active = targets.get('active', torch.ones_like(target_x, dtype=torch.bool, device=self.device))
            
            # 检查是否有目标索引
            if target_indices is None and 'target_idx' in missiles:
                target_indices = missiles['target_idx']
            
            # 如果有目标索引，使用它来计算距离
            if target_indices is not None:
                # 为每个导弹找到其目标
                for i in range(batch_size):
                    # 跳过非活跃的导弹
                    if not missile_active[i]:
                        continue
                        
                    # 获取目标索引
                    idx = target_indices[i].item()
                    
                    # 检查索引是否有效
                    if idx >= 0 and idx < len(target_x) and target_active[idx]:
                        # 计算到目标的距离
                        dx = missile_x[i] - target_x[idx]
                        dy = missile_y[i] - target_y[idx]
                        distance_squared = dx*dx + dy*dy
                        
                        # 检查是否击中
                        hit_distance_squared = self.hit_distance * self.hit_distance
                        if distance_squared < hit_distance_squared:
                            hit_mask[i] = True
            else:
                # 没有目标索引，计算到所有目标的距离
                # 创建广播形状：[missile_batch_size, target_batch_size]
                m_x = missile_x.unsqueeze(1)  # 形状: [missile_batch, 1]
                m_y = missile_y.unsqueeze(1)  # 形状: [missile_batch, 1]
                
                t_x = target_x.unsqueeze(0)  # 形状: [1, target_batch]
                t_y = target_y.unsqueeze(0)  # 形状: [1, target_batch]
                
                # 计算所有导弹到所有目标的距离
                dx = m_x - t_x  # 形状: [missile_batch, target_batch]
                dy = m_y - t_y  # 形状: [missile_batch, target_batch]
                
                # 计算距离平方
                distance_squared = dx*dx + dy*dy  # 形状: [missile_batch, target_batch]
                
                # 检查每个导弹是否击中任何目标
                hit_distance_squared = self.hit_distance * self.hit_distance
                
                # 找出命中的导弹和目标对
                hits = distance_squared < hit_distance_squared  # 形状: [missile_batch, target_batch]
                
                # 任何命中标志
                any_hit_mask = hits.any(dim=1)  # 形状: [missile_batch]
                
                # 将活跃的导弹与命中掩码结合
                hit_mask = any_hit_mask & missile_active
        else:
            raise ValueError("当前版本只支持字典格式的目标")
        
        return hit_mask
    
    def check_missile_self_destruct_batch(self, missiles):
        """检查导弹是否应该自毁（速度过低）（批处理版本）
        
        Args:
            missiles: 导弹的张量字典批次
            
        Returns:
            torch.Tensor: 自毁掩码，形状为[batch_size]，表示哪些导弹应该自毁
        """
        return (missiles['speed'] < self.min_missile_speed) & missiles['active']
    
    def limit_to_battlefield_batch(self, entities):
        """确保实体不会飞出战场边界（批处理版本）
        
        Args:
            entities: 飞行器张量字典批次
            
        Returns:
            dict: 更新后的飞行器张量字典批次
        """
        # 使用原地操作提高效率
        entities['x'].clamp_(0.0, self.battlefield_size)
        entities['y'].clamp_(0.0, self.battlefield_size)
        
        # 检查是否在边界上，如果是则计算反弹
        at_left = entities['x'] <= 0.0
        at_right = entities['x'] >= self.battlefield_size
        at_top = entities['y'] <= 0.0
        at_bottom = entities['y'] >= self.battlefield_size
        
        # 如果实体在边界上，反转速度分量
        if 'vx' in entities:
            entities['vx'] = torch.where(at_left | at_right, -entities['vx'], entities['vx'])
        if 'vy' in entities:
            entities['vy'] = torch.where(at_top | at_bottom, -entities['vy'], entities['vy'])
            
        # 如果设置了速度分量，更新角度
        if 'vx' in entities and 'vy' in entities:
            # 计算新的角度
            new_angle = torch.atan2(entities['vy'], entities['vx']) * self.RAD_TO_DEG
            # 确保角度在0-360范围内
            entities['angle'] = new_angle % 360.0
        
        return entities
    
    def check_game_over_batch(self, aircraft1_batch, aircraft2_batch, missiles_batch=None):
        """检查游戏是否结束（批处理版本）
        
        Args:
            aircraft1_batch: 第一架飞机的张量字典批次（红方）
            aircraft2_batch: 第二架飞机的张量字典批次（蓝方）
            missiles_batch: 导弹列表，每个元素是导弹的张量字典批次
            
        Returns:
            tuple: (game_over_mask, winner_mask)
                game_over_mask: 形状为[batch_size]的布尔张量，表示哪些环境游戏结束
                winner_mask: 形状为[batch_size]的整数张量，0表示平局，1表示红方胜利，2表示蓝方胜利
        """
        batch_size = aircraft1_batch['x'].shape[0]
        
        # 获取飞机活跃状态
        aircraft1_active = aircraft1_batch['active']
        aircraft2_active = aircraft2_batch['active']
        
        # 检查是否有飞机不活跃（被摧毁）
        red_destroyed = ~aircraft1_active
        blue_destroyed = ~aircraft2_active
        
        # 检查是否所有导弹都不活跃或没有导弹
        all_missiles_inactive = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        if missiles_batch:
            for missile in missiles_batch:
                if missile is not None and missile['active'].any():
                    # 如果有任何活跃的导弹，标记为False
                    all_missiles_inactive &= ~missile['active']
        
        # 检查导弹耗尽情况
        missiles_depleted = (aircraft1_batch['missiles'] <= 0) & (aircraft2_batch['missiles'] <= 0)
        
        # 游戏结束条件：
        # 1. 任一方被摧毁 且 没有活跃的导弹，或
        # 2. 双方导弹都耗尽 且 没有活跃的导弹（平局）
        game_over_mask = ((red_destroyed | blue_destroyed) & all_missiles_inactive) | (missiles_depleted & all_missiles_inactive)
        
        # 决定获胜者
        # 0 = 平局（都被摧毁或导弹耗尽）
        # 1 = 红方胜利（蓝方被摧毁）
        # 2 = 蓝方胜利（红方被摧毁）
        winner_mask = torch.zeros(batch_size, dtype=torch.int8, device=self.device)
        
        # 设置蓝方被摧毁，红方存活的情况
        red_win_mask = blue_destroyed & ~red_destroyed & game_over_mask
        winner_mask = torch.where(red_win_mask, torch.ones_like(winner_mask), winner_mask)
        
        # 设置红方被摧毁，蓝方存活的情况
        blue_win_mask = red_destroyed & ~blue_destroyed & game_over_mask
        winner_mask = torch.where(blue_win_mask, torch.full_like(winner_mask, 2), winner_mask)
        
        return game_over_mask, winner_mask
    
    def apply_damage_batch(self, entities, damage_mask):
        """应用伤害到实体（批处理版本）
        
        Args:
            entities: 实体的张量字典批次
            damage_mask: 伤害掩码，形状为[batch_size]，表示哪些实体受到伤害
            
        Returns:
            dict: 更新后的实体张量字典批次
        """
        # 确保damage_mask的大小与entities['active']匹配
        batch_size = entities['active'].shape[0]
        
        # 如果damage_mask是单个值，扩展它以匹配entities的批次大小
        if damage_mask.numel() != batch_size:
            if damage_mask.numel() == 1:
                if damage_mask.item():
                    # 单个True，影响索引为0的实体
                    new_damage_mask = torch.zeros_like(entities['active'])
                    new_damage_mask[0] = True
                    damage_mask = new_damage_mask
                else:
                    # 单个False，不影响任何实体
                    damage_mask = torch.zeros_like(entities['active'])
            else:
                # 如果大小不匹配且不是单个值，则发出警告
                print(f"警告: damage_mask大小 ({damage_mask.numel()}) 与 entities批次大小 ({batch_size}) 不匹配")
                # 创建一个全零掩码作为默认行为
                damage_mask = torch.zeros_like(entities['active'])
        
        # 将受到伤害的实体标记为非活跃
        entities['active'] = entities['active'] & ~damage_mask
        
        return entities
    
    def generate_reward_batch(self, game_state, winner_mask, new_done_mask):
        """生成奖励值（批处理版本）
        
        Args:
            game_state: 游戏状态字典批次
            winner_mask: 获胜者掩码，形状为[batch_size]
            new_done_mask: 新完成环境掩码，形状为[batch_size]
            
        Returns:
            dict: 奖励字典，包含红蓝双方的奖励
        """
        batch_size = winner_mask.shape[0]
        
        # 初始化奖励
        red_reward = torch.zeros(batch_size, device=self.device)
        blue_reward = torch.zeros(batch_size, device=self.device)
        
        # 胜利/失败奖励
        # 红方胜利（蓝方被摧毁）
        red_win_mask = (winner_mask == 1) & new_done_mask
        red_reward = torch.where(red_win_mask, torch.full_like(red_reward, 10.0), red_reward)
        blue_reward = torch.where(red_win_mask, torch.full_like(blue_reward, -10.0), blue_reward)
        
        # 蓝方胜利（红方被摧毁）
        blue_win_mask = (winner_mask == 2) & new_done_mask
        red_reward = torch.where(blue_win_mask, torch.full_like(red_reward, -10.0), red_reward)
        blue_reward = torch.where(blue_win_mask, torch.full_like(blue_reward, 10.0), blue_reward)
        
        # 平局奖励
        draw_mask = (winner_mask == 0) & new_done_mask
        red_reward = torch.where(draw_mask, torch.full_like(red_reward, 0.0), red_reward)
        blue_reward = torch.where(draw_mask, torch.full_like(blue_reward, 0.0), blue_reward)
        
        # 未完成环境的奖励
        # 可以添加基于状态的小奖励，例如根据位置、速度等
        
        return {'red': red_reward, 'blue': blue_reward} 