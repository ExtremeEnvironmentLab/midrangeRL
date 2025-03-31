# -*- coding: utf-8 -*-

import torch
import math

class MissileGuidanceTensor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化导弹制导系统的Tensor批处理版本
        
        Args:
            device: 计算设备，默认使用CUDA（如果可用）
        """
        self.device = device
        
        # 制导参数
        self.MAX_RUDDER = torch.tensor(1.0, device=device)  # 最大舵量
        self.PROPORTIONAL_GAIN = torch.tensor(3.0, device=device)  # 比例增益
        self.LEAD_TIME = torch.tensor(2.0, device=device)  # 前置时间（秒）
        self.guidance_gain = torch.tensor(5.0, device=device)  # 制导增益
        
        # 数学常数
        self.PI = torch.tensor(math.pi, device=device)
        self.TWO_PI = torch.tensor(2.0 * math.pi, device=device)
        self.DEG_TO_RAD = torch.tensor(math.pi / 180.0, device=device)
        self.RAD_TO_DEG = torch.tensor(180.0 / math.pi, device=device)
        
        # 用于存储上次制导计算的时间和历史数据
        self.last_guidance_time = None
        self.guidance_history = {}
        
        # 用于存储视线角历史（支持单个导弹的计算）
        self.last_los_angles = {}
        self.last_times = {}
    
    def reset_guidance_history(self, batch_size=None):
        """重置制导历史数据
        
        Args:
            batch_size: 如果提供，将初始化指定大小的批次历史数据
        """
        self.last_guidance_time = None
        self.guidance_history = {}
        self.last_los_angles = {}
        self.last_times = {}
        
        if batch_size is not None:
            # 初始化批次历史数据
            self.guidance_history = {
                'last_target_x': torch.zeros(batch_size, device=self.device),
                'last_target_y': torch.zeros(batch_size, device=self.device),
                'last_target_vx': torch.zeros(batch_size, device=self.device),
                'last_target_vy': torch.zeros(batch_size, device=self.device),
                'last_time': torch.zeros(batch_size, device=self.device)
            }
    
    def calculate_guidance_batch(self, missiles, targets, current_time=None):
        """计算导弹的制导舵量（批处理版本）
        
        Args:
            missiles: 导弹的张量字典批次
            targets: 目标飞机的张量字典批次
            current_time: 当前模拟时间（可选）
            
        Returns:
            torch.Tensor: 制导舵量张量，形状为[batch_size]
        """
        # 获取批次大小
        batch_size = missiles['x'].shape[0]
        
        # 如果没有提供当前时间，使用默认值
        if current_time is None:
            current_time = torch.ones(batch_size, device=self.device)
        elif not isinstance(current_time, torch.Tensor):
            current_time = torch.tensor(current_time, device=self.device).expand(batch_size)
        
        # 确保制导历史已初始化
        if not self.guidance_history:
            self.reset_guidance_history(batch_size)
            self.guidance_history['last_time'] = current_time.clone()
        
        # 计算自上次制导以来的时间间隔
        dt = current_time - self.guidance_history['last_time']
        # 避免除零
        dt = torch.maximum(dt, torch.tensor(0.01, device=self.device))
        
        # 提取目标信息
        target_x = targets['x']
        target_y = targets['y']
        target_vx = targets['vx'] if 'vx' in targets else torch.zeros_like(target_x)
        target_vy = targets['vy'] if 'vy' in targets else torch.zeros_like(target_y)
        
        # 如果没有显式的速度向量，从角度和速度计算
        if 'vx' not in targets and 'vy' not in targets and 'angle' in targets and 'speed' in targets:
            target_angle_rad = targets['angle'] * self.DEG_TO_RAD
            target_vx = torch.cos(target_angle_rad) * targets['speed']
            target_vy = torch.sin(target_angle_rad) * targets['speed']
        
        # 提取导弹信息
        missile_x = missiles['x']
        missile_y = missiles['y']
        missile_vx = missiles['vx']
        missile_vy = missiles['vy']
        missile_speed = missiles['speed']
        
        # 创建导弹到目标的映射
        # 如果存在target_idx，使用它来指定每个导弹对应的目标
        if 'target_idx' in missiles:
            target_indices = missiles['target_idx'].long()
            valid_mask = target_indices >= 0
            
            # 创建目标位置和速度的张量
            mapped_target_x = torch.zeros_like(missile_x)
            mapped_target_y = torch.zeros_like(missile_y)
            mapped_target_vx = torch.zeros_like(missile_vx)
            mapped_target_vy = torch.zeros_like(missile_vy)
            
            # 为每个有效的导弹分配目标
            for i in range(batch_size):
                if valid_mask[i]:
                    idx = target_indices[i].item()
                    if idx < len(target_x):
                        mapped_target_x[i] = target_x[idx]
                        mapped_target_y[i] = target_y[idx]
                        mapped_target_vx[i] = target_vx[idx]
                        mapped_target_vy[i] = target_vy[idx]
            
            # 使用映射后的目标
            target_x = mapped_target_x
            target_y = mapped_target_y
            target_vx = mapped_target_vx
            target_vy = mapped_target_vy
        
        # 计算目标与导弹之间的向量
        dx = target_x - missile_x
        dy = target_y - missile_y
        
        # 计算目标的速度估计
        # 如果有之前的位置记录，计算平均速度
        if 'last_target_x' in self.guidance_history and torch.any(dt > 0):
            last_target_x = self.guidance_history['last_target_x']
            last_target_y = self.guidance_history['last_target_y']
            
            # 计算位置变化
            dx_target = target_x - last_target_x
            dy_target = target_y - last_target_y
            
            # 估计目标速度（使用dt作为分母）
            est_target_vx = dx_target / dt
            est_target_vy = dy_target / dt
            
            # 更新保存的目标速度（使用平滑滤波）
            alpha = torch.tensor(0.7, device=self.device)  # 平滑系数
            if 'last_target_vx' in self.guidance_history:
                last_target_vx = self.guidance_history['last_target_vx']
                last_target_vy = self.guidance_history['last_target_vy']
                
                # 平滑滤波: v = alpha * v_new + (1-alpha) * v_old
                target_vx = alpha * est_target_vx + (1-alpha) * last_target_vx
                target_vy = alpha * est_target_vy + (1-alpha) * last_target_vy
        
        # 更新制导历史
        self.guidance_history['last_target_x'] = target_x.clone()
        self.guidance_history['last_target_y'] = target_y.clone()
        self.guidance_history['last_target_vx'] = target_vx.clone()
        self.guidance_history['last_target_vy'] = target_vy.clone()
        self.guidance_history['last_time'] = current_time.clone()
        
        # 计算预测的目标位置（考虑目标速度和前置时间）
        predicted_target_x = target_x + target_vx * self.LEAD_TIME
        predicted_target_y = target_y + target_vy * self.LEAD_TIME
        
        # 计算导弹到预测目标位置的向量
        pred_dx = predicted_target_x - missile_x
        pred_dy = predicted_target_y - missile_y
        
        # 计算目标方位角（弧度）
        target_bearing = torch.atan2(pred_dy, pred_dx)
        
        # 计算导弹当前航向（弧度）
        missile_heading = torch.atan2(missile_vy, missile_vx)
        
        # 计算航向差（弧度）
        heading_diff = target_bearing - missile_heading
        
        # 规范化航向差到[-pi, pi]范围
        heading_diff = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff))
        
        # 计算比例导引舵量
        rudder = (heading_diff / self.PI) * self.PROPORTIONAL_GAIN
        
        # 限制舵量在[-1, 1]范围内
        rudder = torch.clamp(rudder, -self.MAX_RUDDER, self.MAX_RUDDER)
        
        # 创建活跃掩码（只对活跃的导弹应用制导）
        active_mask = missiles['active'] if 'active' in missiles else torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        # 使用掩码条件应用制导
        # 对非活跃的导弹，设置舵量为0
        final_rudder = torch.where(active_mask, rudder, torch.zeros_like(rudder))
        
        return final_rudder
    
    def calculate_guidance(self, missile, target, current_time):
        """计算单个导弹的制导指令
        
        Args:
            missile: 导弹的张量字典
            target: 目标的张量字典
            current_time: 当前时间戳（float或tensor）
            
        Returns:
            torch.Tensor: 舵量指令 (-1 到 1)
        """
        # 确保current_time是张量
        if not isinstance(current_time, torch.Tensor):
            current_time = torch.tensor(current_time, device=self.device)
        
        # 计算弹目连线角度
        dx = target['x'] - missile['x']
        dy = target['y'] - missile['y']
        
        # 计算atan2，得到弹目连线角度（弧度）
        # 使用arctan2，这是atan2的张量版本
        current_los_angle_rad = torch.atan2(dy, dx)
        
        # 转换为度
        current_los_angle = current_los_angle_rad * (180.0 / math.pi)
        
        # 获取导弹ID（用于在字典中存储状态）
        missile_id = id(missile)
        
        # 计算视线角速度
        if missile_id in self.last_los_angles and missile_id in self.last_times:
            last_los_angle = self.last_los_angles[missile_id]
            last_time = self.last_times[missile_id]
            
            dt = current_time - last_time
            
            # 确保dt不为零
            if dt > 1e-6:
                # 处理角度跨越360度的情况
                # 计算角度差，考虑到角度可能跨越了360度的边界
                angle_diff = (current_los_angle - last_los_angle + 180) % 360 - 180
                
                # 计算角速度（度/秒）
                los_rate = angle_diff / dt
            else:
                los_rate = torch.tensor(0.0, device=self.device)
        else:
            # 首次计算，没有历史数据，返回零
            los_rate = torch.tensor(0.0, device=self.device)
        
        # 更新历史数据
        self.last_los_angles[missile_id] = current_los_angle
        self.last_times[missile_id] = current_time
        
        # 计算制导指令（比例引导法）
        # 归一化到[-1,1]范围
        rudder = self.guidance_gain * los_rate / 180.0
        
        # 限制舵量范围
        rudder = torch.clamp(rudder, -1.0, 1.0)
        
        return rudder
    
    def clean_guidance_history(self, missile_id):
        """清除指定导弹的制导历史
        
        Args:
            missile_id: 导弹的ID
        """
        if missile_id in self.last_los_angles:
            del self.last_los_angles[missile_id]
        
        if missile_id in self.last_times:
            del self.last_times[missile_id] 