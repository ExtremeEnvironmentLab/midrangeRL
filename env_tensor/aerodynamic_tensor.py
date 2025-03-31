# -*- coding: utf-8 -*-

import torch
import math

class AerodynamicTensor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化气动力学计算的Tensor版本
        
        Args:
            device: 计算设备，默认使用CUDA（如果可用）
        """
        self.device = device
        
        # 将所有物理参数转换为tensor并移动到指定设备
        # 战斗机参数
        self.FIGHTER_TERMINAL_VELOCITY = torch.tensor(400.0, device=device)  # 自由落体终端速度 (m/s)
        self.FIGHTER_MIN_TURN_RADIUS = torch.tensor(1000.0, device=device)   # 最小转弯半径 (m)
        self.FIGHTER_CL_MAX = 1 / self.FIGHTER_MIN_TURN_RADIUS  # 最大升力系数
        self.FIGHTER_MAX_THRUST = torch.tensor(1.5 * 9.8, device=device)  # 最大推力 (m/s²)
        self.FIGHTER_LIFT_DRAG_RATIO = torch.tensor(5.0, device=device)     # 最大升阻比

        # 导弹参数
        self.MISSILE_TERMINAL_VELOCITY = torch.tensor(400.0, device=device) # 自由落体终端速度 (m/s)
        self.MISSILE_MIN_TURN_RADIUS = torch.tensor(1000.0, device=device)  # 最小转弯半径 (m)
        self.MISSILE_CL_MAX = 1 / self.MISSILE_MIN_TURN_RADIUS  # 最大升力系数
        self.MISSILE_THRUST = torch.tensor(15.0 * 9.8, device=device)        # 发动机推力 (m/s²)
        self.MISSILE_ENGINE_DURATION = torch.tensor(10.0, device=device) # 发动机工作时间 (s)
        self.MISSILE_LIFT_DRAG_RATIO = torch.tensor(3.0, device=device)     # 最大升阻比
        
        # 重力加速度
        self.GRAVITY = torch.tensor(9.8, device=device)
        
        # 数学常数
        self.PI = torch.tensor(math.pi, device=device)
        self.DEG_TO_RAD = torch.tensor(math.pi / 180.0, device=device)
        self.RAD_TO_DEG = torch.tensor(180.0 / math.pi, device=device)
    
    def create_aircraft_batch(self, batch_size=1, is_missile=False):
        """创建飞机或导弹的张量表示批次 - 高效版
        
        Args:
            batch_size: 批次大小
            is_missile: 是否为导弹
            
        Returns:
            dict: 包含所有飞行器状态的张量字典
        """
        # 创建基本状态张量
        aircraft = {
            'x': torch.zeros(batch_size, device=self.device),  # x坐标
            'y': torch.zeros(batch_size, device=self.device),  # y坐标
            'vx': torch.zeros(batch_size, device=self.device),  # x方向速度分量
            'vy': torch.zeros(batch_size, device=self.device),  # y方向速度分量
            'angle': torch.zeros(batch_size, device=self.device),  # 朝向角度（度）
            'speed': torch.ones(batch_size, device=self.device) * 300.0,  # 初始速度（米/秒）
            'throttle': torch.ones(batch_size, device=self.device),  # 油门位置（0-1）
            'rudder': torch.zeros(batch_size, device=self.device),  # 舵量（-1到1）
            'n_load': torch.ones(batch_size, device=self.device),  # 过载（G）
            'is_missile': torch.full((batch_size,), is_missile, dtype=torch.bool, device=self.device),  # 是否为导弹
            'active': torch.ones(batch_size, dtype=torch.bool, device=self.device),  # 活跃状态
        }
        
        # 随机生成初始角度
        aircraft['angle'] = torch.rand(batch_size, device=self.device) * 360.0
        
        # 基于角度计算初始速度向量
        angle_rad = aircraft['angle'] * self.DEG_TO_RAD
        aircraft['vx'] = torch.cos(angle_rad) * aircraft['speed']
        aircraft['vy'] = torch.sin(angle_rad) * aircraft['speed']
        
        # 对于战斗机，添加导弹数量
        if not is_missile:
            aircraft['missiles'] = torch.full((batch_size,), 6, device=self.device)  # 导弹数量
        else:
            # 对于导弹，添加引擎工作时间和目标
            aircraft['engine_time'] = torch.full((batch_size,), self.MISSILE_ENGINE_DURATION.item(), device=self.device)
            aircraft['target_idx'] = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)  # -1表示无目标
        
        return aircraft
    
    def calculate_physics_batch(self, aircraft, dt):
        """计算飞行器的物理状态更新（基于矢量物理的批处理版本）
        
        Args:
            aircraft: 飞行器张量字典
            dt: 时间步长（秒）
            
        Returns:
            dict: 更新后的飞行器张量字典
        """
        # 确保dt是张量
        dt_tensor = torch.tensor(dt, device=self.device) if not isinstance(dt, torch.Tensor) else dt.to(self.device)
        
        # 提取状态变量
        x = aircraft['x']
        y = aircraft['y']
        vx = aircraft['vx']
        vy = aircraft['vy']
        speed = aircraft['speed']
        angle = aircraft['angle']
        throttle = aircraft['throttle']
        rudder = aircraft['rudder']
        is_missile = aircraft['is_missile']
        
        # 根据实体类型选择物理参数 - 使用条件选择来处理导弹和飞机的不同参数
        terminal_velocity = torch.where(
            is_missile, 
            self.MISSILE_TERMINAL_VELOCITY, 
            self.FIGHTER_TERMINAL_VELOCITY
        )
        
        min_turn_radius = torch.where(
            is_missile, 
            self.MISSILE_MIN_TURN_RADIUS, 
            self.FIGHTER_MIN_TURN_RADIUS
        )
        
        cl_max = torch.where(
            is_missile, 
            self.MISSILE_CL_MAX, 
            self.FIGHTER_CL_MAX
        )
        
        max_thrust = torch.where(
            is_missile, 
            self.MISSILE_THRUST, 
            self.FIGHTER_MAX_THRUST
        )
        
        lift_drag_ratio = torch.where(
            is_missile, 
            self.MISSILE_LIFT_DRAG_RATIO, 
            self.FIGHTER_LIFT_DRAG_RATIO
        )
        
        # 计算推力
        if 'engine_time' in aircraft and torch.any(is_missile):
            # 更新导弹引擎工作时间
            aircraft['engine_time'] = torch.maximum(
                aircraft['engine_time'] - dt_tensor, 
                torch.tensor(0.0, device=self.device)
            )
            # 引擎工作时提供推力，否则无推力
            thrust_mask = (aircraft['engine_time'] > 0) & is_missile
            thrust = torch.where(thrust_mask, max_thrust, torch.tensor(0.0, device=self.device))
        else:
            # 飞机推力由油门控制
            thrust = throttle * max_thrust
        
        # 计算空气动力参数
        # 1. 计算零升阻力系数（基于终端速度）
        Cd0 = self.GRAVITY / (terminal_velocity ** 2)
        
        # 2. 计算升力系数（与舵量成正比）
        Cl = torch.abs(rudder) * cl_max
        
        # 3. 计算升阻比相关参数
        k = (cl_max - lift_drag_ratio * Cd0) / (lift_drag_ratio * cl_max)
        
        # 4. 总阻力系数
        Cd = Cd0 + k * Cl
        
        # 计算速度矢量的方向
        velocity_angle = torch.atan2(vy, vx)
        
        # === 开始矢量物理计算 ===
        
        # 计算垂直于速度方向的单位矢量（逆时针旋转90度）
        perpendicular_x = -torch.sin(velocity_angle)
        perpendicular_y = torch.cos(velocity_angle)
        
        # 计算升力大小（与舵量和速度平方成正比）
        lift_magnitude = Cl * speed * speed
        
        # 计算阻力大小（与速度平方成正比）
        drag_magnitude = Cd * speed * speed
        
        # 计算前进方向的推力分量
        thrust_x = thrust * torch.cos(velocity_angle)
        thrust_y = thrust * torch.sin(velocity_angle)
        
        # 计算阻力分量（与速度方向相反）
        drag_x = -drag_magnitude * torch.cos(velocity_angle)
        drag_y = -drag_magnitude * torch.sin(velocity_angle)
        
        # 计算升力分量（垂直于速度方向）- 方向基于舵量符号
        lift_direction = torch.sign(rudder)
        lift_x = lift_magnitude * perpendicular_x * lift_direction
        lift_y = lift_magnitude * perpendicular_y * lift_direction
        
        # 计算合力
        total_force_x = thrust_x + drag_x + lift_x
        total_force_y = thrust_y + drag_y + lift_y
        
        # 计算加速度
        ax = total_force_x
        ay = total_force_y
        
        # 更新速度
        vx = vx + ax * dt_tensor
        vy = vy + ay * dt_tensor
        
        # 计算新的速度大小
        new_speed = torch.sqrt(vx*vx + vy*vy)
        
        # 更新位置
        x = x + vx * dt_tensor
        y = y + vy * dt_tensor
        
        # 计算新的朝向角度（度）
        new_angle = torch.atan2(vy, vx) * self.RAD_TO_DEG
        # 确保角度在0-360范围内
        new_angle = new_angle % 360.0
        
        # 计算横向过载（G）
        # 创建一个掩码来避免除以零
        non_zero_rudder = (torch.abs(rudder) > 1e-6)
        
        # 计算向心加速度
        centripetal_acc = torch.where(
            non_zero_rudder,
            lift_magnitude,
            torch.tensor(0.0, device=self.device)
        )
        
        # 计算过载（G）
        n_load = torch.where(
            non_zero_rudder,
            centripetal_acc / self.GRAVITY + 1.0,  # +1.0表示重力引起的1G
            torch.tensor(1.0, device=self.device)  # 直线飞行时过载为1G
        )
        
        # 更新飞行器状态
        aircraft['x'] = x
        aircraft['y'] = y
        aircraft['vx'] = vx
        aircraft['vy'] = vy
        aircraft['speed'] = new_speed
        aircraft['angle'] = new_angle
        aircraft['n_load'] = n_load
        
        return aircraft
    
    def limit_to_battlefield(self, aircraft, battlefield_size):
        """限制飞行器在战场范围内
        
        Args:
            aircraft: 飞行器张量字典
            battlefield_size: 战场大小
            
        Returns:
            dict: 更新后的飞行器张量字典
        """
        # 转换为张量
        if not isinstance(battlefield_size, torch.Tensor):
            battlefield_size = torch.tensor(battlefield_size, device=self.device)
            
        # 限制x, y在战场范围内
        aircraft['x'] = torch.clamp(aircraft['x'], torch.tensor(0.0, device=self.device), battlefield_size)
        aircraft['y'] = torch.clamp(aircraft['y'], torch.tensor(0.0, device=self.device), battlefield_size)
        
        return aircraft 