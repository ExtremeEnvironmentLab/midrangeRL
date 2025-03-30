# -*- coding: utf-8 -*-

import math
import numpy as np

class Aerodynamic:
    def __init__(self):
        # 战斗机参数
        self.FIGHTER_TERMINAL_VELOCITY = 400  # 自由落体终端速度 (m/s)
        self.FIGHTER_MIN_TURN_RADIUS = 1000   # 最小转弯半径 (m)
        self.FIGHTER_CL_MAX = 1 / self.FIGHTER_MIN_TURN_RADIUS  # 最大升力系数
        self.FIGHTER_MAX_THRUST = 1.5 * 9.8  # 最大推力 (m/s²)
        self.FIGHTER_LIFT_DRAG_RATIO = 5     # 最大升阻比

        # 导弹参数
        self.MISSILE_TERMINAL_VELOCITY = 400 # 自由落体终端速度 (m/s)
        self.MISSILE_MIN_TURN_RADIUS = 1000  # 最小转弯半径 (m)
        self.MISSILE_CL_MAX = 1 / self.MISSILE_MIN_TURN_RADIUS  # 最大升力系数
        self.MISSILE_THRUST = 15 * 9.8        # 发动机推力 (m/s²)
        self.MISSILE_ENGINE_DURATION = 10.0 # 发动机工作时间 (s)
        self.MISSILE_LIFT_DRAG_RATIO = 3     # 最大升阻比

    def calculate_aircraft_physics(self, aircraft, dt):
        """计算飞机/导弹的物理参数和运动学
        
        Args:
            aircraft: 飞机或导弹对象
            dt: 时间步长
        """
        # 根据实体类型选择物理参数
        if aircraft.is_missile:
            terminal_velocity = self.MISSILE_TERMINAL_VELOCITY
            cl_max = self.MISSILE_CL_MAX
            min_turn_radius = self.MISSILE_MIN_TURN_RADIUS
            max_thrust = self.MISSILE_THRUST
            lift_drag_ratio = self.MISSILE_LIFT_DRAG_RATIO
            # 更新剩余发动机工作时间
            aircraft.engine_time = max(0, aircraft.engine_time - dt)
            # 发动机燃烧期间提供推力，燃耗完毕后无推力
            thrust = max_thrust if aircraft.engine_time > 0 else 0.0
        else:
            terminal_velocity = self.FIGHTER_TERMINAL_VELOCITY
            cl_max = self.FIGHTER_CL_MAX
            min_turn_radius = self.FIGHTER_MIN_TURN_RADIUS
            max_thrust = self.FIGHTER_MAX_THRUST
            lift_drag_ratio = self.FIGHTER_LIFT_DRAG_RATIO
            # 飞机的推力由油门控制
            thrust = aircraft.throttle * max_thrust
        
        # 计算空气动力参数
        # 1. 计算零升阻力系数（基于终端速度）
        Cd0 = 9.8 / (terminal_velocity ** 2)
        
        # 2. 计算升力系数（与舵量成正比）
        Cl = abs(aircraft.rudder) * cl_max
        
        # 3. 计算升阻比相关参数
        k = (cl_max - lift_drag_ratio * Cd0) / (lift_drag_ratio * cl_max)
        
        # 4. 总阻力系数
        Cd = Cd0 + k * Cl
        
        # 计算阻力
        Drag_force = Cd * aircraft.speed ** 2
        
        # 计算净加速度并更新速度
        acceleration = thrust - Drag_force
        aircraft.speed += acceleration * dt
        
        # 限制最小速度不低于0
        aircraft.speed = max(aircraft.speed, 0)

        # 更新角度（根据舵量和最小转弯半径）
        actual_turn_radius = min_turn_radius / abs(aircraft.rudder) if aircraft.rudder != 0 else float('inf')
        aircraft.turn_rate = math.degrees(aircraft.speed / actual_turn_radius) * aircraft.rudder
        aircraft.angle += aircraft.turn_rate * dt
        aircraft.angle %= 360  # 保持角度在0-360范围内

        # 计算横向过载（单位：G）
        aircraft.n_load = aircraft.speed ** 2 / (actual_turn_radius * 9.8) if aircraft.rudder != 0 else 1.0

        # 更新位置
        rad = math.radians(aircraft.angle)
        aircraft.x += math.cos(rad) * aircraft.speed * dt
        aircraft.y += math.sin(rad) * aircraft.speed * dt
        
        return aircraft

    def create_aircraft(self, x, y, angle, is_missile=False):
        """创建一个新的飞行器属性字典
        
        Args:
            x: 初始x坐标
            y: 初始y坐标
            angle: 初始角度
            is_missile: 是否为导弹
        
        Returns:
            dict: 飞行器属性字典
        """
        aircraft = {
            'x': x,
            'y': y,
            'angle': angle,
            'speed': 300,  # 初始速度
            'throttle': 1.0,  # 油门
            'rudder': 0.0,  # 舵量
            'is_missile': is_missile,
            'turn_rate': 0.0,  # 角速度
            'n_load': 1.0,  # 过载
        }
        
        if is_missile:
            aircraft['engine_time'] = self.MISSILE_ENGINE_DURATION
            aircraft['target'] = None
        else:
            aircraft['missiles'] = 6  # 导弹数量
        
        return aircraft