# -*- coding: utf-8 -*-

import pygame
import math
import time
from collections import deque
from missile_guidance import MissileGuidance

# ========================
# 物理参数配置（可调整区域）
# ========================

# 战斗机参数
FIGHTER_TERMINAL_VELOCITY = 400  # 自由落体终端速度 (m/s)
FIGHTER_MIN_TURN_RADIUS = 1000   # 最小转弯半径 (m)
FIGHTER_CL_MAX = 1 / FIGHTER_MIN_TURN_RADIUS# 最大升力系数
FIGHTER_MAX_THRUST = 1.5 * 9.8  # 最大推力 (m/s²)
FIGHTER_LIFT_DRAG_RATIO = 5     # 最大升阻比

# 导弹参数
MISSILE_TERMINAL_VELOCITY = 400 # 自由落体终端速度 (m/s)
MISSILE_MIN_TURN_RADIUS = 1000  # 最小转弯半径 (m)
MISSILE_CL_MAX = 1 / MISSILE_MIN_TURN_RADIUS# 最大升力系数
MISSILE_THRUST = 15 * 9.8        # 发动机推力 (m/s²)
MISSILE_ENGINE_DURATION = 10.0 # 发动机工作时间 (s)
MISSILE_LIFT_DRAG_RATIO = 3     # 最大升阻比

# 颜色定义
LIGHT_RED = (255, 150, 150)
LIGHT_BLUE = (150, 150, 255)

class Aircraft:
    def __init__(self, x, y, angle, color, is_player1=True, is_missile=False):
        # 根据实体类型选择物理参数
        if is_missile:
            self.terminal_velocity = MISSILE_TERMINAL_VELOCITY
            self.cl_max = MISSILE_CL_MAX
            self.min_turn_radius = MISSILE_MIN_TURN_RADIUS
            self.max_thrust = MISSILE_THRUST
            self.lift_drag_ratio = MISSILE_LIFT_DRAG_RATIO
            self.engine_time = MISSILE_ENGINE_DURATION
            self.target = None  # 导弹的目标
            self.guidance = MissileGuidance()  # 导弹制导系统
        else:
            self.terminal_velocity = FIGHTER_TERMINAL_VELOCITY
            self.cl_max = FIGHTER_CL_MAX
            self.min_turn_radius = FIGHTER_MIN_TURN_RADIUS
            self.max_thrust = FIGHTER_MAX_THRUST
            self.lift_drag_ratio = FIGHTER_LIFT_DRAG_RATIO
            self.engine_time = 0.0

        self.is_missile = is_missile
        self.x = x  # 实际坐标（米）
        self.y = y  # 实际坐标（米）
        self.angle = angle  # 朝向角度（度）
        self.speed = 300  # 初始速度（米/秒）
        self.throttle = 1.0  # 油门位置（0-1）
        self.rudder = 0.0  # 舵量（-1到1）
        self.missiles = 6  # 导弹数量
        self.color = color  # 飞机颜色
        self.is_player1 = is_player1  # 是否为玩家1
        self.trail = deque(maxlen=100)  # 存储轨迹点
        self.trail_update_count = 0  # 轨迹点更新计时器
        self.last_update_time = time.time()
        self.turn_rate = 0.0  # 角速度（度/秒）
        
        # 控制键
        if is_player1:
            self.key_left = pygame.K_a
            self.key_right = pygame.K_d
            self.key_throttle_up = pygame.K_w
            self.key_throttle_down = pygame.K_s
            self.key_fire = pygame.K_t
        else:
            self.key_left = pygame.K_LEFT
            self.key_right = pygame.K_RIGHT
            self.key_throttle_up = pygame.K_UP
            self.key_throttle_down = pygame.K_DOWN
            self.key_fire = pygame.K_EQUALS  # 等号键
        
        # 舵量和油门控制变量
        self.rudder_input = 0  # 当前舵量输入状态
        self.throttle_input = 0  # 当前油门输入状态
        self.last_rudder_time = 0  # 上次舵量变化时间
        self.last_throttle_time = 0  # 上次油门变化时间
    
    def update(self, keys, battlefield_size, game):
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        if self.is_missile and self.target:
            # 使用导弹制导系统计算舵量
            self.rudder = self.guidance.calculate_guidance(self, self.target)
        else:
            # 处理舵量输入
            if keys[self.key_left]:
                self.rudder_input = -1
                # 计算舵量变化速率
                rudder_change_rate = 1.0  # 1秒内从0到1
                self.rudder += rudder_change_rate * dt * self.rudder_input
                # 限制舵量范围
                self.rudder = max(-1.0, min(1.0, self.rudder))
            elif keys[self.key_right]:
                self.rudder_input = 1
                # 计算舵量变化速率
                rudder_change_rate = 1.0  # 1秒内从0到1
                self.rudder += rudder_change_rate * dt * self.rudder_input
                # 限制舵量范围
                self.rudder = max(-1.0, min(1.0, self.rudder))
            else:
                # 松开按键立即开始回正
                if abs(self.rudder) > 0.01:
                    rudder_return_rate = 1.0  # 1秒内回正
                    if self.rudder > 0:
                        self.rudder -= rudder_return_rate * dt
                        self.rudder = max(0, self.rudder)
                    else:
                        self.rudder += rudder_return_rate * dt
                        self.rudder = min(0, self.rudder)
                else:
                    self.rudder = 0
                    self.rudder_input = 0
        
        # 处理油门输入
        if keys[self.key_throttle_up]:
            self.throttle_input = 1
            self.last_throttle_time = current_time
        elif keys[self.key_throttle_down]:
            self.throttle_input = -1
            self.last_throttle_time = current_time
        else:
            self.throttle_input = 0
        
        # 根据输入状态更新油门,两秒内从0到1
        if self.throttle_input != 0:
            throttle_change_rate = 0.5  # 2秒内从0到1
            self.throttle += throttle_change_rate * dt * self.throttle_input
            # 限制油门范围
            self.throttle = max(0.0, min(1.0, self.throttle))
        
        # 计算空气动力参数
        # 1. 计算零升阻力系数（基于终端速度）
        Cd0 = 9.8 / (self.terminal_velocity ** 2)
        
        # 2. 计算升力系数（与舵量成正比）
        Cl = abs(self.rudder) * self.cl_max
        
        # 3. 计算升阻比相关参数
        # 公式推导：k = (Clmax - n*Cd0) / (n*Clmax)
        # 其中n为最大升阻比
        k = (self.cl_max - self.lift_drag_ratio * Cd0) / (self.lift_drag_ratio * self.cl_max)
        
        # 4. 总阻力系数
        Cd = Cd0 + k * Cl
        
        # 计算推力和阻力
        if self.is_missile:
            Thrust = self.max_thrust if self.engine_time > 0 else 0.0
            self.engine_time = max(0, self.engine_time - dt)
        else:
            Thrust = self.throttle * self.max_thrust
        
        Drag_force = Cd * self.speed ** 2
        
        # 计算净加速度并更新速度
        acceleration = Thrust - Drag_force
        self.speed += acceleration * dt
        
        # 限制最小速度不低于0
        self.speed = max(self.speed, 0)

        # 更新角度（根据舵量和最小转弯半径）
        actual_turn_radius = self.min_turn_radius / abs(self.rudder) if self.rudder != 0 else float('inf')
        self.turn_rate = math.degrees(self.speed / actual_turn_radius) * self.rudder
        self.angle += self.turn_rate * dt
        self.angle %= 360  # 保持角度在0-360范围内

        # 计算横向过载（单位：G）
        self.n_load =  self.speed ** 2 / ( actual_turn_radius * 9.8 )

        # 更新位置
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed * dt
        self.y += math.sin(rad) * self.speed * dt
        
        # 添加当前位置到轨迹
        if self.trail_update_count == 10:
            self.trail.append((self.x, self.y))
            self.trail_update_count = 0
        else:
            self.trail_update_count += 1
        
        # 确保飞机不会飞出战场
        self.x = max(0, min(battlefield_size, self.x))
        self.y = max(0, min(battlefield_size, self.y))

    def fire_missile(self, game):
        # 检查是否还有导弹
        if self.missiles <= 0:
            return None
        
        # 减少导弹数量
        self.missiles -= 1
        
        # 创建导弹实体（位置略微前移,以避免碰撞）
        offset = 20  # 前移距离（米）
        missile_x = self.x + math.cos(math.radians(self.angle)) * offset
        missile_y = self.y + math.sin(math.radians(self.angle)) * offset
        
        # 创建导弹（继承战斗机的速度和方向）
        # 降低颜色饱和度
        if self.is_player1:
            # 红方导弹颜色（降低饱和度）
            missile_color = LIGHT_RED
        else:
            # 蓝方导弹颜色（降低饱和度）
            missile_color = LIGHT_BLUE
        
        missile = Aircraft(missile_x, missile_y, self.angle, missile_color, self.is_player1, is_missile=True)
        # 设置导弹的目标为敌方战斗机
        missile.target = game.aircraft2 if self.is_player1 else game.aircraft1
        # 导弹继承战斗机的速度作为初始速度
        missile.speed = self.speed
        # 导弹舵量保持为0
        missile.rudder = 0.0
        
        # 设置导弹的最后更新时间为当前时间
        missile.last_update_time = time.time()
        
        return missile

    def draw(self, window, scale_factor):
        # 将实际坐标转换为屏幕坐标
        screen_x = int(self.x * scale_factor)
        screen_y = int(self.y * scale_factor)
        
        # 绘制轨迹
        if len(self.trail) > 1:
            points = []
            for i, (trail_x, trail_y) in enumerate(self.trail):
                trail_screen_x = int(trail_x * scale_factor)
                trail_screen_y = int(trail_y * scale_factor)
                points.append((trail_screen_x, trail_screen_y))
                
                # 根据点的新旧程度调整透明度
                alpha = int(255 * (i / len(self.trail)))
                trail_color = (*self.color[:3], alpha)
                
                if i > 0:
                    pygame.draw.line(window, trail_color, points[i-1], points[i], 1)
        
        if self.is_missile:
            # 绘制导弹（小三角形）
            triangle_size = 4
            angle_rad = math.radians(self.angle)
            # 计算三角形的三个顶点
            point1 = (screen_x + int(math.cos(angle_rad) * triangle_size * 2), 
                      screen_y + int(math.sin(angle_rad) * triangle_size * 2))
            point2 = (screen_x + int(math.cos(angle_rad + math.pi * 2/3) * triangle_size), 
                      screen_y + int(math.sin(angle_rad + math.pi * 2/3) * triangle_size))
            point3 = (screen_x + int(math.cos(angle_rad + math.pi * 4/3) * triangle_size), 
                      screen_y + int(math.sin(angle_rad + math.pi * 4/3) * triangle_size))
            pygame.draw.polygon(window, self.color, [point1, point2, point3])
        else:
            # 绘制飞机（圆点）
            pygame.draw.circle(window, self.color, (screen_x, screen_y), 5)
        
        # 绘制速度向量（表示方向和大小）
        vector_length = self.speed / 10  # 缩放速度向量长度
        end_x = screen_x + int(math.cos(math.radians(self.angle)) * vector_length)
        end_y = screen_y + int(math.sin(math.radians(self.angle)) * vector_length)
        pygame.draw.line(window, self.color, (screen_x, screen_y), (end_x, end_y), 2) 