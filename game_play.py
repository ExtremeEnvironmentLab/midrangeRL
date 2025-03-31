# -*- coding: utf-8 -*-

'''
游戏背景：
游戏场景是现实中三代机超视距空战的俯视图2D简化版本,
战场是一个边长50km的正方形,交战的红蓝双方各为1架战斗机,
初始位置在正方形对角线的10%与90%处,初始速度300,朝向正方形战场中心,油门为最大,舵量为0。
每架战斗机携带6发中距弹(挂在战斗机上时只是一个数字,不作为实体计算,也不考虑其质量),
胜利条件是在保存自己的实力情况下消灭全部敌机,双方导弹均耗尽时未分出胜负算平局。
'''

import pygame
import math
import time
import random
import numpy as np
from collections import deque
import os

# 导入环境组件
from visualization import Visualizer
from env_numpy.aerodynamic import Aerodynamic
from env_numpy.game_events import GameEvents
from env_numpy.missile_guidance import MissileGuidance

# 初始化Pygame
pygame.init()

# ========================
# 游戏参数配置
# ========================

# 初始距离参数
INITIAL_DISTANCE_RATIO = 0.4  # 初始位置比例（0-0.5）

# 战场大小参数
BATTLEFIELD_SIZE = 50000  # 50km的战场大小

# ========================
# 游戏显示常量
# ========================
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
SCALE_FACTOR = WINDOW_WIDTH / BATTLEFIELD_SIZE

# 设置中文字体
def get_system_font():
    """获取系统中可用的中文字体"""
    # Windows系统字体路径
    windows_fonts = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/simkai.ttf",  # 楷体
    ]
    
    # 尝试加载Windows系统字体
    for font_path in windows_fonts:
        if os.path.exists(font_path):
            try:
                return pygame.font.Font(font_path, 24)
            except:
                continue
    
    # 如果Windows字体都失败，尝试使用系统字体
    try:
        return pygame.font.SysFont('microsoft yahei', 24)
    except:
        try:
            return pygame.font.SysFont('simhei', 24)
        except:
            # 如果都失败，使用默认字体
            return pygame.font.Font(None, 36)

class Aircraft:
    def __init__(self, x, y, angle, color, is_player1=True, is_missile=False):
        self.x = x  # 实际坐标（米）
        self.y = y  # 实际坐标（米）
        self.angle = angle  # 朝向角度（度）
        self.speed = 300  # 初始速度（米/秒）
        # 添加速度向量分量
        rad = math.radians(angle)
        self.vx = math.cos(rad) * self.speed  # 速度x分量
        self.vy = math.sin(rad) * self.speed  # 速度y分量
        self.throttle = 1.0  # 油门位置（0-1）
        self.rudder = 0.0  # 舵量（-1到1）
        self.missiles = 6  # 导弹数量
        self.color = color  # 飞机颜色
        self.is_player1 = is_player1  # 是否为玩家1
        self.is_missile = is_missile  # 是否为导弹
        self.trail = deque(maxlen=100)  # 存储轨迹点
        self.trail_update_count = 0  # 轨迹点更新计时器
        self.last_update_time = time.time()
        self.turn_rate = 0.0  # 角速度（度/秒）
        self.n_load = 1.0  # 横向过载（G）
        self.alive = True  # 存活状态
        
        # 如果是导弹，设置额外属性
        if is_missile:
            self.target = None  # 导弹的目标
            self.engine_time = 10.0  # 发动机工作时间
        
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

class Game:
    def __init__(self):
        # 初始化字体
        self.font = get_system_font()
        
        # 初始化可视化系统
        self.visualizer = Visualizer(WINDOW_WIDTH, WINDOW_HEIGHT, BATTLEFIELD_SIZE, self.font)
        
        # 初始化物理引擎
        self.aero = Aerodynamic()
        
        # 初始化游戏事件系统
        self.events = GameEvents(BATTLEFIELD_SIZE)
        
        # 初始化导弹制导系统
        self.guidance = MissileGuidance()
        
        # 计算初始位置
        red_x = BATTLEFIELD_SIZE * INITIAL_DISTANCE_RATIO
        red_y = BATTLEFIELD_SIZE * INITIAL_DISTANCE_RATIO
        blue_x = BATTLEFIELD_SIZE * (1 - INITIAL_DISTANCE_RATIO)
        blue_y = BATTLEFIELD_SIZE * (1 - INITIAL_DISTANCE_RATIO)
        
        # 计算初始朝向（朝向战场中心）
        red_angle = math.degrees(math.atan2(BATTLEFIELD_SIZE/2 - red_y, BATTLEFIELD_SIZE/2 - red_x))
        blue_angle = math.degrees(math.atan2(BATTLEFIELD_SIZE/2 - blue_y, BATTLEFIELD_SIZE/2 - blue_x))
        
        # 颜色定义
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        
        # 初始化红蓝双方飞机
        self.aircraft1 = Aircraft(red_x, red_y, red_angle, self.RED, True)
        self.aircraft2 = Aircraft(blue_x, blue_y, blue_angle, self.BLUE, False)
        self.missiles = []  # 存储所有导弹
        
        self.game_over = False
        self.winner = None
        
        # 添加发射冷却时间
        self.last_fire_time = 0
        self.fire_cooldown = 0.5  # 发射冷却时间（秒）
        
        # 按键状态记录
        self.rudder_input1 = 0  # 红方舵量输入状态
        self.rudder_input2 = 0  # 蓝方舵量输入状态
        self.throttle_input1 = 0  # 红方油门输入状态
        self.throttle_input2 = 0  # 蓝方油门输入状态
    
    def handle_input(self, keys, dt):
        """处理玩家输入"""
        # 处理红方舵量输入
        if keys[self.aircraft1.key_left]:
            self.rudder_input1 = -1
        elif keys[self.aircraft1.key_right]:
            self.rudder_input1 = 1
        else:
            self.rudder_input1 = 0
        
        # 处理蓝方舵量输入
        if keys[self.aircraft2.key_left]:
            self.rudder_input2 = -1
        elif keys[self.aircraft2.key_right]:
            self.rudder_input2 = 1
        else:
            self.rudder_input2 = 0
        
        # 处理红方油门输入
        if keys[self.aircraft1.key_throttle_up]:
            self.throttle_input1 = 1
        elif keys[self.aircraft1.key_throttle_down]:
            self.throttle_input1 = -1
        else:
            self.throttle_input1 = 0
        
        # 处理蓝方油门输入
        if keys[self.aircraft2.key_throttle_up]:
            self.throttle_input2 = 1
        elif keys[self.aircraft2.key_throttle_down]:
            self.throttle_input2 = -1
        else:
            self.throttle_input2 = 0
        
        # 根据输入更新红方舵量
        if self.rudder_input1 != 0:
            rudder_change_rate = 1.0  # 1秒内从0到1
            self.aircraft1.rudder += rudder_change_rate * dt * self.rudder_input1
            self.aircraft1.rudder = max(-1.0, min(1.0, self.aircraft1.rudder))
        else:
            # 松开按键立即开始回正
            if abs(self.aircraft1.rudder) > 0.01:
                rudder_return_rate = 1.0  # 1秒内回正
                if self.aircraft1.rudder > 0:
                    self.aircraft1.rudder -= rudder_return_rate * dt
                    self.aircraft1.rudder = max(0, self.aircraft1.rudder)
                else:
                    self.aircraft1.rudder += rudder_return_rate * dt
                    self.aircraft1.rudder = min(0, self.aircraft1.rudder)
            else:
                self.aircraft1.rudder = 0
        
        # 根据输入更新蓝方舵量
        if self.rudder_input2 != 0:
            rudder_change_rate = 1.0  # 1秒内从0到1
            self.aircraft2.rudder += rudder_change_rate * dt * self.rudder_input2
            self.aircraft2.rudder = max(-1.0, min(1.0, self.aircraft2.rudder))
        else:
            # 松开按键立即开始回正
            if abs(self.aircraft2.rudder) > 0.01:
                rudder_return_rate = 1.0  # 1秒内回正
                if self.aircraft2.rudder > 0:
                    self.aircraft2.rudder -= rudder_return_rate * dt
                    self.aircraft2.rudder = max(0, self.aircraft2.rudder)
                else:
                    self.aircraft2.rudder += rudder_return_rate * dt
                    self.aircraft2.rudder = min(0, self.aircraft2.rudder)
            else:
                self.aircraft2.rudder = 0
        
        # 根据输入更新红方油门
        if self.throttle_input1 != 0:
            throttle_change_rate = 0.5  # 2秒内从0到1
            self.aircraft1.throttle += throttle_change_rate * dt * self.throttle_input1
            self.aircraft1.throttle = max(0.0, min(1.0, self.aircraft1.throttle))
        
        # 根据输入更新蓝方油门
        if self.throttle_input2 != 0:
            throttle_change_rate = 0.5  # 2秒内从0到1
            self.aircraft2.throttle += throttle_change_rate * dt * self.throttle_input2
            self.aircraft2.throttle = max(0.0, min(1.0, self.aircraft2.throttle))
    
    def update_game_state(self, dt):
        """更新游戏状态"""
        # 更新飞机物理状态
        self.aero.calculate_aircraft_physics(self.aircraft1, dt)
        self.aero.calculate_aircraft_physics(self.aircraft2, dt)
        
        # 确保飞机不会飞出战场
        self.events.update_battlefield_boundary(self.aircraft1)
        self.events.update_battlefield_boundary(self.aircraft2)
        
        # 更新导弹状态
        for missile in self.missiles[:]:
            # 更新导弹物理状态
            self.aero.calculate_aircraft_physics(missile, dt)
            
            # 更新导弹制导
            if missile.target:
                missile.rudder = self.guidance.calculate_guidance(missile, missile.target)
            
            # 确保导弹不会飞出战场
            self.events.update_battlefield_boundary(missile)
            
            # 检查导弹是否击中目标
            if missile.target:
                dx = missile.x - missile.target.x
                dy = missile.y - missile.target.y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 100:  # 击中目标
                    if missile.target == self.aircraft1:
                        self.aircraft1.alive = False
                        self.game_over = True
                        self.winner = 'blue'
                    else:
                        self.aircraft2.alive = False
                        self.game_over = True
                        self.winner = 'red'
                    self.missiles.remove(missile)
                    continue
            
            # 检查导弹是否自毁
            if missile.speed < 200:
                self.missiles.remove(missile)
                continue
            
            # 更新导弹轨迹
            if missile.trail_update_count >= 5:
                missile.trail.append((missile.x, missile.y))
                missile.trail_update_count = 0
            else:
                missile.trail_update_count += 1
        
        # 更新飞机轨迹
        if self.aircraft1.trail_update_count >= 5:
            self.aircraft1.trail.append((self.aircraft1.x, self.aircraft1.y))
            self.aircraft1.trail_update_count = 0
        else:
            self.aircraft1.trail_update_count += 1
            
        if self.aircraft2.trail_update_count >= 5:
            self.aircraft2.trail.append((self.aircraft2.x, self.aircraft2.y))
            self.aircraft2.trail_update_count = 0
        else:
            self.aircraft2.trail_update_count += 1
        
        # 检查游戏是否结束
        if self.aircraft1.missiles == 0 and self.aircraft2.missiles == 0 and not self.missiles:
            self.game_over = True
            self.winner = 'draw'
    
    def handle_fire_missiles(self, keys):
        """处理导弹发射"""
        current_time = time.time()
        
        # 检查导弹发射（添加冷却时间）
        if current_time - self.last_fire_time >= self.fire_cooldown:
            if keys[self.aircraft1.key_fire] and self.aircraft1.alive:
                # 创建新导弹
                missile = Aircraft(
                    self.aircraft1.x + math.cos(math.radians(self.aircraft1.angle)) * 20,
                    self.aircraft1.y + math.sin(math.radians(self.aircraft1.angle)) * 20,
                    self.aircraft1.angle,
                    self.visualizer.LIGHT_RED,
                    True,
                    True
                )
                # 设置导弹速度和速度向量
                missile.speed = self.aircraft1.speed
                missile.vx = self.aircraft1.vx
                missile.vy = self.aircraft1.vy
                missile.target = self.aircraft2
                
                # 减少导弹数量
                if self.aircraft1.missiles > 0:
                    self.aircraft1.missiles -= 1
                    self.missiles.append(missile)
                    self.last_fire_time = current_time
            
            if keys[self.aircraft2.key_fire] and self.aircraft2.alive:
                # 创建新导弹
                missile = Aircraft(
                    self.aircraft2.x + math.cos(math.radians(self.aircraft2.angle)) * 20,
                    self.aircraft2.y + math.sin(math.radians(self.aircraft2.angle)) * 20,
                    self.aircraft2.angle,
                    self.visualizer.LIGHT_BLUE,
                    False,
                    True
                )
                # 设置导弹速度和速度向量
                missile.speed = self.aircraft2.speed
                missile.vx = self.aircraft2.vx
                missile.vy = self.aircraft2.vy
                missile.target = self.aircraft1
                
                # 减少导弹数量
                if self.aircraft2.missiles > 0:
                    self.aircraft2.missiles -= 1
                    self.missiles.append(missile)
                    self.last_fire_time = current_time
    
    def run(self):
        """运行游戏主循环"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # 计算时间步长
            dt = clock.tick(60) / 1000.0  # 将毫秒转换为秒
            
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # 切换视图模式
                    elif event.key == pygame.K_v:
                        self.visualizer.toggle_view_mode()
            
            # 获取按键状态
            keys = pygame.key.get_pressed()
            
            # 处理玩家输入
            self.handle_input(keys, dt)
            
            # 处理导弹发射
            self.handle_fire_missiles(keys)
            
            # 更新游戏状态
            if not self.game_over:
                self.update_game_state(dt)
            
            # 绘制游戏画面
            self.visualizer.clear()
            self.visualizer.draw_battlefield()
            
            # 绘制飞机和导弹
            self.visualizer.draw_aircraft(self.aircraft1)
            self.visualizer.draw_aircraft(self.aircraft2)
            for missile in self.missiles:
                self.visualizer.draw_aircraft(missile)
            
            # 绘制UI
            self.visualizer.draw_ui(self.aircraft1, self.aircraft2, self.game_over, self.winner)
            self.visualizer.update()
        
        pygame.quit()

if __name__ == '__main__':
    game = Game()
    game.run()