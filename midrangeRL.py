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
from aircraft_model import Aircraft
from visualization import Visualizer
import os

# 初始化Pygame
pygame.init()

# ========================
# 物理参数配置（可调整区域）
# ========================

# 初始距离参数
INITIAL_DISTANCE_RATIO = 0.4  # 初始位置比例（0-0.5）

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

# ========================
# 游戏显示常量
# ========================
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
BATTLEFIELD_SIZE = 50000  # 50km的战场大小
SCALE_FACTOR = WINDOW_WIDTH / BATTLEFIELD_SIZE

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
LIGHT_RED = (255, 150, 150)
LIGHT_BLUE = (150, 150, 255)

# 创建游戏窗口
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('中距空战游戏')

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

# 初始化字体
FONT = get_system_font()

'''
控制方式：
导弹由比例引导法控制,比例引导的目的是将导弹与目标之间的相对速度的切向分量尽量消除,
只剩下径向分量,也就是接近率,具体实现方法有很多种,
比如说可以令导弹的舵量正比于弹目连线旋转的角速度。
当导弹与敌方战斗机距离小于100m时发生爆炸摧毁敌机,
当速度下降到小于200m/s时自爆。
战斗机由玩家控制,
两个玩家分别通过AD和左右方向键控制飞机转向舵量,
按得越久转向舵量越大,连续按1秒钟舵量达到最大,
松开按键则舵量会在一秒钟时回正。
两个玩家分别用·键和=键发射导弹。
WS键和上下方向键则用来调节各自的油门位置,要实现一个类似于积分器的效果,
举个例子,当油门位置为0的时候,按下W键后油门应当在两秒的时间内均匀增加到1,
然后松开W键时油门则固定在1,如果按下S键则导致油门下降,松开S键则停止下降。
'''

class Game:
    def __init__(self):
        # 初始化可视化系统
        self.visualizer = Visualizer(WINDOW_WIDTH, WINDOW_HEIGHT, BATTLEFIELD_SIZE, FONT)
        
        # 计算初始位置
        red_x = BATTLEFIELD_SIZE * INITIAL_DISTANCE_RATIO
        red_y = BATTLEFIELD_SIZE * INITIAL_DISTANCE_RATIO
        blue_x = BATTLEFIELD_SIZE * (1 - INITIAL_DISTANCE_RATIO)
        blue_y = BATTLEFIELD_SIZE * (1 - INITIAL_DISTANCE_RATIO)
        
        # 计算初始朝向（朝向战场中心）
        red_angle = math.degrees(math.atan2(BATTLEFIELD_SIZE/2 - red_y, BATTLEFIELD_SIZE/2 - red_x))
        blue_angle = math.degrees(math.atan2(BATTLEFIELD_SIZE/2 - blue_y, BATTLEFIELD_SIZE/2 - blue_x))
        
        # 初始化红蓝双方飞机
        self.aircraft1 = Aircraft(red_x, red_y, red_angle, self.visualizer.RED, True)
        self.aircraft2 = Aircraft(blue_x, blue_y, blue_angle, self.visualizer.BLUE, False)
        self.missiles = []  # 存储所有导弹
        self.game_over = False
        self.winner = None
        
        # 添加发射冷却时间
        self.last_fire_time = 0
        self.fire_cooldown = 0.5  # 发射冷却时间（秒）
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # 获取按键状态
            keys = pygame.key.get_pressed()
            current_time = time.time()
            
            # 更新飞机状态
            self.aircraft1.update(keys, BATTLEFIELD_SIZE, self)
            self.aircraft2.update(keys, BATTLEFIELD_SIZE, self)
            
            # 检查导弹发射（添加冷却时间）
            if current_time - self.last_fire_time >= self.fire_cooldown:
                if keys[self.aircraft1.key_fire]:
                    missile = self.aircraft1.fire_missile(self)
                    if missile:
                        self.missiles.append(missile)
                        self.last_fire_time = current_time
                
                if keys[self.aircraft2.key_fire]:
                    missile = self.aircraft2.fire_missile(self)
                    if missile:
                        self.missiles.append(missile)
                        self.last_fire_time = current_time
            
            # 更新导弹状态
            for missile in self.missiles[:]:
                missile.update(keys, BATTLEFIELD_SIZE, self)
                
                # 检查导弹是否击中目标
                if missile.target:
                    dx = missile.x - missile.target.x
                    dy = missile.y - missile.target.y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance < 100:  # 击中目标
                        if missile.target == self.aircraft1:
                            self.game_over = True
                            self.winner = 'blue'
                        else:
                            self.game_over = True
                            self.winner = 'red'
                
                # 检查导弹是否自爆
                if missile.speed < 200:
                    self.missiles.remove(missile)
            
            # 检查游戏是否结束
            if self.aircraft1.missiles == 0 and self.aircraft2.missiles == 0 and not self.missiles:
                self.game_over = True
                self.winner = 'draw'
            
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
            
            # 控制帧率
            clock.tick(60)
        
        pygame.quit()

if __name__ == '__main__':
    game = Game()
    game.run()