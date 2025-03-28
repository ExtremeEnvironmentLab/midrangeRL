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

# 初始化Pygame
pygame.init()

# ========================
# 物理参数配置（可调整区域）
# ========================

# 初始距离参数
INITIAL_DISTANCE_RATIO = 0.3  # 初始位置比例（0-0.5）

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
            self.guidance_gain = 200  # 比例引导系数
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
    
    def update(self, keys):
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        if self.is_missile and self.target:
            # 计算弹目连线角度
            dx = self.target.x - self.x
            dy = self.target.y - self.y
            los_angle = math.degrees(math.atan2(dy, dx))
            
            # 计算视线角速度（弹目连线角度变化率）
            angle_diff = (los_angle - self.angle + 180) % 360 - 180
            self.rudder = self.guidance_gain * angle_diff / 180.0  # 归一化到[-1,1]范围
            self.rudder = max(-1.0, min(1.0, self.rudder))  # 限制舵量范围
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
        self.x = max(0, min(BATTLEFIELD_SIZE, self.x))
        self.y = max(0, min(BATTLEFIELD_SIZE, self.y))
    
    def fire_missile(self):
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
    
    def draw(self):
        # 将实际坐标转换为屏幕坐标
        screen_x = int(self.x * SCALE_FACTOR)
        screen_y = int(self.y * SCALE_FACTOR)
        
        # 绘制轨迹
        if len(self.trail) > 1:
            points = []
            for i, (trail_x, trail_y) in enumerate(self.trail):
                trail_screen_x = int(trail_x * SCALE_FACTOR)
                trail_screen_y = int(trail_y * SCALE_FACTOR)
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

class Game:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('simhei', 24)  # 使用中文字体
        self.running = True
        self.winner = None  # 记录胜利方
        
        # 创建两架飞机,位于对角线的10%和90%处
        start_x1 = BATTLEFIELD_SIZE * INITIAL_DISTANCE_RATIO
        start_y1 = BATTLEFIELD_SIZE * INITIAL_DISTANCE_RATIO
        start_x2 = BATTLEFIELD_SIZE * (1 - INITIAL_DISTANCE_RATIO)
        start_y2 = BATTLEFIELD_SIZE * (1 - INITIAL_DISTANCE_RATIO)
        
        # 计算朝向战场中心的角度
        angle1 = math.degrees(math.atan2(BATTLEFIELD_SIZE/2 - start_y1, BATTLEFIELD_SIZE/2 - start_x1))
        angle2 = math.degrees(math.atan2(BATTLEFIELD_SIZE/2 - start_y2, BATTLEFIELD_SIZE/2 - start_x2))
        
        self.aircraft1 = Aircraft(start_x1, start_y1, angle1, RED, True, is_missile=False)
        self.aircraft2 = Aircraft(start_x2, start_y2, angle2, BLUE, False, is_missile=False)
        
        # 导弹列表
        self.missiles = []
        
        # 发射冷却时间（秒）
        self.missile_cooldown = 0.5
        self.last_fire1_time = 0
        self.last_fire2_time = 0
    
    def draw_battlefield(self):
        # 绘制战场边界
        pygame.draw.rect(window, WHITE, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), 2)
        
        # 绘制中心点标记
        center_x = WINDOW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2
        pygame.draw.circle(window, WHITE, (center_x, center_y), 5, 1)
        pygame.draw.line(window, WHITE, (center_x - 10, center_y), (center_x + 10, center_y), 1)
        pygame.draw.line(window, WHITE, (center_x, center_y - 10), (center_x, center_y + 10), 1)
        
    def draw_ui(self):
        # 显示红方飞机状态
        speed_text1 = self.font.render(f'红方速度: {int(self.aircraft1.speed)} m/s', True, RED)
        throttle_text1 = self.font.render(f'红方油门: {int(self.aircraft1.throttle * 100)}%', True, RED)
        rudder_text1 = self.font.render(f'红方舵量: {int(self.aircraft1.rudder * 100)}%', True, RED)
        load_text1 = self.font.render(f'红方过载: {self.aircraft1.n_load:.2f} G', True, RED)
        turn_rate_text1 = self.font.render(f'红方角速度: {int(self.aircraft1.turn_rate)}°/s', True, RED)
        missiles_text1 = self.font.render(f'红方导弹: {self.aircraft1.missiles}', True, RED)
        
        window.blit(speed_text1, (10, 10))
        window.blit(throttle_text1, (10, 40))
        window.blit(rudder_text1, (10, 70))
        window.blit(load_text1, (10, 100))
        window.blit(turn_rate_text1, (10, 130))
        window.blit(missiles_text1, (10, 160))
        
        # 显示蓝方飞机状态
        speed_text2 = self.font.render(f'蓝方速度: {int(self.aircraft2.speed)} m/s', True, BLUE)
        throttle_text2 = self.font.render(f'蓝方油门: {int(self.aircraft2.throttle * 100)}%', True, BLUE)
        rudder_text2 = self.font.render(f'蓝方舵量: {int(self.aircraft2.rudder * 100)}%', True, BLUE)
        load_text2 = self.font.render(f'蓝方过载: {self.aircraft2.n_load:.2f} G', True, BLUE)
        turn_rate_text2 = self.font.render(f'蓝方角速度: {int(self.aircraft2.turn_rate)}°/s', True, BLUE)
        missiles_text2 = self.font.render(f'蓝方导弹: {self.aircraft2.missiles}', True, BLUE)
        
        window.blit(speed_text2, (WINDOW_WIDTH - 200, 10))
        window.blit(throttle_text2, (WINDOW_WIDTH - 200, 40))
        window.blit(rudder_text2, (WINDOW_WIDTH - 200, 70))
        window.blit(load_text2, (WINDOW_WIDTH - 200, 100))
        window.blit(turn_rate_text2, (WINDOW_WIDTH - 200, 130))
        window.blit(missiles_text2, (WINDOW_WIDTH - 200, 160))
        
        # 显示控制说明
        controls_text = self.font.render('红方: WASD控制, T发射导弹 | 蓝方: 方向键控制, =发射导弹', True, WHITE)
        window.blit(controls_text, (WINDOW_WIDTH // 2 - 250, WINDOW_HEIGHT - 30))
    
    def run(self):
        while self.running:
            current_time = time.time()
            
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # 获取按键状态
            keys = pygame.key.get_pressed()
            
            # 检测导弹发射
            # 玩家1发射导弹
            if keys[self.aircraft1.key_fire] and current_time - self.last_fire1_time > self.missile_cooldown:
                missile = self.aircraft1.fire_missile()
                if missile:
                    self.missiles.append(missile)
                    self.last_fire1_time = current_time
            
            # 玩家2发射导弹
            if keys[self.aircraft2.key_fire] and current_time - self.last_fire2_time > self.missile_cooldown:
                missile = self.aircraft2.fire_missile()
                if missile:
                    self.missiles.append(missile)
                    self.last_fire2_time = current_time
            
            # 更新飞机状态
            self.aircraft1.update(keys)
            self.aircraft2.update(keys)
            
            # 更新导弹状态
            for missile in self.missiles[:]:  # 使用副本进行迭代,以便安全删除
                missile.update(keys)
                
                # 检查导弹是否击中目标（碰撞检测）
                target = missile.target
                distance = math.sqrt((missile.x - target.x)**2 + (missile.y - target.y)**2)
                if distance < 100:  # 爆炸半径为100米
                    self.missiles.remove(missile)
                    self.winner = '红方' if missile.is_player1 else '蓝方'
                    break
                
                # 检查导弹是否飞出战场或速度过低
                if (missile.x <= 0 or missile.x >= BATTLEFIELD_SIZE or 
                    missile.y <= 0 or missile.y >= BATTLEFIELD_SIZE or 
                    missile.speed < 200):
                    self.missiles.remove(missile)
            
            # 绘制游戏画面
            window.fill(BLACK)
            self.draw_battlefield()
            
            # 绘制导弹
            for missile in self.missiles:
                missile.draw()
                
            # 绘制飞机（在导弹之后绘制,使飞机显示在导弹上层）
            self.aircraft1.draw()
            self.aircraft2.draw()
            
            self.draw_ui()
            
            # 更新显示
            pygame.display.flip()
            
            # 如果游戏结束,显示胜利信息
            if self.winner:
                victory_text = self.font.render(f'{self.winner}胜利！', True, WHITE)
                text_rect = victory_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
                window.blit(victory_text, text_rect)
                pygame.display.flip()
                # 移除自动关闭的等待时间
                while True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                            return
                    self.clock.tick(60)
            
            # 控制帧率
            self.clock.tick(60)

# 运行游戏
if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit()