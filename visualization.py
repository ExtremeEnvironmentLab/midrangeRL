# -*- coding: utf-8 -*-

import pygame
import math
from collections import deque

class Visualizer:
    def __init__(self, window_width, window_height, battlefield_size, font):
        self.window_width = window_width
        self.window_height = window_height
        self.battlefield_size = battlefield_size
        self.scale_factor = window_width / battlefield_size
        self.font = font
        
        # 颜色定义
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.LIGHT_RED = (255, 150, 150)
        self.LIGHT_BLUE = (150, 150, 255)
        self.GREEN = (0, 255, 0)
        
        # 创建游戏窗口
        self.window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('中距空战游戏')
        
        # 视图模式
        self.view_mode = "normal"  # 可以是 "normal" 或 "spacetime"（时空图）
    
    def draw_battlefield(self):
        """绘制战场边界和中心点"""
        # 绘制战场边界
        pygame.draw.rect(self.window, self.WHITE, (0, 0, self.window_width, self.window_height), 2)
        
        # 绘制中心点标记
        center_x = self.window_width // 2
        center_y = self.window_height // 2
        pygame.draw.circle(self.window, self.WHITE, (center_x, center_y), 5, 1)
        pygame.draw.line(self.window, self.WHITE, (center_x - 10, center_y), (center_x + 10, center_y), 1)
        pygame.draw.line(self.window, self.WHITE, (center_x, center_y - 10), (center_x, center_y + 10), 1)
    
    def draw_aircraft(self, aircraft):
        """绘制飞机及其轨迹"""
        if isinstance(aircraft, dict):
            # 处理字典格式的飞行器对象
            screen_x = int(aircraft['x'] * self.scale_factor)
            screen_y = int(aircraft['y'] * self.scale_factor)
            angle = aircraft['angle']
            is_missile = aircraft['is_missile']
            
            # 确定颜色
            if is_missile:
                color = self.LIGHT_RED if aircraft.get('is_player1', False) else self.LIGHT_BLUE
            else:
                color = self.RED if aircraft.get('is_player1', False) else self.BLUE
            
            # 绘制轨迹
            if 'trail' in aircraft and len(aircraft['trail']) > 1:
                points = []
                for i, (trail_x, trail_y) in enumerate(aircraft['trail']):
                    trail_screen_x = int(trail_x * self.scale_factor)
                    trail_screen_y = int(trail_y * self.scale_factor)
                    points.append((trail_screen_x, trail_screen_y))
                    
                    # 根据点的新旧程度调整透明度
                    alpha = int(255 * (i / len(aircraft['trail'])))
                    trail_color = (*color[:3], alpha)
                    
                    if i > 0:
                        pygame.draw.line(self.window, trail_color, points[i-1], points[i], 1)
        else:
            # 处理对象格式的飞行器
            screen_x = int(aircraft.x * self.scale_factor)
            screen_y = int(aircraft.y * self.scale_factor)
            angle = aircraft.angle
            is_missile = aircraft.is_missile
            color = aircraft.color
            
            # 绘制轨迹
            if hasattr(aircraft, 'trail') and len(aircraft.trail) > 1:
                points = []
                for i, (trail_x, trail_y) in enumerate(aircraft.trail):
                    trail_screen_x = int(trail_x * self.scale_factor)
                    trail_screen_y = int(trail_y * self.scale_factor)
                    points.append((trail_screen_x, trail_screen_y))
                    
                    # 根据点的新旧程度调整透明度
                    alpha = int(255 * (i / len(aircraft.trail)))
                    trail_color = (*color[:3], alpha)
                    
                    if i > 0:
                        pygame.draw.line(self.window, trail_color, points[i-1], points[i], 1)
        
        if is_missile:
            # 绘制导弹（小三角形）
            triangle_size = 4
            angle_rad = math.radians(angle)
            # 计算三角形的三个顶点
            point1 = (screen_x + int(math.cos(angle_rad) * triangle_size * 2), 
                      screen_y + int(math.sin(angle_rad) * triangle_size * 2))
            point2 = (screen_x + int(math.cos(angle_rad + math.pi * 2/3) * triangle_size), 
                      screen_y + int(math.sin(angle_rad + math.pi * 2/3) * triangle_size))
            point3 = (screen_x + int(math.cos(angle_rad + math.pi * 4/3) * triangle_size), 
                      screen_y + int(math.sin(angle_rad + math.pi * 4/3) * triangle_size))
            pygame.draw.polygon(self.window, color, [point1, point2, point3])
        else:
            # 绘制飞机（圆点）
            pygame.draw.circle(self.window, color, (screen_x, screen_y), 5)
        
        # 绘制速度向量（表示方向和大小）
        speed = aircraft['speed'] if isinstance(aircraft, dict) else aircraft.speed
        vector_length = speed / 10  # 缩放速度向量长度
        end_x = screen_x + int(math.cos(math.radians(angle)) * vector_length)
        end_y = screen_y + int(math.sin(math.radians(angle)) * vector_length)
        pygame.draw.line(self.window, color, (screen_x, screen_y), (end_x, end_y), 2)
    
    def draw_ui(self, aircraft1, aircraft2, game_over=False, winner=None):
        """绘制用户界面"""
        # 获取飞机属性
        if isinstance(aircraft1, dict):
            red_speed = int(aircraft1['speed'])
            red_rudder = aircraft1.get('rudder', 0) * 100
            red_throttle = aircraft1.get('throttle', 0) * 100
            red_load = aircraft1.get('n_load', 1.0)
            red_turn_rate = aircraft1.get('turn_rate', 0)
            red_missiles = aircraft1.get('missiles', 0)
        else:
            red_speed = int(aircraft1.speed)
            red_rudder = aircraft1.rudder * 100
            red_throttle = aircraft1.throttle * 100
            red_load = getattr(aircraft1, 'n_load', 1.0)
            red_turn_rate = getattr(aircraft1, 'turn_rate', 0)
            red_missiles = aircraft1.missiles
        
        if isinstance(aircraft2, dict):
            blue_speed = int(aircraft2['speed'])
            blue_rudder = aircraft2.get('rudder', 0) * 100
            blue_throttle = aircraft2.get('throttle', 0) * 100
            blue_load = aircraft2.get('n_load', 1.0)
            blue_turn_rate = aircraft2.get('turn_rate', 0)
            blue_missiles = aircraft2.get('missiles', 0)
        else:
            blue_speed = int(aircraft2.speed)
            blue_rudder = aircraft2.rudder * 100
            blue_throttle = aircraft2.throttle * 100
            blue_load = getattr(aircraft2, 'n_load', 1.0)
            blue_turn_rate = getattr(aircraft2, 'turn_rate', 0)
            blue_missiles = aircraft2.missiles
        
        # 显示红方飞机状态（左上角）
        red_status = [
            f"红方状态:",
            f"速度: {red_speed}m/s",
            f"舵量: {red_rudder:.1f}%",
            f"油门: {red_throttle:.1f}%",
            f"过载: {red_load:.1f}G",
            f"角速度: {red_turn_rate:.1f}°/s",
            f"导弹: {red_missiles}"
        ]
        
        for i, text in enumerate(red_status):
            text_surface = self.font.render(text, True, self.RED)
            self.window.blit(text_surface, (10, 10 + i * 25))
        
        # 显示蓝方飞机状态（右上角）
        blue_status = [
            f"蓝方状态:",
            f"速度: {blue_speed}m/s",
            f"舵量: {blue_rudder:.1f}%",
            f"油门: {blue_throttle:.1f}%",
            f"过载: {blue_load:.1f}G",
            f"角速度: {blue_turn_rate:.1f}°/s",
            f"导弹: {blue_missiles}"
        ]
        
        for i, text in enumerate(blue_status):
            text_surface = self.font.render(text, True, self.BLUE)
            text_rect = text_surface.get_rect()
            text_rect.right = self.window_width - 10
            text_rect.top = 10 + i * 25
            self.window.blit(text_surface, text_rect)
        
        # 显示控制说明（底部）
        controls_text = self.font.render('红方: WASD控制, T发射导弹 | 蓝方: 方向键控制, =发射导弹', True, self.WHITE)
        self.window.blit(controls_text, (self.window_width // 2 - controls_text.get_width() // 2, self.window_height - 30))
        
        # 显示视图模式
        view_mode_text = self.font.render(f'视图模式: {"常规" if self.view_mode == "normal" else "时空图"}', 
                                         True, self.GREEN)
        self.window.blit(view_mode_text, (10, self.window_height - 60))
        
        # 如果游戏结束，显示获胜者
        if game_over:
            if winner == 'draw':
                winner_text = "游戏结束！平局！"
            else:
                winner_text = f"游戏结束！{'红方' if winner == 'red' else '蓝方'}获胜！"
            text = self.font.render(winner_text, True, self.WHITE)
            text_rect = text.get_rect(center=(self.window_width/2, self.window_height/2))
            self.window.blit(text, text_rect)
    
    def toggle_view_mode(self):
        """切换视图模式（常规/时空图）"""
        self.view_mode = "spacetime" if self.view_mode == "normal" else "normal"
    
    def draw_spacetime_view(self, entities, time_window=30):
        """绘制时空图视图
        
        Args:
            entities: 实体列表
            time_window: 时间窗口（秒）
        """
        # 待实现
        pass
    
    def clear(self):
        """清空屏幕"""
        self.window.fill(self.BLACK)
    
    def update(self):
        """更新显示"""
        pygame.display.flip()