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
        
        # 创建游戏窗口
        self.window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('中距空战游戏')
    
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
        # 将实际坐标转换为屏幕坐标
        screen_x = int(aircraft.x * self.scale_factor)
        screen_y = int(aircraft.y * self.scale_factor)
        
        # 绘制轨迹
        if len(aircraft.trail) > 1:
            points = []
            for i, (trail_x, trail_y) in enumerate(aircraft.trail):
                trail_screen_x = int(trail_x * self.scale_factor)
                trail_screen_y = int(trail_y * self.scale_factor)
                points.append((trail_screen_x, trail_screen_y))
                
                # 根据点的新旧程度调整透明度
                alpha = int(255 * (i / len(aircraft.trail)))
                trail_color = (*aircraft.color[:3], alpha)
                
                if i > 0:
                    pygame.draw.line(self.window, trail_color, points[i-1], points[i], 1)
        
        if aircraft.is_missile:
            # 绘制导弹（小三角形）
            triangle_size = 4
            angle_rad = math.radians(aircraft.angle)
            # 计算三角形的三个顶点
            point1 = (screen_x + int(math.cos(angle_rad) * triangle_size * 2), 
                      screen_y + int(math.sin(angle_rad) * triangle_size * 2))
            point2 = (screen_x + int(math.cos(angle_rad + math.pi * 2/3) * triangle_size), 
                      screen_y + int(math.sin(angle_rad + math.pi * 2/3) * triangle_size))
            point3 = (screen_x + int(math.cos(angle_rad + math.pi * 4/3) * triangle_size), 
                      screen_y + int(math.sin(angle_rad + math.pi * 4/3) * triangle_size))
            pygame.draw.polygon(self.window, aircraft.color, [point1, point2, point3])
        else:
            # 绘制飞机（圆点）
            pygame.draw.circle(self.window, aircraft.color, (screen_x, screen_y), 5)
        
        # 绘制速度向量（表示方向和大小）
        vector_length = aircraft.speed / 10  # 缩放速度向量长度
        end_x = screen_x + int(math.cos(math.radians(aircraft.angle)) * vector_length)
        end_y = screen_y + int(math.sin(math.radians(aircraft.angle)) * vector_length)
        pygame.draw.line(self.window, aircraft.color, (screen_x, screen_y), (end_x, end_y), 2)
    
    def draw_ui(self, aircraft1, aircraft2, game_over, winner):
        """绘制用户界面"""
        # 显示红方飞机状态（左上角）
        red_status = [
            f"红方状态:",
            f"速度: {int(aircraft1.speed)}m/s",
            f"舵量: {aircraft1.rudder*100:.1f}%",
            f"油门: {aircraft1.throttle*100:.1f}%",
            f"过载: {aircraft1.n_load:.1f}G",
            f"角速度: {aircraft1.turn_rate:.1f}°/s",
            f"导弹: {aircraft1.missiles}"
        ]
        
        for i, text in enumerate(red_status):
            text_surface = self.font.render(text, True, self.RED)
            self.window.blit(text_surface, (10, 10 + i * 25))
        
        # 显示蓝方飞机状态（右上角）
        blue_status = [
            f"蓝方状态:",
            f"速度: {int(aircraft2.speed)}m/s",
            f"舵量: {aircraft2.rudder*100:.1f}%",
            f"油门: {aircraft2.throttle*100:.1f}%",
            f"过载: {aircraft2.n_load:.1f}G",
            f"角速度: {aircraft2.turn_rate:.1f}°/s",
            f"导弹: {aircraft2.missiles}"
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
        
        # 如果游戏结束，显示获胜者
        if game_over:
            if winner == 'draw':
                winner_text = "游戏结束！平局！"
            else:
                winner_text = f"游戏结束！{'红方' if winner == 'red' else '蓝方'}获胜！"
            text = self.font.render(winner_text, True, self.WHITE)
            text_rect = text.get_rect(center=(self.window_width/2, self.window_height/2))
            self.window.blit(text, text_rect)
    
    def clear(self):
        """清空屏幕"""
        self.window.fill(self.BLACK)
    
    def update(self):
        """更新显示"""
        pygame.display.flip() 