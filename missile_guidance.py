# -*- coding: utf-8 -*-

import math
import time

class MissileGuidance:
    def __init__(self, guidance_gain=200):
        self.guidance_gain = guidance_gain
        self.last_los_angle = None  # 存储上一帧的弹目连线角度
        self.last_time = None       # 存储上一帧的时间
    
    def calculate_guidance(self, missile, target):
        """计算导弹制导指令
        
        Args:
            missile: 导弹对象
            target: 目标对象
            
        Returns:
            float: 舵量指令 (-1 到 1)
        """
        if not target:
            return 0.0
            
        # 计算当前弹目连线角度
        dx = target.x - missile.x
        dy = target.y - missile.y
        current_los_angle = math.degrees(math.atan2(dy, dx))
        current_time = time.time()
        
        # 计算视线角速度
        if self.last_los_angle is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                # 处理角度跨越360度的情况
                angle_diff = (current_los_angle - self.last_los_angle + 180) % 360 - 180
                los_rate = angle_diff / dt  # 真正的角速度（度/秒）
            else:
                los_rate = 0
        else:
            los_rate = 0
        
        # 更新上一帧数据
        self.last_los_angle = current_los_angle
        self.last_time = current_time
        
        # 计算制导指令（比例导引）
        rudder = self.guidance_gain * los_rate / 180.0  # 归一化到[-1,1]范围
        rudder = max(-1.0, min(1.0, rudder))  # 限制舵量范围
        
        return rudder 