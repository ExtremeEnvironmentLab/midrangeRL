# -*- coding: utf-8 -*-

import math
# 注意：这个版本仍然使用 Python 内置 math，没有引入 NumPy 或 PyTorch
# 它是为逻辑修改，以便后续更容易迁移到 Tensor 库

class Aerodynamic:
    def __init__(self):
        # 战斗机参数
        self.FIGHTER_TERMINAL_VELOCITY = 400  # 自由落体终端速度 (m/s)
        self.FIGHTER_MIN_TURN_RADIUS = 1000   # 最小转弯半径 (m)
        # cl_max 现在代表 1/Rmin，用于向心加速度计算
        self.FIGHTER_CL_MAX = 1 / self.FIGHTER_MIN_TURN_RADIUS
        self.FIGHTER_MAX_THRUST = 1.5 * 9.8  # 最大推力 (假设为 加速度 m/s²)
        self.FIGHTER_LIFT_DRAG_RATIO = 5     # 最大升阻比 (用于计算 k)

        # 导弹参数
        self.MISSILE_TERMINAL_VELOCITY = 400 # 自由落体终端速度 (m/s)
        self.MISSILE_MIN_TURN_RADIUS = 1000  # 最小转弯半径 (m)
        self.MISSILE_CL_MAX = 1 / self.MISSILE_MIN_TURN_RADIUS
        self.MISSILE_THRUST = 8 * 9.8        # 发动机推力 (假设为 加速度 m/s²)
        self.MISSILE_ENGINE_DURATION = 10.0 # 发动机工作时间 (s)
        self.MISSILE_LIFT_DRAG_RATIO = 3     # 最大升阻比 (用于计算 k)

        self.G = 9.8 # 重力加速度

    def calculate_aircraft_physics(self, aircraft, dt):
        """计算飞机/导弹的物理参数和运动学 (基于矢量)

        Args:
            aircraft: 飞机或导弹对象 (字典或对象)
            dt: 时间步长
        """
        # --- 辅助函数 ---
        def get_attr(obj, attr):
            return obj[attr] if isinstance(obj, dict) else getattr(obj, attr)

        def set_attr(obj, attr, value):
            if isinstance(obj, dict):
                obj[attr] = value
            else:
                setattr(obj, attr, value)

        # --- 获取当前状态 ---
        vx = get_attr(aircraft, 'vx')
        vy = get_attr(aircraft, 'vy')
        rudder = get_attr(aircraft, 'rudder') # 控制输入 [-1, 1]
        is_missile = get_attr(aircraft, 'is_missile')

        # --- 根据实体类型选择物理参数 ---
        if is_missile:
            terminal_velocity = self.MISSILE_TERMINAL_VELOCITY
            cl_max = self.MISSILE_CL_MAX # 1 / Rmin
            max_thrust = self.MISSILE_THRUST
            lift_drag_ratio = self.MISSILE_LIFT_DRAG_RATIO
            engine_time = get_attr(aircraft, 'engine_time')
            engine_time = max(0, engine_time - dt)
            set_attr(aircraft, 'engine_time', engine_time)
            thrust_accel = max_thrust if engine_time > 0 else 0.0
        else:
            terminal_velocity = self.FIGHTER_TERMINAL_VELOCITY
            cl_max = self.FIGHTER_CL_MAX # 1 / Rmin
            max_thrust = self.FIGHTER_MAX_THRUST
            lift_drag_ratio = self.FIGHTER_LIFT_DRAG_RATIO
            throttle = get_attr(aircraft, 'throttle')
            thrust_accel = throttle * max_thrust # 推力加速度

        # --- 核心矢量物理计算 ---
        epsilon = 1e-7 # 防止除以零

        # 1. 计算速度平方和速度倒数
        vSquare = vx*vx + vy*vy
        inv_v = 1.0 / math.sqrt(vSquare + epsilon) # 计算 1/v

        # 2. 计算归一化的速度方向向量 (nx, ny)
        nx = vx * inv_v
        ny = vy * inv_v

        # 3. 计算归一化的垂直(向心)方向向量 (pnx, pny) - 左转为正 rudder
        pnx = -ny
        pny = nx

        # 4. 计算空气动力系数
        Cd0 = self.G / (terminal_velocity ** 2)
        # 假设升力系数 Cl 与舵量绝对值成正比 (这个假设可以调整)
        # 注意: cl_max 本身代表 1/Rmin，不是传统空气动力学 Cl
        # 我们用它来计算诱导阻力相关的 k，这里的 Cl 只是个中间量
        Cl_intermediate = abs(rudder) * cl_max # 这个 Cl 不是直接用，而是影响 k
        # 计算诱导阻力系数 k (基于升阻比 L/D = Cl/Cd)
        # L/D = Cl / (Cd0 + k*Cl^2) --> 在特定条件下（如最大L/D时）推导 k
        # 一个简化的假设：k 正比于 1 / ( aspect_ratio * pi * efficiency_factor)
        # 这里用一个基于 Cl_max 和 L/D_max 的粗略估计，可能需调整
        # D = D0 + Di = q*S*Cd0 + q*S*k*Cl^2
        # L = q*S*Cl
        # L/D = Cl / (Cd0 + k*Cl^2)
        # d(L/D)/dCl = 0 => Cl_at_max_LD = sqrt(Cd0/k)
        # (L/D)_max = 1 / (2*sqrt(Cd0*k)) => k = 1 / (4*Cd0*(L/D)_max^2)
        # 我们这里的 lift_drag_ratio 是 (L/D)_max
        k = 1.0 / (4.0 * Cd0 * (lift_drag_ratio**2) + epsilon) # 估算诱导阻力因子 k
        # (另一种估算 k 的方法，来自原代码逻辑，保留作为参考，但上面基于L/D max的更标准)
        # k_original = (cl_max - lift_drag_ratio * Cd0) / (lift_drag_ratio * cl_max**2 + epsilon) # 假设 lift_drag_ratio 发生在 Cl=Cl_max 时

        # 5. 计算总阻力系数 Cd (注意这里的 Cl_intermediate 用途)
        # Cd = Cd0 + 诱导阻力项。诱导阻力 Cd_i = k * Cl^2
        # 这里的 'Cl' 需要是真正的气动升力系数，与法向加速度有关
        # 法向加速度 a_n = v^2 / R = v^2 * abs(rudder) * cl_max
        # 升力 L = m * a_n = m * v^2 * abs(rudder) * cl_max
        # 升力也等于 L = 0.5 * rho * v^2 * S * Cl_aero
        # 所以 Cl_aero 正比于 abs(rudder) * cl_max
        # 我们就用 Cl_intermediate^2 近似 Cl_aero^2 对诱导阻力的贡献
        Cd = Cd0 + k * (Cl_intermediate**2) # 总阻力系数

        # 6. 计算阻力加速度大小 (Drag / mass)
        drag_accel = Cd * vSquare

        # 7. 计算平行于速度方向的加速度大小
        parallel_accel_mag = thrust_accel - drag_accel

        # 8. 计算垂直于速度方向(向心)的加速度大小
        # a_c = v^2 / R = v^2 * (abs(rudder) / min_turn_radius) = v^2 * abs(rudder) * cl_max
        # 方向由 rudder 符号决定
        centripetal_accel_mag = vSquare * cl_max * rudder

        # 9. 合成总加速度矢量 (ax, ay)
        ax = nx * parallel_accel_mag + pnx * centripetal_accel_mag
        ay = ny * parallel_accel_mag + pny * centripetal_accel_mag

        # --- 更新状态 ---
        # 更新速度矢量
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt

        # 更新位置 (使用更新后的速度进行积分，略优于用旧速度)
        x = get_attr(aircraft, 'x') + vx_new * dt
        y = get_attr(aircraft, 'y') + vy_new * dt

        # 存储新的矢量状态
        set_attr(aircraft, 'vx', vx_new)
        set_attr(aircraft, 'vy', vy_new)
        set_attr(aircraft, 'x', x)
        set_attr(aircraft, 'y', y)

        # --- 计算并存储其他辅助/显示用的标量值 ---
        speed_new = math.sqrt(vx_new**2 + vy_new**2 + epsilon)
        set_attr(aircraft, 'speed', speed_new)

        # 计算角度 (需要 atan2，如果其他地方确实需要角度的话)
        angle_new = math.degrees(math.atan2(vy_new, vx_new))
        set_attr(aircraft, 'angle', angle_new % 360) # 保持在 [0, 360)

        # 计算过载 (法向加速度 / g)
        # 使用 abs(centripetal_accel_mag / rudder) 得到纯粹的大小
        n_load_magnitude = abs(vSquare * cl_max) # v^2/Rmin
        actual_n_load = n_load_magnitude * abs(rudder) / self.G if self.G > 0 else 0 # 横向G值
        set_attr(aircraft, 'n_load', actual_n_load)

        # 计算角速度 (度/秒)
        # omega = v / R = v * (abs(rudder) / min_turn_radius) = v * abs(rudder) * cl_max
        # 方向由 rudder 符号决定
        turn_rate_rad_per_sec = speed_new * cl_max * rudder
        turn_rate_deg_per_sec = math.degrees(turn_rate_rad_per_sec)
        set_attr(aircraft, 'turn_rate', turn_rate_deg_per_sec)

        # 限制最小速度 (虽然物理上应该由阻力自然处理，但以防万一)
        # 这个检查现在意义不大了，因为速度是矢量计算的结果
        # if speed_new < 0: # 速度小于0物理上不太可能，除非推力为负且很大
        #     set_attr(aircraft, 'speed', 0)
        #     set_attr(aircraft, 'vx', 0)
        #     set_attr(aircraft, 'vy', 0)

        return aircraft

    def create_aircraft(self, x, y, angle, speed=300, is_missile=False):
        """创建一个新的飞行器属性字典 (使用矢量速度)

        Args:
            x: 初始x坐标
            y: 初始y坐标
            angle: 初始角度 (度)
            speed: 初始速度 (m/s)
            is_missile: 是否为导弹

        Returns:
            dict: 飞行器属性字典
        """
        rad = math.radians(angle)
        vx_initial = math.cos(rad) * speed
        vy_initial = math.sin(rad) * speed

        aircraft = {
            'x': x,
            'y': y,
            'vx': vx_initial, # 使用速度矢量
            'vy': vy_initial, # 使用速度矢量
            # 'angle': angle, # 存储，但不由物理核心更新直接使用
            # 'speed': speed, # 存储，但不由物理核心更新直接使用
            'throttle': 1.0,  # 油门 (仅飞机)
            'rudder': 0.0,  # 舵量 [-1, 1]
            'is_missile': is_missile,
            # 'turn_rate': 0.0, # 将由物理计算更新
            # 'n_load': 1.0,  # 将由物理计算更新
        }

        # 计算并添加初始的辅助标量值
        aircraft['speed'] = speed
        aircraft['angle'] = angle % 360
        aircraft['turn_rate'] = 0.0
        aircraft['n_load'] = 0.0 # 初始无转向

        if is_missile:
            aircraft['engine_time'] = self.MISSILE_ENGINE_DURATION
            aircraft['target'] = None
            # 导弹可能没有油门概念
            if 'throttle' in aircraft: del aircraft['throttle']
        else:
            aircraft['missiles'] = 6  # 导弹数量 (仅飞机)

        return aircraft