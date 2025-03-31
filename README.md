# MidRangeRL - 中距空战强化学习环境

## 项目介绍

你好啊，赛博飞行员，你是否时常对这些问题感到疑惑：中距弹的射程为什么一会儿写10千米一会儿写100千米？现代导弹都能拉出30G过载了，战斗机的机动性还有意义吗？

如果你希望这些困惑得到解答，那么欢迎来到midrangeRL。它是一个娱乐+科普性质的中距空战AI强化学习平台，最大的特色是“时空图”UI，用于估计导弹真实射程。

在这里，你可以设计和测试你的空战策略，同时还还可以手动游玩，亲自体验与各式各样的空战AI对抗。物理参数的真实性调试也是一项很好玩的工作，后面我们还会引入更多的物理机制，比如更科学的气动计算、超音速气动修正、雷达RCS等。这些都需要你根据在安东星激情对射的经验来提出批评与改进建议！

Gemini、Cursor与TRAE等LLM代码助手正在帮我们处理最令人掉头发的程序实现部分。

## 项目特点

- **高性能并行模拟**：支持在单个GPU上同时模拟超过100万个环境实例
- **矢量化物理引擎**：基于向量的物理计算，提供更真实的飞行动力学
- **高效内存管理**：针对大规模批处理进行了内存优化
- **导弹制导系统**：实现了现实的导弹制导逻辑
- **可扩展架构**：易于扩展和自定义的模块化设计

## 项目结构

```
midrangeRL/
├── env_numpy/             # NumPy环境实现（单例模式）
├── env_tensor/            # PyTorch张量环境实现（批处理、GPU加速）
├── visualization/         # 可视化相关组件
├── game_play.py           # 游戏实际运行入口
└── run_tensor_test.py      # 批量性能测试脚本
```

## 核心模块

### env_tensor - 高性能张量环境

- **aerodynamic_tensor.py**：物理引擎
- **missile_guidance_tensor.py**：张量化的导弹制导系统
- **game_events_tensor.py**：游戏事件处理系统（碰撞检测、边界处理等）
- **tensor_env_manager.py**：环境管理器，整合所有组件
- **test_million_envs.py**：百万级并行环境测试脚本

## 快速开始

### 安装依赖

```bash
pip install torch numpy pygame
```

### 运行游戏

```bash
python game_play.py
```

### 运行百万级环境测试

```bash
python -m env_tensor.test_million_envs --num_envs 1000000 --steps 10
```



## 环境接口

环境管理器提供了标准的强化学习接口：

```python
from env_tensor.tensor_env_manager import TensorEnvManager

# 创建环境
env = TensorEnvManager(num_envs=1024, device='cuda')

# 重置环境
observations = env.reset()

# 执行步骤
actions = {
    'red': {
        'throttle': torch.rand(1024, device='cuda'),  # 0到1
        'rudder': torch.rand(1024, device='cuda')*2-1,  # -1到1
        'fire': torch.rand(1024, device='cuda') > 0.9  # 布尔值
    },
    'blue': {
        'throttle': torch.rand(1024, device='cuda'),
        'rudder': torch.rand(1024, device='cuda')*2-1,
        'fire': torch.rand(1024, device='cuda') > 0.9
    }
}

observations, rewards, dones, info = env.step(actions)

# 关闭环境
env.close()
```
## 路线图

🚀 对接强化学习框架，实现基础训练流程 (train.py)。

✨ 开发 HTML5 前端原型，实现主菜单，战斗回放，时空图可视化 (web_frontend)。

🔧 (可选/评估中) 探索 Isaac Gym 以获得极致并行性能。

📦 模型导出与集成，完成可玩的 HTML5 版本。

📈 持续迭代物理模型、AI 策略和功能。

## 许可证

MIT
