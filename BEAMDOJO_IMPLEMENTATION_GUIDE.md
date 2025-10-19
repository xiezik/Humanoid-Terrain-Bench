# BEAMDOJO Implementation Guide
# BEAMDOJO实现使用指南

本文档介绍如何使用基于BEAMDOJO论文实现的双Critic网络和两阶段训练功能。

## 功能概述

### 1. 双Critic网络架构 (Double Critic Network)
- **密集奖励Critic**: 学习基于传统奖励函数的价值函数
- **稀疏奖励Critic**: 学习基于Foothold采样的稀疏奖励价值函数
- **优势函数合并**: 根据配置权重合并两个Critic的优势估计

### 2. Foothold奖励系统
- **脚掌采样**: 在每个脚掌周围采样多个点评估落脚稳定性
- **地形高度查询**: 实时查询地形高度信息
- **稳定性评估**: 基于高度差异和接触状态计算奖励

### 3. 两阶段训练系统 (Two-Stage Training)
- **Stage 1**: 软约束训练，平坦地形，踩空不终止episode
- **Stage 2**: 硬约束训练，复杂地形，踩空立即终止episode
- **自动切换**: 基于成功率和训练步数自动切换阶段

## 文件结构

```
Humanoid-Terrain-Bench/
├── rsl_rl/rsl_rl/
│   ├── modules/
│   │   └── actor_critic.py              # 双Critic网络实现
│   ├── algorithms/
│   │   └── ppo_double_reward.py         # 双Critic PPO算法
│   └── storage/
│       └── rollout_storage.py           # 扩展存储支持双Critic
├── legged_gym/legged_gym/
│   ├── envs/
│   │   ├── base/
│   │   │   ├── humanoid_robot.py        # 添加Foothold奖励函数
│   │   │   └── legged_robot_config.py   # 更新配置支持BEAMDOJO
│   │   └── humanoid/
│   │       ├── humanoid_beamdojo_config.py  # BEAMDOJO配置示例
│   │       └── __init__.py
│   └── utils/
│       ├── foothold_reward.py           # Foothold奖励计算器
│       └── two_stage_training.py        # 两阶段训练管理器
└── scripts/
    └── train_beamdojo.py               # BEAMDOJO训练示例脚本
```

## 使用方法

### 1. 基础双Critic训练

```python
# 在配置文件中启用双Critic
class YourRobotCfgPPO(LeggedRobotCfgPPO):
    class policy:
        use_double_critic = True
    
    class algorithm:
        use_double_critic = True
        dense_value_loss_coef = 1.0
        sparse_value_loss_coef = 1.0
        advantage_merge_weight = 0.5
    
    class runner:
        policy_class_name = 'ActorCriticRMADoubleReward'
        algorithm_class_name = 'PPODoubleReward'
```

### 2. 奖励分离配置

```python
class rewards:
    # 定义哪些奖励属于密集/稀疏类别
    dense_rewards = [
        'tracking_lin_vel', 'tracking_ang_vel', 'orientation',
        'torques', 'action_rate', 'dof_vel'
    ]
    sparse_rewards = [
        'foothold'  # BEAMDOJO Foothold奖励
    ]
    
    class scales:
        # 添加Foothold奖励权重
        foothold = 1.0
    
    # Foothold奖励配置
    class foothold:
        num_sample_points = 4
        sample_radius = 0.02
        height_tolerance = 0.05
```

### 3. 两阶段训练配置

```python
class training:
    enable_two_stage = True
    
    class stage1:
        min_steps = 1000000
        max_steps = 5000000
        success_threshold = 0.8
        use_soft_termination = True
        terrain_type = "flat_with_target_perception"
    
    class stage2:
        use_soft_termination = False
        terrain_type = "sparse_terrain"
```

### 4. 运行训练

```bash
# 使用预配置的BEAMDOJO环境
python scripts/train_beamdojo.py --task=humanoid_beamdojo_full --headless

# 或使用阶段管理演示
python scripts/train_beamdojo.py --task=humanoid_beamdojo_full --stage-demo --headless
```

## 核心组件详解

### ActorCriticRMADoubleReward

双Critic网络的核心实现：

```python
class ActorCriticRMADoubleReward(nn.Module):
    def __init__(self, num_obs, num_privileged_obs, num_actions, **kwargs):
        # 创建两个独立的Critic网络
        self.critic1 = build_critic_net(...)  # 密集奖励
        self.critic2 = build_critic_net(...)  # 稀疏奖励
        
    def evaluate(self, obs, privileged_obs=None):
        # 返回两个价值估计
        value1 = self.critic1(obs)  # 密集价值
        value2 = self.critic2(obs)  # 稀疏价值
        return value1, value2
```

### PPODoubleReward

支持双Critic的PPO算法：

```python
class PPODoubleReward(PPO):
    def compute_returns(self, last_values1, last_values2):
        # 分别计算两个奖励流的GAE
        dense_advantages = compute_gae(dense_rewards, values1, ...)
        sparse_advantages = compute_gae(sparse_rewards, values2, ...)
        
        # 合并优势函数
        combined_advantages = merge_advantages(dense_advantages, sparse_advantages)
        return combined_advantages
```

### TwoStageTrainingManager

两阶段训练的核心管理器：

```python
class TwoStageTrainingManager:
    def check_stage_transition(self, success_rate, training_step):
        if self.current_stage == 1:
            # 检查是否满足Stage2切换条件
            if success_rate >= self.stage1_success_threshold:
                self.current_stage = 2
                return True, 2
        return False, self.current_stage
    
    def get_stage_config(self):
        # 返回当前阶段的环境配置
        return {
            'use_soft_termination': self.should_use_soft_termination(),
            'terrain_type': self.get_terrain_type(),
            'command_ranges': self.get_command_ranges()
        }
```

## 配置参数说明

### 双Critic参数

- `use_double_critic`: 是否启用双Critic架构
- `dense_value_loss_coef`: 密集奖励价值损失系数
- `sparse_value_loss_coef`: 稀疏奖励价值损失系数  
- `advantage_merge_weight`: 优势函数合并权重 (0.0=仅密集, 1.0=仅稀疏)

### Foothold奖励参数

- `num_sample_points`: 每个脚掌采样点数量
- `sample_radius`: 采样半径
- `height_tolerance`: 高度容忍度
- `stability_weight`: 稳定性权重
- `safety_weight`: 安全性权重

### 两阶段训练参数

- `enable_two_stage`: 是否启用两阶段训练
- `min_steps`: Stage1最小训练步数
- `max_steps`: Stage1最大训练步数
- `success_threshold`: 切换到Stage2的成功率阈值

## 注意事项

1. **向后兼容性**: 所有修改都保持与现有单Critic系统的兼容性
2. **内存使用**: 双Critic会增加约2倍的Critic网络内存使用
3. **训练稳定性**: 建议先使用单Critic验证环境，再启用双Critic
4. **奖励设计**: 确保稀疏奖励确实稀疏，密集奖励提供足够的学习信号
5. **阶段切换**: Stage1到Stage2的切换是不可逆的，确保切换条件合理

## 调试建议

1. **检查奖励分离**: 确保密集和稀疏奖励正确分类
2. **监控两个价值函数**: 观察两个Critic的学习曲线
3. **验证Foothold计算**: 检查地形高度查询是否正确
4. **阶段切换时机**: 确保Stage1训练充分再切换到Stage2

## 扩展功能

本实现提供了扩展框架，可以轻松添加：
- 更多稀疏奖励类型
- 自定义阶段切换条件
- 不同的优势函数合并策略
- 多阶段训练（超过2个阶段）

通过修改相应的配置类和管理器，可以适应不同的研究需求。