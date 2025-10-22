# G1机器人BEAMDOJO训练指南

## 🤖 G1机器人配置概述

### 📊 G1机器人规格
- **自由度**: 12DoF (每条腿6个关节)
- **身高**: 约0.8m站立高度
- **关节配置**: 髋关节(偏航、滚转、俯仰) + 膝关节 + 踝关节(俯仰、滚转)
- **控制方式**: 位置控制 (PD控制器)

### 🎯 G1专用优化
- **脚部前向对齐奖励**: 针对G1步态特点优化
- **基座高度目标**: 0.7m (适配G1身高)
- **动作缩放**: 0.25 (适中的关节运动范围)
- **控制刚度**: 髋关节100, 膝关节150, 踝关节40

---

## 🚀 G1训练方案

### 方案1: G1 Stage1训练 (推荐新手)
```bash
# 基础平地运动学习
python scripts/train_g1.py \
    --task humanoid_beamdojo_g1_stage1 \
    --stage stage1 \
    --max_iterations 3000 \
    --headless \
    --experiment_name "g1_stage1_basic"
```

### 方案2: G1完整BEAMDOJO训练
```bash
# 包含双Critic和复杂地形
python scripts/train_g1.py \
    --task humanoid_beamdojo_g1_full \
    --stage full \
    --use_double_critic \
    --enable_beamdojo \
    --max_iterations 5000 \
    --experiment_name "g1_beamdojo_full"
```

### 方案3: G1快速测试
```bash
# 小规模验证训练
python scripts/train_g1.py \
    --task humanoid_beamdojo_g1_stage1 \
    --stage stage1 \
    --max_iterations 100 \
    --num_envs 1024 \
    --experiment_name "g1_test"
```

---

## 📋 G1训练参数说明

| 参数 | 默认值 | G1优化值 | 说明 |
|------|--------|----------|------|
| `--task` | humanoid_beamdojo_g1_stage1 | - | G1任务配置 |
| `--stage` | stage1 | stage1/stage2/full | 训练阶段 |
| `--max_iterations` | 3000 | 3000 | G1适中训练量 |
| `--num_envs` | 4096 | 2048-4096 | 根据GPU调整 |
| `base_height_target` | 1.0 | 0.7 | G1实际高度 |
| `action_scale` | 0.5 | 0.25 | G1关节范围 |

---

## 🎯 G1训练阶段详解

### Stage1: 平地基础运动 (推荐起点)
- **目标**: 学习行走、转向、平衡
- **地形**: 100%平坦地形
- **算法**: 标准PPO
- **速度范围**: 保守设置 (x: -0.5~0.8 m/s)
- **训练时间**: 约2-3小时
- **成功标准**: 能在平地稳定行走

### Stage2: 复杂地形导航
- **目标**: 足点选择和地形适应
- **地形**: 粗糙地面、楼梯、障碍物
- **算法**: 双Critic PPO (可选)
- **速度范围**: 全速度范围
- **训练时间**: 约4-6小时
- **成功标准**: 能穿越复杂地形

### Full: 完整BEAMDOJO训练
- **目标**: 两阶段自动切换 + 全功能
- **算法**: 完整BEAMDOJO (双Critic + 两阶段)
- **训练时间**: 约6-8小时
- **成功标准**: 完整地形适应能力

---

## 🛠️ G1训练配置特点

### 🤖 G1物理配置
```python
# G1默认关节角度 (站立姿态)
default_joint_angles = {
    'left_hip_yaw_joint': 0.0,
    'left_hip_roll_joint': 0.0,
    'left_hip_pitch_joint': -0.1,
    'left_knee_joint': 0.3,
    'left_ankle_pitch_joint': -0.2,
    'left_ankle_roll_joint': 0.0,
    # 右腿对称设置
}

# G1控制参数
stiffness = {
    'hip_yaw': 100,    # 髋关节偏航
    'hip_roll': 100,   # 髋关节滚转
    'hip_pitch': 100,  # 髋关节俯仰
    'knee': 150,       # 膝关节 (更高刚度)
    'ankle': 40,       # 踝关节 (更低刚度)
}
```

### 🎯 G1奖励配置
```python
# G1特有奖励
scales = {
    'feet_forward_alignment': 1.5,  # 脚部前向对齐 (G1特有)
    'feet_distance': -2.0,          # 脚间距离 (G1优化)
    'feet_air_time': 2.5,           # 腾空时间 (G1优化)
    'orientation': -1.25,           # 姿态控制 (G1优化)
}

# G1物理参数
base_height_target = 0.7           # G1目标高度
min_dist = 0.08                    # G1最小脚间距
max_dist = 0.25                    # G1最大脚间距
```

---

## 🔧 G1故障排除

### 如果遇到G1配置错误
```bash
# 检查G1配置文件
ls legged_gym/legged_gym/envs/humanoid/humanoid_beamdojo_g1_config.py

# 重新生成G1配置 (如果丢失)
# 使用原始配置脚本重新创建
```

### 如果遇到G1 URDF错误
```bash
# 检查G1模型文件
ls legged_gym/resources/robots/g1/g1_12dof_with_hand.urdf

# 检查G1关节名称匹配
grep -E "joint.*revolute" legged_gym/resources/robots/g1/g1_12dof_with_hand.urdf
```

### 如果G1运动异常
1. **关节角度检查**: 确认默认姿态合理
2. **控制参数调整**: 调整PD控制器刚度和阻尼
3. **速度范围限制**: 初期使用保守的速度设置
4. **地形适配**: 从平地开始，逐步增加复杂度

---

## 📈 G1训练监控

### 关键指标
- **Episode Length**: G1平地应达到1000+ steps
- **Linear Velocity Tracking**: 跟踪误差 < 0.1 m/s
- **Orientation**: 姿态偏差 < 0.1 rad
- **Feet Air Time**: 腾空时间接近0.25s
- **Base Height**: 基座高度稳定在0.7m附近

### 成功标准
- **Stage1成功**: 平地连续行走30秒以上
- **Stage2成功**: 能穿越高度0.1m的台阶
- **Full成功**: 在复杂地形中保持90%以上成功率

---

## 📁 G1输出文件

训练完成后生成：
```
logs/
├── beamdojo_g1_stage1_YYYYMMDD_HHMMSS/
│   ├── model_*.pt              # G1训练模型
│   ├── config.yaml             # G1配置
│   └── train/
│       └── summaries/          # G1训练日志
```

---

## 🎓 G1下一步计划

### Stage1完成后
1. **评估基础技能**: 在平地测试各种步态
2. **参数微调**: 根据G1表现调整控制参数
3. **准备Stage2**: 检查复杂地形性能

### Stage2完成后  
1. **综合测试**: 在混合地形中评估
2. **性能对比**: 与其他humanoid机器人比较
3. **实机部署**: 准备向真实G1机器人迁移

### 实机部署建议
1. **Sim2Real Gap**: 考虑仿真与现实差异
2. **安全测试**: 从低速低难度开始
3. **参数适配**: 根据实际硬件微调
4. **渐进部署**: 逐步增加任务复杂度

---

## 💡 G1训练小贴士

1. **首次训练**: 建议使用Stage1配置熟悉G1特性
2. **参数调优**: G1的踝关节需要特别关注刚度设置
3. **地形渐进**: 从平地 → 小坡度 → 台阶 → 复杂地形
4. **监控关节**: 注意G1髋关节的运动范围和负载
5. **备份策略**: 定期保存Stage1的成功模型作为baseline

现在可以开始您的G1机器人训练之旅! 🚀🤖