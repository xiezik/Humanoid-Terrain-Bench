# BEAMDOJO Checkpoint 继续训练指南

## 🎯 概述

本指南介绍如何使用BEAMDOJO训练脚本从之前的checkpoint继续训练，支持多种checkpoint加载方式。

## 🔧 使用方法

### 方法1：使用完整checkpoint路径 (推荐)

```bash
# 使用绝对路径指定checkpoint
python scripts/train_beamdojo.py \
    --task humanoid_beamdojo_full \
    --headless \
    --proj_name "beamdojo_resume" \
    --resume_path "/path/to/your/checkpoint/model_XXXX.pt"
```

**示例**:
```bash
python scripts/train_beamdojo.py \
    --task humanoid_beamdojo_full \
    --headless \
    --proj_name "beamdojo_resume" \
    --resume_path "/home/cft/zikang/Humanoid-Terrain-Bench/legged_gym/logs/beamdojo/Oct24_14-23-45--humanoid_beamdojo_full_1024_1423/model_5000.pt"
```

### 方法2：使用run ID自动查找

```bash
# 自动从第0个run的最新checkpoint继续
python scripts/train_beamdojo.py \
    --task humanoid_beamdojo_full \
    --headless \
    --proj_name "beamdojo_resume" \
    --load_run 0

# 从第1个run的特定checkpoint继续
python scripts/train_beamdojo.py \
    --task humanoid_beamdojo_full \
    --headless \
    --proj_name "beamdojo_resume" \
    --load_run 1 \
    --checkpoint 3000
```

### 方法3：使用简单resume标志

```bash
# 自动从最新的run和checkpoint继续
python scripts/train_beamdojo.py \
    --task humanoid_beamdojo_full \
    --headless \
    --proj_name "beamdojo_resume" \
    --resume
```

## 📂 Checkpoint文件结构

典型的checkpoint文件结构如下：

```
legged_gym/logs/beamdojo/
├── Oct24_14-23-45--humanoid_beamdojo_full_1024_1423/
│   ├── model_1000.pt
│   ├── model_2000.pt
│   ├── model_3000.pt
│   ├── model_4000.pt
│   └── model_5000.pt
└── Oct24_15-30-22--humanoid_beamdojo_full_1024_1530/
    ├── model_1000.pt
    ├── model_2000.pt
    └── model_3000.pt
```

## 🔍 查找您的Checkpoint

### 1. 列出所有可用的训练run

```bash
ls -la legged_gym/logs/beamdojo/
```

### 2. 查看特定run的checkpoint

```bash
ls -la legged_gym/logs/beamdojo/Oct24_14-23-45--humanoid_beamdojo_full_1024_1423/
```

### 3. 找到最新的checkpoint

```bash
ls -lt legged_gym/logs/beamdojo/*/model_*.pt | head -5
```

## 💡 推荐的继续训练流程

### Stage 1: 平坦地形基础训练

```bash
# 开始Stage1训练
python scripts/train_beamdojo.py \
    --task humanoid_beamdojo_full \
    --headless \
    --proj_name "beamdojo_stage1" \
    --stage stage1

# 从Stage1 checkpoint继续
python scripts/train_beamdojo.py \
    --task humanoid_beamdojo_full \
    --headless \
    --proj_name "beamdojo_stage1_resume" \
    --stage stage1 \
    --resume_path "/path/to/stage1/model_XXXX.pt"
```

### Stage 2: 复杂地形微调

```bash
# 从Stage1最佳checkpoint开始Stage2
python scripts/train_beamdojo.py \
    --task humanoid_beamdojo_full \
    --headless \
    --proj_name "beamdojo_stage2" \
    --stage stage2 \
    --resume_path "/path/to/stage1/best_model.pt"

# 继续Stage2训练
python scripts/train_beamdojo.py \
    --task humanoid_beamdojo_full \
    --headless \
    --proj_name "beamdojo_stage2_resume" \
    --stage stage2 \
    --resume_path "/path/to/stage2/model_XXXX.pt"
```

## ⚙️ 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--resume` | 简单恢复标志，自动找最新checkpoint | `--resume` |
| `--resume_path` | 指定checkpoint完整路径 | `--resume_path "/path/to/model.pt"` |
| `--load_run` | 指定run ID (从0开始) | `--load_run 0` |
| `--checkpoint` | 指定checkpoint编号 | `--checkpoint 5000` |
| `--stage` | 指定训练阶段 | `--stage stage1` 或 `--stage stage2` |

## 🚨 注意事项

### 1. **兼容性检查**
- 确保checkpoint与当前配置兼容
- 如果修改了网络结构，可能无法加载旧checkpoint

### 2. **GPU设备**
- 确保checkpoint的设备与当前训练设备匹配
- 如果需要，可以添加 `--device cuda:0` 参数

### 3. **学习率调度**
- 继续训练时学习率会从checkpoint保存的状态继续
- 如需重置学习率，可能需要修改代码

### 4. **实验追踪**
- 建议为继续训练使用不同的项目名称
- 这样便于在wandb中区分不同的训练阶段

## 🛠️ 故障排除

### Checkpoint加载失败

```bash
❌ Checkpoint加载失败: RuntimeError: Error(s) in loading state_dict
```

**解决方案**:
1. 检查checkpoint文件是否完整
2. 确认配置与checkpoint训练时一致
3. 检查设备匹配 (CPU vs GPU)

### 找不到Checkpoint文件

```bash
❌ Checkpoint文件不存在: /path/to/model.pt
```

**解决方案**:
1. 确认路径拼写正确
2. 检查文件权限
3. 使用绝对路径而非相对路径

### Run ID超出范围

```bash
❌ Run ID 5超出范围，共有3个run
```

**解决方案**:
1. 使用 `ls legged_gym/logs/beamdojo/` 查看可用run
2. Run ID从0开始计数
3. 使用 `-1` 表示最新的run

## 📊 监控继续训练

继续训练时，建议监控以下指标：

1. **Loss趋势**: 确保loss继续下降
2. **性能指标**: 比较继续训练前后的性能
3. **资源使用**: GPU/CPU使用率
4. **收敛速度**: 是否比从头训练更快收敛

## 🎯 最佳实践

1. **定期保存**: 设置合理的保存间隔
2. **多checkpoint备份**: 保留多个历史checkpoint
3. **实验记录**: 记录每次继续训练的原因和设置
4. **性能测试**: 定期测试模型在验证环境中的表现

使用这些方法，您可以灵活地从任何保存的checkpoint继续BEAMDOJO训练！