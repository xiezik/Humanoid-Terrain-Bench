# 🎯 从Checkpoint恢复BEAMDOJO训练 - 简化指南

## 使用您的具体checkpoint

您的checkpoint路径是：
```
/home/cft/zikang/Humanoid-Terrain-Bench/legged_gym/logs/beamdojo_test/Oct23_23-57-35--humanoid_beamdojo_full_1023_2357/model_3400.pt
```

## 🚀 恢复训练的三种方法

### 方法1：使用resume_path (推荐)

```bash
python scripts/train_beamdojo.py \
    --task humanoid_beamdojo_full \
    --headless \
    --proj_name "beamdojo_resume" \
    --no_wandb \
    --resume_path "/home/cft/zikang/Humanoid-Terrain-Bench/legged_gym/logs/beamdojo_test/Oct23_23-57-35--humanoid_beamdojo_full_1023_2357/model_3400.pt"
```

### 方法2：使用标准resume + load_run + checkpoint

```bash
python scripts/train_beamdojo.py \
    --task humanoid_beamdojo_full \
    --headless \
    --proj_name "beamdojo_resume" \
    --no_wandb \
    --resume \
    --load_run "beamdojo_test/Oct23_23-57-35--humanoid_beamdojo_full_1023_2357" \
    --checkpoint 3400
```

### 方法3：快速测试 (短路径)

如果您想要用简短的命令，可以创建软链接：

```bash
# 创建软链接
ln -sf "/home/cft/zikang/Humanoid-Terrain-Bench/legged_gym/logs/beamdojo_test/Oct23_23-57-35--humanoid_beamdojo_full_1023_2357/model_3400.pt" \
       "/tmp/latest_checkpoint.pt"

# 使用简短路径
python scripts/train_beamdojo.py \
    --task humanoid_beamdojo_full \
    --headless \
    --proj_name "beamdojo_resume" \
    --no_wandb \
    --resume_path "/tmp/latest_checkpoint.pt"
```

## 📋 参数说明

| 参数 | 说明 | 备注 |
|------|------|------|
| `--resume_path` | 完整checkpoint路径 | 最直接的方法 |
| `--resume` | 启用恢复模式 | 需配合load_run使用 |
| `--load_run` | 指定run名称 | 例如："beamdojo_test/Oct23_23-57-35--..." |
| `--checkpoint` | checkpoint编号 | 例如：3400 |
| `--proj_name` | 新的项目名 | 避免与原训练混淆 |

## ✅ 验证恢复成功

当checkpoint成功加载时，您会看到类似的输出：

```
📂 从指定checkpoint恢复: /path/to/your/checkpoint/model_3400.pt
🏃 开始BEAMDOJO训练...
   🎯 任务: humanoid_beamdojo_full
   🔢 迭代数: [剩余的迭代数]
   🌍 环境数: 1024
   💾 日志路径: [新的日志路径]
```

训练会从iteration 3400继续，而不是从0开始。

## 🚨 常见问题

### 1. 路径错误
```bash
❌ 错误: FileNotFoundError: model_3400.pt
```
**解决**: 确认文件路径正确，使用绝对路径

### 2. 设备不匹配
```bash
❌ 错误: CUDA device mismatch
```
**解决**: 添加正确的设备参数
```bash
--device cuda:1  # 如果您想使用GPU 1
```

### 3. 配置不匹配
```bash
❌ 错误: Configuration mismatch
```
**解决**: 确保当前配置与checkpoint训练时一致

## 💡 最佳实践

1. **使用不同的项目名**: 避免与原训练日志混淆
2. **备份checkpoint**: 在恢复前备份重要的checkpoint
3. **验证设备**: 确保GPU设备与训练时一致
4. **监控指标**: 观察训练是否从正确的iteration继续

## 🎯 立即使用

复制以下命令，将路径替换为您的实际路径：

```bash
python scripts/train_beamdojo.py \
    --task humanoid_beamdojo_full \
    --headless \
    --proj_name "beamdojo_resume_$(date +%m%d_%H%M)" \
    --no_wandb \
    --resume_path "/home/cft/zikang/Humanoid-Terrain-Bench/legged_gym/logs/beamdojo_test/Oct23_23-57-35--humanoid_beamdojo_full_1023_2357/model_3400.pt"
```

这将创建一个带时间戳的新项目名，方便您跟踪不同的恢复训练！