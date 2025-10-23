#!/usr/bin/env python3
"""
BEAMDOJO训练脚本
基于legged_gym的train.py结构，适配BEAMDOJO双Critic网络和两阶段训练
解决IsaacGym和PyTorch导入顺序问题
"""

import numpy as np
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 🔥 关键：按照legged_gym的方式导入，确保正确的导入顺序
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

# 导入BEAMDOJO相关模块
try:
    from rsl_rl.algorithms.ppo_double_reward import PPODoubleReward
    from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
    print("✅ BEAMDOJO双Critic模块导入成功")
except ImportError as e:
    print(f"⚠️ BEAMDOJO双Critic模块导入失败: {e}")
    PPODoubleReward = None
    ActorCriticRMADoubleReward = None

def setup_beamdojo_config(args, env_cfg, ppo_cfg):
    """设置BEAMDOJO专用配置"""
    print(f"\n🎯 配置BEAMDOJO训练参数...")
    
    # 检查是否启用BEAMDOJO功能
    enable_beamdojo = getattr(args, 'enable_beamdojo', False)
    use_double_critic = getattr(args, 'use_double_critic', False)
    stage = getattr(args, 'stage', 'auto')
    
    # 根据任务名判断是否启用BEAMDOJO
    if 'beamdojo_full' in args.task:
        enable_beamdojo = True
        use_double_critic = True
        print(f"   🌟 检测到full任务，自动启用完整BEAMDOJO功能")
    elif 'beamdojo' in args.task:
        use_double_critic = True
        print(f"   🎯 检测到beamdojo任务，启用双Critic")
    
    # 根据阶段调整配置
    if stage == 'stage1' or 'stage1' in args.task:
        print(f"   📝 Stage1配置: 平坦地形基础运动学习")
        # 强制使用平坦地形
        if hasattr(env_cfg.terrain, 'terrain_types'):
            env_cfg.terrain.terrain_types = ["flat"]
            env_cfg.terrain.terrain_proportions = [1.0]
            print(f"   ✓ 地形: 100%平坦地形")
        
        # Stage1不强制使用双Critic
        use_double_critic = False
        
    elif stage == 'stage2':
        print(f"   📝 Stage2配置: 复杂地形足点导航")
        if hasattr(env_cfg.terrain, 'terrain_types'):
            env_cfg.terrain.terrain_types = ["rough", "stairs", "obstacles"]
            env_cfg.terrain.terrain_proportions = [0.4, 0.3, 0.3]
            print(f"   ✓ 地形: 复杂地形组合")
        use_double_critic = True
    
    # 启用双Critic配置
    if use_double_critic:
        print(f"   🔄 启用双Critic配置...")
        if hasattr(ppo_cfg.policy, 'use_double_critic'):
            ppo_cfg.policy.use_double_critic = True
        if hasattr(ppo_cfg.algorithm, 'use_double_critic'):
            ppo_cfg.algorithm.use_double_critic = True
        if hasattr(ppo_cfg.runner, 'policy_class_name'):
            ppo_cfg.runner.policy_class_name = 'ActorCriticRMADoubleReward'
        if hasattr(ppo_cfg.runner, 'algorithm_class_name'):
            ppo_cfg.runner.algorithm_class_name = 'PPODoubleReward'
        print(f"   ✓ 双Critic: 已启用")
    
    # 启用完整BEAMDOJO
    if enable_beamdojo:
        print(f"   🌟 启用完整BEAMDOJO功能...")
        if hasattr(ppo_cfg.training, 'enable_two_stage'):
            ppo_cfg.training.enable_two_stage = True
            print(f"   ✓ 两阶段训练: 已启用")
    
    return env_cfg, ppo_cfg

def train_beamdojo(args):
    """BEAMDOJO训练主函数，基于legged_gym的train函数"""
    
    # 设置无头模式
    args.headless = True
    
    # 创建日志路径 - 完全按照train.py的方式
    from legged_gym import LEGGED_GYM_ROOT_DIR
    
    # 生成实验ID
    if not hasattr(args, 'exptid') or not args.exptid:
        args.exptid = f"{args.task}_{datetime.now().strftime('%m%d_%H%M')}"
    
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(
        getattr(args, 'proj_name', 'beamdojo')) + \
        datetime.now().strftime('%b%d_%H-%M-%S--') + args.exptid
    
    try:
        os.makedirs(log_pth)
    except:
        pass
    
    # Wandb设置 - 完全按照train.py的方式
    debug_mode = getattr(args, 'debug', False)
    no_wandb = getattr(args, 'no_wandb', False)
    
    if debug_mode:
        mode = "disabled"
        args.rows = 10
        args.cols = 8  
        args.num_envs = 64
    else:
        mode = "online"
    
    if no_wandb:
        mode = "disabled"
    
    # 初始化wandb - 与train.py保持一致
    import wandb
    proj_name = getattr(args, 'proj_name', 'beamdojo')
    wandb.init(project=proj_name, name=args.exptid, group=args.exptid[:3], mode=mode, dir="../../logs")
    
    # 保存重要文件到wandb - 与train.py保持一致
    from legged_gym import LEGGED_GYM_ENVS_DIR
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")
    
    # 保存BEAMDOJO配置文件
    try:
        wandb.save(LEGGED_GYM_ENVS_DIR + "/humanoid/humanoid_beamdojo_config.py", policy="now")
        wandb.save(LEGGED_GYM_ENVS_DIR + "/humanoid/humanoid_beamdojo_g1_config.py", policy="now")
        print(f"✅ BEAMDOJO配置文件已保存到wandb")
    except:
        pass
    
    print(f"✅ Wandb初始化成功: {mode}模式，项目: {proj_name}")
    
    # 创建环境和算法 (完全按照legged_gym的方式)
    print(f"🔄 创建{args.task}训练环境...")
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # 设置BEAMDOJO配置
    env_cfg, _ = setup_beamdojo_config(args, env_cfg, task_registry.get_cfgs(name=args.task)[1])
    
    # 创建算法runner (使用task_registry的标准方式)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        log_root=log_pth, env=env, name=args.task, args=args)
    
    print(f"\n🏃 开始BEAMDOJO训练...")
    print(f"   🎯 任务: {args.task}")
    print(f"   🔢 迭代数: {train_cfg.runner.max_iterations}")
    print(f"   🌍 环境数: {env_cfg.env.num_envs}")
    print(f"   💾 日志路径: {log_pth}")
    
    # 开始训练 (完全按照legged_gym的方式)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, 
                    init_at_random_ep_len=True)
    
    print(f"\n✅ BEAMDOJO训练完成!")
    print(f"📁 模型保存在: {log_pth}")

def get_beamdojo_args():
    """扩展legged_gym的get_args，添加BEAMDOJO专用参数"""
    
    # 先获取标准参数
    args = get_args()
    
    # 添加BEAMDOJO专用属性
    if not hasattr(args, 'stage'):
        args.stage = 'auto'
    if not hasattr(args, 'use_double_critic'):
        args.use_double_critic = False  
    if not hasattr(args, 'enable_beamdojo'):
        args.enable_beamdojo = False
    if not hasattr(args, 'proj_name'):
        args.proj_name = 'beamdojo'
    
    # 添加wandb相关参数（与train.py保持一致）
    if not hasattr(args, 'debug'):
        args.debug = False
    if not hasattr(args, 'no_wandb'):
        args.no_wandb = False
    if not hasattr(args, 'exptid'):
        args.exptid = f"{args.task}_{datetime.now().strftime('%m%d_%H%M')}"
        
    return args

if __name__ == '__main__':
    print("🚀 BEAMDOJO训练脚本启动")
    print("="*50)
    
    try:
        # 获取参数 (使用legged_gym的标准方式)
        args = get_beamdojo_args()
        
        print(f"📋 训练配置:")
        print(f"   🎯 任务: {args.task}")
        print(f"   💻 设备: {args.rl_device}")
        print(f"   👻 无头模式: {args.headless}")
        
        # 开始训练
        train_beamdojo(args)
        
        print(f"\n🎉 BEAMDOJO训练任务完成!")
        
    except Exception as e:
        print(f"\n❌ 训练错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)