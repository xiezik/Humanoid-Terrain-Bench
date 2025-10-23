#!/usr/bin/env python3
"""
BEAMDOJO Stage1训练脚本
专门用于Stage1训练，使用平坦地形和密集奖励
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 先导入torch
import torch

# 再导入其他模块
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

def create_stage1_args():
    """创建Stage1训练专用参数"""
    parser = argparse.ArgumentParser(description='BEAMDOJO Stage1 Training')
    
    # 基本参数
    parser.add_argument('--task', type=str, default='humanoid_beamdojo', 
                       help='任务名称')
    parser.add_argument('--resume', action='store_true', default=False,
                       help='恢复训练')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称')
    parser.add_argument('--run_name', type=str, default='',
                       help='运行名称')
    parser.add_argument('--load_run', type=str, default='',
                       help='加载的运行路径')
    parser.add_argument('--checkpoint', type=int, default=-1,
                       help='检查点编号')
    
    # Stage1专用参数
    parser.add_argument('--max_iterations', type=int, default=3000,
                       help='Stage1最大训练迭代数')
    parser.add_argument('--num_envs', type=int, default=4096,
                       help='并行环境数量')
    parser.add_argument('--headless', action='store_true', default=False,
                       help='无头模式运行')
    parser.add_argument('--rl_device', type=str, default='cuda',
                       help='RL设备')
    parser.add_argument('--sim_device', type=str, default='cuda',
                       help='仿真设备')
    
    # 记录参数
    parser.add_argument('--log_freq', type=int, default=10,
                       help='日志记录频率')
    parser.add_argument('--save_freq', type=int, default=100,
                       help='模型保存频率')
    
    # Wandb参数（与train.py保持一致）
    parser.add_argument('--proj_name', type=str, default='beamdojo_stage1',
                       help='wandb项目名称')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='调试模式（禁用wandb，减少环境数）')
    parser.add_argument('--no_wandb', action='store_true', default=False,
                       help='禁用wandb记录')
    
    return parser.parse_args()

def main():
    """Stage1训练主函数"""
    print("🚀 BEAMDOJO Stage1 训练启动")
    print("="*50)
    
    # 获取参数
    args = create_stage1_args()
    
    # 设置实验名称
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"beamdojo_stage1_{timestamp}"
    
    # 设置实验ID用于wandb
    args.exptid = args.experiment_name
    
    print(f"📋 训练配置:")
    print(f"   任务: {args.task}")
    print(f"   实验名称: {args.experiment_name}")
    print(f"   最大迭代: {args.max_iterations}")
    print(f"   环境数量: {args.num_envs}")
    print(f"   设备: {args.rl_device}")
    print(f"   无头模式: {args.headless}")
    
    try:
        # Wandb初始化 - 完全按照train.py的方式
        import wandb
        
        # 设置wandb模式
        if args.debug:
            mode = "disabled"
            args.rows = 10
            args.cols = 8
            args.num_envs = 64
        else:
            mode = "online"
        
        if args.no_wandb:
            mode = "disabled"
        
        # 创建日志路径
        from legged_gym import LEGGED_GYM_ROOT_DIR
        log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + \
                  datetime.now().strftime('%b%d_%H-%M-%S--') + args.exptid
        
        try:
            os.makedirs(log_pth)
        except:
            pass
            
        # 初始化wandb
        wandb.init(project=args.proj_name, name=args.exptid, group=args.exptid[:3], mode=mode, dir="../../logs")
        
        # 保存重要文件到wandb
        from legged_gym import LEGGED_GYM_ENVS_DIR
        wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
        wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")
        
        # 保存BEAMDOJO配置文件
        try:
            wandb.save(LEGGED_GYM_ENVS_DIR + "/humanoid/humanoid_beamdojo_config.py", policy="now")
            print(f"✅ BEAMDOJO配置文件已保存到wandb")
        except:
            pass
        
        print(f"✅ Wandb初始化成功: {mode}模式，项目: {args.proj_name}")
        
        # 动态导入训练模块
        from rsl_rl.runners import OnPolicyRunner
        
        # 获取环境和配置
        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        ppo_cfg = task_registry.get_cfgs(name=args.task)[1]
        
        # Stage1专用配置调整
        print(f"\n🎯 应用Stage1专用配置...")
        
        # 强制使用平坦地形
        if hasattr(env_cfg.terrain, 'stage1_terrain_types'):
            env_cfg.terrain.terrain_types = env_cfg.terrain.stage1_terrain_types
            env_cfg.terrain.terrain_proportions = [1.0]  # 100%平坦地形
            print(f"   ✓ 地形类型: {env_cfg.terrain.terrain_types}")
        
        # 禁用两阶段训练（仅Stage1）
        if hasattr(ppo_cfg.training, 'enable_two_stage'):
            ppo_cfg.training.enable_two_stage = False
            print(f"   ✓ 禁用两阶段训练")
        
        # 使用标准PPO算法（Stage1不需要双Critic）
        ppo_cfg.runner.algorithm_class_name = 'PPO'
        ppo_cfg.runner.policy_class_name = 'ActorCritic'
        print(f"   ✓ 使用标准PPO算法")
        
        # 调整训练参数
        ppo_cfg.runner.max_iterations = args.max_iterations
        ppo_cfg.runner.save_interval = args.save_freq
        print(f"   ✓ 最大迭代数: {args.max_iterations}")
        
        # 设置实验名称
        ppo_cfg.runner.experiment_name = args.experiment_name
        ppo_cfg.runner.run_name = args.run_name
        
        # 创建runner并开始训练
        print(f"\n🏃 启动训练...")
        runner = OnPolicyRunner(env, ppo_cfg, log_dir=None, device=args.rl_device)
        
        if args.resume:
            runner.load(resume=True)
            print(f"   ✓ 恢复训练")
        elif args.load_run:
            runner.load(resume=False, load_run=args.load_run, checkpoint=args.checkpoint)
            print(f"   ✓ 加载检查点: {args.load_run}")
        
        print(f"\n🎯 开始Stage1训练...")
        print(f"   目标: 在平坦地形上学习基础运动技能")
        print(f"   预期时间: 约 {args.max_iterations * 24 // 3600} 小时")
        
        runner.learn(num_learning_iterations=args.max_iterations, 
                    init_at_random_ep_len=True)
        
        print(f"\n✅ Stage1训练完成!")
        print(f"📁 模型保存路径: {runner.log_dir}")
        
    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print(f"💡 建议:")
        print(f"   1. 检查IsaacGym是否正确安装")
        print(f"   2. 检查环境变量设置")
        print(f"   3. 尝试重新激活conda环境")
        return False
        
    except Exception as e:
        print(f"\n❌ 训练错误: {e}")
        print(f"💡 检查配置和环境设置")
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)