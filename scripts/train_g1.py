#!/usr/bin/env python3
"""
BEAMDOJO G1机器人训练脚本
专门针对G1机器人的训练配置和参数优化
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 🔥 CRITICAL: 使用特殊的导入策略避免IsaacGym冲突
# 方法：延迟导入所有涉及IsaacGym的模块直到真正需要时

def delayed_torch_import():
    """延迟导入torch，确保在IsaacGym模块之前"""
    global torch
    import torch as torch_module
    torch = torch_module
    print(f"✅ PyTorch imported: {torch.__version__}")
    return torch

def delayed_import_training_modules():
    """延迟导入所有训练相关模块"""
    global task_registry
    
    # 使用环境变量强制IsaacGym接受PyTorch已导入的状态
    os.environ['ISAACGYM_TORCH_IMPORTED'] = '1'
    
    try:
        # 先导入torch
        torch = delayed_torch_import()
        
        # 然后导入legged_gym相关模块
        import legged_gym.envs
        from legged_gym.utils import task_registry as tr
        task_registry = tr
        
        print("✅ Training modules imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to import training modules: {e}")
        
        # 尝试替代方案：直接导入不依赖IsaacGym的部分
        try:
            print("🔄 尝试替代导入方案...")
            from legged_gym.utils.task_registry import TaskRegistry
            task_registry = TaskRegistry()
            print("✅ 使用替代导入方案成功")
            return True
        except:
            return False

# 全局变量
torch = None
task_registry = None

def create_g1_args():
    """创建G1训练专用参数"""
    parser = argparse.ArgumentParser(description='BEAMDOJO G1 Robot Training')
    
    # 基本参数
    parser.add_argument('--task', type=str, default='humanoid_beamdojo_g1_stage1', 
                       choices=[
                           'humanoid_beamdojo_g1', 
                           'humanoid_beamdojo_g1_full', 
                           'humanoid_beamdojo_g1_stage1'
                       ],
                       help='G1任务配置')
    parser.add_argument('--stage', type=str, default='stage1',
                       choices=['stage1', 'stage2', 'full'],
                       help='训练阶段')
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
    
    # G1专用训练参数
    parser.add_argument('--max_iterations', type=int, default=3000,
                       help='最大训练迭代数')
    parser.add_argument('--num_envs', type=int, default=4096,
                       help='并行环境数量')
    parser.add_argument('--headless', action='store_true', default=False,
                       help='无头模式运行')
    parser.add_argument('--rl_device', type=str, default='cuda',
                       help='RL设备')
    parser.add_argument('--sim_device', type=str, default='cuda',
                       help='仿真设备')
    
    # G1特定参数
    parser.add_argument('--use_double_critic', action='store_true', default=False,
                       help='启用双Critic网络')
    parser.add_argument('--enable_beamdojo', action='store_true', default=False,
                       help='启用完整BEAMDOJO功能')
    
    # 记录参数
    parser.add_argument('--log_freq', type=int, default=10,
                       help='日志记录频率')
    parser.add_argument('--save_freq', type=int, default=100,
                       help='模型保存频率')
    
    # Wandb参数（与train.py保持一致）
    parser.add_argument('--proj_name', type=str, default='beamdojo_g1',
                       help='wandb项目名称')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='调试模式（禁用wandb，减少环境数）')
    parser.add_argument('--no_wandb', action='store_true', default=False,
                       help='禁用wandb记录')
    
    return parser.parse_args()

def register_g1_tasks():
    """注册G1任务配置 - 使用动态导入避免IsaacGym冲突"""
    try:
        print("🔄 动态导入G1配置...")
        
        # 动态导入G1配置
        import importlib
        g1_config_module = importlib.import_module('legged_gym.envs.humanoid.humanoid_beamdojo_g1_config')
        
        HumanoidBEAMDOJOG1Cfg = g1_config_module.HumanoidBEAMDOJOG1Cfg
        HumanoidBEAMDOJOG1CfgPPO = g1_config_module.HumanoidBEAMDOJOG1CfgPPO
        HumanoidBEAMDOJOG1FullCfg = g1_config_module.HumanoidBEAMDOJOG1FullCfg
        HumanoidBEAMDOJOG1FullCfgPPO = g1_config_module.HumanoidBEAMDOJOG1FullCfgPPO
        HumanoidBEAMDOJOG1Stage1Cfg = g1_config_module.HumanoidBEAMDOJOG1Stage1Cfg
        HumanoidBEAMDOJOG1Stage1CfgPPO = g1_config_module.HumanoidBEAMDOJOG1Stage1CfgPPO
        
        print("✅ G1配置导入成功")
        
        # 尝试导入BEAMDOJO环境，回退到基础环境
        try:
            beamdojo_env_module = importlib.import_module('legged_gym.envs.humanoid.humanoid_beamdojo_env')
            env_class = beamdojo_env_module.HumanoidBEAMDOJOEnv
            print("✅ 使用BEAMDOJO环境")
        except ImportError:
            print("   ⚠️  BEAMDOJO环境不可用，使用基础humanoid环境")
            try:
                humanoid_env_module = importlib.import_module('legged_gym.envs.humanoid.humanoid_env')
                env_class = humanoid_env_module.HumanoidEnv
            except ImportError:
                # 最后回退到基础环境
                base_env_module = importlib.import_module('legged_gym.envs.base.legged_robot')
                env_class = base_env_module.LeggedRobot
                print("   ⚠️  使用基础LeggedRobot环境")
        
        # 注册G1基础配置
        task_registry.register("humanoid_beamdojo_g1", 
                               env_class, 
                               HumanoidBEAMDOJOG1Cfg(), 
                               HumanoidBEAMDOJOG1CfgPPO())
        
        # 注册G1完整BEAMDOJO配置
        task_registry.register("humanoid_beamdojo_g1_full", 
                               env_class, 
                               HumanoidBEAMDOJOG1FullCfg(), 
                               HumanoidBEAMDOJOG1FullCfgPPO())
        
        # 注册G1 Stage1配置
        task_registry.register("humanoid_beamdojo_g1_stage1", 
                               env_class, 
                               HumanoidBEAMDOJOG1Stage1Cfg(), 
                               HumanoidBEAMDOJOG1Stage1CfgPPO())
        
        print("✅ G1任务配置注册成功")
        return True
        
    except ImportError as e:
        print(f"❌ 无法导入G1配置: {e}")
        print("💡 请确保G1配置文件存在且格式正确")
        return False
    except Exception as e:
        print(f"❌ G1任务注册失败: {e}")
        return False

def setup_g1_training_config(args, env_cfg, ppo_cfg):
    """设置G1专用训练配置"""
    print(f"\n🎯 配置G1专用训练参数...")
    
    # 根据训练阶段调整配置
    if args.stage == 'stage1':
        # Stage1: 平坦地形，基础运动
        print(f"   📝 Stage1配置: 平坦地形基础运动学习")
        
        # 强制使用平坦地形
        if hasattr(env_cfg.terrain, 'terrain_types'):
            env_cfg.terrain.terrain_types = ["flat"]
            env_cfg.terrain.terrain_proportions = [1.0]
            env_cfg.terrain.curriculum = False
            print(f"   ✓ 地形: 100%平坦地形")
        
        # 使用标准PPO (Stage1不需要双Critic)
        ppo_cfg.runner.algorithm_class_name = 'PPO'
        ppo_cfg.runner.policy_class_name = 'ActorCritic'
        print(f"   ✓ 算法: 标准PPO")
        
        # 保守的命令范围
        if hasattr(env_cfg.commands, 'ranges'):
            env_cfg.commands.ranges.lin_vel_x = [-0.5, 0.8]
            env_cfg.commands.ranges.lin_vel_y = [-0.3, 0.3]
            env_cfg.commands.ranges.ang_vel_yaw = [-0.6, 0.6]
            print(f"   ✓ 速度范围: 保守设置适合基础学习")
            
    elif args.stage == 'stage2':
        # Stage2: 复杂地形，foothold奖励
        print(f"   📝 Stage2配置: 复杂地形足点导航")
        
        # 复杂地形配置
        if hasattr(env_cfg.terrain, 'terrain_types'):
            env_cfg.terrain.terrain_types = ["rough", "stairs", "obstacles"]
            env_cfg.terrain.terrain_proportions = [0.4, 0.3, 0.3]
            env_cfg.terrain.curriculum = True
            print(f"   ✓ 地形: 复杂地形组合")
        
        # 启用双Critic
        if args.use_double_critic:
            ppo_cfg.runner.algorithm_class_name = 'PPODoubleReward'
            ppo_cfg.runner.policy_class_name = 'ActorCriticRMADoubleReward'
            print(f"   ✓ 算法: 双Critic PPO")
        
    elif args.stage == 'full':
        # 完整训练: 两阶段 + 所有功能
        print(f"   📝 完整训练配置: 两阶段+双Critic+全功能")
        
        # 启用所有BEAMDOJO功能
        if hasattr(ppo_cfg.training, 'enable_two_stage'):
            ppo_cfg.training.enable_two_stage = True
            print(f"   ✓ 两阶段训练: 启用")
        
        # 启用双Critic
        ppo_cfg.runner.algorithm_class_name = 'PPODoubleReward'
        ppo_cfg.runner.policy_class_name = 'ActorCriticRMADoubleReward'
        print(f"   ✓ 算法: 完整BEAMDOJO")
    
    # 通用G1优化
    print(f"\n🤖 应用G1机器人优化...")
    
    # G1物理参数优化
    if hasattr(env_cfg.rewards.scales, 'feet_forward_alignment'):
        env_cfg.rewards.scales.feet_forward_alignment = 1.5  # G1特有
        print(f"   ✓ 脚部前向对齐奖励: 1.5 (G1优化)")
    
    if hasattr(env_cfg.rewards, 'base_height_target'):
        env_cfg.rewards.base_height_target = 0.7  # G1高度
        print(f"   ✓ 目标基座高度: 0.7m (G1规格)")
    
    # G1控制参数
    if hasattr(env_cfg.control, 'action_scale'):
        env_cfg.control.action_scale = 0.25  # G1适中的动作幅度
        print(f"   ✓ 动作缩放: 0.25 (G1优化)")
    
    # 训练参数调整
    ppo_cfg.runner.max_iterations = args.max_iterations
    ppo_cfg.runner.save_interval = args.save_freq
    env_cfg.env.num_envs = args.num_envs
    
    print(f"   ✓ 训练迭代: {args.max_iterations}")
    print(f"   ✓ 并行环境: {args.num_envs}")
    
    return env_cfg, ppo_cfg

def main():
    """G1训练主函数"""
    global task_registry
    
    print("🤖 BEAMDOJO G1机器人训练启动")
    print("="*50)
    
    # 获取参数
    args = create_g1_args()
    
    try:
        # 首先导入训练模块
        print("🔄 初始化训练环境...")
        if not delayed_import_training_modules():
            print("❌ 训练模块导入失败")
            return False
        
        # 注册G1任务
        if not register_g1_tasks():
            return False
        
        # 设置实验名称
        if args.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.experiment_name = f"beamdojo_g1_{args.stage}_{timestamp}"
        
        # 设置实验ID用于wandb
        args.exptid = args.experiment_name
        
        # 设置项目名称
        args.proj_name = 'beamdojo_g1'
        
        print(f"\n📋 G1训练配置:")
        print(f"   🤖 机器人: G1 (12DoF)")
        print(f"   🎯 任务: {args.task}")
        print(f"   📈 阶段: {args.stage}")
        print(f"   🏷️  实验名称: {args.experiment_name}")
        print(f"   🔢 最大迭代: {args.max_iterations}")
        print(f"   🌍 环境数量: {args.num_envs}")
        print(f"   💻 设备: {args.rl_device}")
        print(f"   👻 无头模式: {args.headless}")
        
        # Wandb初始化 - 完全按照train.py的方式
        import wandb
        
        # 设置wandb模式
        if hasattr(args, 'debug') and args.debug:
            mode = "disabled"
            args.rows = 10
            args.cols = 8
            args.num_envs = 64
        else:
            mode = "online"
        
        if hasattr(args, 'no_wandb') and args.no_wandb:
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
        
        # 保存G1配置文件
        try:
            wandb.save(LEGGED_GYM_ENVS_DIR + "/humanoid/humanoid_beamdojo_g1_config.py", policy="now")
            print(f"✅ G1配置文件已保存到wandb")
        except:
            pass
        
        print(f"✅ Wandb初始化成功: {mode}模式，项目: {args.proj_name}")
        
        # 动态导入训练模块
        print("🔄 导入训练模块...")
        import importlib
        rsl_rl_module = importlib.import_module('rsl_rl.runners')
        OnPolicyRunner = rsl_rl_module.OnPolicyRunner
        
        # 获取环境和配置
        print("🔄 创建训练环境...")
        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        ppo_cfg = task_registry.get_cfgs(name=args.task)[1]
        
        # 应用G1专用配置
        env_cfg, ppo_cfg = setup_g1_training_config(args, env_cfg, ppo_cfg)
        
        # 设置实验名称
        ppo_cfg.runner.experiment_name = args.experiment_name
        ppo_cfg.runner.run_name = args.run_name
        
        # 创建runner并开始训练
        print(f"\n🏃 启动G1训练...")
        runner = OnPolicyRunner(env, ppo_cfg, log_dir=None, device=args.rl_device)
        
        if args.resume:
            runner.load(resume=True)
            print(f"   ✓ 恢复训练")
        elif args.load_run:
            runner.load(resume=False, load_run=args.load_run, checkpoint=args.checkpoint)
            print(f"   ✓ 加载检查点: {args.load_run}")
        
        # 显示训练目标
        stage_goals = {
            'stage1': "在平坦地形上学习基础运动技能 (行走、转向、平衡)",
            'stage2': "在复杂地形上学习足点导航和地形适应",
            'full': "完整两阶段训练，包含所有BEAMDOJO功能"
        }
        
        print(f"\n🎯 开始G1 {args.stage.upper()}训练...")
        print(f"   目标: {stage_goals.get(args.stage, '自定义训练目标')}")
        print(f"   预期时间: 约 {args.max_iterations * 24 // 3600} 小时")
        
        # 开始训练
        runner.learn(num_learning_iterations=args.max_iterations, 
                    init_at_random_ep_len=True)
        
        print(f"\n✅ G1 {args.stage.upper()}训练完成!")
        print(f"📁 模型保存路径: {runner.log_dir}")
        print(f"🎉 G1机器人已学会{stage_goals.get(args.stage, '相关技能')}!")
        
    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print(f"💡 建议:")
        print(f"   1. 检查IsaacGym是否正确安装")
        print(f"   2. 检查G1配置文件是否存在")
        print(f"   3. 检查环境变量设置")
        print(f"   4. 尝试重新激活conda环境")
        print(f"   5. 确保PyTorch在IsaacGym之前导入")
        return False
        
    except Exception as e:
        print(f"\n❌ 训练错误: {e}")
        print(f"💡 检查G1配置和环境设置")
        import traceback
        traceback.print_exc()
        return False
    
    return True
    
    return True

if __name__ == '__main__':
    success = main()
    if success:
        print(f"\n🎊 G1训练任务完成! 机器人已准备好展示技能!")
    else:
        print(f"\n😞 G1训练遇到问题，请检查配置和环境")
    sys.exit(0 if success else 1)