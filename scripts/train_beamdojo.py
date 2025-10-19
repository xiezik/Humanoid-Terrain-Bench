#!/usr/bin/env python3
"""
BEAMDOJO训练示例脚本
展示如何使用双Critic网络和两阶段训练功能

使用方式:
python train_beamdojo.py --task=humanoid_beamdojo_full --headless
"""

import os
import sys
import argparse
from datetime import datetime
import torch

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.two_stage_training import TwoStageTrainingManager, StageAwareEnvironment

# RSL RL imports
from rsl_rl.algorithms import PPO
from rsl_rl.algorithms.ppo_double_reward import PPODoubleReward
from rsl_rl.modules import ActorCritic
from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner


def main():
    """主训练函数"""
    args = get_args()
    
    # 创建环境
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_cfg = task_registry.get_cfgs(name=args.task)[1]
    
    print(f"Training task: {args.task}")
    print(f"Number of environments: {env.num_envs}")
    
    # 检查是否启用BEAMDOJO功能
    use_double_critic = getattr(ppo_cfg.algorithm, 'use_double_critic', False)
    enable_two_stage = getattr(ppo_cfg, 'training', None) and getattr(ppo_cfg.training, 'enable_two_stage', False)
    
    print(f"Double Critic enabled: {use_double_critic}")
    print(f"Two-stage training enabled: {enable_two_stage}")
    
    # 初始化两阶段训练管理器（如果启用）
    stage_manager = None
    if enable_two_stage:
        stage_manager = TwoStageTrainingManager(ppo_cfg, device=env.device)
        
        # 用阶段感知环境包装原环境
        env = StageAwareEnvironment(env, stage_manager)
        print("Two-stage training manager initialized")
        print(f"Current stage: {stage_manager.get_current_stage()}")
        print(f"Terrain type: {stage_manager.get_terrain_type()}")
    
    # 创建策略网络
    if use_double_critic:
        policy_class = ActorCriticRMADoubleReward
        algorithm_class = PPODoubleReward
        print("Using BEAMDOJO Double Critic architecture")
    else:
        policy_class = ActorCritic
        algorithm_class = PPO
        print("Using standard single Critic architecture")
    
    # 初始化策略网络
    actor_critic = policy_class(
        env.cfg.env.num_observations,
        env.cfg.env.num_privileged_obs,
        env.cfg.env.num_actions,
        **ppo_cfg.policy.__dict__
    ).to(env.device)
    
    # 初始化PPO算法
    alg = algorithm_class(actor_critic, device=env.device, **ppo_cfg.algorithm.__dict__)
    
    # 创建训练运行器
    runner = OnPolicyRunner(env, actor_critic, alg, **ppo_cfg.runner.__dict__)
    runner.learn(num_learning_iterations=ppo_cfg.runner.max_iterations, init_at_random_ep_len=True)


def train_with_stage_management():
    """带有阶段管理的训练示例"""
    args = get_args()
    
    # 强制使用BEAMDOJO完整配置
    if args.task != 'humanoid_beamdojo_full':
        print("Switching to humanoid_beamdojo_full for stage management demo")
        args.task = 'humanoid_beamdojo_full'
    
    # 创建环境和配置
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_cfg = task_registry.get_cfgs(name=args.task)[1]
    
    # 初始化两阶段训练管理器
    stage_manager = TwoStageTrainingManager(ppo_cfg, device=env.device)
    env = StageAwareEnvironment(env, stage_manager)
    
    # 创建BEAMDOJO双Critic策略
    actor_critic = ActorCriticRMADoubleReward(
        env.cfg.env.num_observations,
        env.cfg.env.num_privileged_obs,
        env.cfg.env.num_actions,
        **ppo_cfg.policy.__dict__
    ).to(env.device)
    
    # 创建双Critic PPO算法
    alg = PPODoubleReward(actor_critic, device=env.device, **ppo_cfg.algorithm.__dict__)
    
    # 自定义训练循环，支持阶段切换
    print("Starting BEAMDOJO two-stage training...")
    
    for iteration in range(ppo_cfg.runner.max_iterations):
        # 收集数据
        env.reset()
        
        # 更新阶段管理器
        stage_manager.update_training_step(iteration)
        
        # 检查是否需要切换阶段
        success_rate = getattr(env, 'success_rate', 0.0)  # 需要环境实现success_rate计算
        should_switch, new_stage = stage_manager.check_stage_transition(success_rate, iteration)
        
        if should_switch:
            print(f"Switching to Stage {new_stage} at iteration {iteration}")
            # 重新配置环境
            stage_config = stage_manager.get_stage_config()
            # 这里需要根据stage_config重新配置环境参数
            
        # 执行PPO更新
        alg.update()
        
        # 定期保存和记录
        if iteration % ppo_cfg.runner.save_interval == 0:
            print(f"Iteration {iteration}, Stage {stage_manager.get_current_stage()}, Success Rate: {success_rate:.3f}")
            
            # 保存阶段checkpoint
            model_state = actor_critic.state_dict()
            optimizer_state = alg.optimizer.state_dict()
            training_info = {
                'iteration': iteration,
                'success_rate': success_rate
            }
            checkpoint = stage_manager.save_stage_checkpoint(model_state, optimizer_state, training_info)
            
            # 这里可以实际保存checkpoint到文件
            save_path = f"logs/{args.task}/stage_{stage_manager.get_current_stage()}_iter_{iteration}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(checkpoint, save_path)
    
    print("BEAMDOJO training completed!")


if __name__ == '__main__':
    # 检查命令行参数，决定使用哪种训练模式
    if len(sys.argv) > 1 and '--stage-demo' in sys.argv:
        train_with_stage_management()
    else:
        main()