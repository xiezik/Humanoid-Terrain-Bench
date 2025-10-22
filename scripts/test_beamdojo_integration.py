#!/usr/bin/env python3
"""
BEAMDOJO集成测试脚本
验证双Critic网络、Foothold奖励和两阶段训练系统的集成

运行方式:
python test_beamdojo_integration.py
"""

import sys
import os
import torch
import numpy as np

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_double_critic_import():
    """测试双Critic网络导入"""
    print("=== 测试双Critic网络导入 ===")
    try:
        from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
        print("✓ ActorCriticRMADoubleReward导入成功")
        
        from rsl_rl.algorithms.ppo_double_reward import PPODoubleReward
        print("✓ PPODoubleReward导入成功")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_foothold_reward_import():
    """测试Foothold奖励系统导入"""
    print("\n=== 测试Foothold奖励系统导入 ===")
    try:
        from legged_gym.utils.foothold_reward import FootholdRewardCalculator, FootholdRewardWrapper
        print("✓ FootholdRewardCalculator导入成功")
        print("✓ FootholdRewardWrapper导入成功")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_two_stage_training_import():
    """测试两阶段训练系统导入"""
    print("\n=== 测试两阶段训练系统导入 ===")
    try:
        from legged_gym.utils.two_stage_training import TwoStageTrainingManager, StageAwareEnvironment
        print("✓ TwoStageTrainingManager导入成功")
        print("✓ StageAwareEnvironment导入成功")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_beamdojo_config_import():
    """测试BEAMDOJO配置导入"""
    print("\n=== 测试BEAMDOJO配置导入 ===")
    try:
        from legged_gym.envs.humanoid.humanoid_beamdojo_config import (
            HumanoidBEAMDOJOCfg,
            HumanoidBEAMDOJOCfgPPO,
            HumanoidBEAMDOJOFullCfg,
            HumanoidBEAMDOJOFullCfgPPO
        )
        print("✓ BEAMDOJO配置类导入成功")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_double_critic_creation():
    """测试双Critic网络创建"""
    print("\n=== 测试双Critic网络创建 ===")
    try:
        from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
        
        # 创建简单的双Critic网络
        num_obs = 100
        num_privileged_obs = 50
        num_actions = 12
        
        actor_critic = ActorCriticRMADoubleReward(
            num_obs=num_obs,
            num_privileged_obs=num_privileged_obs,
            num_actions=num_actions,
            actor_hidden_dims=[128, 64],
            critic_hidden_dims=[128, 64],
            activation='elu',
            use_double_critic=True
        )
        
        print(f"✓ 双Critic网络创建成功")
        print(f"  - 观测维度: {num_obs}")
        print(f"  - 特权观测维度: {num_privileged_obs}")
        print(f"  - 动作维度: {num_actions}")
        
        # 测试前向传播
        obs = torch.randn(1, num_obs)
        privileged_obs = torch.randn(1, num_privileged_obs)
        
        actions, value1, value2, actions_log_prob, mu, sigma = actor_critic.act(obs, privileged_obs)
        print(f"✓ 前向传播成功")
        print(f"  - 动作形状: {actions.shape}")
        print(f"  - 密集价值形状: {value1.shape}") 
        print(f"  - 稀疏价值形状: {value2.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 双Critic网络测试失败: {e}")
        return False

def test_foothold_reward_calculation():
    """测试Foothold奖励计算"""
    print("\n=== 测试Foothold奖励计算 ===")
    try:
        from legged_gym.utils.foothold_reward import FootholdRewardCalculator
        
        # 创建计算器
        calculator = FootholdRewardCalculator(
            num_sample_points=4,
            sample_radius=0.02,
            height_tolerance=0.05,
            device='cpu'
        )
        
        print("✓ FootholdRewardCalculator创建成功")
        
        # 模拟脚部数据
        batch_size = 2
        num_feet = 2
        
        foot_positions = torch.randn(batch_size, num_feet, 3)
        foot_orientations = torch.randn(batch_size, num_feet, 4)  # 四元数
        foot_contacts = torch.randint(0, 2, (batch_size, num_feet)).float()
        
        # 模拟地形高度查询函数
        def mock_terrain_height_fn(positions):
            return torch.zeros(positions.shape[0])
        
        # 计算奖励
        rewards = calculator.compute_foothold_reward(
            foot_positions, foot_orientations, foot_contacts, mock_terrain_height_fn
        )
        
        print(f"✓ Foothold奖励计算成功")
        print(f"  - 奖励形状: {rewards.shape}")
        print(f"  - 奖励范围: [{rewards.min().item():.3f}, {rewards.max().item():.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ Foothold奖励测试失败: {e}")
        return False

def test_two_stage_manager():
    """测试两阶段训练管理器"""
    print("\n=== 测试两阶段训练管理器 ===")
    try:
        from legged_gym.utils.two_stage_training import TwoStageTrainingManager
        
        # 创建模拟配置
        class MockConfig:
            class training:
                class stage1:
                    min_steps = 1000
                    max_steps = 5000
                    success_threshold = 0.8
                dense_rewards = ['reward1', 'reward2']
                sparse_rewards = ['foothold']
        
        cfg = MockConfig()
        
        # 创建管理器
        manager = TwoStageTrainingManager(cfg, device='cpu')
        print("✓ TwoStageTrainingManager创建成功")
        print(f"  - 当前阶段: {manager.get_current_stage()}")
        print(f"  - 地形类型: {manager.get_terrain_type()}")
        
        # 测试奖励分离
        rewards_dict = {
            'reward1': torch.tensor([1.0, 2.0]),
            'reward2': torch.tensor([0.5, 1.5]),
            'foothold': torch.tensor([0.1, 0.2])
        }
        
        dense_rewards, sparse_rewards = manager.separate_rewards(rewards_dict)
        print(f"✓ 奖励分离成功")
        print(f"  - 密集奖励: {list(dense_rewards.keys())}")
        print(f"  - 稀疏奖励: {list(sparse_rewards.keys())}")
        
        # 测试阶段切换检查
        should_switch, new_stage = manager.check_stage_transition(success_rate=0.9, training_step=2000)
        print(f"✓ 阶段切换检查成功")
        print(f"  - 应该切换: {should_switch}")
        print(f"  - 新阶段: {new_stage}")
        
        return True
    except Exception as e:
        print(f"✗ 两阶段管理器测试失败: {e}")
        return False

def test_beamdojo_config():
    """测试BEAMDOJO配置"""
    print("\n=== 测试BEAMDOJO配置 ===")
    try:
        from legged_gym.envs.humanoid.humanoid_beamdojo_config import HumanoidBEAMDOJOFullCfgPPO
        
        # 创建配置实例
        cfg = HumanoidBEAMDOJOFullCfgPPO()
        
        print("✓ BEAMDOJO配置创建成功")
        print(f"  - 双Critic启用: {cfg.algorithm.use_double_critic}")
        print(f"  - 两阶段训练启用: {cfg.training.enable_two_stage}")
        print(f"  - 策略类名: {cfg.runner.policy_class_name}")
        print(f"  - 算法类名: {cfg.runner.algorithm_class_name}")
        
        # 检查奖励配置
        if hasattr(cfg, 'rewards'):
            print(f"  - 密集奖励: {getattr(cfg.rewards, 'dense_rewards', [])}")
            print(f"  - 稀疏奖励: {getattr(cfg.rewards, 'sparse_rewards', [])}")
        
        return True
    except Exception as e:
        print(f"✗ BEAMDOJO配置测试失败: {e}")
        return False

def test_integration_compatibility():
    """测试集成兼容性"""
    print("\n=== 测试集成兼容性 ===")
    try:
        # 测试是否可以同时导入所有组件
        from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
        from rsl_rl.algorithms.ppo_double_reward import PPODoubleReward
        from legged_gym.utils.foothold_reward import FootholdRewardCalculator
        from legged_gym.utils.two_stage_training import TwoStageTrainingManager
        from legged_gym.envs.humanoid.humanoid_beamdojo_config import HumanoidBEAMDOJOFullCfgPPO
        
        print("✓ 所有组件可以同时导入")
        
        # 测试组件间的基本交互
        cfg = HumanoidBEAMDOJOFullCfgPPO()
        
        # 创建双Critic网络
        actor_critic = ActorCriticRMADoubleReward(
            num_obs=100,
            num_privileged_obs=50,
            num_actions=12,
            use_double_critic=True,
            **cfg.policy.__dict__
        )
        
        # 创建PPO算法
        ppo = PPODoubleReward(actor_critic, device='cpu', **cfg.algorithm.__dict__)
        
        print("✓ 双Critic PPO系统创建成功")
        
        # 创建两阶段管理器
        stage_manager = TwoStageTrainingManager(cfg, device='cpu')
        
        print("✓ 两阶段管理器集成成功")
        
        return True
    except Exception as e:
        print(f"✗ 集成兼容性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("BEAMDOJO系统集成测试")
    print("=" * 50)
    
    tests = [
        test_double_critic_import,
        test_foothold_reward_import,
        test_two_stage_training_import,
        test_beamdojo_config_import,
        test_double_critic_creation,
        test_foothold_reward_calculation,
        test_two_stage_manager,
        test_beamdojo_config,
        test_integration_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有BEAMDOJO组件集成测试通过！")
        print("\n建议下一步:")
        print("1. 运行实际训练测试环境设置")
        print("2. 使用小规模环境验证训练流程")
        print("3. 监控双Critic价值函数学习曲线")
        print("4. 验证两阶段切换机制")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
        print("\n故障排除建议:")
        print("1. 确保所有依赖包已正确安装")
        print("2. 检查Python路径配置")
        print("3. 验证torch版本兼容性")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)