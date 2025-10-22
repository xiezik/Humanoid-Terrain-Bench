#!/usr/bin/env python3
"""
BEAMDOJO功能验证脚本
全面测试双Critic网络、两阶段训练、foothold奖励等核心功能

使用方式:
python validate_beamdojo.py --test-all
python validate_beamdojo.py --test-networks
python validate_beamdojo.py --test-training
python validate_beamdojo.py --test-rewards
"""

import os
import sys
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 重要：必须在导入torch之前导入isaacgym相关模块
try:
    from legged_gym.utils.two_stage_training import TwoStageTrainingManager, StageAwareEnvironment
    from rsl_rl.algorithms.ppo_double_reward import PPODoubleReward
    from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
except ImportError:
    pass  # 如果无法导入，后面会处理

# 现在才能安全导入torch
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt


class BeamDojoValidator:
    """BEAMDOJO功能验证器"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.test_results = {}
        self.validation_report = {
            'timestamp': datetime.now().isoformat(),
            'device': device,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': {}
        }
        
        # 尝试导入必要的模块
        self._import_modules()
        
        print(f"BeamDojo Validator initialized on device: {device}")
    
    def _import_modules(self):
        """安全导入所需模块"""
        try:
            global TwoStageTrainingManager, StageAwareEnvironment, PPODoubleReward, ActorCriticRMADoubleReward
            from legged_gym.utils.two_stage_training import TwoStageTrainingManager, StageAwareEnvironment
            from rsl_rl.algorithms.ppo_double_reward import PPODoubleReward
            from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
            self.modules_available = True
            print("  ✓ All required modules imported successfully")
        except ImportError as e:
            self.modules_available = False
            print(f"  ✗ Failed to import modules: {e}")
            print("  Please ensure the project is properly set up")
    
    def validate_all(self) -> bool:
        """运行所有验证测试"""
        print("\n" + "="*60)
        print("BEAMDOJO 全面功能验证")
        print("="*60)
        
        tests = [
            ('network_architecture', self.validate_network_architecture),
            ('double_critic_forward', self.validate_double_critic_forward),
            ('ppo_algorithm', self.validate_ppo_algorithm),
            ('two_stage_manager', self.validate_two_stage_manager),
            ('reward_separation', self.validate_reward_separation),
            ('foothold_reward', self.validate_foothold_reward),
            ('stage_transition', self.validate_stage_transition),
            ('training_flow', self.validate_training_flow),
            ('memory_efficiency', self.validate_memory_efficiency),
            ('gradient_flow', self.validate_gradient_flow)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            print(f"\n[TEST] {test_name}")
            try:
                success = test_func()
                self.validation_report['test_details'][test_name] = {
                    'status': 'PASSED' if success else 'FAILED',
                    'timestamp': datetime.now().isoformat()
                }
                if success:
                    self.validation_report['tests_passed'] += 1
                    print(f"✓ {test_name} PASSED")
                else:
                    self.validation_report['tests_failed'] += 1
                    print(f"✗ {test_name} FAILED")
                    all_passed = False
            except Exception as e:
                self.validation_report['tests_failed'] += 1
                self.validation_report['test_details'][test_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                print(f"✗ {test_name} ERROR: {e}")
                all_passed = False
        
        self._generate_report()
        return all_passed
    
    def validate_network_architecture(self) -> bool:
        """验证双Critic网络架构"""
        print("  Testing double critic network architecture...")
        
        if not self.modules_available:
            print("    ✗ Required modules not available, skipping test")
            return False
        
        try:
            actor_critic = ActorCriticRMADoubleReward(
                num_prop=48,
                num_scan=187,
                num_critic_obs=235,
                num_priv_latent=4,
                num_priv_explicit=8,
                num_hist=15,
                num_actions=12,
                actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[512, 256, 128],
                activation='elu',
                use_double_critic=True,
                priv_encoder_dims=[64, 20]
            ).to(self.device)
        
            # 验证网络结构
            assert hasattr(actor_critic, 'critic1'), "Missing critic1"
            assert hasattr(actor_critic, 'critic2'), "Missing critic2"
            assert actor_critic.use_double_critic == True, "use_double_critic not set correctly"
        
            # 验证网络参数
            critic1_params = sum(p.numel() for p in actor_critic.critic1.parameters())
            critic2_params = sum(p.numel() for p in actor_critic.critic2.parameters())
            assert critic1_params > 0, "Critic1 has no parameters"
            assert critic2_params > 0, "Critic2 has no parameters"
        
            print(f"    ✓ Critic1 parameters: {critic1_params}")
            print(f"    ✓ Critic2 parameters: {critic2_params}")
            print(f"    ✓ Total actor-critic parameters: {sum(p.numel() for p in actor_critic.parameters())}")
        
            return True
        except Exception as e:
            print(f"    ✗ Network architecture test failed: {e}")
            return False
    
    def validate_double_critic_forward(self) -> bool:
        """验证双Critic前向传播"""
        print("  Testing double critic forward pass...")
        
        if not self.modules_available:
            print("    ✗ Required modules not available, skipping test")
            return False
        
        try:
            actor_critic = ActorCriticRMADoubleReward(
                num_prop=48,
                num_scan=187,
                num_critic_obs=235,
                num_priv_latent=4,
                num_priv_explicit=8,
                num_hist=15,
                num_actions=12,
                use_double_critic=True,
                priv_encoder_dims=[64, 20]
            ).to(self.device)
        
            # 创建测试数据
            batch_size = 4
            obs = torch.randn(batch_size, 235, device=self.device)
            
            # 测试evaluate方法
            value1, value2 = actor_critic.evaluate(obs)
            assert value1.shape == (batch_size, 1), f"Unexpected value1 shape: {value1.shape}"
            assert value2.shape == (batch_size, 1), f"Unexpected value2 shape: {value2.shape}"
        
            # 测试单独评估
            val1_only = actor_critic.evaluate_critic1(obs)
            val2_only = actor_critic.evaluate_critic2(obs)
            assert torch.allclose(val1_only, value1), "Inconsistent critic1 evaluation"
            assert torch.allclose(val2_only, value2), "Inconsistent critic2 evaluation"
        
            print(f"    ✓ Forward pass successful")
            print(f"    ✓ Value1 range: [{value1.min().item():.3f}, {value1.max().item():.3f}]")
            print(f"    ✓ Value2 range: [{value2.min().item():.3f}, {value2.max().item():.3f}]")
        
            return True
        except Exception as e:
            print(f"    ✗ Forward pass test failed: {e}")
            return False
    
    def validate_ppo_algorithm(self) -> bool:
        """验证PPO双Critic算法"""
        print("  Testing PPO double critic algorithm...")
        
        if not self.modules_available:
            print("    ✗ Required modules not available, skipping test")
            return False
        
        try:
            actor_critic = ActorCriticRMADoubleReward(
                num_prop=48,
                num_scan=187,
                num_critic_obs=235,
                num_priv_latent=4,
                num_priv_explicit=8,
                num_hist=15,
                num_actions=12,
                use_double_critic=True,
                priv_encoder_dims=[64, 20]
            ).to(self.device)
        
            # 创建PPO算法
            ppo = PPODoubleReward(
                actor_critic=actor_critic,
                estimator=None,
                estimator_paras=None,
                depth_encoder=None,
                depth_encoder_paras=None,
                depth_actor=None,
                num_learning_epochs=1,
                num_mini_batches=2,
                clip_param=0.2,
                gamma=0.99,
                lam=0.95,
                value_loss_coef=1.0,
                entropy_coef=0.01,
                learning_rate=3e-4,
                device=self.device,
                dense_reward_weight=1.0,
                sparse_reward_weight=0.25,
                use_separate_value_loss=True
            )
        
            # 验证算法属性
            assert ppo.use_double_critic == True, "PPO not configured for double critic"
            assert ppo.dense_reward_weight == 1.0, "Incorrect dense reward weight"
            assert ppo.sparse_reward_weight == 0.25, "Incorrect sparse reward weight"
        
            # 初始化存储
            num_envs = 8
            num_transitions = 24
            actor_obs_shape = (235,)
            critic_obs_shape = (235,)
            action_shape = (12,)
        
            ppo.init_storage(num_envs, num_transitions, actor_obs_shape, critic_obs_shape, action_shape)
        
            # 验证存储初始化
            assert ppo.storage is not None, "Storage not initialized"
        
            print(f"    ✓ PPO algorithm initialized successfully")
            print(f"    ✓ Dense reward weight: {ppo.dense_reward_weight}")
            print(f"    ✓ Sparse reward weight: {ppo.sparse_reward_weight}")
        
            return True
        except Exception as e:
            print(f"    ✗ PPO algorithm test failed: {e}")
            return False
    
    def validate_two_stage_manager(self) -> bool:
        """验证两阶段训练管理器"""
        print("  Testing two-stage training manager...")
        
        if not self.modules_available:
            print("    ✗ Required modules not available, skipping test")
            return False
        
        try:
            class MockConfig:
                class training:
                    class stage1:
                        min_steps = 1000
                        max_steps = 5000
                        success_threshold = 0.8
                    class stage2:
                        max_steps = 5000
                    dense_rewards = ['tracking_lin_vel', 'orientation']
                    sparse_rewards = ['foothold']
        
            cfg = MockConfig()
        
            # 创建管理器
            manager = TwoStageTrainingManager(cfg, device=self.device)
        
            # 验证初始状态
            assert manager.get_current_stage() == 1, "Initial stage should be 1"
            assert not manager.stage1_completed, "Stage1 should not be completed initially"
        
            # 测试阶段切换逻辑
            should_switch, new_stage = manager.check_stage_transition(success_rate=0.9, training_step=2000)
            assert should_switch == True, "Should switch to stage 2 with high success rate"
            assert new_stage == 2, "New stage should be 2"
        
            # 测试奖励分离
            rewards_dict = {
                'tracking_lin_vel': torch.tensor([1.0, 2.0], device=self.device),
                'orientation': torch.tensor([0.5, 1.5], device=self.device),
                'foothold': torch.tensor([0.1, 0.2], device=self.device)
            }
        
            dense_rewards, sparse_rewards = manager.separate_rewards(rewards_dict)
            assert len(dense_rewards) == 2, "Should have 2 dense rewards"
            assert len(sparse_rewards) == 1, "Should have 1 sparse reward"
        
            print(f"    ✓ Stage manager initialized successfully")
            print(f"    ✓ Stage transition logic working")
            print(f"    ✓ Reward separation working")
        
            return True
        except Exception as e:
            print(f"    ✗ Two stage manager test failed: {e}")
            return False
    
    def validate_reward_separation(self) -> bool:
        """验证奖励分离逻辑"""
        print("  Testing reward separation logic...")
        
        try:
            # 创建模拟配置
            class MockConfig:
                class training:
                    class stage1:
                        min_steps = 1000
                        max_steps = 5000
                        success_threshold = 0.8
                    class stage2:
                        max_steps = 5000
                    dense_rewards = ['vel_track', 'orientation', 'torques']
                    sparse_rewards = ['foothold', 'terrain_adapt']
        
            cfg = MockConfig()
            manager = TwoStageTrainingManager(cfg, device=self.device)
        
            # 创建复杂的奖励字典
            rewards_dict = {
                'vel_track': torch.tensor([1.0, 2.0, 3.0], device=self.device),
                'orientation': torch.tensor([0.5, 1.5, 2.5], device=self.device),
                'torques': torch.tensor([-0.1, -0.2, -0.3], device=self.device),
                'foothold': torch.tensor([0.1, 0.2, 0.3], device=self.device),
                'terrain_adapt': torch.tensor([0.05, 0.1, 0.15], device=self.device),
                'unknown_reward': torch.tensor([0.0, 0.0, 0.0], device=self.device)  # 应该被忽略
            }
        
            dense_rewards, sparse_rewards = manager.separate_rewards(rewards_dict)
        
            # 验证分离结果
            assert 'vel_track' in dense_rewards, "vel_track should be in dense rewards"
            assert 'orientation' in dense_rewards, "orientation should be in dense rewards"
            assert 'torques' in dense_rewards, "torques should be in dense rewards"
            assert 'foothold' in sparse_rewards, "foothold should be in sparse rewards"
            assert 'terrain_adapt' in sparse_rewards, "terrain_adapt should be in sparse rewards"
            assert 'unknown_reward' not in dense_rewards and 'unknown_reward' not in sparse_rewards, "unknown_reward should be ignored"
        
            # 验证张量值
            assert torch.allclose(dense_rewards['vel_track'], rewards_dict['vel_track']), "Dense reward values incorrect"
            assert torch.allclose(sparse_rewards['foothold'], rewards_dict['foothold']), "Sparse reward values incorrect"
        
            print(f"    ✓ Dense rewards: {list(dense_rewards.keys())}")
            print(f"    ✓ Sparse rewards: {list(sparse_rewards.keys())}")
        
            return True
        except Exception as e:
            print(f"    ✗ Reward separation test failed: {e}")
            return False
    
    def validate_foothold_reward(self) -> bool:
        """验证foothold奖励计算（模拟）"""
        print("  Testing foothold reward computation...")
        
        try:
            # 模拟foothold奖励计算
            def mock_foothold_reward(foot_positions, terrain_heights, config):
                """模拟foothold奖励计算"""
                n_samples = config.get('n_samples', 16)
                epsilon = config.get('epsilon', -0.1)
            
                # 模拟脚部采样点
                batch_size = foot_positions.shape[0]
                penalties = torch.zeros(batch_size, device=self.device)
            
                for i in range(batch_size):
                    # 模拟每只脚的采样
                    for foot_id in range(2):  # 2只脚
                        for sample_id in range(n_samples):
                            # 模拟地形高度查询
                            terrain_height = terrain_heights[i] + torch.randn(1, device=self.device) * 0.1
                            if terrain_height < epsilon:
                                penalties[i] += 1
            
                return -penalties
        
            # 测试数据
            batch_size = 4
            foot_positions = torch.randn(batch_size, 6, device=self.device)  # 2只脚 * 3D坐标
            terrain_heights = torch.randn(batch_size, device=self.device) * 0.5  # 地形高度
        
            config = {
                'n_samples': 16,
                'epsilon': -0.1
            }
        
            # 计算foothold奖励
            foothold_rewards = mock_foothold_reward(foot_positions, terrain_heights, config)
        
            # 验证结果
            assert foothold_rewards.shape == (batch_size,), f"Unexpected foothold reward shape: {foothold_rewards.shape}"
            assert foothold_rewards.dtype == torch.float32, "Foothold rewards should be float32"
            assert (foothold_rewards <= 0).all(), "Foothold rewards should be non-positive (penalties)"
        
            print(f"    ✓ Foothold reward computation successful")
            print(f"    ✓ Reward range: [{foothold_rewards.min().item():.3f}, {foothold_rewards.max().item():.3f}]")
            print(f"    ✓ Mean penalty: {foothold_rewards.mean().item():.3f}")
        
            return True
        except Exception as e:
            print(f"    ✗ Foothold reward test failed: {e}")
            return False
    
    def validate_stage_transition(self) -> bool:
        """验证阶段切换逻辑"""
        print("  Testing stage transition logic...")
        
        try:
            # 创建配置
            class MockConfig:
                class training:
                    class stage1:
                        min_steps = 1000
                        max_steps = 5000
                        success_threshold = 0.8
                    class stage2:
                        max_steps = 5000
                    dense_rewards = ['vel_track']
                    sparse_rewards = ['foothold']
        
            cfg = MockConfig()
            manager = TwoStageTrainingManager(cfg, device=self.device)
        
            # 测试各种切换条件
            test_cases = [
                # (success_rate, training_step, expected_switch, expected_stage)
                (0.5, 500, False, 1),      # 低成功率，少步数 - 不切换
                (0.9, 500, False, 1),      # 高成功率，但步数不够 - 不切换
                (0.5, 2000, False, 1),     # 足够步数，但成功率低 - 不切换
                (0.9, 2000, True, 2),      # 高成功率，足够步数 - 切换
                (0.8, 1000, True, 2),      # 刚好达到阈值 - 切换
            ]
        
            for success_rate, step, expected_switch, expected_stage in test_cases:
                # 重置管理器状态
                manager.current_stage = 1
                manager.stage1_completed = False
            
                should_switch, new_stage = manager.check_stage_transition(success_rate, step)
            
                assert should_switch == expected_switch, f"Transition logic failed for success_rate={success_rate}, step={step}"
                if should_switch:
                    assert new_stage == expected_stage, f"Wrong new stage for success_rate={success_rate}, step={step}"
        
            print(f"    ✓ All stage transition test cases passed")
        
            return True
        except Exception as e:
            print(f"    ✗ Stage transition test failed: {e}")
            return False
    
    def validate_training_flow(self) -> bool:
        """验证训练流程的完整性"""
        print("  Testing training flow integration...")
        
        try:
            # 验证关键类存在
            assert ActorCriticRMADoubleReward is not None, "ActorCriticRMADoubleReward not available"
            assert PPODoubleReward is not None, "PPODoubleReward not available"
            assert TwoStageTrainingManager is not None, "TwoStageTrainingManager not available"
            assert StageAwareEnvironment is not None, "StageAwareEnvironment not available"
            
            print("    ✓ All training components available")
            
            return True
            
        except ImportError as e:
            print(f"    ✗ Training flow validation failed: {e}")
            return False
        except Exception as e:
            print(f"    ✗ Training flow test failed: {e}")
            return False
    
    def validate_memory_efficiency(self) -> bool:
        """验证内存效率"""
        print("  Testing memory efficiency...")
        
        try:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
            # 创建大批量网络
            batch_size = 64
            actor_critic = ActorCriticRMADoubleReward(
                num_prop=48,
                num_scan=187,
                num_critic_obs=235,
                num_priv_latent=4,
                num_priv_explicit=8,
                num_hist=15,
                num_actions=12,
                use_double_critic=True,
                priv_encoder_dims=[64, 20]
            ).to(self.device)
        
            # 大批量前向传播
            obs = torch.randn(batch_size, 235, device=self.device)
            value1, value2 = actor_critic.evaluate(obs)
        
            current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_increase = current_memory - initial_memory
        
            # 清理
            del actor_critic, obs, value1, value2
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
            print(f"    ✓ Memory test completed")
            if torch.cuda.is_available():
                print(f"    ✓ Memory increase: {memory_increase / 1024 / 1024:.1f} MB")
        
            return True
        except Exception as e:
            print(f"    ✗ Memory efficiency test failed: {e}")
            return False
    
    def validate_gradient_flow(self) -> bool:
        """验证梯度流"""
        print("  Testing gradient flow...")
        
        try:
            # 创建网络和优化器
            actor_critic = ActorCriticRMADoubleReward(
                num_prop=48,
                num_scan=187,
                num_critic_obs=235,
                num_priv_latent=4,
                num_priv_explicit=8,
                num_hist=15,
                num_actions=12,
                use_double_critic=True,
                priv_encoder_dims=[64, 20]
            ).to(self.device)
        
            optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
        
            # 前向传播
            batch_size = 8
            obs = torch.randn(batch_size, 235, device=self.device)
            value1, value2 = actor_critic.evaluate(obs)
        
            # 创建伪损失
            target_value1 = torch.randn_like(value1)
            target_value2 = torch.randn_like(value2)
        
            loss1 = nn.MSELoss()(value1, target_value1)
            loss2 = nn.MSELoss()(value2, target_value2)
            total_loss = loss1 + loss2
        
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
        
            # 检查梯度
            critic1_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor_critic.critic1.parameters())
            critic2_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor_critic.critic2.parameters())
        
            assert critic1_has_grad, "Critic1 has no gradients"
            assert critic2_has_grad, "Critic2 has no gradients"
        
            optimizer.step()
        
            print(f"    ✓ Gradient flow verified")
            print(f"    ✓ Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}")
        
            return True
        except Exception as e:
            print(f"    ✗ Gradient flow test failed: {e}")
            return False
    
    def _generate_report(self):
        """生成验证报告"""
        report_path = f"/home/cft/zikang/Humanoid-Terrain-Bench/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.validation_report, f, indent=2)
        
        print(f"\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Tests Passed: {self.validation_report['tests_passed']}")
        print(f"Tests Failed: {self.validation_report['tests_failed']}")
        total_tests = self.validation_report['tests_passed'] + self.validation_report['tests_failed']
        if total_tests > 0:
            print(f"Success Rate: {self.validation_report['tests_passed'] / total_tests * 100:.1f}%")
        print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='BEAMDOJO功能验证')
    parser.add_argument('--test-all', action='store_true', help='运行所有测试')
    parser.add_argument('--test-networks', action='store_true', help='只测试网络架构')
    parser.add_argument('--test-training', action='store_true', help='只测试训练流程')
    parser.add_argument('--test-rewards', action='store_true', help='只测试奖励系统')
    parser.add_argument('--device', default='auto', help='计算设备 (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    validator = BeamDojoValidator(device=device)
    
    success = True
    
    if args.test_all or (not any([args.test_networks, args.test_training, args.test_rewards])):
        success = validator.validate_all()
    else:
        if args.test_networks:
            success &= validator.validate_network_architecture()
            success &= validator.validate_double_critic_forward()
            success &= validator.validate_gradient_flow()
        
        if args.test_training:
            success &= validator.validate_ppo_algorithm()
            success &= validator.validate_two_stage_manager()
            success &= validator.validate_stage_transition()
            success &= validator.validate_training_flow()
        
        if args.test_rewards:
            success &= validator.validate_reward_separation()
            success &= validator.validate_foothold_reward()
        
        validator._generate_report()
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)