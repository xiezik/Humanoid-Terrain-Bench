#!/usr/bin/env python3
"""
BEAMDOJO端到端集成测试
验证完整训练流程的正确性和鲁棒性
"""

import os
import sys
from datetime import datetime
import json
import traceback
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 重要：必须在导入torch之前导入legged_gym模块
try:
    from legged_gym.utils.two_stage_training import TwoStageTrainingManager, StageAwareEnvironment
    from rsl_rl.algorithms.ppo_double_reward import PPODoubleReward
    from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
except ImportError:
    pass

# 现在才能安全导入torch
import torch
import numpy as np


class BeamDojoIntegrationTester:
    """BEAMDOJO集成测试器"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.test_results = {}
        self.mini_batch_size = 4  # 小批量测试
        self.mini_steps = 10      # 少量步数测试
        
        print(f"BeamDojo Integration Tester initialized on device: {device}")
    
    def test_full_training_pipeline(self) -> bool:
        """测试完整训练管道"""
        print("\n[INTEGRATION TEST] Full Training Pipeline")
        
        try:
            # 1. 创建模拟环境配置
            env_config = self._create_mock_env_config()
            ppo_config = self._create_mock_ppo_config()
            
            # 2. 创建双Critic网络
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
            
            print("  ✓ Double critic network created")
            
            # 3. 创建PPO算法
            ppo = PPODoubleReward(
                actor_critic=actor_critic,
                estimator=None,
                estimator_paras=None,
                depth_encoder=None,
                depth_encoder_paras=None,
                depth_actor=None,
                **ppo_config,
                device=self.device
            )
            
            print("  ✓ PPO algorithm created")
            
            # 4. 初始化存储
            num_envs = self.mini_batch_size
            num_transitions = 24
            actor_obs_shape = (235,)
            critic_obs_shape = (235,)
            action_shape = (12,)
            
            ppo.init_storage(num_envs, num_transitions, actor_obs_shape, critic_obs_shape, action_shape)
            print("  ✓ Storage initialized")
            
            # 5. 创建两阶段管理器
            stage_manager = TwoStageTrainingManager(env_config, device=self.device)
            print("  ✓ Two-stage manager created")
            
            # 6. 模拟训练循环
            success = self._run_mini_training_loop(ppo, stage_manager, num_envs)
            
            if success:
                print("  ✓ Mini training loop completed successfully")
                return True
            else:
                print("  ✗ Mini training loop failed")
                return False
                
        except Exception as e:
            print(f"  ✗ Pipeline test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_stage_transition_integration(self) -> bool:
        """测试阶段切换集成"""
        print("\n[INTEGRATION TEST] Stage Transition Integration")
        
        try:
            # 创建环境配置
            env_config = self._create_mock_env_config()
            stage_manager = TwoStageTrainingManager(env_config, device=self.device)
            
            # 创建模拟环境
            mock_env = self._create_mock_environment()
            stage_aware_env = StageAwareEnvironment(mock_env, stage_manager)
            
            print("  ✓ Stage-aware environment created")
            
            # 测试Stage1行为
            initial_stage = stage_manager.get_current_stage()
            self.assertEqual(initial_stage, 1, "Should start in Stage 1")
            
            terrain_type = stage_manager.get_terrain_type()
            self.assertIn('flat', terrain_type.lower(), "Stage 1 should use flat terrain")
            print(f"  ✓ Stage 1 terrain: {terrain_type}")
            
            # 模拟高成功率触发切换
            should_switch, new_stage = stage_manager.check_stage_transition(
                success_rate=0.9, training_step=2000
            )
            
            self.assertTrue(should_switch, "Should switch to Stage 2")
            self.assertEqual(new_stage, 2, "Should switch to Stage 2")
            print("  ✓ Stage transition logic works")
            
            # 执行切换
            stage_manager.current_stage = 2
            stage_manager.stage1_completed = True
            
            # 验证Stage2行为
            terrain_type_stage2 = stage_manager.get_terrain_type()
            command_ranges = stage_manager.get_command_ranges()
            
            print(f"  ✓ Stage 2 terrain: {terrain_type_stage2}")
            print(f"  ✓ Stage 2 commands: {command_ranges}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Stage transition test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_reward_computation_integration(self) -> bool:
        """测试奖励计算集成"""
        print("\n[INTEGRATION TEST] Reward Computation Integration")
        
        try:
            # 创建环境和管理器
            env_config = self._create_mock_env_config()
            stage_manager = TwoStageTrainingManager(env_config, device=self.device)
            
            # 模拟复杂奖励数据
            batch_size = self.mini_batch_size
            rewards_dict = {
                'tracking_lin_vel': torch.randn(batch_size, device=self.device),
                'tracking_ang_vel': torch.randn(batch_size, device=self.device) * 0.5,
                'orientation': torch.randn(batch_size, device=self.device) * 0.3,
                'base_height': torch.randn(batch_size, device=self.device) * 0.2,
                'torques': torch.randn(batch_size, device=self.device) * -0.1,
                'foothold': torch.randn(batch_size, device=self.device) * -0.05,  # 稀疏惩罚
                'action_rate': torch.randn(batch_size, device=self.device) * -0.02,
            }
            
            # 测试奖励分离
            dense_rewards, sparse_rewards = stage_manager.separate_rewards(rewards_dict)
            
            print(f"  ✓ Dense rewards separated: {list(dense_rewards.keys())}")
            print(f"  ✓ Sparse rewards separated: {list(sparse_rewards.keys())}")
            
            # 测试奖励计算
            reward_scales = {name: 1.0 for name in rewards_dict.keys()}
            total_dense, total_sparse = stage_manager.compute_separated_rewards(
                rewards_dict, reward_scales
            )
            
            self.assertEqual(total_dense.shape, (batch_size,))
            self.assertEqual(total_sparse.shape, (batch_size,))
            print(f"  ✓ Dense reward range: [{total_dense.min():.3f}, {total_dense.max():.3f}]")
            print(f"  ✓ Sparse reward range: [{total_sparse.min():.3f}, {total_sparse.max():.3f}]")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Reward computation test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_memory_management_integration(self) -> bool:
        """测试内存管理集成"""
        print("\n[INTEGRATION TEST] Memory Management Integration")
        
        try:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # 创建完整系统
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
            
            ppo_config = self._create_mock_ppo_config()
            ppo = PPODoubleReward(
                actor_critic=actor_critic,
                estimator=None,
                estimator_paras=None,
                depth_encoder=None,
                depth_encoder_paras=None,
                depth_actor=None,
                **ppo_config,
                device=self.device
            )
            
            # 初始化存储
            batch_size = 16  # 较大批量测试内存
            ppo.init_storage(batch_size, 48, (235,), (235,), (12,))
            
            # 多次前向传播测试内存泄漏
            for i in range(5):
                obs = torch.randn(batch_size, 235, device=self.device)
                critic_obs = torch.randn(batch_size, 235, device=self.device)
                
                # 模拟act调用
                actions = ppo.act(obs, critic_obs, {})
                
                # 清理中间变量
                del obs, critic_obs, actions
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_increase = current_memory - initial_memory
            
            print(f"  ✓ Memory management test completed")
            if torch.cuda.is_available():
                print(f"  ✓ Total memory increase: {memory_increase / 1024 / 1024:.1f} MB")
                # 内存增长应该在合理范围内
                self.assertLess(memory_increase / 1024 / 1024, 500, "Memory increase too large")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Memory management test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_configuration_compatibility(self) -> bool:
        """测试配置兼容性"""
        print("\n[INTEGRATION TEST] Configuration Compatibility")
        
        try:
            # 测试不同配置组合
            test_configs = [
                {'use_double_critic': True, 'enable_two_stage': True},
                {'use_double_critic': True, 'enable_two_stage': False},
                {'use_double_critic': False, 'enable_two_stage': False},  # 向后兼容
            ]
            
            for i, config in enumerate(test_configs):
                print(f"  Testing config {i+1}: {config}")
                
                # 创建网络
                if config['use_double_critic']:
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
                    
                    ppo_class = PPODoubleReward
                else:
                    # 测试向后兼容性
                    actor_critic = ActorCriticRMADoubleReward(
                        num_prop=48,
                        num_scan=187,
                        num_critic_obs=235,
                        num_priv_latent=4,
                        num_priv_explicit=8,
                        num_hist=15,
                        num_actions=12,
                        use_double_critic=False,
                        priv_encoder_dims=[64, 20]
                    ).to(self.device)
                    
                    ppo_class = PPO
                
                # 验证网络创建
                self.assertIsNotNone(actor_critic)
                print(f"    ✓ Network created for config {i+1}")
                
                # 简单前向传播测试
                obs = torch.randn(4, 235, device=self.device)
                if config['use_double_critic']:
                    value1, value2 = actor_critic.evaluate(obs)
                    self.assertEqual(value1.shape, (4, 1))
                    self.assertEqual(value2.shape, (4, 1))
                else:
                    value = actor_critic.evaluate(obs)
                    self.assertEqual(value.shape, (4, 1))
                
                print(f"    ✓ Forward pass works for config {i+1}")
            
            print("  ✓ All configuration combinations work")
            return True
            
        except Exception as e:
            print(f"  ✗ Configuration compatibility test failed: {e}")
            traceback.print_exc()
            return False
    
    def _create_mock_env_config(self):
        """创建模拟环境配置"""
        class MockConfig:
            class training:
                class stage1:
                    min_steps = 100  # 减少用于测试
                    max_steps = 500
                    success_threshold = 0.8
                class stage2:
                    max_steps = 500
                dense_rewards = [
                    'tracking_lin_vel', 'tracking_ang_vel', 'orientation', 
                    'base_height', 'torques', 'action_rate'
                ]
                sparse_rewards = ['foothold']
        
        return MockConfig()
    
    def _create_mock_ppo_config(self):
        """创建模拟PPO配置"""
        return {
            'num_learning_epochs': 1,
            'num_mini_batches': 2,
            'clip_param': 0.2,
            'gamma': 0.99,
            'lam': 0.95,
            'value_loss_coef': 1.0,
            'entropy_coef': 0.01,
            'learning_rate': 3e-4,
            'dense_reward_weight': 1.0,
            'sparse_reward_weight': 0.25,
            'use_separate_value_loss': True
        }
    
    def _create_mock_environment(self):
        """创建模拟环境"""
        class MockEnvironment:
            def __init__(self):
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.num_envs = 4
            
            def check_termination(self):
                return torch.tensor([False] * self.num_envs, device=self.device)
            
            def compute_reward(self):
                return {
                    'tracking_lin_vel': torch.randn(self.num_envs, device=self.device),
                    'foothold': torch.randn(self.num_envs, device=self.device) * -0.1
                }
        
        return MockEnvironment()
    
    def _run_mini_training_loop(self, ppo, stage_manager, num_envs):
        """运行迷你训练循环"""
        try:
            # 模拟训练数据收集
            for step in range(self.mini_steps):
                # 创建观测数据
                obs = torch.randn(num_envs, 235, device=self.device)
                critic_obs = torch.randn(num_envs, 235, device=self.device)
                
                # 执行动作
                actions = ppo.act(obs, critic_obs, {})
                
                # 模拟奖励
                rewards_dict = {
                    'tracking_lin_vel': torch.randn(num_envs, device=self.device),
                    'orientation': torch.randn(num_envs, device=self.device) * 0.5,
                    'foothold': torch.randn(num_envs, device=self.device) * -0.1,
                }
                
                # 分离奖励
                dense_rewards, sparse_rewards = stage_manager.separate_rewards(rewards_dict)
                
                # 模拟环境步骤
                done = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                ppo.process_env_step(list(dense_rewards.values())[0], done, [{}] * num_envs)
                
                # 存储转换
                if hasattr(ppo.storage, 'add_transitions'):
                    ppo.storage.add_transitions()
            
            # 计算returns
            last_critic_obs = torch.randn(num_envs, 235, device=self.device)
            ppo.compute_returns(last_critic_obs)
            
            # 执行一次更新
            if hasattr(ppo, 'update'):
                mean_value_loss, mean_surrogate_loss = ppo.update()
                print(f"    ✓ Update completed - Value Loss: {mean_value_loss:.4f}, Policy Loss: {mean_surrogate_loss:.4f}")
            
            return True
            
        except Exception as e:
            print(f"    ✗ Mini training loop failed: {e}")
            return False
    
    def assertEqual(self, a, b, msg=""):
        """简单断言"""
        if a != b:
            raise AssertionError(f"{msg}: {a} != {b}")
    
    def assertTrue(self, condition, msg=""):
        """简单断言"""
        if not condition:
            raise AssertionError(f"{msg}: condition is False")
    
    def assertFalse(self, condition, msg=""):
        """简单断言"""
        if condition:
            raise AssertionError(f"{msg}: condition is True")
    
    def assertIn(self, item, container, msg=""):
        """简单断言"""
        if item not in container:
            raise AssertionError(f"{msg}: {item} not in {container}")
    
    def assertLess(self, a, b, msg=""):
        """简单断言"""
        if a >= b:
            raise AssertionError(f"{msg}: {a} >= {b}")
    
    def assertIsNotNone(self, obj, msg=""):
        """简单断言"""
        if obj is None:
            raise AssertionError(f"{msg}: object is None")


def run_integration_tests():
    """运行所有集成测试"""
    print("="*60)
    print("BEAMDOJO 集成测试套件")
    print("="*60)
    
    tester = BeamDojoIntegrationTester()
    
    tests = [
        ('Full Training Pipeline', tester.test_full_training_pipeline),
        ('Stage Transition Integration', tester.test_stage_transition_integration),
        ('Reward Computation Integration', tester.test_reward_computation_integration),
        ('Memory Management Integration', tester.test_memory_management_integration),
        ('Configuration Compatibility', tester.test_configuration_compatibility),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            if success:
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} ERROR: {e}")
            traceback.print_exc()
    
    # 生成报告
    print(f"\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Tests Passed: {passed}")
    print(f"Tests Failed: {failed}")
    print(f"Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    # 保存报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'tests_passed': passed,
        'tests_failed': failed,
        'success_rate': passed / (passed + failed) * 100,
        'device': tester.device
    }
    
    report_path = f"/home/cft/zikang/Humanoid-Terrain-Bench/integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {report_path}")
    
    return passed > 0 and failed == 0


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)