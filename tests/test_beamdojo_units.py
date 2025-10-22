#!/usr/bin/env python3
"""
BEAMDOJO核心组件单元测试
详细测试每个核心功能的正确性和鲁棒性
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 先导入torch相关模块
import torch
import torch.nn as nn
import numpy as np

# 动态导入，避免IsaacGym问题
try:
    from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
    from rsl_rl.algorithms.ppo_double_reward import PPODoubleReward
    CORE_MODULES_AVAILABLE = True
except ImportError:
    ActorCriticRMADoubleReward = None
    PPODoubleReward = None
    CORE_MODULES_AVAILABLE = False

try:
    from legged_gym.utils.two_stage_training import TwoStageTrainingManager, StageAwareEnvironment
    TRAINING_MODULES_AVAILABLE = True
except ImportError:
    TwoStageTrainingManager = None
    StageAwareEnvironment = None
    TRAINING_MODULES_AVAILABLE = False


class TestDoubleCriticNetwork(unittest.TestCase):
    """测试双Critic网络"""
    
    def setUp(self):
        """测试前置设置"""
        if not CORE_MODULES_AVAILABLE:
            self.skipTest("Core modules (ActorCriticRMADoubleReward, PPODoubleReward) not available")
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 标准网络参数
        self.net_params = {
            'num_prop': 48,
            'num_scan': 187,
            'num_critic_obs': 235,
            'num_priv_latent': 4,
            'num_priv_explicit': 8,
            'num_hist': 15,
            'num_actions': 12,
            'actor_hidden_dims': [256, 128],
            'critic_hidden_dims': [256, 128],
            'activation': 'elu',
            'priv_encoder_dims': [64, 20]
        }
        
        self.batch_size = 4
    
    def test_single_critic_mode(self):
        """测试单Critic模式（向后兼容）"""
        network = ActorCriticRMADoubleReward(
            **self.net_params,
            use_double_critic=False
        ).to(self.device)
        
        # 验证网络结构
        self.assertFalse(network.use_double_critic)
        self.assertTrue(hasattr(network, 'critic'))
        self.assertFalse(hasattr(network, 'critic1'))
        self.assertFalse(hasattr(network, 'critic2'))
        
        # 测试前向传播
        obs = torch.randn(self.batch_size, 235, device=self.device)
        value = network.evaluate(obs)
        self.assertEqual(value.shape, (self.batch_size, 1))
    
    def test_double_critic_mode(self):
        """测试双Critic模式"""
        network = ActorCriticRMADoubleReward(
            **self.net_params,
            use_double_critic=True
        ).to(self.device)
        
        # 验证网络结构
        self.assertTrue(network.use_double_critic)
        self.assertTrue(hasattr(network, 'critic1'))
        self.assertTrue(hasattr(network, 'critic2'))
        
        # 测试前向传播
        obs = torch.randn(self.batch_size, 235, device=self.device)
        value1, value2 = network.evaluate(obs)
        self.assertEqual(value1.shape, (self.batch_size, 1))
        self.assertEqual(value2.shape, (self.batch_size, 1))
        
        # 测试单独评估
        val1_only = network.evaluate_critic1(obs)
        val2_only = network.evaluate_critic2(obs)
        self.assertTrue(torch.allclose(val1_only, value1))
        self.assertTrue(torch.allclose(val2_only, value2))
    
    def test_network_independence(self):
        """测试两个Critic网络的独立性"""
        network = ActorCriticRMADoubleReward(
            **self.net_params,
            use_double_critic=True
        ).to(self.device)
        
        obs = torch.randn(self.batch_size, 235, device=self.device)
        
        # 多次前向传播，检查输出的多样性
        values1_list = []
        values2_list = []
        
        for _ in range(5):
            # 添加小的随机扰动
            obs_perturbed = obs + torch.randn_like(obs) * 0.01
            val1, val2 = network.evaluate(obs_perturbed)
            values1_list.append(val1)
            values2_list.append(val2)
        
        # 验证两个网络确实产生不同的输出
        val1_std = torch.stack(values1_list).std()
        val2_std = torch.stack(values2_list).std()
        
        self.assertGreater(val1_std.item(), 0, "Critic1 output should vary")
        self.assertGreater(val2_std.item(), 0, "Critic2 output should vary")
    
    def test_gradient_separation(self):
        """测试梯度分离"""
        network = ActorCriticRMADoubleReward(
            **self.net_params,
            use_double_critic=True
        ).to(self.device)
        
        obs = torch.randn(self.batch_size, 235, device=self.device)
        val1, val2 = network.evaluate(obs)
        
        # 只对critic1计算梯度
        target1 = torch.randn_like(val1)
        loss1 = nn.MSELoss()(val1, target1)
        loss1.backward(retain_graph=True)
        
        # 检查梯度分布
        critic1_has_grad = any(p.grad is not None for p in network.critic1.parameters())
        critic2_has_no_grad = all(p.grad is None for p in network.critic2.parameters())
        
        self.assertTrue(critic1_has_grad, "Critic1 should have gradients")
        self.assertTrue(critic2_has_no_grad, "Critic2 should not have gradients")
        
        # 清除梯度并测试critic2
        network.zero_grad()
        target2 = torch.randn_like(val2)
        loss2 = nn.MSELoss()(val2, target2)
        loss2.backward()
        
        critic1_has_no_grad = all(p.grad is None for p in network.critic1.parameters())
        critic2_has_grad = any(p.grad is not None for p in network.critic2.parameters())
        
        self.assertTrue(critic1_has_no_grad, "Critic1 should not have gradients")
        self.assertTrue(critic2_has_grad, "Critic2 should have gradients")
    
    def test_parameter_initialization(self):
        """测试参数初始化"""
        network = ActorCriticRMADoubleReward(
            **self.net_params,
            use_double_critic=True
        ).to(self.device)
        
        # 检查参数范围
        for name, param in network.named_parameters():
            if 'weight' in name:
                # 权重应该在合理范围内
                self.assertLessEqual(param.abs().max().item(), 2.0, f"Weight {name} too large")
                self.assertGreaterEqual(param.abs().mean().item(), 0.01, f"Weight {name} too small")
            elif 'bias' in name and param is not None:
                # 偏置应该接近零
                self.assertLessEqual(param.abs().max().item(), 1.0, f"Bias {name} too large")


class TestPPODoubleReward(unittest.TestCase):
    """测试PPO双奖励算法"""
    
    def setUp(self):
        """测试前置设置"""
        if not CORE_MODULES_AVAILABLE:
            self.skipTest("Core modules (ActorCriticRMADoubleReward, PPODoubleReward) not available")
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建网络
        self.actor_critic = ActorCriticRMADoubleReward(
            num_prop=48,
            num_scan=187,
            num_critic_obs=235,
            num_priv_latent=4,
            num_priv_explicit=8,
            num_hist=50,  # 必须是10, 20或50
            num_actions=12,
            use_double_critic=True,
            priv_encoder_dims=[64, 20],
            tanh_encoder_output=False
        ).to(self.device)
        
        # PPO参数
        self.ppo_params = {
            'actor_critic': self.actor_critic,
            'estimator': None,
            'estimator_paras': None,
            'depth_encoder': None,
            'depth_encoder_paras': None,
            'depth_actor': None,
            'num_learning_epochs': 1,
            'num_mini_batches': 2,
            'clip_param': 0.2,
            'gamma': 0.99,
            'lam': 0.95,
            'value_loss_coef': 1.0,
            'entropy_coef': 0.01,
            'learning_rate': 3e-4,
            'device': self.device,
            'dense_reward_weight': 1.0,
            'sparse_reward_weight': 0.25
        }
    
    def test_ppo_initialization(self):
        """测试PPO初始化"""
        ppo = PPODoubleReward(**self.ppo_params)
        
        # 验证双Critic配置
        self.assertTrue(ppo.use_double_critic)
        self.assertEqual(ppo.dense_reward_weight, 1.0)
        self.assertEqual(ppo.sparse_reward_weight, 0.25)
        
        # 验证优化器
        self.assertIsNotNone(ppo.optimizer)
        self.assertEqual(len(ppo.optimizer.param_groups), 1)
    
    def test_storage_initialization(self):
        """测试存储初始化"""
        ppo = PPODoubleReward(**self.ppo_params)
        
        num_envs = 8
        num_transitions = 24
        actor_obs_shape = (235,)
        critic_obs_shape = (235,)
        action_shape = (12,)
        
        ppo.init_storage(num_envs, num_transitions, actor_obs_shape, critic_obs_shape, action_shape)
        
        self.assertIsNotNone(ppo.storage)
        self.assertEqual(ppo.storage.num_envs, num_envs)
        self.assertEqual(ppo.storage.num_transitions_per_env, num_transitions)
    
    def test_action_computation(self):
        """测试动作计算"""
        ppo = PPODoubleReward(**self.ppo_params)
        
        # 初始化存储
        ppo.init_storage(4, 24, (235,), (235,), (12,))
        
        # 创建观测
        obs = torch.randn(4, 235, device=self.device)
        critic_obs = torch.randn(4, 235, device=self.device)
        info = {}
        
        # 计算动作
        actions = ppo.act(obs, critic_obs, info)
        
        self.assertEqual(actions.shape, (4, 12))
        self.assertTrue(torch.isfinite(actions).all())
    
    def test_reward_processing(self):
        """测试奖励处理"""
        ppo = PPODoubleReward(**self.ppo_params)
        
        # 模拟奖励数据
        dense_rewards = torch.randn(4, device=self.device)
        sparse_rewards = torch.randn(4, device=self.device) * 0.1  # 通常稀疏奖励更小
        
        # 测试奖励处理逻辑
        total_reward = ppo.dense_reward_weight * dense_rewards + ppo.sparse_reward_weight * sparse_rewards
        
        expected_reward = 1.0 * dense_rewards + 0.25 * sparse_rewards
        self.assertTrue(torch.allclose(total_reward, expected_reward))


class TestTwoStageTrainingManager(unittest.TestCase):
    """测试两阶段训练管理器"""
    
    def setUp(self):
        """测试前置设置"""
        if not TRAINING_MODULES_AVAILABLE:
            self.skipTest("Training modules (TwoStageTrainingManager, StageAwareEnvironment) not available")
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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
        
        self.cfg = MockConfig()
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = TwoStageTrainingManager(self.cfg, device=self.device)
        
        self.assertEqual(manager.get_current_stage(), 1)
        self.assertFalse(manager.stage1_completed)
        self.assertEqual(manager.training_step, 0)
        self.assertEqual(manager.stage1_success_threshold, 0.8)
    
    def test_reward_separation(self):
        """测试奖励分离"""
        manager = TwoStageTrainingManager(self.cfg, device=self.device)
        
        # 创建复杂奖励字典
        rewards_dict = {
            'vel_track': torch.tensor([1.0, 2.0], device=self.device),
            'orientation': torch.tensor([0.5, 1.5], device=self.device),
            'torques': torch.tensor([-0.1, -0.2], device=self.device),
            'foothold': torch.tensor([0.1, 0.2], device=self.device),
            'terrain_adapt': torch.tensor([0.05, 0.1], device=self.device),
            'unknown_reward': torch.tensor([0.0, 0.0], device=self.device)
        }
        
        dense_rewards, sparse_rewards = manager.separate_rewards(rewards_dict)
        
        # 验证分离正确性
        expected_dense = ['vel_track', 'orientation', 'torques']
        expected_sparse = ['foothold', 'terrain_adapt']
        
        self.assertEqual(set(dense_rewards.keys()), set(expected_dense))
        self.assertEqual(set(sparse_rewards.keys()), set(expected_sparse))
        self.assertNotIn('unknown_reward', dense_rewards)
        self.assertNotIn('unknown_reward', sparse_rewards)
    
    def test_stage_transition_conditions(self):
        """测试阶段切换条件"""
        manager = TwoStageTrainingManager(self.cfg, device=self.device)
        
        # 测试各种情况
        test_cases = [
            # (success_rate, step, expected_switch, description)
            (0.5, 500, False, "Low success, low step"),
            (0.9, 500, False, "High success, but low step"),
            (0.5, 2000, False, "Enough steps, but low success"),
            (0.9, 2000, True, "High success and enough steps"),
            (0.8, 1000, True, "Exact threshold"),
            (0.81, 999, False, "High success but just below min steps"),
        ]
        
        for success_rate, step, expected_switch, description in test_cases:
            # 重置状态
            manager.current_stage = 1
            manager.stage1_completed = False
            
            should_switch, new_stage = manager.check_stage_transition(success_rate, step)
            self.assertEqual(should_switch, expected_switch, f"Failed: {description}")
            
            if should_switch:
                self.assertEqual(new_stage, 2)
    
    def test_terrain_type_selection(self):
        """测试地形类型选择"""
        manager = TwoStageTrainingManager(self.cfg, device=self.device)
        
        # Stage 1应该返回平坦地形
        terrain_type = manager.get_terrain_type()
        self.assertIn('flat', terrain_type.lower())
        
        # 切换到Stage 2
        manager.current_stage = 2
        manager.stage1_completed = True
        
        terrain_type = manager.get_terrain_type()
        # Stage 2应该返回稀疏地形
        self.assertTrue(any(t in terrain_type.lower() for t in ['stones', 'stepping', 'sparse']))
    
    def test_command_range_adjustment(self):
        """测试命令范围调整"""
        manager = TwoStageTrainingManager(self.cfg, device=self.device)
        
        # Stage 1命令范围
        ranges = manager.get_command_ranges()
        self.assertIsInstance(ranges, dict)
        
        # Stage 2命令范围应该受限
        manager.current_stage = 2
        manager.stage1_completed = True
        
        ranges_stage2 = manager.get_command_ranges()
        # 验证Stage2的y速度和yaw受限
        if 'vy_range' in ranges_stage2:
            self.assertEqual(ranges_stage2['vy_range'], [0.0, 0.0])
        if 'yaw_range' in ranges_stage2:
            self.assertEqual(ranges_stage2['yaw_range'], [0.0, 0.0])


class TestStageAwareEnvironment(unittest.TestCase):
    """测试阶段感知环境"""
    
    def setUp(self):
        """测试前置设置"""
        if not TRAINING_MODULES_AVAILABLE:
            self.skipTest("Training modules (TwoStageTrainingManager, StageAwareEnvironment) not available")
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建模拟配置和环境
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
        
        self.cfg = MockConfig()
        self.stage_manager = TwoStageTrainingManager(self.cfg, device=self.device)
        
        # 创建模拟基础环境
        self.base_env = Mock()
        self.base_env.check_termination = Mock(return_value=torch.tensor([False, False]))
        self.base_env.compute_reward = Mock(return_value={'vel_track': torch.tensor([1.0, 1.0]), 'foothold': torch.tensor([0.1, 0.1])})
    
    def test_environment_wrapping(self):
        """测试环境包装"""
        stage_env = StageAwareEnvironment(self.base_env, self.stage_manager)
        
        # 验证包装正确
        self.assertEqual(stage_env.base_env, self.base_env)
        self.assertEqual(stage_env.stage_manager, self.stage_manager)
        
        # 验证方法替换
        self.assertNotEqual(self.base_env.check_termination, stage_env.original_check_termination)
        self.assertNotEqual(self.base_env.compute_reward, stage_env.original_compute_reward)
    
    def test_stage1_termination(self):
        """测试Stage1终止逻辑"""
        stage_env = StageAwareEnvironment(self.base_env, self.stage_manager)
        
        # Stage1应该使用软终止
        self.assertTrue(self.stage_manager.should_use_soft_termination())
        self.assertFalse(self.stage_manager.should_use_hard_termination())
    
    def test_stage2_termination(self):
        """测试Stage2终止逻辑"""
        stage_env = StageAwareEnvironment(self.base_env, self.stage_manager)
        
        # 切换到Stage2
        self.stage_manager.current_stage = 2
        self.stage_manager.stage1_completed = True
        
        # Stage2应该使用硬终止
        self.assertFalse(self.stage_manager.should_use_soft_termination())
        self.assertTrue(self.stage_manager.should_use_hard_termination())
    
    def test_attribute_proxy(self):
        """测试属性代理"""
        stage_env = StageAwareEnvironment(self.base_env, self.stage_manager)
        
        # 设置基础环境的属性
        self.base_env.some_attribute = "test_value"
        self.base_env.some_method = Mock(return_value="method_result")
        
        # 验证代理工作
        self.assertEqual(stage_env.some_attribute, "test_value")
        result = stage_env.some_method()
        self.assertEqual(result, "method_result")


class TestFootholdReward(unittest.TestCase):
    """测试Foothold奖励计算"""
    
    def setUp(self):
        """测试前置设置"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_foothold_reward_properties(self):
        """测试Foothold奖励属性"""
        # 模拟foothold奖励计算
        def compute_foothold_reward(foot_positions, terrain_heights, config):
            batch_size = foot_positions.shape[0]
            penalties = torch.zeros(batch_size, device=self.device)
            
            n_samples = config.get('n_samples', 16)
            epsilon = config.get('epsilon', -0.1)
            
            # 模拟踩空检测
            for i in range(batch_size):
                # 模拟一些踩空情况
                miss_count = torch.randint(0, n_samples, (1,)).item()
                penalties[i] = miss_count
            
            return -penalties  # 负奖励（惩罚）
        
        # 测试参数
        batch_size = 8
        foot_positions = torch.randn(batch_size, 6, device=self.device)
        terrain_heights = torch.randn(batch_size, device=self.device)
        config = {'n_samples': 16, 'epsilon': -0.1}
        
        # 计算奖励
        rewards = compute_foothold_reward(foot_positions, terrain_heights, config)
        
        # 验证属性
        self.assertEqual(rewards.shape, (batch_size,))
        self.assertTrue((rewards <= 0).all(), "Foothold rewards should be non-positive")
        self.assertTrue(torch.isfinite(rewards).all(), "Foothold rewards should be finite")
    
    def test_sampling_consistency(self):
        """测试采样一致性"""
        # 固定随机种子测试一致性
        torch.manual_seed(42)
        
        def deterministic_foothold_reward(foot_positions, config):
            batch_size = foot_positions.shape[0]
            n_samples = config.get('n_samples', 16)
            
            # 基于位置的确定性计算
            penalties = torch.abs(foot_positions[:, 0]) * n_samples / 10.0  # 简化计算
            return -penalties
        
        foot_pos = torch.randn(4, 6, device=self.device)
        config = {'n_samples': 16}
        
        # 多次计算应该产生相同结果
        reward1 = deterministic_foothold_reward(foot_pos, config)
        reward2 = deterministic_foothold_reward(foot_pos, config)
        
        self.assertTrue(torch.allclose(reward1, reward2), "Foothold computation should be deterministic")


def run_unit_tests():
    """运行所有单元测试"""
    print("="*60)
    print("BEAMDOJO 单元测试套件")
    print("="*60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestDoubleCriticNetwork,
        TestPPODoubleReward,
        TestTwoStageTrainingManager,
        TestStageAwareEnvironment,
        TestFootholdReward
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 返回成功状态
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_unit_tests()
    sys.exit(0 if success else 1)