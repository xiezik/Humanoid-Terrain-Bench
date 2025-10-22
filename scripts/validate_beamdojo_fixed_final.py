#!/usr/bin/env python3
"""
BEAMDOJO简化验证脚本
专门测试核心网络功能，避免IsaacGym依赖问题
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_import_structure():
    """测试导入结构"""
    print("测试 1: 验证模块导入结构")
    
    try:
        # 测试基础导入
        from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
        print("  ✓ ActorCriticRMADoubleReward 导入成功")
        
        from rsl_rl.algorithms.ppo_double_reward import PPODoubleReward  
        print("  ✓ PPODoubleReward 导入成功")
        
        return True
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        return False

def test_network_creation():
    """测试网络创建"""
    print("\n测试 2: 验证网络创建")
    
    try:
        import torch
        from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
        
        # 创建双Critic网络
        actor_critic = ActorCriticRMADoubleReward(
            num_prop=48,
            num_scan=187,
            num_critic_obs=235,
            num_priv_latent=4,
            num_priv_explicit=8,
            num_hist=50,  # 必须是10, 20或50
            num_actions=12,
            use_double_critic=True,
            priv_encoder_dims=[64, 20],
            tanh_encoder_output=False  # 添加必需参数
        )
        
        print("  ✓ 双Critic网络创建成功")
        print(f"  ✓ use_double_critic: {actor_critic.use_double_critic}")
        print(f"  ✓ has critic1: {hasattr(actor_critic, 'critic1')}")
        print(f"  ✓ has critic2: {hasattr(actor_critic, 'critic2')}")
        
        return True
    except Exception as e:
        print(f"  ✗ 网络创建失败: {e}")
        return False

def test_forward_pass():
    """测试前向传播"""
    print("\n测试 3: 验证前向传播")
    
    try:
        import torch
        from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建网络
        actor_critic = ActorCriticRMADoubleReward(
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
        ).to(device)
        
        # 创建测试数据
        batch_size = 4
        obs = torch.randn(batch_size, 235, device=device)
        
        # 测试双Critic评估
        value1, value2 = actor_critic.evaluate(obs)
        print(f"  ✓ 双Critic评估成功: value1.shape={value1.shape}, value2.shape={value2.shape}")
        
        # 测试单独评估
        val1_only = actor_critic.evaluate_critic1(obs)
        val2_only = actor_critic.evaluate_critic2(obs)
        print("  ✓ 单独Critic评估成功")
        
        # 测试动作生成
        actions = actor_critic.act(obs)
        print(f"  ✓ 动作生成成功: actions.shape={actions.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n测试 4: 验证向后兼容性")
    
    try:
        import torch
        from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建单Critic网络（向后兼容模式）
        actor_critic = ActorCriticRMADoubleReward(
            num_prop=48,
            num_scan=187,
            num_critic_obs=235,
            num_priv_latent=4,
            num_priv_explicit=8,
            num_hist=50,  # 必须是10, 20或50
            num_actions=12,
            use_double_critic=False,  # 单Critic模式
            priv_encoder_dims=[64, 20],
            tanh_encoder_output=False
        ).to(device)
        
        print(f"  ✓ 单Critic网络创建成功")
        print(f"  ✓ use_double_critic: {actor_critic.use_double_critic}")
        print(f"  ✓ has critic: {hasattr(actor_critic, 'critic')}")
        
        # 测试单Critic前向传播
        obs = torch.randn(4, 235, device=device)
        value = actor_critic.evaluate(obs)
        print(f"  ✓ 单Critic评估成功: value.shape={value.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ 向后兼容性测试失败: {e}")
        return False

def test_gradient_flow():
    """测试梯度流"""
    print("\n测试 5: 验证梯度流")
    
    try:
        import torch
        import torch.nn as nn
        from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建网络和优化器
        actor_critic = ActorCriticRMADoubleReward(
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
        ).to(device)
        
        optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
        
        # 前向传播
        obs = torch.randn(4, 235, device=device)
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
        critic1_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                              for p in actor_critic.critic1.parameters())
        critic2_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                              for p in actor_critic.critic2.parameters())
        
        print(f"  ✓ Critic1梯度存在: {critic1_has_grad}")
        print(f"  ✓ Critic2梯度存在: {critic2_has_grad}")
        print(f"  ✓ 总损失: {total_loss.item():.4f}")
        
        optimizer.step()
        print("  ✓ 参数更新成功")
        
        return critic1_has_grad and critic2_has_grad
    except Exception as e:
        print(f"  ✗ 梯度流测试失败: {e}")
        return False

def test_ppo_algorithm():
    """测试PPO算法"""
    print("\n测试 6: 验证PPO算法")
    
    try:
        import torch
        from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
        from rsl_rl.algorithms.ppo_double_reward import PPODoubleReward
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建网络
        actor_critic = ActorCriticRMADoubleReward(
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
        ).to(device)
        
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
            device=device,
            dense_reward_weight=1.0,
            sparse_reward_weight=0.25
        )
        
        print(f"  ✓ PPO算法创建成功")
        print(f"  ✓ use_double_critic: {ppo.use_double_critic}")
        print(f"  ✓ dense_reward_weight: {ppo.dense_reward_weight}")
        print(f"  ✓ sparse_reward_weight: {ppo.sparse_reward_weight}")
        
        # 初始化存储
        ppo.init_storage(4, 24, (235,), (235,), (12,))
        print("  ✓ 存储初始化成功")
        
        return True
    except Exception as e:
        print(f"  ✗ PPO算法测试失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BEAMDOJO简化验证')
    parser.add_argument('--test', choices=['all', 'import', 'network', 'forward', 'compat', 'gradient', 'ppo'], 
                        default='all', help='选择测试类型')
    
    args = parser.parse_args()
    
    print("BEAMDOJO 简化验证脚本")
    print("="*40)
    
    tests = []
    if args.test in ['all', 'import']:
        tests.append(('import', test_import_structure))
    if args.test in ['all', 'network']:
        tests.append(('network', test_network_creation))
    if args.test in ['all', 'forward']:
        tests.append(('forward', test_forward_pass))
    if args.test in ['all', 'compat']:
        tests.append(('compat', test_backward_compatibility))
    if args.test in ['all', 'gradient']:
        tests.append(('gradient', test_gradient_flow))
    if args.test in ['all', 'ppo']:
        tests.append(('ppo', test_ppo_algorithm))
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"测试 {test_name} 发生异常: {e}")
            failed += 1
    
    print(f"\n{'='*40}")
    print("测试总结")
    print("="*40)
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"成功率: {passed/(passed+failed)*100:.1f}%" if (passed+failed) > 0 else "0%")
    
    if failed == 0:
        print("\n✅ 所有核心功能验证通过！")
        print("BEAMDOJO双Critic系统已准备就绪。")
    else:
        print(f"\n❌ 发现{failed}个问题，请检查上述错误信息。")
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)