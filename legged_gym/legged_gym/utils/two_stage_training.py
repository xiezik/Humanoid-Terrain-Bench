"""
BEAMDOJO两阶段训练系统
实现Stage1软约束训练和Stage2硬约束训练
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional


class TwoStageTrainingManager:
    """
    BEAMDOJO两阶段训练管理器
    """
    
    def __init__(self, 
                 cfg,
                 device='cuda'):
        """
        Args:
            cfg: 训练配置
            device: 计算设备
        """
        self.cfg = cfg
        self.device = device
        
        # 训练阶段状态
        self.current_stage = 1  # 1: Stage1, 2: Stage2
        self.stage1_completed = False
        self.training_step = 0
        
        # 阶段切换条件
        self.stage1_success_threshold = getattr(cfg.training.stage1, 'success_threshold', 0.8)
        self.stage1_min_steps = getattr(cfg.training.stage1, 'min_steps', 1000000)
        self.stage1_max_steps = getattr(cfg.training.stage1, 'max_steps', 5000000)
        
        # 奖励组配置
        self.dense_rewards = getattr(cfg.training, 'dense_rewards', [
            'tracking_lin_vel', 'tracking_ang_vel', 'orientation', 'base_height',
            'lin_vel_z', 'ang_vel_xy', 'torques', 'action_rate', 'smoothness',
            'dof_vel', 'dof_acc', 'feet_air_time', 'feet_ground_parallel',
            'feet_distance', 'feet_clearance'
        ])
        
        self.sparse_rewards = getattr(cfg.training, 'sparse_rewards', [
            'foothold'
        ])
        
        print(f"TwoStageTrainingManager initialized:")
        print(f"  Dense rewards: {self.dense_rewards}")
        print(f"  Sparse rewards: {self.sparse_rewards}")
        print(f"  Stage1 success threshold: {self.stage1_success_threshold}")
        print(f"  Stage1 step range: {self.stage1_min_steps} - {self.stage1_max_steps}")
    
    def get_current_stage(self):
        """获取当前训练阶段"""
        return self.current_stage
    
    def should_use_soft_termination(self):
        """是否使用软终止（Stage1）"""
        return self.current_stage == 1
    
    def should_use_hard_termination(self):
        """是否使用硬终止（Stage2）"""
        return self.current_stage == 2
    
    def get_terrain_type(self):
        """获取当前阶段应使用的地形类型"""
        if self.current_stage == 1:
            # Stage1: 平坦地形 + 目标地形感知
            return "flat_with_target_perception"
        else:
            # Stage2: 真实稀疏地形
            return "sparse_terrain"
    
    def get_command_ranges(self):
        """获取当前阶段的命令范围"""
        if self.current_stage == 1:
            # Stage1: 全方向命令
            return {
                'lin_vel_x': [-1.0, 1.0],
                'lin_vel_y': [-1.0, 1.0],
                'ang_vel_yaw': [-1.0, 1.0]
            }
        else:
            # Stage2: 仅前进命令
            return {
                'lin_vel_x': [-1.0, 1.0],
                'lin_vel_y': [0.0, 0.0],
                'ang_vel_yaw': [0.0, 0.0]
            }
    
    def separate_rewards(self, rewards_dict: Dict[str, torch.Tensor]):
        """
        分离密集和稀疏奖励
        
        Args:
            rewards_dict: 包含所有奖励的字典
            
        Returns:
            dense_rewards: 密集奖励字典
            sparse_rewards: 稀疏奖励字典
        """
        dense_rewards = {}
        sparse_rewards = {}
        
        for name, reward in rewards_dict.items():
            if name in self.dense_rewards:
                dense_rewards[name] = reward
            elif name in self.sparse_rewards:
                sparse_rewards[name] = reward
            else:
                # 默认分类到密集奖励
                dense_rewards[name] = reward
        
        return dense_rewards, sparse_rewards
    
    def compute_separated_rewards(self, rewards_dict: Dict[str, torch.Tensor], reward_scales: Dict[str, float]):
        """
        计算分离的奖励总和
        
        Args:
            rewards_dict: 奖励字典
            reward_scales: 奖励权重字典
            
        Returns:
            dense_total: 密集奖励总和
            sparse_total: 稀疏奖励总和
        """
        dense_rewards, sparse_rewards = self.separate_rewards(rewards_dict)
        
        # 计算密集奖励总和
        dense_total = torch.zeros(rewards_dict[list(rewards_dict.keys())[0]].shape[0], device=self.device)
        for name, reward in dense_rewards.items():
            if name in reward_scales:
                dense_total += reward * reward_scales[name]
        
        # 计算稀疏奖励总和
        sparse_total = torch.zeros(rewards_dict[list(rewards_dict.keys())[0]].shape[0], device=self.device)
        for name, reward in sparse_rewards.items():
            if name in reward_scales:
                sparse_total += reward * reward_scales[name]
        
        return dense_total, sparse_total
    
    def update_training_step(self, step: int):
        """更新训练步数"""
        self.training_step = step
    
    def check_stage_transition(self, success_rate: float, training_step: int):
        """
        检查是否应该切换训练阶段
        
        Args:
            success_rate: 当前成功率
            training_step: 当前训练步数
            
        Returns:
            should_switch: 是否应该切换阶段
            new_stage: 新的训练阶段
        """
        if self.current_stage == 1 and not self.stage1_completed:
            # 检查Stage1是否完成
            min_steps_met = training_step >= self.stage1_min_steps
            success_met = success_rate >= self.stage1_success_threshold
            max_steps_reached = training_step >= self.stage1_max_steps
            
            if (min_steps_met and success_met) or max_steps_reached:
                print(f"Stage1 completed at step {training_step}")
                print(f"  Success rate: {success_rate:.3f} (threshold: {self.stage1_success_threshold})")
                print(f"  Min steps met: {min_steps_met}")
                print(f"  Max steps reached: {max_steps_reached}")
                
                self.stage1_completed = True
                self.current_stage = 2
                return True, 2
        
        return False, self.current_stage
    
    def get_stage_config(self):
        """获取当前阶段的配置"""
        config = {
            'stage': self.current_stage,
            'terrain_type': self.get_terrain_type(),
            'command_ranges': self.get_command_ranges(),
            'use_soft_termination': self.should_use_soft_termination(),
            'use_target_perception': self.current_stage == 1,
            'dense_rewards': self.dense_rewards,
            'sparse_rewards': self.sparse_rewards
        }
        return config
    
    def save_stage_checkpoint(self, model_state, optimizer_state, training_info):
        """保存阶段checkpoint"""
        checkpoint = {
            'stage': self.current_stage,
            'training_step': self.training_step,
            'stage1_completed': self.stage1_completed,
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'training_info': training_info
        }
        return checkpoint
    
    def load_stage_checkpoint(self, checkpoint):
        """加载阶段checkpoint"""
        self.current_stage = checkpoint.get('stage', 1)
        self.training_step = checkpoint.get('training_step', 0)
        self.stage1_completed = checkpoint.get('stage1_completed', False)
        
        return checkpoint['model_state'], checkpoint['optimizer_state'], checkpoint['training_info']


class StageAwareEnvironment:
    """
    阶段感知的环境包装器
    根据训练阶段调整环境行为
    """
    
    def __init__(self, base_env, stage_manager: TwoStageTrainingManager):
        """
        Args:
            base_env: 基础环境
            stage_manager: 阶段管理器
        """
        self.base_env = base_env
        self.stage_manager = stage_manager
        
        # 保存原始方法
        self.original_check_termination = base_env.check_termination
        self.original_compute_reward = base_env.compute_reward
        
        # 替换环境方法
        base_env.check_termination = self.stage_aware_check_termination
        base_env.compute_reward = self.stage_aware_compute_reward
    
    def stage_aware_check_termination(self):
        """阶段感知的终止检查"""
        if self.stage_manager.should_use_soft_termination():
            # Stage1: 软终止 - 踩空不终止episode
            self._check_termination_stage1()
        else:
            # Stage2: 硬终止 - 踩空立即终止
            self.original_check_termination()
    
    def _check_termination_stage1(self):
        """Stage1的终止检查 - 只因姿态和高度终止，不因踩空终止"""
        self.base_env.reset_buf = torch.zeros((self.base_env.num_envs,), dtype=torch.bool, device=self.base_env.device)
        
        # 姿态检查（保持原有逻辑）
        roll_cutoff = torch.abs(self.base_env.roll) > 1.5
        pitch_cutoff = torch.abs(self.base_env.pitch) > 1.5
        
        # 高度检查（保持原有逻辑）
        height_cutoff = self.base_env.root_states[:, 2] < 0.5
        
        # 目标到达检查（保持原有逻辑）
        reach_goal_cutoff = self.base_env.cur_goal_idx >= self.base_env.cfg.terrain.num_goals
        
        # 超时检查（保持原有逻辑）
        self.base_env.time_out_buf = self.base_env.episode_length_buf > self.base_env.max_episode_length
        self.base_env.time_out_buf |= reach_goal_cutoff
        
        # 组合终止条件（不包括踩空）
        self.base_env.reset_buf |= self.base_env.time_out_buf
        self.base_env.reset_buf |= roll_cutoff
        self.base_env.reset_buf |= pitch_cutoff
        self.base_env.reset_buf |= height_cutoff
        
        # 更新统计信息
        self.base_env.total_times += len(self.base_env.reset_buf.nonzero(as_tuple=False).flatten())
        self.base_env.success_times += len(reach_goal_cutoff.nonzero(as_tuple=False).flatten())
        if hasattr(self.base_env, 'complete_times'):
            reset_envs = self.base_env.reset_buf.nonzero(as_tuple=False).flatten()
            if len(reset_envs) > 0:
                self.base_env.complete_times += (self.base_env.cur_goal_idx[reset_envs] / self.base_env.cfg.terrain.num_goals).sum()
    
    def stage_aware_compute_reward(self):
        """阶段感知的奖励计算"""
        # 调用原始奖励计算
        self.original_compute_reward()
        
        # 如果使用双Critic，需要分离奖励
        if hasattr(self.base_env, 'reward_functions') and hasattr(self.stage_manager, 'separate_rewards'):
            # 收集所有奖励
            rewards_dict = {}
            for i, name in enumerate(self.base_env.reward_names):
                if name in self.base_env.reward_scales:
                    reward = self.base_env.reward_functions[i]()
                    rewards_dict[name] = reward
            
            # 分离密集和稀疏奖励
            dense_total, sparse_total = self.stage_manager.compute_separated_rewards(
                rewards_dict, self.base_env.reward_scales
            )
            
            # 存储分离的奖励供PPO使用
            self.base_env.dense_rewards = dense_total
            self.base_env.sparse_rewards = sparse_total
    
    def __getattr__(self, name):
        """代理其他属性和方法到基础环境"""
        return getattr(self.base_env, name)