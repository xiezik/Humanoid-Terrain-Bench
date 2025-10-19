"""
BEAMDOJO Foothold Reward System
基于多边形脚采样的稀疏foothold奖励计算
"""

import torch
import torch.nn as nn
import numpy as np


class FootholdRewardCalculator:
    """
    BEAMDOJO论文中的Foothold奖励计算器
    基于多边形脚部几何形状进行采样，计算踩空惩罚
    """
    
    def __init__(self, 
                 device='cuda',
                 n_samples=16,
                 epsilon=-0.1,
                 foot_length=0.2,
                 foot_width=0.1,
                 num_feet=2):
        """
        Args:
            device: 计算设备
            n_samples: 每只脚的采样点数
            epsilon: 深度容忍阈值(m)，低于此值视为踩空
            foot_length: 脚部长度(m)
            foot_width: 脚部宽度(m)
            num_feet: 脚的数量
        """
        self.device = device
        self.n_samples = n_samples
        self.epsilon = epsilon
        self.foot_length = foot_length
        self.foot_width = foot_width
        self.num_feet = num_feet
        
        # 预计算脚部采样点的相对位置（脚坐标系）
        self._generate_foot_sample_points()
    
    def _generate_foot_sample_points(self):
        """生成脚部采样点的相对位置"""
        # 在脚部矩形区域内生成采样点
        # 使用网格采样或随机采样
        samples_per_axis = int(np.sqrt(self.n_samples))
        
        # 在脚部矩形内均匀采样
        x_coords = torch.linspace(-self.foot_length/2, self.foot_length/2, samples_per_axis, device=self.device)
        y_coords = torch.linspace(-self.foot_width/2, self.foot_width/2, samples_per_axis, device=self.device)
        
        # 创建网格
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        # 扁平化为采样点列表 [n_samples, 2]
        self.foot_sample_points = torch.stack([
            x_grid.flatten(),
            y_grid.flatten()
        ], dim=1)  # [n_samples, 2]
        
        # 如果采样点数不够，随机生成更多点
        if len(self.foot_sample_points) < self.n_samples:
            additional_samples = self.n_samples - len(self.foot_sample_points)
            random_x = torch.rand(additional_samples, device=self.device) * self.foot_length - self.foot_length/2
            random_y = torch.rand(additional_samples, device=self.device) * self.foot_width - self.foot_width/2
            additional_points = torch.stack([random_x, random_y], dim=1)
            self.foot_sample_points = torch.cat([self.foot_sample_points, additional_points], dim=0)
        
        # 如果采样点太多，截取
        self.foot_sample_points = self.foot_sample_points[:self.n_samples]
    
    def compute_foothold_reward(self, 
                              foot_positions,      # [num_envs, num_feet, 3] 脚部世界坐标
                              foot_orientations,   # [num_envs, num_feet, 4] 脚部四元数方向
                              contact_forces,      # [num_envs, num_feet, 3] 接触力
                              terrain_heights_func, # 函数：根据世界坐标(x,y)获取地形高度
                              contact_threshold=1.0):
        """
        计算Foothold奖励
        
        Args:
            foot_positions: 脚部世界坐标 [num_envs, num_feet, 3]
            foot_orientations: 脚部方向（四元数） [num_envs, num_feet, 4]
            contact_forces: 接触力 [num_envs, num_feet, 3]
            terrain_heights_func: 地形高度查询函数
            contact_threshold: 接触力阈值，超过此值认为脚部接触地面
        
        Returns:
            foothold_reward: [num_envs] 每个环境的foothold奖励
        """
        num_envs = foot_positions.shape[0]
        total_penalty = torch.zeros(num_envs, device=self.device)
        
        for env_idx in range(num_envs):
            for foot_idx in range(self.num_feet):
                # 检查脚部是否接触地面
                contact_force_magnitude = torch.norm(contact_forces[env_idx, foot_idx])
                
                if contact_force_magnitude > contact_threshold:
                    # 脚部接触地面，进行采样检查
                    foot_pos = foot_positions[env_idx, foot_idx]  # [3]
                    foot_quat = foot_orientations[env_idx, foot_idx]  # [4]
                    
                    # 将脚部采样点转换到世界坐标系
                    world_sample_points = self._transform_sample_points_to_world(
                        foot_pos, foot_quat, self.foot_sample_points
                    )  # [n_samples, 3]
                    
                    # 对每个采样点检查地形高度
                    for sample_point in world_sample_points:
                        terrain_height = terrain_heights_func(sample_point[0], sample_point[1])
                        
                        # 如果采样点高度低于地形高度 + epsilon，视为踩空
                        if sample_point[2] < terrain_height + self.epsilon:
                            total_penalty[env_idx] += 1
        
        # 转换为奖励（惩罚越多，奖励越低）
        foothold_reward = -total_penalty
        return foothold_reward
    
    def _transform_sample_points_to_world(self, foot_pos, foot_quat, sample_points):
        """
        将脚部采样点从脚坐标系转换到世界坐标系
        
        Args:
            foot_pos: 脚部世界位置 [3]
            foot_quat: 脚部四元数方向 [4] (w, x, y, z)
            sample_points: 脚坐标系下的采样点 [n_samples, 2]
        
        Returns:
            world_points: 世界坐标系下的采样点 [n_samples, 3]
        """
        # 将2D采样点扩展为3D（z=0，在脚平面上）
        sample_points_3d = torch.cat([
            sample_points,  # [n_samples, 2]
            torch.zeros(len(sample_points), 1, device=self.device)  # [n_samples, 1]
        ], dim=1)  # [n_samples, 3]
        
        # 应用四元数旋转
        rotated_points = self._quat_rotate(foot_quat, sample_points_3d)
        
        # 平移到脚部世界位置
        world_points = rotated_points + foot_pos.unsqueeze(0)  # [n_samples, 3]
        
        return world_points
    
    def _quat_rotate(self, quat, points):
        """
        使用四元数旋转点
        
        Args:
            quat: 四元数 [4] (w, x, y, z)
            points: 点 [n_points, 3]
        
        Returns:
            rotated_points: 旋转后的点 [n_points, 3]
        """
        # 标准化四元数
        quat = quat / torch.norm(quat)
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # 构建旋转矩阵
        rotation_matrix = torch.tensor([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ], device=self.device)
        
        # 应用旋转
        rotated_points = torch.matmul(points, rotation_matrix.T)
        return rotated_points


class FootholdRewardWrapper:
    """
    Foothold奖励的包装器，用于集成到现有的环境奖励系统中
    """
    
    def __init__(self, 
                 env,
                 n_samples=16,
                 epsilon=-0.1,
                 foot_length=0.2,
                 foot_width=0.1,
                 weight=1.0):
        """
        Args:
            env: 环境实例
            n_samples: 采样点数
            epsilon: 深度容忍阈值
            foot_length: 脚长
            foot_width: 脚宽
            weight: 奖励权重
        """
        self.env = env
        self.weight = weight
        
        self.calculator = FootholdRewardCalculator(
            device=env.device,
            n_samples=n_samples,
            epsilon=epsilon,
            foot_length=foot_length,
            foot_width=foot_width,
            num_feet=len(env.feet_indices)
        )
    
    def compute_reward(self):
        """
        计算当前环境状态的foothold奖励
        
        Returns:
            reward: [num_envs] foothold奖励
        """
        # 获取脚部位置和方向
        foot_positions = self.env.rigid_body_states[:, self.env.feet_indices, :3]  # [num_envs, num_feet, 3]
        foot_orientations = self.env.rigid_body_states[:, self.env.feet_indices, 3:7]  # [num_envs, num_feet, 4]
        
        # 获取接触力
        contact_forces = self.env.contact_forces[:, self.env.feet_indices, :]  # [num_envs, num_feet, 3]
        
        # 定义地形高度查询函数
        def terrain_heights_func(x, y):
            # 这里需要根据具体环境实现地形高度查询
            # 简化版本：假设地形是平的
            if hasattr(self.env, 'get_terrain_height'):
                return self.env.get_terrain_height(x, y)
            else:
                return 0.0
        
        # 计算foothold奖励
        foothold_reward = self.calculator.compute_foothold_reward(
            foot_positions=foot_positions,
            foot_orientations=foot_orientations,
            contact_forces=contact_forces,
            terrain_heights_func=terrain_heights_func
        )
        
        return self.weight * foothold_reward


def add_foothold_reward_to_env(env_class):
    """
    装饰器：为环境类添加foothold奖励功能
    
    Usage:
        @add_foothold_reward_to_env
        class MyEnv(BaseEnv):
            pass
    """
    
    original_init = env_class.__init__
    original_compute_reward = getattr(env_class, '_compute_rewards', None)
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        
        # 添加foothold奖励计算器
        foothold_config = getattr(self.cfg.rewards, 'foothold', None)
        if foothold_config is not None:
            self.foothold_reward_wrapper = FootholdRewardWrapper(
                env=self,
                n_samples=getattr(foothold_config, 'n_samples', 16),
                epsilon=getattr(foothold_config, 'epsilon', -0.1),
                foot_length=getattr(foothold_config, 'foot_length', 0.2),
                foot_width=getattr(foothold_config, 'foot_width', 0.1),
                weight=getattr(foothold_config, 'weight', 1.0)
            )
        else:
            self.foothold_reward_wrapper = None
    
    def new_reward_foothold(self):
        """Foothold奖励函数"""
        if self.foothold_reward_wrapper is not None:
            return self.foothold_reward_wrapper.compute_reward()
        else:
            return torch.zeros(self.num_envs, device=self.device)
    
    # 替换方法
    env_class.__init__ = new_init
    env_class._reward_foothold = new_reward_foothold
    
    return env_class