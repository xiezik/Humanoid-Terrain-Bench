# BEAMDOJO Humanoid Robot Configuration
# 基于BEAMDOJO论文的人形机器人配置示例
# 展示双Critic网络、两阶段训练和Foothold奖励的完整配置

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HumanoidBEAMDOJOCfg(LeggedRobotCfg):
    """
    BEAMDOJO人形机器人环境配置
    实现双Critic网络和两阶段训练的完整配置
    """
    
    class env(LeggedRobotCfg.env):
        """环境配置"""
        num_envs = 4096
        episode_length_s = 40
        
        # 观测配置（根据实际humanoid机器人调整）
        n_scan = 132
        n_priv = 3 + 3 + 3  # 位置(3) + 速度(3) + 其他(3)
        n_priv_latent = 4 + 1 + 12 + 12  # 潜在状态维度
        n_proprio = 3 + 2 + 3 + 4 + 36 + 5  # 本体感受维度
        history_len = 10
        
        # 重新计算总观测维度
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv
        num_actions = 12  # 12个关节动作
        
        # 启用接触信息
        include_foot_contacts = True
        
    class terrain:
        """地形配置 - 支持两阶段训练"""
        mesh_type = 'trimesh'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        
        # 地形类型配置
        terrain_types = ["flat", "rough", "stairs", "obstacles"]
        terrain_proportions = [0.3, 0.3, 0.2, 0.2]
        
        # 支持两阶段地形切换
        stage1_terrain_types = ["flat"]  # Stage1仅使用平坦地形
        stage2_terrain_types = ["rough", "stairs", "obstacles"]  # Stage2使用复杂地形
        
    class commands:
        """命令配置 - 支持两阶段不同的命令范围"""
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.  # 命令重新采样时间 [s]
        heading_command = True
        
        # 默认命令范围（将被两阶段训练动态调整）
        class ranges:
            lin_vel_x = [-1.0, 1.0]   # x方向线速度 [m/s]
            lin_vel_y = [-1.0, 1.0]   # y方向线速度 [m/s]
            ang_vel_yaw = [-1.0, 1.0] # yaw角速度 [rad/s]
            heading = [-3.14, 3.14]
            
    class rewards(LeggedRobotCfg.rewards):
        """BEAMDOJO奖励配置"""
        class scales:
            """奖励权重配置"""
            # 密集奖励（用于第一个Critic）
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            orientation = -2.0
            base_height = -10.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            action_rate = -0.01
            smoothness = -1e-3
            stand_still = -0.05
            dof_vel = -1e-4
            dof_acc = -2.5e-8
            dof_pos_limits = -5.0
            dof_vel_limits = -1e-3
            dof_power = -2e-5
            feet_ground_parallel = -0.02
            feet_distance = 0.5
            feet_air_time = 1.0
            feet_clearance = -1.0
            # 稀疏奖励（用于第二个Critic）
            foothold = 1.0  # BEAMDOJO Foothold奖励
            
        # BEAMDOJO双Critic奖励分离
        dense_rewards = [
            'tracking_lin_vel', 'tracking_ang_vel', 'orientation', 'base_height',
            'lin_vel_z', 'ang_vel_xy', 'torques', 'action_rate', 'dof_vel', 
            'dof_acc', 'feet_air_time', 'collision', 'feet_stumble', 'stand_still'
        ]
        sparse_rewards = [
            'foothold'
        ]
        
        # Foothold奖励配置
        class foothold:
            num_sample_points = 4      # 每个脚掌采样点数量
            sample_radius = 0.02       # 采样半径 [m]
            height_tolerance = 0.05    # 高度容忍度 [m]
            stability_weight = 1.0     # 稳定性权重
            safety_weight = 0.5        # 安全性权重
            
        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100.
        
        # 覆盖基础配置中的参数，设置为论文中的值
        min_dist = 0.18                   # d_min: Minimum allowable distance between two feet (论文值)
        max_dist = 0.25                   # 最大脚间距，用于计算目标距离
        
    class normalization:
        """归一化配置"""
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        """噪声配置"""
        add_noise = True
        noise_level = 1.0
        
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05

    class sim:
        """仿真配置"""
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1
        
        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class HumanoidBEAMDOJOCfgPPO(LeggedRobotCfgPPO):
    """
    BEAMDOJO人形机器人PPO训练配置
    包含双Critic网络和两阶段训练的完整配置
    """
    
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    
    class policy:
        """策略网络配置"""
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        
        # 扫描编码器配置
        scan_encoder_dims = [128, 64, 32]
        priv_encoder_dims = [64, 20]
        
        # 支持双Critic的编码器
        use_double_critic = False  # 在这里可以启用双Critic
        
    class algorithm:
        """BEAMDOJO PPO算法配置"""
        # 基础PPO参数
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        
        # BEAMDOJO双Critic配置
        use_double_critic = False      # 设置为True启用双Critic
        dense_value_loss_coef = 1.0    # 密集奖励价值损失系数
        sparse_value_loss_coef = 1.0   # 稀疏奖励价值损失系数
        advantage_merge_weight = 0.5   # 优势函数合并权重
        
    class runner:
        """训练运行器配置"""
        policy_class_name = 'ActorCritic'  # 使用 'ActorCriticRMADoubleReward' 启用双Critic
        algorithm_class_name = 'PPO'       # 使用 'PPODoubleReward' 启用双Critic算法
        num_steps_per_env = 24
        max_iterations = 100000
        
        save_interval = 100
        experiment_name = 'humanoid_beamdojo'
        run_name = ''
        
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    # BEAMDOJO两阶段训练配置
    class training:
        """两阶段训练配置"""
        enable_two_stage = False  # 设置为True启用两阶段训练
        
        class stage1:
            """Stage1软约束训练配置"""
            min_steps = 1000000            # 最小训练步数
            max_steps = 5000000            # 最大训练步数  
            success_threshold = 0.8        # 成功率阈值
            terrain_type = "flat_with_target_perception"
            use_soft_termination = True    # 软终止：踩空不终止episode
            use_target_perception = True   # 使用目标地形感知
            
            # Stage1命令范围（全方向）
            class command_ranges:
                lin_vel_x = [-1.0, 1.0]
                lin_vel_y = [-1.0, 1.0]
                ang_vel_yaw = [-1.0, 1.0]
        
        class stage2:
            """Stage2硬约束训练配置"""
            terrain_type = "sparse_terrain"
            use_soft_termination = False   # 硬终止：踩空立即终止
            use_target_perception = False  # 不使用目标地形感知
            
            # Stage2命令范围（仅前进）
            class command_ranges:
                lin_vel_x = [-1.0, 1.0]
                lin_vel_y = [0.0, 0.0]     # 固定为0
                ang_vel_yaw = [0.0, 0.0]   # 固定为0


# 启用BEAMDOJO功能的完整配置示例
class HumanoidBEAMDOJOFullCfg(HumanoidBEAMDOJOCfg):
    """启用所有BEAMDOJO功能的完整配置"""
    pass

class HumanoidBEAMDOJOFullCfgPPO(HumanoidBEAMDOJOCfgPPO):
    """启用所有BEAMDOJO功能的完整PPO配置"""
    
    class policy(HumanoidBEAMDOJOCfgPPO.policy):
        use_double_critic = True  # 启用双Critic
        
    class algorithm(HumanoidBEAMDOJOCfgPPO.algorithm):
        use_double_critic = True  # 启用双Critic算法
        
    class runner(HumanoidBEAMDOJOCfgPPO.runner):
        policy_class_name = 'ActorCriticRMADoubleReward'  # 使用双Critic策略
        algorithm_class_name = 'PPODoubleReward'          # 使用双Critic算法
        experiment_name = 'humanoid_beamdojo_full'
        
    class training(HumanoidBEAMDOJOCfgPPO.training):
        enable_two_stage = True   # 启用两阶段训练