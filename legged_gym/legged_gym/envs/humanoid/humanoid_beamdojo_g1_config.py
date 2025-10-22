# BEAMDOJO G1 Robot Configuration
# 基于BEAMDOJO论文的G1人形机器人配置
# 适配G1机器人的物理特性和运动学参数

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HumanoidBEAMDOJOG1Cfg(LeggedRobotCfg):
    """
    BEAMDOJO G1机器人环境配置
    实现双Critic网络和两阶段训练的完整配置，适配G1机器人
    """
    
    class init_state(LeggedRobotCfg.init_state):
        """G1机器人初始状态配置"""
        pos = [0.0, 0.0, 0.80]  # x,y,z [m] - G1机器人站立高度约0.8m
        default_joint_angles = { # G1机器人默认关节角度 [rad]
           'left_hip_yaw_joint': 0.0,
           'left_hip_roll_joint': 0.0,
           'left_hip_pitch_joint': -0.1,
           'left_knee_joint': 0.3,
           'left_ankle_pitch_joint': -0.2,
           'left_ankle_roll_joint': 0.0,
           'right_hip_yaw_joint': 0.0,
           'right_hip_roll_joint': 0.0,
           'right_hip_pitch_joint': -0.1,
           'right_knee_joint': 0.3,
           'right_ankle_pitch_joint': -0.2,
           'right_ankle_roll_joint': 0.0,
        }
    
    class env(LeggedRobotCfg.env):
        """环境配置 - 根据G1机器人调整"""
        num_envs = 4096
        episode_length_s = 40
        
        # G1机器人观测配置
        n_scan = 132                    # 激光雷达扫描点数
        n_priv = 3 + 3 + 3             # 私有状态：位置(3) + 速度(3) + 其他(3) = 9
        n_priv_latent = 4 + 1 + 12 + 12  # 潜在状态：质量(4) + 摩擦(1) + 电机强度(12+12) = 29
        n_proprio = 51                  # G1本体感受维度 (根据G1实际传感器配置)
        history_len = 10                # 历史长度
        
        # 计算总观测维度: 本体感受 + 激光雷达 + 历史 + 潜在状态 + 私有状态
        num_observations = n_proprio + n_scan + history_len * n_proprio + n_priv_latent + n_priv
        num_actions = 12               # G1机器人12个关节动作 (6DoF per leg)
        
        # 启用接触信息
        include_foot_contacts = True
        contact_buf_len = 100
        
    class control(LeggedRobotCfg.control):
        """G1机器人控制配置"""
        control_type = 'P'  # 位置控制
        
        # G1机器人PD控制参数 (根据实际硬件调整)
        stiffness = {
            'hip_yaw': 100,     # 髋关节偏航刚度
            'hip_roll': 100,    # 髋关节滚转刚度  
            'hip_pitch': 100,   # 髋关节俯仰刚度
            'knee': 150,        # 膝关节刚度
            'ankle': 40,        # 踝关节刚度
        }  # [N*m/rad]
        
        damping = {
            'hip_yaw': 2,       # 髋关节偏航阻尼
            'hip_roll': 2,      # 髋关节滚转阻尼
            'hip_pitch': 2,     # 髋关节俯仰阻尼
            'knee': 4,          # 膝关节阻尼
            'ankle': 2,         # 踝关节阻尼
        }  # [N*m*s/rad]
        
        # 动作缩放：目标角度 = actionScale * action + defaultAngle
        action_scale = 0.25
        # 控制频率分频：每个策略DT的控制动作更新次数
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        """G1机器人资产配置"""
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof_with_hand.urdf'
        name = "g1_beamdojo"
        foot_name = "ankle_roll"        # G1脚部连接名称
        knee_name = "knee"              # G1膝关节名称
        penalize_contacts_on = ["hip", "knee"]  # 惩罚接触的部位
        terminate_after_contacts_on = ["pelvis"]  # 接触后终止的部位
        self_collisions = 0             # 启用自碰撞检测
        flip_visual_attachments = False

    class terrain:
        """地形配置 - 支持两阶段训练"""
        mesh_type = 'trimesh'
        horizontal_scale = 0.1          # 水平分辨率
        vertical_scale = 0.005          # 垂直分辨率
        border_size = 25                # 边界大小
        curriculum = True               # 启用课程学习
        static_friction = 1.0           # 静摩擦系数
        dynamic_friction = 1.0          # 动摩擦系数
        restitution = 0.0               # 恢复系数
        
        # 地形类型配置
        terrain_types = ["flat", "rough", "stairs", "obstacles"]
        terrain_proportions = [0.3, 0.3, 0.2, 0.2]
        
        # 支持两阶段地形切换
        stage1_terrain_types = ["flat"]  # Stage1仅使用平坦地形
        stage2_terrain_types = ["rough", "stairs", "obstacles"]  # Stage2使用复杂地形
        
    class commands(LeggedRobotCfg.commands):
        """运动命令配置 - 适配G1运动能力"""
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0          # 命令重新采样时间 [s]
        heading_command = True          # 启用朝向命令
        
        # G1特有的命令属性
        ang_vel_clip = 0.1              # 角速度命令死区阈值
        lin_vel_clip = 0.1              # 线速度命令死区阈值
        height_adaptive_speed = False    # 启用基于高度的自适应速度
        speed_complexity_weight = 0.4   # 地形复杂度权重
        speed_gradient_weight = 0.4     # 高度梯度权重  
        speed_roughness_weight = 0.2    # 地形粗糙度权重
        
        # G1机器人命令范围 (根据实际运动能力调整)
        class ranges:
            lin_vel_x = [-0.8, 1.2]     # x方向线速度 [m/s] - G1前进能力较强
            lin_vel_y = [-0.6, 0.6]     # y方向线速度 [m/s] - 侧向移动
            ang_vel_yaw = [-1.0, 1.0]   # yaw角速度 [rad/s] - 转向能力
            heading = [-3.14, 3.14]     # 朝向角度 [rad]
            
        # 最大命令范围（必需的属性）
        class max_ranges:
            lin_vel_x = [-0.8, 1.2]     # x方向线速度 [m/s] - G1前进能力较强
            lin_vel_y = [-0.6, 0.6]     # y方向线速度 [m/s] - 侧向移动
            ang_vel_yaw = [-1.0, 1.0]   # yaw角速度 [rad/s] - 转向能力
            heading = [-3.14, 3.14]     # 朝向角度 [rad]
            
    class rewards(LeggedRobotCfg.rewards):
        """BEAMDOJO奖励配置 - 针对G1优化"""
        class scales:
            """奖励权重配置"""
            # 密集奖励（用于第一个Critic）
            tracking_lin_vel = 2.0      # 线速度跟踪奖励
            tracking_ang_vel = 1.0      # 角速度跟踪奖励
            orientation = -1.25         # 姿态奖励（G1优化）
            base_height = -10.0         # 基座高度奖励
            lin_vel_z = -2.0           # Z方向速度惩罚
            ang_vel_xy = -0.05         # XY轴角速度惩罚
            torques = -2.5e-6          # 力矩惩罚
            action_rate = -0.01        # 动作变化率惩罚
            smoothness = -1e-3         # 平滑度奖励
            stand_still = -0.05        # 静止惩罚
            dof_vel = -1e-4           # 关节速度惩罚
            dof_acc = -2.5e-8         # 关节加速度惩罚
            dof_pos_limits = -5.0     # 关节位置限制惩罚
            dof_vel_limits = -1e-3    # 关节速度限制惩罚
            dof_power = -2e-5         # 关节功率惩罚
            feet_ground_parallel = -0.02  # 脚部平行地面奖励
            feet_distance = -2.0      # 脚间距离奖励（G1优化）
            feet_air_time = 2.5       # 腾空时间奖励（G1优化）
            feet_clearance = -1.0     # 脚部离地高度奖励
            
            # 稀疏奖励（用于第二个Critic）
            foothold = 1.0            # BEAMDOJO Foothold奖励
            
        # BEAMDOJO双Critic奖励分离
        dense_rewards = [
            'tracking_lin_vel', 'tracking_ang_vel', 'orientation', 'base_height',
            'lin_vel_z', 'ang_vel_xy', 'torques', 'action_rate', 'dof_vel', 
            'dof_acc', 'feet_air_time', 'collision', 'feet_stumble', 'stand_still',
            'feet_distance', 'feet_forward_alignment'
        ]
        sparse_rewards = [
            'foothold', 'reach_goal', 'heading_tracking', 'next_heading_tracking'
        ]
        
        # Foothold奖励配置（针对G1脚部）
        class foothold:
            num_sample_points = 4      # 每个脚掌采样点数量
            sample_radius = 0.02       # 采样半径 [m]
            height_tolerance = 0.05    # 高度容忍度 [m]
            stability_weight = 1.0     # 稳定性权重
            safety_weight = 0.5        # 安全性权重
            
        only_positive_rewards = False  # 允许负奖励
        tracking_sigma = 0.25         # 跟踪奖励标准差
        soft_dof_pos_limit = 1.0      # 软关节位置限制
        soft_dof_vel_limit = 1.0      # 软关节速度限制
        soft_torque_limit = 1.0       # 软力矩限制
        base_height_target = 0.7      # G1目标基座高度
        max_contact_force = 100.0     # 最大接触力
        
        # G1特定参数
        feet_air_time_target = 0.25   # 目标腾空时间 [s]
        min_dist = 0.08              # 最小脚间距 [m]
        max_dist = 0.25              # 最大脚间距 [m]
        target_feet_height = 0.1      # 目标脚部高度 [m]
        is_play = False              # 是否为测试模式
        
    class normalization:
        """归一化配置"""
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            
        clip_observations = 100.0
        clip_actions = 100.0

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
        dt = 0.005                    # 仿真时间步长
        substeps = 1                  # 子步数
        gravity = [0., 0., -9.81]     # 重力加速度
        up_axis = 1                   # 上轴索引 (Z轴)
        
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


class HumanoidBEAMDOJOG1CfgPPO(LeggedRobotCfgPPO):
    """
    BEAMDOJO G1机器人PPO训练配置
    包含双Critic网络和两阶段训练的完整配置
    """
    
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    
    class policy:
        """策略网络配置 - 适配G1"""
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]   # Actor网络隐藏层
        critic_hidden_dims = [512, 256, 128]  # Critic网络隐藏层
        activation = 'elu'                    # 激活函数
        
        # 扫描编码器配置
        scan_encoder_dims = [128, 64, 32]     # 激光雷达编码器
        priv_encoder_dims = [64, 20]          # 私有状态编码器
        
        # 网络参数配置
        tanh_encoder_output = False           # 编码器输出是否使用tanh激活
        
        # 支持双Critic的编码器
        use_double_critic = False             # 在这里可以启用双Critic
        
    class algorithm:
        """BEAMDOJO PPO算法配置"""
        # 基础PPO参数
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3              # 学习率
        schedule = 'adaptive'             # 学习率调度
        gamma = 0.99                      # 折扣因子
        lam = 0.95                        # GAE lambda
        desired_kl = 0.01                 # 目标KL散度
        max_grad_norm = 1.0               # 梯度裁剪
        
        # BEAMDOJO双Critic配置
        use_double_critic = False         # 设置为True启用双Critic
        dense_value_loss_coef = 1.0       # 密集奖励价值损失系数
        sparse_value_loss_coef = 1.0      # 稀疏奖励价值损失系数
        advantage_merge_weight = 0.5      # 优势函数合并权重
        
        # DAgger参数
        dagger_update_freq = 20           # DAgger更新频率
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]        # 特权正则化系数调度
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]       # 恢复训练时的特权正则化系数调度
        
    class runner:
        """训练运行器配置"""
        policy_class_name = 'ActorCritic'    # 使用 'ActorCriticRMADoubleReward' 启用双Critic
        algorithm_class_name = 'PPO'         # 使用 'PPODoubleReward' 启用双Critic算法
        num_steps_per_env = 24
        max_iterations = 100000
        
        save_interval = 200                   # G1训练保存间隔
        experiment_name = 'humanoid_beamdojo_g1'
        run_name = ''
        
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    class estimator(LeggedRobotCfgPPO.estimator):
        """状态估计器配置 - 适配G1"""
        train_with_estimated_states = True
        learning_rate = 1e-4
        hidden_dims = [128, 64]
        priv_states_dim = HumanoidBEAMDOJOG1Cfg.env.n_priv        # G1私有状态维度
        num_prop = HumanoidBEAMDOJOG1Cfg.env.n_proprio            # G1本体感受维度
        num_scan = HumanoidBEAMDOJOG1Cfg.env.n_scan               # G1激光雷达维度

    # BEAMDOJO两阶段训练配置 - 针对G1优化
    class training:
        """两阶段训练配置"""
        enable_two_stage = False  # 设置为True启用两阶段训练
        
        class stage1:
            """Stage1软约束训练配置 - G1基础运动学习"""
            min_steps = 1000000            # 最小训练步数
            max_steps = 5000000            # 最大训练步数  
            success_threshold = 0.8        # 成功率阈值
            terrain_type = "flat_with_target_perception"
            use_soft_termination = True    # 软终止：G1踩空不终止episode
            use_target_perception = True   # 使用目标地形感知
            
            # Stage1命令范围（G1全方向移动）
            class command_ranges:
                lin_vel_x = [-0.6, 1.0]    # G1前进能力较强
                lin_vel_y = [-0.4, 0.4]    # G1侧向移动能力
                ang_vel_yaw = [-0.8, 0.8]  # G1转向能力
        
        class stage2:
            """Stage2硬约束训练配置 - G1复杂地形导航"""
            terrain_type = "sparse_terrain"
            use_soft_termination = False   # 硬终止：踩空立即终止
            use_target_perception = False  # 不使用目标地形感知
            
            # Stage2命令范围（G1仅前进）
            class command_ranges:
                lin_vel_x = [-0.8, 1.2]    # G1最大前进速度
                lin_vel_y = [0.0, 0.0]     # 固定为0
                ang_vel_yaw = [0.0, 0.0]   # 固定为0


# 启用BEAMDOJO功能的完整G1配置
class HumanoidBEAMDOJOG1FullCfg(HumanoidBEAMDOJOG1Cfg):
    """启用所有BEAMDOJO功能的完整G1配置"""
    pass

class HumanoidBEAMDOJOG1FullCfgPPO(HumanoidBEAMDOJOG1CfgPPO):
    """启用所有BEAMDOJO功能的完整G1 PPO配置"""
    
    class policy(HumanoidBEAMDOJOG1CfgPPO.policy):
        use_double_critic = True  # 启用双Critic
        
    class algorithm(HumanoidBEAMDOJOG1CfgPPO.algorithm):
        use_double_critic = True  # 启用双Critic算法
        
    class runner(HumanoidBEAMDOJOG1CfgPPO.runner):
        policy_class_name = 'ActorCriticRMADoubleReward'  # 使用双Critic策略
        algorithm_class_name = 'PPODoubleReward'          # 使用双Critic算法
        experiment_name = 'humanoid_beamdojo_g1_full'
        
    class training(HumanoidBEAMDOJOG1CfgPPO.training):
        enable_two_stage = True   # 启用两阶段训练


# G1机器人专用的Stage1训练配置
class HumanoidBEAMDOJOG1Stage1Cfg(HumanoidBEAMDOJOG1Cfg):
    """G1 Stage1专用配置 - 平地基础运动学习"""
    
    class terrain(HumanoidBEAMDOJOG1Cfg.terrain):
        # 强制使用平坦地形
        terrain_types = ["flat"]
        terrain_proportions = [1.0]
        curriculum = False  # Stage1不使用课程学习
        
    class commands(HumanoidBEAMDOJOG1Cfg.commands):
        # 继承父类的所有属性，只覆盖ranges和max_ranges
        # Stage1命令范围 - 保守设置
        class ranges:
            lin_vel_x = [-0.5, 0.8]     # 降低速度要求
            lin_vel_y = [-0.3, 0.3]     # 降低侧向速度
            ang_vel_yaw = [-0.6, 0.6]   # 降低转向速度
            heading = [-3.14, 3.14]
            
        # 最大命令范围（必需的属性）
        class max_ranges:
            lin_vel_x = [-0.5, 0.8]     # 降低速度要求
            lin_vel_y = [-0.3, 0.3]     # 降低侧向速度
            ang_vel_yaw = [-0.6, 0.6]   # 降低转向速度
            heading = [-3.14, 3.14]

class HumanoidBEAMDOJOG1Stage1CfgPPO(HumanoidBEAMDOJOG1CfgPPO):
    """G1 Stage1专用PPO配置"""
    
    class runner(HumanoidBEAMDOJOG1CfgPPO.runner):
        experiment_name = 'humanoid_beamdojo_g1_stage1'
        max_iterations = 3000  # Stage1较少迭代数
        save_interval = 100    # 更频繁保存