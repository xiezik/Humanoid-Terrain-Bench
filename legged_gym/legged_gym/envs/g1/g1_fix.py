# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# class G1FixCfg( LeggedRobotCfg ):
#     class init_state( LeggedRobotCfg.init_state ):
#         pos = [0.0, 0.0, 0.80]  # x,y,z [m]
#         default_joint_angles = { # = target angles [rad] when action = 0.0
#            'left_hip_yaw_joint' : 0. ,
#            'left_hip_roll_joint' : 0,
#            'left_hip_pitch_joint' : -0.1,
#            'left_knee_joint' : 0.3,
#            'left_ankle_pitch_joint' : -0.2,
#            'left_ankle_roll_joint' : 0,
#            'right_hip_yaw_joint' : 0.,
#            'right_hip_roll_joint' : 0,
#            'right_hip_pitch_joint' : -0.1,
#            'right_knee_joint' : 0.3,
#            'right_ankle_pitch_joint': -0.2,
#            'right_ankle_roll_joint' : 0,
#            'torso_joint' : 0.
#         }

#     class env( LeggedRobotCfg.env ):
#         num_envs = 4096
#         n_scan = 132
#         n_priv = 3 + 3 + 3 # = 9 base velocity 3个
#         # n_priv_latent = 4 + 1 + 12 +12
#         n_priv_latent = 4 + 1 + 12 + 12 # mass, fraction, motor strength1 and 2
        
#         n_proprio = 51 # 所有本体感知信息，即obs_buf
#         history_len = 10

#         # num obs = 53+132+10*53+43+9 = 187+47+530+43+9 = 816
#         num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
#         num_actions = 12
#         env_spacing = 3.  # not used with heightfields/trimeshes 

#         contact_buf_len = 100
        
#     class control( LeggedRobotCfg.control ):
#         # PD Drive parameters:
#         control_type = 'P'
#         # PD Drive parameters:
#         stiffness = {'hip_yaw': 100,
#                      'hip_roll': 100,
#                      'hip_pitch': 100,
#                      'knee': 150,
#                      'ankle': 40,
#                      }  # [N*m/rad]
#         damping = {  'hip_yaw': 2,
#                      'hip_roll': 2,
#                      'hip_pitch': 2,
#                      'knee': 4,
#                      'ankle': 2,
#                      }  # [N*m/rad]  # [N*m*s/rad]
#         # action scale: target angle = actionScale * action + defaultAngle
#         action_scale = 0.25
#         # decimation: Number of control action updates @ sim DT per policy DT
#         decimation = 4

#     class asset( LeggedRobotCfg.asset ):
#         file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof_with_hand.urdf'
#         name = "g1_fix_upper"
#         foot_name = "ankle_roll"
#         knee_name = "knee"
#         penalize_contacts_on = ["hip", "knee"]
#         terminate_after_contacts_on = ["pelvis"]
#         self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
#         flip_visual_attachments = False

#     class commands( LeggedRobotCfg.commands ):
#         """运动命令配置"""
#         # resampling_time = 1.0         # 命令重采样时间间隔（秒）
#         # heading_command = True         # 启用朝向命令模式
#         # ang_vel_clip = 0.05            # 角速度命令死区阈值
#         # lin_vel_clip = 0.2            # 线速度命令死区阈值
        
#         # 策略1：智能速度生成配置
#         height_adaptive_speed = False   # 启用基于高度的自适应速度
#         speed_complexity_weight = 0.4  # 地形复杂度权重
#         speed_gradient_weight = 0.4   # 高度梯度权重  
#         speed_roughness_weight = 0.2  # 地形粗糙度权重
#         class ranges( LeggedRobotCfg.commands.ranges ):
#             lin_vel_x = [0.1, 0.6]  # min max [m/s]
#             lin_vel_y = [0.0, 0.0]   # min max [m/s]
#             ang_vel_yaw = [0, 0]    # min max [rad/s]
#             heading = [0, 0]

  
#     class rewards:
#         class scales:
#             termination = -0.0
#             tracking_lin_vel = 1.0
#             tracking_ang_vel = 0.5
#             lin_vel_z = -2.0
#             ang_vel_xy = -0.05
#             orientation = -0.
#             torques = -0.00001
#             dof_vel = -0.
#             dof_acc = -2.5e-7
#             base_height = -0. 
#             feet_air_time =  1.0
#             collision = -1.
#             feet_stumble = -0.0 
#             action_rate = -0.01
#             stand_still = -0.

#             # reach_goal = 1.0           # 到达目标奖励
#             # heading_tracking = 0.8      # 朝向跟踪奖励
#             # next_heading_tracking = 0.4  # 下一朝向跟踪奖励

#         only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
#         tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
#         soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
#         soft_dof_vel_limit = 1.
#         soft_torque_limit = 1.
#         base_height_target = 1.
#         max_contact_force = 100. # forces above this value are penalized
#         is_play = False

# class G1FixCfgPPO( LeggedRobotCfgPPO ):
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         entropy_coef = 0.01
#     class runner( LeggedRobotCfgPPO.runner ):
#         run_name = ''
#         experiment_name = 'g1_fix'
#         max_iterations = 100001 # number of policy updates
#         save_interval = 200

#     class estimator(LeggedRobotCfgPPO.estimator):
#         train_with_estimated_states = True
#         learning_rate = 1.e-4
#         hidden_dims = [128, 64]
#         priv_states_dim = G1FixCfg.env.n_priv
#         num_prop = G1FixCfg.env.n_proprio
#         num_scan = G1FixCfg.env.n_scan


# align with unitree code
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1FixCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.80]  # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,
           'left_hip_roll_joint' : 0,
           'left_hip_pitch_joint' : -0.1,
           'left_knee_joint' : 0.3,
           'left_ankle_pitch_joint' : -0.2,
           'left_ankle_roll_joint' : 0,
           'right_hip_yaw_joint' : 0.,
           'right_hip_roll_joint' : 0,
           'right_hip_pitch_joint' : -0.1,
           'right_knee_joint' : 0.3,
           'right_ankle_pitch_joint': -0.2,
           'right_ankle_roll_joint' : 0,
           'torso_joint' : 0.
        }

    class env( LeggedRobotCfg.env ):
        num_envs = 2048
        n_scan = 132
        n_priv = 3 + 3 + 3 # = 9 base velocity 3个
        # n_priv_latent = 4 + 1 + 12 +12
        n_priv_latent = 4 + 1 + 12 + 12 # mass, fraction, motor strength1 and 2
        
        n_proprio = 51 # 所有本体感知信息，即obs_buf
        history_len = 10

        # num obs = 53+132+10*53+43+9 = 187+47+530+43+9 = 816
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 

        contact_buf_len = 100
        
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof_with_hand.urdf'
        name = "g1_fix_upper"
        foot_name = "ankle_roll"
        knee_name = "knee"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class commands( LeggedRobotCfg.commands ):
        """运动命令配置"""
        # resampling_time = 1.0         # 命令重采样时间间隔（秒）
        # heading_command = True         # 启用朝向命令模式
        # ang_vel_clip = 0.05            # 角速度命令死区阈值
        # lin_vel_clip = 0.2            # 线速度命令死区阈值
        
        # 策略1：智能速度生成配置
        height_adaptive_speed = False   # 启用基于高度的自适应速度
        speed_complexity_weight = 0.4  # 地形复杂度权重
        speed_gradient_weight = 0.4   # 高度梯度权重  
        speed_roughness_weight = 0.2  # 地形粗糙度权重
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0.1, 0.8] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [-1.2, 1.2]

  
    # class rewards:
    #     class scales:
    #         # termination = -0.0
    #         tracking_lin_vel = 1.0
    #         tracking_ang_vel = 1.0
    #         base_height = -10.0
    #         orientation = -2.0
    #         lin_vel_z = -2.0
    #         ang_vel_xy = -0.05
    #         # torques = -0.00001
    #         action_rate = -0.01
    #         # smoothness = -1e-3
    #         stand_still = -0.05
    #         dof_vel = -1e-4
    #         dof_acc = -2.5e-8
    #         dof_pos_limits = -5.0
    #         dof_vel_limits = -1e-3
    #         dof_power = -2e-5
    #         feet_ground_parallel = -0.02
    #         feet_distance = 0.5
    #         feet_air_time =  1.0
    #         feet_clearance = -1.0 
    #         # alive = 0.15
    #         # hip_pos = -1.0
    #         # contact_no_vel = -0.2
    #         # feet_swing_height = -20.0
    #         # contact = 0.18
    #         feet_air_time =  2.5 
    #         reach_goal = 1.0           # 到达目标奖励
    #         heading_tracking = 1.0      # 朝向跟踪奖励
    #         # collision = -0.

    #     only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
    #     tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    #     soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
    #     soft_dof_vel_limit = 1.
    #     soft_torque_limit = 1.
    #     feet_air_time_target = 0.5  # 目标腾空时间 (秒)
    #     min_dist = 0.18  # 最小距离，用于feet_distance奖励计算
    #     target_feet_height = 0.2  # 目标脚部高度，用于feet_clearance奖励计算
    #     base_height_target = 0.725
    #     max_contact_force = 100. # forces above this value are penalized
    #     is_play = False

    # class rewards:
    #     class scales:
    #         # termination = -0.0
    #         tracking_lin_vel = 2.0
    #         tracking_ang_vel = 1.0
    #         # base_height = -10.0
    #         orientation = -1.25
    #         # lin_vel_z = -2.0
    #         ang_vel_xy = -0.05
    #         torques = -2.5e-6
    #         action_rate = -0.01
    #         # smoothness = -2e-6
    #         # stand_still = -0.05
    #         dof_vel = -1e-4
    #         dof_acc = -2.5e-8
    #         dof_pos_limits = -5.0
    #         dof_vel_limits = -1e-3
    #         dof_power = -2e-5
    #         feet_ground_parallel = -0.1
    #         feet_distance = -2.0
    #         feet_air_time =  2.5
    #         feet_clearance = -1.0 
    #         # feet_parallel = -2.0
    #         feet_forward_alignment = 1.5
    #         # alive = 0.15
    #         # hip_pos = -1.0
    #         # feet_swing_height = -20.0
    #         # contact = 0.18
    #         # collision = -0.
            
    #         reach_goal = 2.0
    #         heading_tracking = 0.5
    #         next_heading_tracking = 0.3
            

    #     only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
    #     tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    #     soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
    #     soft_dof_vel_limit = 1.
    #     soft_torque_limit = 1.
    #     feet_air_time_target = 0.5  # 目标腾空时间 (秒)
    #     min_dist = 0.1  # 最小距离，用于feet_distance奖励计算
    #     max_dist = 0.35  # 最大距离，用于feet_distance奖励计算
    #     target_feet_height = 0.1  # 目标脚部高度，用于feet_clearance奖励计算
    #     base_height_target = 0.725
    #     max_contact_force = 100. # forces above this value are penalized
    #     is_play = False

    class rewards:
        """奖励函数配置 - G1机器人多地形训练优化版本"""
        class scales:
            """各项奖励的权重系数 - 针对G1机器人多地形训练优化"""
            # === 核心跟踪奖励 ===
            termination = -0.0          # 终止惩罚（设为0，避免早期终止）
            tracking_lin_vel = 2.0      # G1更灵活，提升速度跟踪权重
            tracking_ang_vel = 1.0      # G1转向能力强，高权重角速度跟踪
            
            # === 基础运动约束 ===
            lin_vel_z = -1.2           # G1较轻，允许更多垂直运动
            ang_vel_xy = -0.03         # 降低侧翻惩罚（G1更稳定）
            orientation = -0.6         # G1适应性强，降低姿态惩罚
            
            # === 关节和动作控制 ===
            torques = -1.5e-6         # G1力矩较小，轻微惩罚
            dof_vel_limits = -0.15    # G1关节速度限制稍宽松
            dof_pos_limits = -3.0     # 适度关节位置限制
            # dof_vel = -5e-5           # 轻微关节速度惩罚
            # dof_acc = -1e-8           # 关节加速度惩罚
            # dof_power = -1e-5         # 功率惩罚
            action_rate = -0.008      # 允许G1更灵活的动作变化
            
            # === 步态和接触控制 ===
            feet_air_time = 2.0       # G1腾空能力好，适度奖励
            feet_slip = -0.15         # 轻微脚滑惩罚
            # feet_stumble = -1.5       # 降低绊倒惩罚
            # feet_distance = -1.2      # G1步幅灵活，降低脚间距惩罚
            feet_clearance = -2.5     # 脚部离地高度控制
            feet_ground_parallel = -1.0  # 脚部平行地面奖励
            feet_contact_force = -8e-5    # 接触力惩罚
            feet_parallel = -1.5      # 脚部平行惩罚
            # feet_forward_alignment = 1.0  # 脚部前向对齐奖励
            
            # === 稳定性和平衡 ===
            no_fly = 0.3              # G1防飞行奖励（允许适度跳跃）
            contact_momentum = -8e-5   # 接触动量惩罚
            
            # === 新增：地形适应性奖励 ===
            safe_foot_placement = 0.8     # 安全落脚点奖励
            base_height = -0.2        # 基座高度控制
            # smoothness = -0.003       # 动作平滑性奖励
            
            # === 任务导向奖励 ===
            reach_goal = 2.2          # 目标到达奖励
            heading_tracking = 0.5    # 朝向跟踪奖励
            # next_heading_tracking = 0.5  # 下一朝向跟踪奖励
            
            # === 多地形特定奖励 ===
            # terrain_adaptation = 0.6  # G1地形适应性奖励
            # energy_efficiency = 0.4   # G1能量效率奖励
            

        # === G1机器人多地形奖励函数参数配置 ===
        only_positive_rewards = False    # 允许负奖励（帮助学习约束）
        tracking_sigma = 0.25            # G1精度要求，保持标准跟踪容忍度
        
        # === G1关节限制参数 ===
        soft_dof_pos_limit = 0.95        # G1关节限制稍宽松
        soft_dof_vel_limit = 0.9         # G1速度限制适中
        soft_torque_limit = 0.9          # G1力矩限制适中
        
        # === G1脚部控制参数 ===
        feet_air_time_target = 0.5       # G1目标腾空时间
        min_dist = 0.06                  # G1最小脚间距（更紧凑）
        max_dist = 0.3                   # G1最大脚间距
        target_feet_height = 0.5        # G1目标脚部离地高度（较低）
        base_height_target = 0.75        # G1目标基座高度
        max_contact_force = 400.         # G1最大接触力（较小机器人）
        
        # === 新增：G1多地形特定参数 ===
        slope_adaptation_threshold = 0.15  # G1坡度适应阈值
        rough_terrain_tolerance = 0.12   # G1粗糙地形容忍度
        agility_bonus_threshold = 0.5    # G1敏捷性奖励阈值
        
        is_play = False                  # 训练模式

class G1FixCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'g1_fix'
        max_iterations = 100001 # number of policy updates
        save_interval = 200

    class estimator(LeggedRobotCfgPPO.estimator):
        train_with_estimated_states = True
        learning_rate = 2.e-4
        hidden_dims = [128, 64]
        priv_states_dim = G1FixCfg.env.n_priv
        num_prop = G1FixCfg.env.n_proprio
        num_scan = G1FixCfg.env.n_scan


# unitree config
# from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# class G1FixCfg( LeggedRobotCfg ):
#     class init_state( LeggedRobotCfg.init_state ):
#         pos = [0.0, 0.0, 0.80]  # x,y,z [m]
#         default_joint_angles = { # = target angles [rad] when action = 0.0
#            'left_hip_yaw_joint' : 0. ,
#            'left_hip_roll_joint' : 0,
#            'left_hip_pitch_joint' : -0.1,
#            'left_knee_joint' : 0.3,
#            'left_ankle_pitch_joint' : -0.2,
#            'left_ankle_roll_joint' : 0,
#            'right_hip_yaw_joint' : 0.,
#            'right_hip_roll_joint' : 0,
#            'right_hip_pitch_joint' : -0.1,
#            'right_knee_joint' : 0.3,
#            'right_ankle_pitch_joint': -0.2,
#            'right_ankle_roll_joint' : 0,
#            'torso_joint' : 0.
#         }

#     class env( LeggedRobotCfg.env ):
#         num_envs = 4096
#         n_scan = 132
#         n_priv = 3 + 3 + 3 # = 9 base velocity 3个
#         # n_priv_latent = 4 + 1 + 12 +12
#         n_priv_latent = 4 + 1 + 12 + 12 # mass, fraction, motor strength1 and 2
        
#         n_proprio = 51 # 所有本体感知信息，即obs_buf
#         history_len = 10

#         # num obs = 53+132+10*53+43+9 = 187+47+530+43+9 = 816
#         num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
#         num_actions = 12
#         env_spacing = 3.  # not used with heightfields/trimeshes 

#         contact_buf_len = 100
        
#     class control( LeggedRobotCfg.control ):
#         # PD Drive parameters:
#         control_type = 'P'
#         # PD Drive parameters:
#         stiffness = {'hip_yaw': 100,
#                      'hip_roll': 100,
#                      'hip_pitch': 100,
#                      'knee': 150,
#                      'ankle': 40,
#                      }  # [N*m/rad]
#         damping = {  'hip_yaw': 2,
#                      'hip_roll': 2,
#                      'hip_pitch': 2,
#                      'knee': 4,
#                      'ankle': 2,
#                      }  # [N*m/rad]  # [N*m*s/rad]
#         # action scale: target angle = actionScale * action + defaultAngle
#         action_scale = 0.25
#         # decimation: Number of control action updates @ sim DT per policy DT
#         decimation = 4

#     class asset( LeggedRobotCfg.asset ):
#         file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof_with_hand.urdf'
#         name = "g1_fix_upper"
#         foot_name = "ankle_roll"
#         knee_name = "knee"
#         penalize_contacts_on = ["hip", "knee"]
#         terminate_after_contacts_on = ["pelvis"]
#         self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
#         flip_visual_attachments = False

#     class commands( LeggedRobotCfg.commands ):
#         """运动命令配置"""
#         # resampling_time = 1.0         # 命令重采样时间间隔（秒）
#         # heading_command = True         # 启用朝向命令模式
#         # ang_vel_clip = 0.05            # 角速度命令死区阈值
#         # lin_vel_clip = 0.2            # 线速度命令死区阈值
        
#         # 策略1：智能速度生成配置
#         height_adaptive_speed = False   # 启用基于高度的自适应速度
#         speed_complexity_weight = 0.4  # 地形复杂度权重
#         speed_gradient_weight = 0.4   # 高度梯度权重  
#         speed_roughness_weight = 0.2  # 地形粗糙度权重
#         class ranges( LeggedRobotCfg.commands.ranges ):
#             lin_vel_x = [0.1, 1.2] # min max [m/s]
#             lin_vel_y = [0.0, 0.0]   # min max [m/s]
#             ang_vel_yaw = [0, 0]    # min max [rad/s]
#             heading = [-1.2, 1.2]

  
#     class rewards:
#         class scales:
#             termination = -0.0
#             tracking_lin_vel = 1.5
#             tracking_ang_vel = 0.5
#             lin_vel_z = -2.0
#             ang_vel_xy = -0.05
#             orientation = -1.0
#             torques = -0.00001
#             base_height = -10.0
#             dof_vel = -1e-3
#             dof_acc = -2.5e-7
#             feet_air_time =  0.0 
#             collision = -0.
#             feet_stumble = -0.0 
#             action_rate = -0.01
#             dof_pos_limits = -5.0
#             alive = 0.15
#             hip_pos = -1.0
#             contact_no_vel = -0.2
#             feet_swing_height = -20.0
#             contact = 0.18
#             stand_still = -0.

#             dof_pos_limits = -5.0
#             alive = 0.15
#             hip_pos = -1.0
#             contact_no_vel = -0.2
#             feet_swing_height = -20.0
#             contact = 0.18

#         only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
#         tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
#         soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
#         soft_dof_vel_limit = 1.
#         soft_torque_limit = 1.
#         base_height_target = 1.
#         max_contact_force = 100. # forces above this value are penalized
#         is_play = False

# class G1FixCfgPPO( LeggedRobotCfgPPO ):
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         value_loss_coef = 1.0
#         use_clipped_value_loss = True
#         clip_param = 0.2
#         entropy_coef = 0.01
#         num_learning_epochs = 5
#         num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
#         learning_rate = 1.e-3 #5.e-4
#         schedule = 'adaptive' # could be adaptive, fixed
#         gamma = 0.99
#         lam = 0.95
#         desired_kl = 0.01
#         max_grad_norm = 1.

#     class runner( LeggedRobotCfgPPO.runner ):
#         run_name = ''
#         experiment_name = 'g1_fix'
#         max_iterations = 100001 # number of policy updates
#         save_interval = 200

#     class estimator(LeggedRobotCfgPPO.estimator):
#         train_with_estimated_states = True
#         learning_rate = 1.e-4
#         hidden_dims = [128, 64]
#         priv_states_dim = G1FixCfg.env.n_priv
#         num_prop = G1FixCfg.env.n_proprio
#         num_scan = G1FixCfg.env.n_scan