from .single_terrain import single_terrain
from .config import terrain_config
import numpy as np
class combine_config:
        single = [
                single_terrain.parkour, #0 跳跃 
                single_terrain.hurdle,#1
                single_terrain.bridge,#2
                single_terrain.flat,#3
                single_terrain.uneven,#4
                single_terrain.stair,#5 ！！
                single_terrain.wave,#6
                single_terrain.slope,#7
                single_terrain.gap,#8  ！！
                single_terrain.plot#9 S型 
        ]

        multiplication = [
                [single[3], single[1], single[8], single[4], single[2]],                    # 组合1: stair + hurdle + gap + uneven + bridge
                [single[6], single[8], single[1], single[7], single[2]],                    # 组合2: wave + gap + hurdle + slope + bridge
                [single[8], single[4], single[6], single[5], single[2]],                    # 组合3: gap + uneven + wave + stair + bridge
                [single[7], single[1], single[4], single[8], single[2]],                    # 组合4: slope + hurdle + uneven + gap + bridge
                [single[4], single[6], single[8], single[5], single[2]],                    # 组合5: uneven + wave + gap + stair + bridge
                [single[8], single[5], single[1], single[7], single[2]],                    # 组合6: gap + stair + hurdle + slope + bridge
                [single[6], single[4], single[8], single[7], single[2]],                    # 组合7: wave + uneven + gap + slope + bridge
                [single[1], single[6], single[4], single[5], single[8], single[2]],         # 组合8: hurdle + wave + uneven + stair + gap + bridge
                [single[5], single[7], single[1], single[6], single[8], single[2]],         # 组合9: stair + slope + hurdle + wave + gap + bridge
                [single[1], single[4], single[6], single[7], single[5], single[2]]          # 组合10: gap + uneven + wave + slope + stair + bridge
        ]

        addition = [
                [single[1], single[4], single[5], single[6], single[7], single[2]]  # 组合11: 所有地形 + bridge最后
        ]

        # proportions = [
        #         ("multiplication", 0, 0.1),   # hurdle + uneven + stair + bridge
        #         ("multiplication", 1, 0.1),   # hurdle + wave + slope + bridge
        #         ("multiplication", 2, 0.1),   # uneven + stair + wave + bridge
        #         ("multiplication", 3, 0.1),   # hurdle + uneven + wave + bridge
        #         ("multiplication", 4, 0.1),   # stair + wave + slope + bridge
        #         ("multiplication", 5, 0.1),   # hurdle + stair + slope + bridge
        #         ("multiplication", 6, 0.1),   # uneven + stair + slope + bridge
        #         ("multiplication", 7, 0.1),   # hurdle + uneven + stair + wave + bridge
        #         ("multiplication", 8, 0.1),   # hurdle + stair + wave + slope + bridge
        #         ("multiplication", 9, 0.1)    # uneven + stair + wave + slope + bridge
        # ]
        
        proportions = [
                ("single", 4, 0.5),  
                ("single", 6, 0.5), 
                # ("single", 7, 0.33),  
        ]
                
        # proportions = [
        #         # 只使用单一地形进行多教师蒸馏，确保每个地形都有对应的教师模型
        #         ("single", 5, 1.0),  
        # ]
        
class generator:
        def __init__(self, cfg: terrain_config) -> None:
                self.cfg = cfg

        def single_create(terrain,id=0,difficulty=0.5):
                length_x = terrain_config.terrain_length
                length_y = terrain_config.terrain_width
                num_goals = terrain_config.num_goals
                horizontal_scale = terrain.horizontal_scale
                platform_size = terrain_config.platform_size
                terrain , goals, final_x= combine_config.single[id](
                                                                terrain=terrain, 
                                                                length_x=length_x, 
                                                                length_y=length_y, 
                                                                num_goals=num_goals, 
                                                                platform_size=platform_size, 
                                                                difficulty=difficulty)
                terrain.goals = goals * horizontal_scale
                terrain.idx = id
                return terrain

        def addition_create(terrain,id=0,difficulty=0.5):
                terrain_list = combine_config.addition[id]
                num_terrain = len(terrain_list)
                platform_size = terrain_config.platform_size
                length_x = (terrain_config.terrain_length) // num_terrain
                length_y = terrain_config.terrain_width
                num_goals = terrain_config.num_goals // num_terrain
                horizontal_scale = terrain.horizontal_scale
                goals = []
                final_x = 20
                for i in range(num_terrain):
                        if(i == num_terrain-1):
                                num_goals = terrain_config.num_goals - i*num_goals
                        terrain , sub_goals, final_x = terrain_list[i](
                                                                        terrain=terrain, 
                                                                        length_x=length_x, 
                                                                        length_y=length_y, 
                                                                        num_goals=num_goals, 
                                                                        start_x=final_x, 
                                                                        platform_size=platform_size, 
                                                                        difficulty=difficulty)
                        final_x -= round(platform_size / horizontal_scale)
                        goals.append(sub_goals)

                goals = np.vstack(goals)
                terrain.goals = goals * horizontal_scale
                terrain.idx = id + len(combine_config.single)
                return terrain

        def multiplication_create(terrain,id=0,difficulty=0.5):

                terrain_list = combine_config.multiplication[id]
                num_terrain = len(terrain_list)
                platform_size = terrain_config.platform_size
                length_x = terrain_config.terrain_length- platform_size
                length_y = terrain_config.terrain_width
                num_goals = terrain_config.num_goals
                horizontal_scale = terrain.horizontal_scale
                goals = []
                start_x = 20
                for i in range(num_terrain):
                        terrain , goals, final_x = terrain_list[i](
                                                                terrain=terrain, 
                                                                length_x=length_x, 
                                                                length_y=length_y, 
                                                                num_goals=num_goals, 
                                                                start_x=start_x, 
                                                                platform_size=platform_size, 
                                                                difficulty=difficulty)

                terrain.goals = goals * horizontal_scale
                terrain.idx = id+ len(combine_config.single) + len(combine_config.addition)
                return terrain
