#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import math
from diy_qi.config import GameConfig

# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, player_id):
        self.player_id = player_id
        self.last_frame_state = None

    def result(self, frame_state):
        """
        计算当前帧的奖励
        :param frame_state: 当前帧的状态信息
        :return: 奖励字典
        """
        reward_dict = {
            "hp_point": 0.0,
            "tower_hp_point": 0.0,
            "money": 0.0,
            "ep_rate": 0.0,
            "death": 0.0,
            "kill": 0.0,
            "exp": 0.0,
            "last_hit": 0.0,
        }

        if self.last_frame_state is not None:
            # 计算奖励项
            reward_dict["hp_point"] = self.calculate_hp_reward(frame_state)
            reward_dict["tower_hp_point"] = self.calculate_tower_hp_reward(frame_state)
            reward_dict["money"] = self.calculate_money_reward(frame_state)
            reward_dict["ep_rate"] = self.calculate_ep_rate_reward(frame_state)
            reward_dict["death"] = self.calculate_death_reward(frame_state)
            reward_dict["kill"] = self.calculate_kill_reward(frame_state)
            reward_dict["exp"] = self.calculate_exp_reward(frame_state)
            reward_dict["last_hit"] = self.calculate_last_hit_reward(frame_state)

            # 计算最终奖励
            reward_dict["reward_sum"] = self.calculate_final_reward(reward_dict)

        # 更新上一帧状态
        self.last_frame_state = frame_state
        return reward_dict

    def calculate_hp_reward(self, frame_state):
        # 计算英雄生命值的奖励
        current_hp = frame_state["hero"]["hp"]
        last_hp = self.last_frame_state["hero"]["hp"] if self.last_frame_state else current_hp
        return current_hp - last_hp  # 生命值变化

    def calculate_tower_hp_reward(self, frame_state):
        # 计算塔的生命值奖励
        current_tower_hp = frame_state["tower"]["hp"]
        last_tower_hp = self.last_frame_state["tower"]["hp"] if self.last_frame_state else current_tower_hp
        return current_tower_hp - last_tower_hp  # 塔生命值变化

    def calculate_money_reward(self, frame_state):
        # 计算获得的金币奖励
        current_money = frame_state["hero"]["money"]
        last_money = self.last_frame_state["hero"]["money"] if self.last_frame_state else current_money
        return current_money - last_money  # 金币变化

    def calculate_ep_rate_reward(self, frame_state):
        # 计算法力值的奖励
        current_ep = frame_state["hero"]["ep"]
        last_ep = self.last_frame_state["hero"]["ep"] if self.last_frame_state else current_ep
        return current_ep - last_ep  # 法力值变化

    def calculate_death_reward(self, frame_state):
        # 计算死亡的惩罚
        if frame_state["hero"]["is_dead"]:
            return -1.0  # 死亡惩罚
        return 0.0

    def calculate_kill_reward(self, frame_state):
        # 计算击杀的奖励
        if frame_state["hero"]["killed_enemy"]:
            return 1.0  # 击杀奖励
        return 0.0

    def calculate_exp_reward(self, frame_state):
        # 计算经验值的奖励
        current_exp = frame_state["hero"]["exp"]
        last_exp = self.last_frame_state["hero"]["exp"] if self.last_frame_state else current_exp
        return current_exp - last_exp  # 经验值变化

    def calculate_last_hit_reward(self, frame_state):
        # 计算最后一击的奖励
        if frame_state["hero"]["last_hit"]:
            return 0.5  # 最后一击奖励
        return 0.0

    def calculate_final_reward(self, reward_dict):
        # 计算最终奖励
        final_reward = 0.0
        # 加权求和
        final_reward += reward_dict["hp_point"] * 0.1
        final_reward += reward_dict["tower_hp_point"] * 0.2
        final_reward += reward_dict["money"] * 0.1
        final_reward += reward_dict["ep_rate"] * 0.1
        final_reward += reward_dict["death"] * -1.0
        final_reward += reward_dict["kill"] * 1.0
        final_reward += reward_dict["exp"] * 0.1
        final_reward += reward_dict["last_hit"] * 0.5
        return final_reward
