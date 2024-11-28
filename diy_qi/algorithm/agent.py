#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
)
from diy.model.model import Model
from kaiwu_agent.utils.common_func import attached
from diy.feature.definition import *
from diy.config import Config
from diy.feature.reward_manager import GameRewardManager


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)
        self.cur_model_name = ""
        self.device = device
        # 创建模型并将其转换为通道后内存格式以获得更好的性能
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # 环境信息
        self.hero_camp = 0
        self.player_id = 0
        self.game_id = None

        # 工具
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor

    def _model_inference(self, list_obs_data):
        # 实现模型推理的代码
        obs_tensor = torch.tensor([obs.feature for obs in list_obs_data]).to(self.device)
        with torch.no_grad():
            actions = self.model(obs_tensor)
        list_act_data = [ActData(action=action) for action in actions]
        return list_act_data

    @predict_wrapper
    def predict(self, list_obs_data):
        return self._model_inference(list_obs_data)

    @exploit_wrapper
    def exploit(self, state_dict):
        game_id = state_dict["game_id"]
        if self.game_id != game_id:
            player_id = state_dict["player_id"]
            camp = state_dict["player_camp"]
            self.reset(camp, player_id)
            self.game_id = game_id

        obs_data = self.observation_process(state_dict)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(state_dict, act_data, False)

    def train_predict(self, state_dict):
        obs_data = self.observation_process(state_dict)
        act_data = self.predict([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(state_dict, act_data, True)

    def eval_predict(self, state_dict):
        obs_data = self.observation_process(state_dict)
        act_data = self.predict([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(state_dict, act_data, False)

    @learn_wrapper
    def learn(self, list_sample_data):
        # 实现模型训练的代码
        # 将样本数据转换为张量
        sample_tensors = [torch.tensor(sample.npdata).to(self.device) for sample in list_sample_data]
        # 进行模型训练
        # 这里可以添加具体的训练逻辑，例如前向传播、损失计算和反向传播
        # 例如：
        # loss = self.model(sample_tensors)
        # loss.backward()
        # optimizer.step()
        return

    def action_process(self, state_dict, act_data, is_stochastic):
        # 实现ActData到action的转换
        if is_stochastic:
            # 采用随机采样动作
            action = act_data.action  # 这里可以根据需要进行随机选择
        else:
            # 采用最大概率动作
            action = act_data.d_action
        return action

    def observation_process(self, state_dict):
        # 实现State到ObsData的转换
        feature_vec = state_dict["observation"]
        legal_action = state_dict["legal_action"]
        return ObsData(feature=feature_vec, legal_action=legal_action, lstm_cell=None, lstm_hidden=None)

    def reset(self, hero_camp, player_id):
        self.hero_camp = hero_camp
        self.player_id = player_id
        self.reward_manager = GameRewardManager(player_id)

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            self.model.load_state_dict(
                torch.load(
                    model_file_path,
                    map_location=self.device,
                )
            )
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")
