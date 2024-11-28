#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import time
import random
from kaiwu_agent.utils.common_func import attached
from diy_qi.feature.definition import (
    sample_process,
    lineup_iterator_roundrobin_camp_heroes,
    FrameCollector,
    NONE_ACTION,
)
from diy_qi.config import GameConfig


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env = envs[0]
    agent_num = len(agents)
    frame_collector = FrameCollector(agent_num)

    while True:
        usr_conf = {
            "diy": {
                "monitor_side": 0,
                "monitor_label": "selfplay",
                "lineups": next(lineup_iter),
            }
        }

        _, state_dicts = env.reset(usr_conf=usr_conf)
        for i, agent in enumerate(agents):
            player_id = state_dicts[i]["player_id"]
            camp = state_dicts[i]["player_camp"]
            agent.reset(camp, player_id)
            agent.load_model(id="latest" if i == 0 else "common_ai")

        while True:
            actions = [NONE_ACTION] * agent_num
            for index, agent in enumerate(agents):
                actions[index] = agent.train_predict(state_dicts[index])

            frame_no, _, _, terminated, truncated, state_dicts = env.step(actions)

            if terminated or truncated:
                break
