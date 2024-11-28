#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, attached

# 使用create_cls动态创建类
ObsData = create_cls("ObsData", feature=None, legal_action=None, lstm_cell=None, lstm_hidden=None)
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None, lstm_cell=None, lstm_hidden=None)
SampleData = create_cls("SampleData", npdata=None)

@attached
def sample_process(collector):
    return collector.sample_process()

@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata

@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)
