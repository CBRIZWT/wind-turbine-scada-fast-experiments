# -*- coding: utf-8 -*-
"""22 多层感知机 —— 基础深度学习; 与树模型吃同一份93维特征, ML→NN的桥"""
from torch import nn                                   # 神经网络层

from _torch import run_flat                            # 统一扁平特征训练循环(标准化/加权BCE/早停/评测)

build = lambda d: nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),   # 93→256+ReLU+Dropout0.2
                                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),  # 256→128+ReLU+Dropout0.2
                                nn.Linear(128, 1))                                # 128→1 输出logit
run_flat("22_MLP", build)                              # 交统一训练循环(默认15epoch/lr1e-3)
