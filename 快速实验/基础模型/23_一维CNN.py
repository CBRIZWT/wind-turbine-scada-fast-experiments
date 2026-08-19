# -*- coding: utf-8 -*-
"""23 一维CNN —— 卷积核=残差上升沿等局部形态的检测器"""
from torch import nn                                   # 神经网络层

from _torch import run_seq                             # 统一序列训练循环


class CNN(nn.Module):                                  # 一维卷积分类器
    def __init__(self, C, W):                          # C=通道数(87), W=窗宽(36)
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(C, 64, 5, padding=2), nn.ReLU(),   # 卷积核宽5(≈50min形态)→64通道
                                  nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),  # 第二层卷积64→64
                                  nn.AdaptiveAvgPool1d(1))                     # 全局平均池化聚合整窗证据
        self.head = nn.Linear(64, 1)                   # 分类头→logit

    def forward(self, x):                              # 输入 (B, W, C)
        return self.head(self.conv(x.transpose(1, 2)).squeeze(-1))  # 转成(B,C,W)卷积→压掉长度维→分类


run_seq("23_一维CNN", CNN)                             # 交统一序列训练循环
