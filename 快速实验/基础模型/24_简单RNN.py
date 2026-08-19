# -*- coding: utf-8 -*-
"""24 简单RNN —— 最基础的循环网络(Elman); 无门控, 长依赖易梯度消失, 作LSTM/GRU对照"""
from torch import nn                                   # 神经网络层

from _torch import run_seq                             # 统一序列训练循环


class RNN(nn.Module):                                  # Elman循环网络
    def __init__(self, C, W):                          # C=通道数, W=窗宽(未用)
        super().__init__()
        self.rnn = nn.RNN(C, 64, batch_first=True)     # 单层tanh循环, 隐藏64(与LSTM/GRU对齐)
        self.head = nn.Linear(64, 1)                   # 分类头→logit

    def forward(self, x):                              # 输入 (B, W, C)
        _, h = self.rnn(x)                             # 取最后时刻隐状态 h
        return self.head(h[-1])                        # 用末隐状态分类


run_seq("24_简单RNN", RNN)                             # 交统一序列训练循环
