# -*- coding: utf-8 -*-
"""25 LSTM —— 门控记忆保持缓慢升温趋势"""
from torch import nn                                   # 神经网络层

from _torch import run_seq                             # 统一序列训练循环


class LSTM(nn.Module):                                 # 长短期记忆网络
    def __init__(self, C, W):                          # C=通道数, W=窗宽(未用)
        super().__init__()
        self.rnn = nn.LSTM(C, 64, batch_first=True)    # 单层LSTM(三门控), 隐藏64
        self.head = nn.Linear(64, 1)                   # 分类头→logit

    def forward(self, x):                              # 输入 (B, W, C)
        _, (h, _) = self.rnn(x)                        # 取最后时刻隐状态 h(丢弃细胞状态)
        return self.head(h[-1])                        # 用末隐状态分类


run_seq("25_LSTM", LSTM)                               # 交统一序列训练循环
