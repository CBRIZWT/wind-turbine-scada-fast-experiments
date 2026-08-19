# -*- coding: utf-8 -*-
"""26 GRU —— LSTM的轻量版, 参数少1/4, 短窗口常持平或更好"""
from torch import nn                                   # 神经网络层

from _torch import run_seq                             # 统一序列训练循环


class GRU(nn.Module):                                  # 门控循环单元(2门)
    def __init__(self, C, W):                          # C=通道数, W=窗宽(未用)
        super().__init__()
        self.rnn = nn.GRU(C, 64, batch_first=True)     # 单层GRU(更新门+重置门), 隐藏64
        self.head = nn.Linear(64, 1)                   # 分类头→logit

    def forward(self, x):                              # 输入 (B, W, C)
        _, h = self.rnn(x)                             # 取最后时刻隐状态 h
        return self.head(h[-1])                        # 用末隐状态分类


run_seq("26_GRU", GRU)                                 # 交统一序列训练循环(本任务亚军)
