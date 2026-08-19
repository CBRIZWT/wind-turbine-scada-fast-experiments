# -*- coding: utf-8 -*-
"""28 Transformer —— 小编码器(2层/d64/4头): 自注意力捕多通道联动与长程依赖 (本任务冠军)"""
import torch                                           # 张量/参数
from torch import nn                                   # 神经网络层

from _torch import run_seq                             # 统一序列训练循环


class TransformerNet(nn.Module):                       # Transformer编码器分类器
    def __init__(self, C, W, d=64):                    # C=通道数, W=窗宽, d=嵌入维
        super().__init__()
        self.proj = nn.Linear(C, d)                    # 把87通道线性投影到d=64
        self.pos = nn.Parameter(torch.zeros(1, W, d))  # 可学习位置编码(保留时序顺序)
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=128,   # 单层: 4头自注意力+FFN128
                                           dropout=0.1, batch_first=True)     # Dropout0.1防过拟合
        self.enc = nn.TransformerEncoder(layer, 2)     # 堆叠2层编码器(小模型防过拟合)
        self.head = nn.Linear(d, 1)                    # 分类头→logit

    def forward(self, x):                              # 输入 (B, W, C)
        return self.head(self.enc(self.proj(x) + self.pos).mean(dim=1))  # 投影+位置编码→自注意力→时间平均池化→分类


run_seq("28_Transformer", TransformerNet, lr=5e-4)     # 交统一训练循环; 学习率降至5e-4稳住注意力训练
