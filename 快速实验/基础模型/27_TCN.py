# -*- coding: utf-8 -*-
"""27 TCN —— 因果膨胀卷积(膨胀1/2/4/8, 感受野31步≈窗宽); 左填充保证不偷看未来"""
from torch import nn                                   # 神经网络层
from torch.nn import functional as F                   # 函数式接口(pad/relu)

from _torch import run_seq                             # 统一序列训练循环


class Block(nn.Module):                                # 单个膨胀因果卷积残差块
    def __init__(self, cin, cout, d):                  # cin/cout=输入/输出通道, d=膨胀率
        super().__init__()
        self.pad = 2 * d                               # (k-1)*膨胀 的因果左填充(k=3)
        self.conv = nn.Conv1d(cin, cout, 3, dilation=d)  # 膨胀卷积核3
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()  # 残差支路(通道变则1x1对齐)

    def forward(self, x):                              # 输入 (B, C, W)
        return F.relu(self.conv(F.pad(x, (self.pad, 0))) + self.res(x))  # 仅左填充(不看未来)+残差+ReLU


class TCN(nn.Module):                                  # 时序卷积网络
    def __init__(self, C, W):                          # C=通道数, W=窗宽
        super().__init__()
        self.blocks = nn.Sequential(Block(C, 64, 1), Block(64, 64, 2),   # 膨胀1→2
                                    Block(64, 64, 4), Block(64, 64, 8))  # →4→8, 感受野≈31步
        self.head = nn.Linear(64, 1)                   # 分类头→logit

    def forward(self, x):                              # 输入 (B, W, C)
        return self.head(self.blocks(x.transpose(1, 2))[:, :, -1])  # 转(B,C,W)过块→取窗口末时刻(因果对齐标签)分类


run_seq("27_TCN", TCN)                                 # 交统一序列训练循环
