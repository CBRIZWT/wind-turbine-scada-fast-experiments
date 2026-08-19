# -*- coding: utf-8 -*-
"""79 液态神经网络 —— CfC(闭式连续时间)实现; ncps 未安装, 按 Hasani 2022 闭式解手写

与 LSTM/GRU 的本质区别: 门控 RNN 的时间常数是训练后固定的; 液态网络的时间常数
τ(x,h) 由当前输入现算 —— 同一个网络对不同样本、不同时刻用不同的"反应快慢"。
对本任务的意义: 温度残差在健康段几乎不动(需长 τ 积累缓漂), 在事件前会加速偏离
(需短 τ 快速响应), 固定时间常数的模型必须在两者间折中, 液态网络不必。

用 CfC 闭式解而非 LTC 数值 ODE: LTC 每步要跑 ODE 求解器, 慢且需 torchdiffeq;
CfC 有解析形式, 单步一次前向即可, 精度相当而速度可用于快速实验口径。
"""
import torch                                            # 张量/递推
from torch import nn                                    # 神经网络层
from torch.nn import functional as F                    # softplus/tanh

from _torch import run_seq                              # 统一序列训练循环


class CfCCell(nn.Module):                               # 闭式连续时间单元
    def __init__(self, in_dim, hid):
        super().__init__()
        z = in_dim + hid                                # 拼接维 = 输入 + 隐状态
        self.ff1 = nn.Linear(z, hid)                    # 慢分支 (t→∞ 的稳态)
        self.ff2 = nn.Linear(z, hid)                    # 快分支 (t→0 的初值)
        self.tau = nn.Linear(z, hid)                    # ★液态核心: 输入相关的时间常数
        # τ 与 A 的初值决定初始记忆长度。tau.bias=0 → softplus(0)=0.693 → gate=σ(-0.693)=0.33,
        # 隐态每步被大幅改写, 36步窗内有效记忆仅 3 步 (2026-08-09 实测剖面确认)。
        # 改为 bias=-3 → softplus(-3)≈0.049, 配合 A=1 → gate=σ(0.951)≈0.72 偏稳态分支, 记忆变长。
        nn.init.constant_(self.tau.bias, -3.0)          # 初始 τ 小 = 慢反应 = 长记忆
        self.A = nn.Parameter(torch.ones(hid))          # 相位偏置初值 1 (Hasani 2022 口径)

    def forward(self, x, h, dt=1.0):                    # x:(B,in) h:(B,hid)
        z = torch.cat([x, h], -1)                       # 拼接输入与状态
        f1 = torch.tanh(self.ff1(z))                    # 稳态目标
        f2 = torch.tanh(self.ff2(z))                    # 瞬时目标
        tau = F.softplus(self.tau(z)) + 1e-3            # τ>0; 由 (x,h) 现算 —— 每步每样本都不同
        gate = torch.sigmoid(-tau * dt + self.A)        # 闭式时间插值门 σ(-τ·Δt + A)
        return f1 * gate + f2 * (1.0 - gate)            # τ小→gate→σ(A)偏稳态(长记忆); τ大→gate→0偏瞬时


class LiquidNet(nn.Module):                             # 两层液态 RNN + 分类头
    def __init__(self, C, W, hid=64, n_layer=2):
        super().__init__()
        self.cells = nn.ModuleList(                     # 第一层吃原始通道, 其余层吃上层隐态
            CfCCell(C if i == 0 else hid, hid) for i in range(n_layer))
        self.hid, self.n_layer = hid, n_layer
        self.norm = nn.LayerNorm(hid)                   # 末端归一化(稳训练)
        self.head = nn.Linear(2 * hid, 1)               # 读出=[均值池化, 末刻] → logit

    def forward(self, x):                               # 输入 (B, W, C)
        B, W, _ = x.shape
        hs = [x.new_zeros(B, self.hid) for _ in range(self.n_layer)]  # 各层初始状态 h₀=0
        seq = []                                        # 收集顶层各时刻隐态(供池化读出)
        for t in range(W):                              # 沿时间因果递推(不看未来)
            inp = x[:, t]                               # 当前时刻输入
            for i, cell in enumerate(self.cells):       # 逐层更新
                hs[i] = cell(inp, hs[i])                # 液态单元前向
                inp = hs[i]                             # 本层隐态作为下层输入
            seq.append(hs[-1])                          # 记录顶层隐态
        h = self.norm(torch.stack(seq, 1))              # (B,W,hid) 归一化
        # 与 Mamba 同口径: 只读末刻会让 tanh 反复改写把早期信息冲掉(实测有效记忆 3 步),
        # 用 [整窗均值, 末刻] 拼接强制整窗参与; 窗口整体在标签之前, 因果性不变。
        return self.head(torch.cat([h.mean(1), h[:, -1]], -1))


run_seq("79_液态神经网络", LiquidNet)                      # 交统一序列训练循环
