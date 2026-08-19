# -*- coding: utf-8 -*-
"""78 Mamba —— 选择性状态空间模型(S6); 纯PyTorch实现(mamba-ssm 需编CUDA核, Windows不可靠)

核心与Transformer的区别: 注意力是O(W²)全对全比较, SSM是O(W)因果递推 —— 状态 h 沿时间
单向累积, 天然不偷看未来。Mamba相对S4的关键改动是"选择性": Δ/B/C 由当前输入生成,
使模型能按内容决定"记住多久" —— 对温度残差这种"平时稳、异常前缓慢漂移"的信号,
Δ小=长记忆(积累缓漂), Δ大=快遗忘(忽略瞬时噪声)。
"""
import torch                                            # 张量/递推
from torch import nn                                    # 神经网络层
from torch.nn import functional as F                    # softplus/silu/pad

from _torch import run_seq                              # 统一序列训练循环


class S6(nn.Module):                                    # 选择性扫描核心 (Mamba 的 S6 层)
    def __init__(self, D, N=16):                        # D=内部通道, N=状态维
        super().__init__()
        self.D, self.N = D, N
        A = torch.arange(1, N + 1, dtype=torch.float32).repeat(D, 1)  # HiPPO式初始化 A=-(1..N)
        self.A_log = nn.Parameter(torch.log(A))         # 存log保证 A=-exp(A_log)<0 (稳定衰减)
        self.D_skip = nn.Parameter(torch.ones(D))       # 直通残差项 (y += D*x)
        self.x_proj = nn.Linear(D, 1 + 2 * N, bias=False)  # 由输入生成 Δ/B/C —— "选择性"所在
        # Δ 初始化必须让 softplus(dt_bias) 落在 [0.001,0.1] (官方 Mamba 口径)。
        # 若用 zeros → softplus(0)=0.693 → dA=exp(-0.693·A) 每步衰减过半, 36步窗内
        # 有效记忆仅 2 步, Mamba 退化成短窗模型 (2026-08-09 实测剖面确认过此坑)。
        dt = torch.exp(torch.rand(D) * (torch.log(torch.tensor(0.1))
                                        - torch.log(torch.tensor(1e-3)))
                       + torch.log(torch.tensor(1e-3)))          # 对数均匀采样 Δ∈[1e-3,0.1]
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # softplus 的逆

    def forward(self, x):                               # x:(B,W,D)
        Bsz, W, D = x.shape
        A = -torch.exp(self.A_log)                      # (D,N) 负实部 → 状态指数衰减
        dbc = self.x_proj(x)                            # (B,W,1+2N) 输入相关的 Δ/B/C
        dt, Bm, Cm = dbc[..., :1], dbc[..., 1:1 + self.N], dbc[..., 1 + self.N:]  # 切三段
        dt = F.softplus(dt + self.dt_bias)              # (B,W,D) 步长必须为正; 小=长记忆, 大=快遗忘
        dA = torch.exp(dt.unsqueeze(-1) * A)            # (B,W,D,N) 零阶保持离散化 Ā=exp(ΔA)
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)  # (B,W,D,N) B̄x=ΔB·x
        h = x.new_zeros(Bsz, D, self.N)                 # 初始状态 h₀=0
        ys = []                                         # 逐步输出
        for t in range(W):                              # 因果递推 (W=36 步, 短窗直接循环)
            h = dA[:, t] * h + dBx[:, t]                # hₜ = Ā hₜ₋₁ + B̄ xₜ
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))  # yₜ = C hₜ (对状态维求和)
        return torch.stack(ys, 1) + self.D_skip * x     # (B,W,D) 加直通项


class Block(nn.Module):                                 # Mamba 块: 门控 + 因果卷积 + S6
    def __init__(self, d_model, d_inner, N=16, k=4):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_inner)  # 一次投影出 主支x 与 门控z
        self.conv = nn.Conv1d(d_inner, d_inner, k, groups=d_inner)  # 深度可分离因果卷积(局部特征)
        self.pad = k - 1                                # 左填充量 = k-1, 只看过去
        self.ssm = S6(d_inner, N)                       # 选择性状态空间
        self.out_proj = nn.Linear(d_inner, d_model)     # 投影回模型维
        self.norm = nn.LayerNorm(d_model)               # 前置归一化

    def forward(self, x):                               # x:(B,W,d_model)
        r = x                                           # 残差支
        x, z = self.in_proj(self.norm(x)).chunk(2, -1)  # 归一化后分成 主支/门控支
        x = self.conv(F.pad(x.transpose(1, 2), (self.pad, 0)))[:, :, :x.shape[1]]  # 仅左填充=因果
        x = F.silu(x.transpose(1, 2))                   # 转回(B,W,D)并激活
        x = self.ssm(x) * F.silu(z)                     # S6 输出被门控支调制 (SiLU门)
        return r + self.out_proj(x)                     # 残差相加


class Mamba(nn.Module):                                 # 两层 Mamba + 分类头
    def __init__(self, C, W, d_model=64, expand=2, n_layer=2):
        super().__init__()
        self.inp = nn.Linear(C, d_model)                # 87通道 → d_model
        self.blocks = nn.ModuleList(Block(d_model, d_model * expand) for _ in range(n_layer))
        self.norm = nn.LayerNorm(d_model)               # 末端归一化
        self.head = nn.Linear(2 * d_model, 1)           # 读出=[均值池化, 末刻] → logit

    def forward(self, x):                               # 输入 (B, W, C)
        h = self.inp(x)                                 # 升维
        for b in self.blocks:                           # 逐块
            h = b(h)
        h = self.norm(h)
        # 读出用 [整窗均值, 末刻] 拼接而非只取末刻: Block 的残差直通会把末刻原始输入
        # 原样送到输出(实测末刻敏感度=扰动幅度), 只读末刻则历史贡献被淹没, 36步窗内
        # 有效记忆退化到 3 步。均值池化让整窗强制参与, 且窗口整体在标签之前, 不破坏因果。
        return self.head(torch.cat([h.mean(1), h[:, -1]], -1))


run_seq("78_Mamba", Mamba)                              # 交统一序列训练循环
