# -*- coding: utf-8 -*-
"""30 变分自编码器 —— 无监督概率生成式; KL正则潜空间, 分数=重构误差"""
import numpy as np                                     # 数组/分块
import torch                                           # 深度学习框架
from torch import nn                                   # 神经网络层

from _common import load_flat, needs_external_train_scores, now, report, standardize  # 统一数据/计时/评测/标准化
from _torch import DEV, seed                           # 设备与随机种子

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 标准化
Xn = Xtr[ytr == 0]                                     # 只用正常样本(无监督)


class VAE(nn.Module):                                  # 变分自编码器
    def __init__(self, d=93, z=16):                    # d=输入维, z=潜维
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU())   # 编码器 93→64
        self.mu, self.lv = nn.Linear(64, z), nn.Linear(64, z)   # 潜分布的均值/对数方差两个头
        self.dec = nn.Sequential(nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, d))  # 解码器 z→64→93

    def forward(self, x):
        h = self.enc(x)                                # 编码
        mu, lv = self.mu(h), self.lv(h)                # 潜分布参数
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp() if self.training else mu  # 训练时重参数采样, 推理用均值
        return self.dec(z), mu, lv                     # 返回重构、均值、对数方差


seed()                                                 # 固定种子
m = VAE(d=Xtr.shape[1]).to(DEV)                        # 建模放到设备(输入维随farm动态)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)        # Adam优化器
rng = np.random.default_rng(0)                         # 固定随机源

t0 = now()                                             # 起始计时
for ep in range(10):                                   # 10轮训练
    for ids in np.array_split(rng.permutation(len(Xn)), len(Xn) // 4096):  # 打乱切4096一批
        x = torch.from_numpy(Xn[ids]).to(DEV)          # 该批正常样本
        xh, mu, lv = m(x)                              # 前向: 重构与潜参数
        kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean() / x.shape[1]  # KL散度(潜分布向N(0,1)靠)
        loss = ((xh - x) ** 2).mean() + 0.1 * kl       # 重构MSE + 0.1×KL正则(弱正则防后验坍塌)
        opt.zero_grad(); loss.backward(); opt.step()   # 清梯度→反传→更新


@torch.no_grad()                                       # 推理不建图
def err(X):
    out = []                                           # 收集每批分数
    for i in range(0, len(X), 16384):                  # 分块推理
        x = torch.from_numpy(X[i:i + 16384]).to(DEV)   # 该块
        out.append(((m(x)[0] - x) ** 2).mean(1).cpu().numpy())  # 重构误差=异常分数
    return np.concatenate(out)                         # 拼接


m.eval()                                               # 评估模式
report("30_变分自编码器", yva, err(Xva), yte, err(Xte), now() - t0,
       train_scores=err(Xn[::4]) if needs_external_train_scores() else None)  # 用重构误差评测
