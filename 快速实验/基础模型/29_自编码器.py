# -*- coding: utf-8 -*-
"""29 自编码器 —— 无监督: 只学正常样本的压缩重构, 分数=重构误差"""
import numpy as np                                     # 数组/分块拼接
import torch                                           # 深度学习框架
from torch import nn                                   # 神经网络层

from _common import load_flat, needs_external_train_scores, now, report, standardize  # 统一数据/计时/评测/标准化
from _torch import DEV, seed                           # 设备与随机种子

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 标准化(仅train统计量)
Xn = Xtr[ytr == 0]                                     # 只用正常样本(无监督)

seed()                                                 # 固定随机种子
D = Xtr.shape[1]                                       # 输入维随farm动态(kel=93/pen=95/hot=59)
m = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(),   # 编码 D→64→16(瓶颈)
                  nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, D)).to(DEV)      # 解码 16→64→D, 放到设备
opt = torch.optim.Adam(m.parameters(), lr=1e-3)        # Adam优化器
rng = np.random.default_rng(0)                         # 固定随机源(批次打乱)

t0 = now()                                             # 起始计时
for ep in range(10):                                   # 10轮MSE重构训练
    for ids in np.array_split(rng.permutation(len(Xn)), len(Xn) // 4096):  # 打乱后切成4096一批
        x = torch.from_numpy(Xn[ids]).to(DEV)          # 该批正常样本
        loss = ((m(x) - x) ** 2).mean()                # 重构误差(MSE)
        opt.zero_grad(); loss.backward(); opt.step()   # 清梯度→反传→更新


@torch.no_grad()                                       # 推理不建图
def err(X):                                            # 逐样本重构误差=异常分数
    return np.concatenate([((m(torch.from_numpy(X[i:i + 16384]).to(DEV))       # 分块前向重构
                             - torch.from_numpy(X[i:i + 16384]).to(DEV)) ** 2)  # 与原输入求平方差
                           .mean(1).cpu().numpy() for i in range(0, len(X), 16384)])  # 每样本均值→拼接


m.eval()                                               # 评估模式
report("29_自编码器", yva, err(Xva), yte, err(Xte), now() - t0,
       train_scores=err(Xn[::4]) if needs_external_train_scores() else None)  # 用重构误差评测
