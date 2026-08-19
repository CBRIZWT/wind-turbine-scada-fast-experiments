# -*- coding: utf-8 -*-
"""33 REINFORCE —— 策略型强化学习: 策略网络直接输出报警概率,
采样动作得奖励(判对+1/判错-1), 带基线的策略梯度 + 熵正则鼓励探索。
诚实备注同32: 转移与动作无关 → 上下文老虎机。评测分数 = π(报警|s)。
"""
import numpy as np                                     # 抽样/分块
import torch                                           # 深度学习框架
from torch import nn                                   # 神经网络层

from _common import load_flat, now, report, standardize  # 统一数据/计时/评测/标准化
from _torch import DEV, seed                           # 设备与随机种子

STEPS, BATCH = 2000, 1024                              # 训练步数/批大小

seed()                                                 # 固定种子
Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 标准化
ipos, ineg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]  # 正/负样本索引

D = Xtr.shape[1]                                       # 输入维随farm动态
m = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),  # 策略网络
                  nn.Linear(64, 2)).to(DEV)                                       # 输出2动作logits
opt = torch.optim.Adam(m.parameters(), lr=1e-3)        # Adam优化器
rng = np.random.default_rng(0)                         # 固定随机源

t0 = now()                                             # 起始计时
for step in range(STEPS):                              # 逐步训练
    ids = np.concatenate([rng.choice(ipos, BATCH // 2), rng.choice(ineg, BATCH // 2)])  # 平衡采样
    s = torch.from_numpy(Xtr[ids]).to(DEV)             # 状态
    y = torch.from_numpy(ytr[ids].astype(np.int64)).to(DEV)  # 真标签(算奖励)
    dist = torch.distributions.Categorical(logits=m(s))  # 策略给出的动作分布
    a = dist.sample()                                  # 按策略采样动作
    r = torch.where(a == y, 1.0, -1.0)                 # 奖励: 判对+1判错-1
    adv = r - r.mean()                                 # 批均值作基线降方差(优势)
    loss = -(dist.log_prob(a) * adv).mean() - 0.01 * dist.entropy().mean()  # 策略梯度 + 熵正则鼓励探索
    opt.zero_grad(); loss.backward(); opt.step()       # 清梯度→反传→更新
    if step % 500 == 0:                                # 每500步
        print(f"  step {step}: 平均奖励={float(r.mean()):.3f}")  # 打印平均奖励


@torch.no_grad()                                       # 推理不建图
def prob(X):                                           # 分数 = π(报警|s)
    return np.concatenate([torch.softmax(m(torch.from_numpy(X[i:i + 16384]).to(DEV)), 1)[:, 1]  # 报警概率
                           .cpu().numpy() for i in range(0, len(X), 16384)])  # 分块→拼接


m.eval()                                               # 评估模式
report("33_REINFORCE", yva, prob(Xva), yte, prob(Xte), now() - t0, extra={"范式": "强化学习-策略型"})  # 评测
