# -*- coding: utf-8 -*-
"""34 Actor-Critic —— 演员π(a|s)选动作 + 评论家V(s)估值,
一步优势 adv = r + γV(下一时刻) - V(s) 同时更新两头。
诚实备注同32: 转移与动作无关 → γV项只学到标签的时间自相关。评测分数 = π(报警|s)。
"""
import numpy as np                                     # 抽样/分块
import torch                                           # 深度学习框架
from torch import nn                                   # 神经网络层

from _common import load_flat, now, report, standardize  # 统一数据/计时/评测/标准化
from _torch import DEV, seed                           # 设备与随机种子

STEPS, BATCH, GAMMA = 2000, 1024, 0.9                  # 训练步数/批大小/折扣因子


class AC(nn.Module):                                   # 演员评论家网络(共享躯干)
    def __init__(self, d):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())  # 共享特征提取
        self.pi, self.v = nn.Linear(64, 2), nn.Linear(64, 1)  # 演员头(2动作) + 评论家头(状态值)

    def forward(self, x):
        h = self.body(x)                               # 共享特征
        return self.pi(h), self.v(h).squeeze(-1)       # 返回动作logits与状态值


seed()                                                 # 固定种子
Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 标准化
ipos, ineg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]  # 正/负样本索引
m = AC(Xtr.shape[1]).to(DEV)                           # 建模放到设备(输入维随farm动态)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)        # Adam优化器
rng = np.random.default_rng(0)                         # 固定随机源

t0 = now()                                             # 起始计时
for step in range(STEPS):                              # 逐步训练
    ids = np.concatenate([rng.choice(ipos, BATCH // 2), rng.choice(ineg, BATCH // 2)])  # 平衡采样
    s = torch.from_numpy(Xtr[ids]).to(DEV)             # 当前状态
    s2 = torch.from_numpy(Xtr[np.minimum(ids + 1, len(Xtr) - 1)]).to(DEV)  # 下一时刻状态
    y = torch.from_numpy(ytr[ids].astype(np.int64)).to(DEV)  # 真标签(算奖励)
    logits, v = m(s)                                   # 前向: 动作分布参数与状态值
    dist = torch.distributions.Categorical(logits=logits)  # 动作分布
    a = dist.sample()                                  # 采样动作
    r = torch.where(a == y, 1.0, -1.0)                 # 奖励: 判对+1判错-1
    with torch.no_grad():
        v2 = m(s2)[1]                                  # 下一状态值
    adv = r + GAMMA * v2 - v                           # 一步TD优势
    loss = (-(dist.log_prob(a) * adv.detach()).mean()  # 演员: 策略梯度(优势不回传评论家)
            + 0.5 * (adv ** 2).mean()                  # 评论家: TD误差
            - 0.01 * dist.entropy().mean())            # 熵正则鼓励探索
    opt.zero_grad(); loss.backward(); opt.step()       # 清梯度→反传→更新
    if step % 500 == 0:                                # 每500步
        print(f"  step {step}: 平均奖励={float(r.mean()):.3f}")  # 打印平均奖励


@torch.no_grad()                                       # 推理不建图
def prob(X):                                           # 分数 = π(报警|s)
    return np.concatenate([torch.softmax(m(torch.from_numpy(X[i:i + 16384]).to(DEV))[0], 1)[:, 1]  # 报警概率
                           .cpu().numpy() for i in range(0, len(X), 16384)])  # 分块→拼接


m.eval()                                               # 评估模式
report("34_ActorCritic", yva, prob(Xva), yte, prob(Xte), now() - t0,  # 评测
       extra={"范式": "强化学习-演员评论家"})           # 记录范式
