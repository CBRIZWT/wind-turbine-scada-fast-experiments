# -*- coding: utf-8 -*-
"""35 PPO —— 近端策略优化: Actor-Critic + 比率裁剪(限制每次策略更新幅度, 更稳),
每批数据用旧策略采样后做4轮裁剪更新。诚实备注同32: 上下文老虎机。分数 = π(报警|s)。
"""
import numpy as np                                     # 抽样/分块
import torch                                           # 深度学习框架
from torch import nn                                   # 神经网络层

from _common import load_flat, now, report, standardize  # 统一数据/计时/评测/标准化
from _torch import DEV, seed                           # 设备与随机种子

ITERS, BATCH, GAMMA, CLIP = 500, 1024, 0.9, 0.2        # 迭代数/批大小/折扣/裁剪幅度


class AC(nn.Module):                                   # 演员评论家网络(同34)
    def __init__(self, d):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())  # 共享躯干
        self.pi, self.v = nn.Linear(64, 2), nn.Linear(64, 1)  # 演员头 + 评论家头

    def forward(self, x):
        h = self.body(x)                               # 共享特征
        return self.pi(h), self.v(h).squeeze(-1)       # 动作logits与状态值


seed()                                                 # 固定种子
Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 标准化
ipos, ineg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]  # 正/负样本索引
m = AC(Xtr.shape[1]).to(DEV)                           # 建模放到设备(输入维随farm动态)
opt = torch.optim.Adam(m.parameters(), lr=3e-4)        # Adam(PPO惯例低学习率)
rng = np.random.default_rng(0)                         # 固定随机源

t0 = now()                                             # 起始计时
for it in range(ITERS):                                # 逐迭代
    ids = np.concatenate([rng.choice(ipos, BATCH // 2), rng.choice(ineg, BATCH // 2)])  # 平衡采样
    s = torch.from_numpy(Xtr[ids]).to(DEV)             # 状态
    s2 = torch.from_numpy(Xtr[np.minimum(ids + 1, len(Xtr) - 1)]).to(DEV)  # 下一状态
    y = torch.from_numpy(ytr[ids].astype(np.int64)).to(DEV)  # 真标签
    with torch.no_grad():                              # 旧策略采样一批经验(冻结)
        logits_old, v = m(s)                           # 旧策略动作分布与状态值
        dist_old = torch.distributions.Categorical(logits=logits_old)
        a = dist_old.sample()                          # 采样动作
        logp_old = dist_old.log_prob(a)                # 旧策略对数概率
        r = torch.where(a == y, 1.0, -1.0)             # 奖励
        adv = r + GAMMA * m(s2)[1] - v                 # 一步TD优势
        ret = r + GAMMA * m(s2)[1]                     # 评论家回归目标
    for _ in range(4):                                 # 同一批上4轮裁剪更新(提高样本效率)
        logits, vv = m(s)                              # 新策略前向
        dist = torch.distributions.Categorical(logits=logits)
        ratio = (dist.log_prob(a) - logp_old).exp()    # 新旧策略概率比
        actor = -torch.min(ratio * adv, ratio.clamp(1 - CLIP, 1 + CLIP) * adv).mean()  # 裁剪代理目标
        loss = actor + 0.5 * ((vv - ret) ** 2).mean() - 0.01 * dist.entropy().mean()  # +评论家损失+熵正则
        opt.zero_grad(); loss.backward(); opt.step()   # 清梯度→反传→更新
    if it % 100 == 0:                                  # 每100迭代
        print(f"  iter {it}: 平均奖励={float(r.mean()):.3f}")  # 打印平均奖励


@torch.no_grad()                                       # 推理不建图
def prob(X):                                           # 分数 = π(报警|s)
    return np.concatenate([torch.softmax(m(torch.from_numpy(X[i:i + 16384]).to(DEV))[0], 1)[:, 1]  # 报警概率
                           .cpu().numpy() for i in range(0, len(X), 16384)])  # 分块→拼接


m.eval()                                               # 评估模式
report("35_PPO", yva, prob(Xva), yte, prob(Xte), now() - t0, extra={"范式": "强化学习-PPO"})  # 评测(RL家族最佳)
