# -*- coding: utf-8 -*-
"""32 DQN —— 价值型强化学习: 状态=93维特征, 动作={0不报警,1报警}, 奖励=判对+1/判错-1

诚实备注: 报警不改变风机状态 → 转移与动作无关, MDP退化为上下文老虎机;
因此数据集本身即经验池(免去回放缓冲), 每步平衡采样状态、ε-贪婪选动作、
目标网络算 r+γ·maxQ(下一时刻)。评测分数 = Q(报警)-Q(不报警)。
"""
import numpy as np                                     # 抽样/分块
import torch                                           # 深度学习框架
from torch import nn                                   # 神经网络层

from _common import load_flat, now, report, standardize  # 统一数据/计时/评测/标准化
from _torch import DEV, seed                           # 设备与随机种子

STEPS, BATCH, GAMMA = 4000, 256, 0.9                   # 训练步数/批大小/折扣因子

seed()                                                 # 固定种子
Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 标准化
ipos, ineg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]  # 正/负样本索引(用于平衡采样)

D = Xtr.shape[1]                                       # 输入维随farm动态
qnet = lambda: nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),  # Q网络工厂
                             nn.Linear(64, 2)).to(DEV)                                       # 输出2个动作的Q值
m, tgt = qnet(), qnet()                                # 在线网络 + 目标网络
tgt.load_state_dict(m.state_dict())                    # 目标网络初始同步
opt = torch.optim.Adam(m.parameters(), lr=1e-3)        # Adam优化器
rng = np.random.default_rng(0)                         # 固定随机源

t0 = now()                                             # 起始计时
for step in range(STEPS):                              # 逐步训练
    ids = np.concatenate([rng.choice(ipos, BATCH // 2), rng.choice(ineg, BATCH // 2)])  # 正负各半平衡采样
    s = torch.from_numpy(Xtr[ids]).to(DEV)             # 当前状态
    s2 = torch.from_numpy(Xtr[np.minimum(ids + 1, len(Xtr) - 1)]).to(DEV)  # 下一时刻状态(时间+1)
    y = torch.from_numpy(ytr[ids].astype(np.int64)).to(DEV)  # 真标签(用于算奖励)
    eps = 0.5 - 0.45 * step / STEPS                    # ε 0.5→0.05 线性退火(探索递减)
    with torch.no_grad():
        a = m(s).argmax(1)                             # 贪婪动作
    a = torch.where(torch.rand(BATCH, device=DEV) < eps,       # 以ε概率
                    torch.randint(0, 2, (BATCH,), device=DEV), a)  # 随机探索, 否则贪婪
    r = torch.where(a == y, 1.0, -1.0)                 # 奖励: 判对+1判错-1
    with torch.no_grad():
        target = r + GAMMA * tgt(s2).max(1).values     # 贝尔曼目标 r+γ·maxQ(s')
    q = m(s).gather(1, a.unsqueeze(1)).squeeze(1)       # 当前动作的Q值
    loss = nn.functional.smooth_l1_loss(q, target)     # Huber损失拟合目标
    opt.zero_grad(); loss.backward(); opt.step()       # 清梯度→反传→更新
    if step % 500 == 0:                                # 每500步
        tgt.load_state_dict(m.state_dict())            # 同步目标网络
        print(f"  step {step}: eps={eps:.2f} loss={float(loss):.4f}")  # 打印进度


@torch.no_grad()                                       # 推理不建图
def qdiff(X):                                          # 分数 = Q(报警) - Q(不报警)
    out = []                                           # 收集
    for i in range(0, len(X), 16384):                  # 分块推理
        q = m(torch.from_numpy(X[i:i + 16384]).to(DEV))  # 该块Q值
        out.append((q[:, 1] - q[:, 0]).cpu().numpy())  # 报警优势=异常分数
    return np.concatenate(out)                         # 拼接


m.eval()                                               # 评估模式
report("32_DQN", yva, qdiff(Xva), yte, qdiff(Xte), now() - t0, extra={"范式": "强化学习-价值型"})  # 评测
