# -*- coding: utf-8 -*-
"""41 自监督预测误差 —— 无标签: GRU 用前35步预测第36步(自监督, 监督信号来自数据本身),
只在正常窗上训练, 分数 = 一步预测误差。与重构AE的区别 = 预测范式(战役证明预测>重构)"""
import numpy as np                                     # 数组/分块
import torch                                           # 深度学习框架
from torch import nn                                   # 神经网络层

from _common import load_seq, needs_external_train_scores, now, report, take_windows  # 序列数据/计时/评测/取窗
from _torch import DEV, seed                           # 设备与随机种子


class Forecaster(nn.Module):                           # 一步预测器
    def __init__(self, C):                             # C=通道数
        super().__init__()
        self.rnn = nn.GRU(C, 64, batch_first=True)     # GRU编码历史
        self.out = nn.Linear(64, C)                    # 输出层→预测下一步C通道

    def forward(self, x):                              # (B, 35, C) → 预测第36步
        _, h = self.rnn(x)                             # 取末隐状态
        return self.out(h[-1])                         # 映射为下一步预测值


seed()                                                 # 固定种子
d, W = load_seq()                                      # 加载序列数据与窗宽
(btr, itr, ytr), (bva, iva, yva), (bte, ite, yte) = d["train"], d["val"], d["test"]  # 解包三split
idx_n = itr[ytr == 0][::2]                             # 只用正常窗(无监督/自监督)
m = Forecaster(btr.shape[1]).to(DEV)                   # 建模放到设备
opt = torch.optim.Adam(m.parameters(), lr=1e-3)        # Adam优化器
rng = np.random.default_rng(0)                         # 固定随机源

t0 = now()                                             # 起始计时
for ep in range(6):                                    # 6轮训练
    for ids in np.array_split(rng.permutation(len(idx_n)), len(idx_n) // 512):  # 打乱切512一批
        w = torch.from_numpy(take_windows(btr, idx_n[ids], W)).to(DEV)  # 惰性取窗
        loss = ((m(w[:, :-1]) - w[:, -1]) ** 2).mean()  # 用前35步预测第36步的MSE
        opt.zero_grad(); loss.backward(); opt.step()   # 清梯度→反传→更新


@torch.no_grad()                                       # 推理不建图
def err(base, idx):
    out = []                                           # 收集每批分数
    for i in range(0, len(idx), 2048):                 # 分块取窗推理
        w = torch.from_numpy(take_windows(base, idx[i:i + 2048], W)).to(DEV)  # 该批窗口
        out.append(((m(w[:, :-1]) - w[:, -1]) ** 2).mean(1).cpu().numpy())  # 一步预测误差=异常分数
    return np.concatenate(out)                         # 拼接


m.eval()                                               # 评估模式
report("41_自监督_预测误差", yva, err(bva, iva), yte, err(bte, ite), now() - t0,  # 用预测误差评测
       extra={"范式": "自监督(无标签)"},
       train_scores=err(btr, idx_n[::4]) if needs_external_train_scores() else None)  # 记录范式
