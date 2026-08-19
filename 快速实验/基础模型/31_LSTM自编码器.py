# -*- coding: utf-8 -*-
"""31 LSTM自编码器 —— 无监督序列重构: 编码6h窗→潜向量→解码重构, 分数=窗口重构误差"""
import numpy as np                                     # 数组/分块
import torch                                           # 深度学习框架
from torch import nn                                   # 神经网络层

from _common import load_seq, needs_external_train_scores, now, report, take_windows  # 序列数据/计时/评测/取窗
from _torch import DEV, seed                           # 设备与随机种子


class LSTMAE(nn.Module):                               # 序列自编码器
    def __init__(self, C, W, h=64):                    # C=通道数, W=窗宽, h=隐藏维
        super().__init__()
        self.W = W                                     # 记住窗宽(解码展开用)
        self.enc = nn.LSTM(C, h, batch_first=True)     # 编码LSTM
        self.dec = nn.LSTM(h, h, batch_first=True)     # 解码LSTM
        self.out = nn.Linear(h, C)                     # 输出层→重构C通道

    def forward(self, x):                              # (B, W, C)
        _, (hn, _) = self.enc(x)                       # 编码取末隐状态作潜向量
        z = hn[-1].unsqueeze(1).repeat(1, self.W, 1)   # 潜向量沿时间复制W份(展开)
        return self.out(self.dec(z)[0])                # 解码并映射回C通道重构序列


seed()                                                 # 固定种子
d, W = load_seq()                                      # 加载序列数据与窗宽
(btr, itr, ytr), (bva, iva, yva), (bte, ite, yte) = d["train"], d["val"], d["test"]  # 解包三split
idx_n = itr[ytr == 0][::2]                             # 只用正常窗(隔2下采样, ~4万)
m = LSTMAE(btr.shape[1], W).to(DEV)                    # 建模放到设备
opt = torch.optim.Adam(m.parameters(), lr=1e-3)        # Adam优化器
rng = np.random.default_rng(0)                         # 固定随机源

t0 = now()                                             # 起始计时
for ep in range(6):                                    # 6轮训练
    for ids in np.array_split(rng.permutation(len(idx_n)), len(idx_n) // 512):  # 打乱切512一批
        x = torch.from_numpy(take_windows(btr, idx_n[ids], W)).to(DEV)  # 惰性取该批窗口
        loss = ((m(x) - x) ** 2).mean()                # 整窗重构MSE
        opt.zero_grad(); loss.backward(); opt.step()   # 清梯度→反传→更新


@torch.no_grad()                                       # 推理不建图
def err(base, idx):
    out = []                                           # 收集每批分数
    for i in range(0, len(idx), 2048):                 # 分块取窗推理
        x = torch.from_numpy(take_windows(base, idx[i:i + 2048], W)).to(DEV)  # 该批窗口
        out.append(((m(x) - x) ** 2).mean((1, 2)).cpu().numpy())  # 整窗重构误差=异常分数
    return np.concatenate(out)                         # 拼接


m.eval()                                               # 评估模式
report("31_LSTM自编码器", yva, err(bva, iva), yte, err(bte, ite), now() - t0,
       train_scores=err(btr, idx_n[::4]) if needs_external_train_scores() else None)  # 用窗口重构误差评测
