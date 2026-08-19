# -*- coding: utf-8 -*-
"""_torch.py — 深度模型共用训练循环 (扁平/序列一套代码)

统一做法: BCE损失+正例权重(治类不平衡) → 每epoch在val算AUPRC早停并回滚最优
→ val/test打分交给 _common.report。Windows教训: 不用多进程DataLoader。
"""
import copy                       # 深拷贝最优权重用

import numpy as np                # 数组/随机打乱
import torch                      # 深度学习框架
from sklearn.metrics import average_precision_score   # 早停监控指标(val AUPRC)
from torch import nn              # 神经网络层/损失

from _common import load_flat, load_seq, now, report, standardize, take_windows  # 复用统一数据与评测

DEV = "cuda" if torch.cuda.is_available() else "cpu"     # 有GPU用cuda, 否则cpu


def seed(k=0):
    """固定随机种子, 保证结果可复现"""
    torch.manual_seed(k)          # 固定torch随机数(参数初始化/dropout)
    np.random.seed(k)             # 固定numpy随机数(批次打乱)


def scores(model, batch_of, n, batch=4096):
    """分批推理 → P(报警) 概率数组。batch_of(ids)->(B,...)输入张量"""
    model.eval()                  # 评估模式(关dropout等)
    out = []                      # 收集每批概率
    with torch.no_grad():         # 推理不建计算图, 省显存
        for i in range(0, n, batch):                          # 按batch大小遍历全部样本
            out.append(torch.sigmoid(model(batch_of(np.arange(i, min(i + batch, n)))))  # logit过sigmoid得概率
                       .squeeze(-1).float().cpu().numpy())    # 去最后一维, 转float, 回CPU转numpy
    return np.concatenate(out)    # 拼成完整概率数组


def _run(name, model, tr, ytr, va, yva, te, yte, epochs, lr, batch, patience):
    """核心循环。tr/va/te 都是 batch_of(ids)->输入张量 的函数"""
    t0 = now()                                                # 起始计时
    opt = torch.optim.Adam(model.parameters(), lr=lr)         # Adam优化器
    pw = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), device=DEV).float()  # 正例权重=负/正样本数比
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)               # 带正例加权的二分类交叉熵(治不平衡)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(DEV)     # 训练标签放到设备
    best, state, bad = -1.0, None, 0                          # 最优val分数/最优权重/连续未提升计数
    rng = np.random.default_rng(0)                            # 固定随机源
    for ep in range(1, epochs + 1):                           # 逐epoch训练
        model.train()                                         # 训练模式
        for ids in np.array_split(rng.permutation(len(ytr)), max(len(ytr) // batch, 1)):  # 打乱后切成小批
            opt.zero_grad()                                   # 清空梯度
            loss = lossf(model(tr(ids)).squeeze(-1), yt[ids]) # 前向: 该批logit vs 标签算损失
            loss.backward()                                   # 反向传播
            opt.step()                                        # 更新参数
        ap = average_precision_score(yva, scores(model, va, len(yva)))  # 每epoch末在val算AUPRC
        print(f"  epoch {ep}: val_AUPRC={ap:.4f}")            # 打印进度
        if ap > best:                       # 只看val, 不碰test —— val提升则
            best, state, bad = ap, copy.deepcopy(model.state_dict()), 0  # 记录最优分数与权重, 计数清零
        elif (bad := bad + 1) >= patience:                   # 否则未提升计数+1, 达耐心值则早停
            break
    model.load_state_dict(state)                              # 回滚到val最优权重
    report(name, yva, scores(model, va, len(yva)), yte, scores(model, te, len(yte)),  # 交统一评测(test只此一次)
           now() - t0, extra={"device": DEV, "val_auprc": round(best, 4)})            # 记录设备与最优val AUPRC


def run_flat(name, build, epochs=15, lr=1e-3, batch=4096, patience=3):
    """扁平93维特征训练。build(输入维数)->模型(输出(B,1) logit)"""
    seed()                                                    # 固定种子
    Xtr, ytr, Xva, yva, Xte, yte = load_flat()                # 加载扁平数据
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)                # 标准化(train统计量)
    model = build(Xtr.shape[1]).to(DEV)                       # 按特征维数建模并放到设备
    b = lambda X: (lambda ids: torch.from_numpy(X[ids]).to(DEV))  # 造 batch_of: 按索引取行成张量
    _run(name, model, b(Xtr), ytr, b(Xva), yva, b(Xte), yte, epochs, lr, batch, patience)  # 进核心循环


def run_seq(name, build, epochs=6, lr=1e-3, batch=512, patience=2):
    """序列窗口(36步×87通道)训练。build(通道数, 窗宽)->模型(输入(B,W,C), 输出(B,1) logit)"""
    seed()                                                    # 固定种子
    d, W = load_seq()                                         # 加载序列数据与窗宽
    (btr, itr, ytr), (bva, iva, yva), (bte, ite, yte) = d["train"], d["val"], d["test"]  # 解包三split
    model = build(btr.shape[1], W).to(DEV)                    # 按通道数与窗宽建模
    b = lambda base, idx: (lambda ids: torch.from_numpy(take_windows(base, idx[ids], W)).to(DEV))  # 造惰性取窗batch_of
    _run(name, model, b(btr, itr), ytr, b(bva, iva), yva, b(bte, ite), yte,  # 进核心循环
         epochs, lr, batch, patience)
