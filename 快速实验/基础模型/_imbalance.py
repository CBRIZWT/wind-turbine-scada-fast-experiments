# -*- coding: utf-8 -*-
"""_imbalance.py — 针对 AUPRC 的类不平衡处理工具。

为什么普通 class_weight 对 AUPRC 帮助有限 (2026-07-26 实测认识):
    AUPRC 只看【分数排序】, 与阈值和类先验无关。给正类加权主要平移决策边界,
    若排序能力不变, AUPRC 也不变 —— 本项目实测: EWMA 后处理 (+19% AUPRC) 远比
    调类权重有效。类权重的真正作用是【防止模型把稀有正类当噪声直接忽略】。

本模块提供两种比 balanced 更对症的权重方案:

  1. event_balanced —— 事件级重加权 (推荐)
     问题: 正例点按【事件】聚簇 (每事件约 72 个连续点)。用 balanced 时,
           一个 72 点的长事件对损失的贡献是 36 点短事件的 2 倍 →
           模型被少数长事件主导, 泛化到新事件差。
     做法: 同一事件内的点共享总权重 1 (各点 1/事件长度), 再整体放大到与负类等总重。
           → 每个【事件】等权, 而非每个【点】等权。这与 event_f1/AUPRC 的评测单位一致。

  2. capped_balanced —— 截断类权重
     问题: 0.13% 正例率下 balanced 给正类权重约 740 倍, 梯度被极少数点支配,
           易过拟合到这些点的噪声。
     做法: 把正类权重上限截到 cap (默认 50), 兼顾"别忽略正类"与"别被少数点绑架"。
"""
from __future__ import annotations

import numpy as np


def _event_ids(y: np.ndarray, turbines=None) -> np.ndarray:
    """给每个正例点分配事件 id (连续正例段为一个事件); 负例为 -1。

    给定 turbines 时按机组隔离 (同一时刻不同机组不会被误并为一个事件)。
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    ids = np.full(n, -1, dtype=np.int64)
    if turbines is not None:
        tb = np.asarray(turbines).astype(str)
        new_block = np.empty(n, dtype=bool)
        new_block[0] = True
        new_block[1:] = tb[1:] != tb[:-1]
    else:
        new_block = np.zeros(n, dtype=bool)
        new_block[0] = True
    cur = -1
    prev_pos = False
    for i in range(n):
        if y[i] == 1:
            if not prev_pos or new_block[i]:
                cur += 1
            ids[i] = cur
            prev_pos = True
        else:
            prev_pos = False
    return ids


def event_balanced_weights(y, turbines=None) -> np.ndarray:
    """事件级重加权: 每个正例事件等权(而非每个正例点等权), 正类总重 = 负类总重。"""
    y = np.asarray(y).astype(int)
    w = np.ones(len(y), dtype=np.float64)
    pos = y == 1
    n_neg = int((~pos).sum())
    if pos.sum() == 0 or n_neg == 0:
        return w
    ids = _event_ids(y, turbines)
    _, inv, cnt = np.unique(ids[pos], return_inverse=True, return_counts=True)
    w[pos] = 1.0 / cnt[inv]                 # 同事件内均分, 每事件总重 = 1
    w[pos] *= n_neg / w[pos].sum()          # 正类总重放大到与负类等总重
    return w


def capped_balanced_weights(y, cap: float = 50.0) -> np.ndarray:
    """截断 balanced: 正类权重上限 cap, 防极端权重让梯度被少数点支配。"""
    y = np.asarray(y).astype(int)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.ones(len(y), dtype=np.float64)
    w = np.ones(len(y), dtype=np.float64)
    w[pos] = min(float(cap), n_neg / n_pos)
    return w


SCHEMES = {
    "event_balanced": event_balanced_weights,
    "capped_balanced": lambda y, turbines=None: capped_balanced_weights(y, cap=50.0),
}
