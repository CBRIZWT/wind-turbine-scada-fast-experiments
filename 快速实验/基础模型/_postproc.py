# -*- coding: utf-8 -*-
"""_postproc.py — 分数后处理增强层: 针对性提升 event_F1 与 AUPRC。

诊断依据 (2026-07-26 归因分析):
    冠军 38_软投票集成 在 test 上检出 8/10 事件, 但产生 804 个报警段其中 794 个是误报,
    点精确率仅 4.29%。→ **瓶颈是精确率, 不是召回**。AUPRC = 精确率对召回的积分,
    故降低误报是提升 AUPRC / event_F1 的唯一有效方向。

三种针对性方法 (全部为因果、逐机组、只用过去):
    1. EWMA 平滑     —— 抑制孤立毛刺; 真实退化是持续过程, 噪声是瞬时的
    2. 持续性门控     —— 要求连续 K 步高分才算报警(滚动最小值), 直接杀掉单点误报
    3. 共形健康分位阈 —— 用 train 健康段分数分布定校准阈值, 给出误报率的形式化保证

超参 (span / K / alpha) 一律【只在 val 上】按 event_F1 选优, test 只评一次。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _grp(scores: np.ndarray, turbines: np.ndarray) -> pd.core.groupby.SeriesGroupBy:
    s = pd.Series(np.asarray(scores, dtype=float), index=np.arange(len(scores)))
    return s.groupby(pd.Series(np.asarray(turbines).astype(str), index=s.index))


def _restore(res: pd.Series, n: int) -> np.ndarray:
    """还原原始行序 (groupby 结果按组键排序, 必须按原索引还原)。"""
    if isinstance(res.index, pd.MultiIndex):
        res = res.reset_index(level=0, drop=True)
    out = res.sort_index().to_numpy()
    assert len(out) == n
    return out


def ewma_smooth(scores: np.ndarray, turbines: np.ndarray, span: int) -> np.ndarray:
    """逐机组因果 EWMA 平滑。span=1 时退化为原分数。"""
    if span is None or int(span) <= 1:
        return np.asarray(scores, dtype=float)
    n = len(scores)
    return _restore(_grp(scores, turbines).ewm(span=int(span), adjust=False).mean(), n)


def persistence_gate(scores: np.ndarray, turbines: np.ndarray, k: int) -> np.ndarray:
    """持续性门控: 分数 ← 过去 K 步的滚动最小值。

    只有连续 K 步都高, 输出才高 → 单点尖峰被压掉。K=1 时退化为原分数。
    """
    if k is None or int(k) <= 1:
        return np.asarray(scores, dtype=float)
    n = len(scores)
    r = _grp(scores, turbines).rolling(int(k), min_periods=1).min()
    return _restore(r, n)


def conformal_quantile(calib_scores: np.ndarray, alpha: float = 0.01) -> float:
    """共形健康分位阈: 在健康校准分数上取 (1-alpha) 分位。

    保证在校准分布下误报率 ≈ alpha (分布无关的形式化保证)。
    """
    c = np.asarray(calib_scores, dtype=float)
    c = c[np.isfinite(c)]
    if c.size == 0:
        return float("inf")
    return float(np.quantile(c, 1.0 - float(alpha)))


# 超参搜索空间 (快速实验口径, 保持轻量)
EWMA_SPANS = (1, 6, 18, 36, 72, 144)
PERSIST_KS = (1, 3, 6, 12, 18)


def tune_on_val(val_scores, val_labels, val_turbines, eval_fn,
                spans=EWMA_SPANS, ks=PERSIST_KS):
    """在 val 上网格搜索 (span, K), 按 eval_fn 返回的 event_F1 选优。

    eval_fn(scores) -> float (越大越好)。只用 val, 不碰 test。
    返回 (best_span, best_k, best_score, 全部记录)。
    """
    best = (1, 1, -np.inf)
    records = []
    for sp in spans:
        sm = ewma_smooth(val_scores, val_turbines, sp)
        for k in ks:
            g = persistence_gate(sm, val_turbines, k)
            v = float(eval_fn(g))
            records.append({"span": sp, "k": k, "val_event_f1": v})
            if v > best[2]:
                best = (sp, k, v)
    return best[0], best[1], best[2], records


def apply(scores, turbines, span: int, k: int) -> np.ndarray:
    """按选定超参施加后处理 (先 EWMA 后门控, 与 tune_on_val 顺序一致)。"""
    return persistence_gate(ewma_smooth(scores, turbines, span), turbines, k)
