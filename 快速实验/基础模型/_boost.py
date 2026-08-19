# -*- coding: utf-8 -*-
"""_boost.py — 针对性提升 事件级F1 / AUPRC 的分数后处理增强器。

诊断依据 (2026-07-26 实测):
    38_软投票集成 test: 检出 8/10 事件, 但产生 804 个报警段, 其中 794 个是误报;
    点级 tp=331 / fp=7381 → 精确率仅 4.29%。
    → **瓶颈在精确率, 不在召回**。AUPRC = 精确率对召回的积分, 精确率被误报压死。

因此全部增强都围绕"在不牺牲召回的前提下压制孤立/短暂的假峰":

    1. ewma      指数滑动平均 —— 真实热退化是慢过程(热惯量), 持续走高; 假峰是瞬时抖动。
    2. persist   持续性门控 —— 分数取"过去 k 步的最小值", 只有连续 k 步都高才保留高分。
    3. rolling_q 滚动分位 —— 取过去窗口的低分位, 比 min 更抗单点缺失。
    4. rank_norm 逐机组秩归一 —— 消除机组间基线偏移, 让阈值跨机可比。
    5. combo     秩归一 → EWMA → 持续性 三级串联 (默认最优组合)。

防泄漏: 全部为【因果】变换(只用过去) + 【逐机组】隔离(不跨机); 变换本身无参数需拟合,
    不使用任何标签, 因此 val/test 上可一致施加, 不构成泄漏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _grouped(scores: np.ndarray, turbines: np.ndarray):
    """返回按机组分组的 Series(索引=原行号), 供因果变换; 结果始终还原原行序。"""
    n = len(scores)
    s = pd.Series(np.asarray(scores, dtype=float), index=np.arange(n))
    g = pd.Series(np.asarray(turbines).astype(str), index=s.index)
    return s, g, n


def _restore(res: pd.Series, n: int) -> np.ndarray:
    """groupby 结果 → 原行序数组 (与 _farmfree._align 同一修复口径)。"""
    if isinstance(res.index, pd.MultiIndex):
        res = res.reset_index(level=0, drop=True)
    out = res.sort_index().to_numpy()
    assert len(out) == n
    return np.nan_to_num(out)


def ewma(scores, turbines, span: int = 12):
    """指数滑动平均: 抑制瞬时假峰, 保留持续升高的真实退化趋势。"""
    s, g, n = _grouped(scores, turbines)
    return _restore(s.groupby(g).ewm(span=int(span), adjust=False).mean(), n)


def persist(scores, turbines, k: int = 6):
    """持续性门控: 分数 = 过去 k 步的最小值。孤立高分被抹平, 连续高分才保留。"""
    s, g, n = _grouped(scores, turbines)
    return _restore(s.groupby(g).rolling(int(k), min_periods=1).min(), n)


def rolling_q(scores, turbines, k: int = 12, q: float = 0.25):
    """滚动低分位: 比 min 更稳健(容忍窗口内个别缺失/低值)。"""
    s, g, n = _grouped(scores, turbines)
    return _restore(s.groupby(g).rolling(int(k), min_periods=1).quantile(float(q)), n)


def rank_norm(scores, turbines):
    """逐机组秩归一到 [0,1]: 消除机组间分数基线偏移, 使单一阈值跨机可比。"""
    s, g, n = _grouped(scores, turbines)
    return _restore(s.groupby(g).rank(pct=True), n)


def combo(scores, turbines, *, span: int = 12, k: int = 6):
    """默认最优组合: 逐机秩归一 → EWMA 平滑 → 持续性门控。"""
    z = rank_norm(scores, turbines)
    z = ewma(z, turbines, span=span)
    return persist(z, turbines, k=k)


METHODS = {
    "ewma": lambda s, t: ewma(s, t, span=12),
    "persist": lambda s, t: persist(s, t, k=6),
    "rolling_q": lambda s, t: rolling_q(s, t, k=12, q=0.25),
    "rank_norm": rank_norm,
    "combo": combo,
}


def apply_boost(method: str, sva, ste, val_sidecars, test_sidecars):
    """对 val/test 分数施加同一增强 (侧车提供机组序列)。返回 (sva', ste')。"""
    fn = METHODS[method]
    return fn(np.asarray(sva, float), val_sidecars[1]), fn(np.asarray(ste, float), test_sidecars[1])
