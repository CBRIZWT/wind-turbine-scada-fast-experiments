# -*- coding: utf-8 -*-
"""_farmfree.py — 富"农场无关"表示 (channel-count independent representation)。

动机 (2026-07-26): 46_零样本_跨场迁移 排名 51/51 (AUPRC 0.0046)。其表示只有 6 维,
    无法区分"跨场迁移本身不可行"与"6 维太薄"。本模块把农场无关特征扩到 ~26 维,
    作为【预训练/微调/迁移/联邦】四大范式的统一输入 —— 因为 kel 87 / pen 89 / hot 53
    通道数不一致, 跨域方法必须先落到与通道数无关的公共表示上。

设计约束 (与主线口径一致, 不放松):
    · 逐机组 (groupby _turbine) 计算, 窗口不跨机组;
    · 仅用过去 (rolling/shift 均为因果), 无未来泄漏;
    · 不使用任何标签 (label-free), 故可在 train/val/test 上一致计算;
    · 结果按 idx_flat 对齐到扁平评测索引, 与 y_flat 逐行对应。

特征构成 (26 维):
    A. 跨通道横截面统计 (7): max / mean / std / q90 / q10 / 正残差能量 / 超阈通道数
    B. 主序列(跨通道max)的因果滚动统计 (12): 3 个窗口 × {mean, max, std, 相对增量}
    C. EWMA 与趋势 (7): 3 档 EWMA + EWMA 偏离 + 短/中/长斜率
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 窗口配方与主线预处理 S7-C 对齐 (10min 采样): 6步=1h, 72步=12h, 144步=24h
WINDOWS = (6, 72, 144)
EWMA_SPANS = (6, 72, 144)
SIGMA_THRESH = 2.0
N_FEATURES = 26


def _cross_channel(X: np.ndarray) -> np.ndarray:
    """(T,C) → (T,7) 跨通道横截面统计; 与通道数 C 无关。"""
    return np.column_stack([
        X.max(axis=1),                                   # 最热通道
        X.mean(axis=1),                                  # 整机平均偏离
        X.std(axis=1),                                   # 通道间离散度
        np.quantile(X, 0.90, axis=1),                    # 高分位(抗单通道毛刺)
        np.quantile(X, 0.10, axis=1),                    # 低分位
        np.mean(np.maximum(0.0, X) ** 2, axis=1),        # 正残差能量(只罚偏热)
        (X > SIGMA_THRESH).sum(axis=1).astype(np.float32),  # 超阈通道计数
    ]).astype(np.float32)


def _align(res: pd.Series, n: int) -> np.ndarray:
    """把 groupby 结果还原成【原始行序】的一维数组。

    [BUG 修复 2026-07-26] groupby().rolling()/.ewm() 返回 (组键, 原索引) 的 MultiIndex,
    且按【组键排序】。此前直接 .to_numpy() 只在"机组块连续且按键有序"时恰好正确;
    机组交错时会整体错位 → 跨机组串扰。现显式按原索引排序还原, 与行序无关。
    """
    if isinstance(res.index, pd.MultiIndex):
        res = res.reset_index(level=0, drop=True)
    out = res.sort_index().to_numpy()
    assert len(out) == n, f"行序还原后长度不符: {len(out)} != {n}"
    return out


def _causal_temporal(main: np.ndarray, turb: np.ndarray) -> np.ndarray:
    """主序列 → (T,19) 因果滚动/EWMA/趋势; 逐机组分组, 只用过去。"""
    n = len(main)
    s = pd.Series(main, index=np.arange(n))
    g = s.groupby(pd.Series(turb, index=s.index))
    cols = []
    for w in WINDOWS:
        r = g.rolling(w, min_periods=1)
        mean = _align(r.mean(), n)
        mx = _align(r.max(), n)
        sd = np.nan_to_num(_align(r.std(), n))          # 首行 std=NaN → 0
        cols += [mean, mx, sd, main - mean]             # 4 × 3窗 = 12
    for sp in EWMA_SPANS:
        cols.append(_align(g.ewm(span=sp, adjust=False).mean(), n))   # 3
    cols.append(main - cols[-1])                                      # EWMA(长)偏离  1
    for lag in (6, 72, 144):
        prev = _align(g.shift(lag), n)
        cols.append(main - np.nan_to_num(prev, nan=main[0]))          # 斜率 3
    return np.nan_to_num(np.column_stack(cols)).astype(np.float32)


def farmfree_features(X: np.ndarray, turbines: np.ndarray) -> np.ndarray:
    """(T,C) 残差矩阵 + 机组序列 → (T,26) 农场无关表示。C 可为 87/89/53 任意值。"""
    X = np.asarray(X, dtype=np.float32)
    cross = _cross_channel(X)
    temporal = _causal_temporal(cross[:, 0].astype(np.float64), np.asarray(turbines).astype(str))
    out = np.column_stack([cross, temporal]).astype(np.float32)
    assert out.shape[1] == N_FEATURES, f"农场无关特征维度应为 {N_FEATURES}, 实际 {out.shape[1]}"
    return out


def load_farmfree(farm: str, split: str, *, variant=None, aligned: str = "flat") -> np.ndarray:
    """读某 farm 某 split 的农场无关特征, 带磁盘缓存。

    aligned="flat" → 按 idx_flat 对齐 (与 y_flat 逐行对应, 供扁平模型/评测);
    aligned="base" → 全行 (供序列模型自行取窗)。
    """
    from _common import quick_data_dir                      # 延迟导入避免循环
    d = quick_data_dir(farm, variant)
    cache = d / f"farmfree26_{split}.npy"
    if cache.exists():
        base_feat = np.load(cache)
    else:
        X = np.load(d / f"X_base_{split}.npy", mmap_mode="r")
        turb = np.load(d / f"turbines_base_{split}.npy")
        base_feat = farmfree_features(np.asarray(X), turb)
        np.save(cache, base_feat)
    if aligned == "base":
        return base_feat
    idx = np.load(d / f"idx_flat_{split}.npy")
    return base_feat[idx]


def load_farmfree_xy(farm: str, *, variant=None):
    """便捷接口: 返回 (Ftr,ytr, Fva,yva, Fte,yte) 农场无关特征 + 扁平标签。"""
    from _common import quick_data_dir
    d = quick_data_dir(farm, variant)
    y = lambda s: np.load(d / f"y_flat_{s}.npy").astype(int)
    return (load_farmfree(farm, "train", variant=variant), y("train"),
            load_farmfree(farm, "val", variant=variant), y("val"),
            load_farmfree(farm, "test", variant=variant), y("test"))
