# -*- coding: utf-8 -*-
"""真实故障快速实验的逐机组数据派生纯函数。"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def sort_by_turbine_time(
    X: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    turbines: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按机组、时间稳定排序，返回排序后的数组及原行号。"""
    X = np.asarray(X)
    labels = np.asarray(labels)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    turbines = np.asarray(turbines).astype(str)
    n = len(X)
    if not (len(labels) == len(timestamps) == len(turbines) == n):
        raise ValueError("X/labels/timestamps/turbines 必须等长")
    # np.lexsort 最后一个键为主键：先 turbine，再 timestamp；原行号保证完全稳定。
    order = np.lexsort((np.arange(n, dtype=np.int64), timestamps, turbines))
    return X[order], labels[order], timestamps[order], turbines[order], order


def per_turbine_causal_features(
    X: np.ndarray,
    turbines: np.ndarray,
    *,
    w_feat: int = 72,
    k_recent: int = 6,
) -> np.ndarray:
    """每台机组独立计算6个因果温度残差特征，绝不共享滚动状态。"""
    X = np.asarray(X, dtype=np.float32)
    turbines = np.asarray(turbines).astype(str)
    if X.ndim != 2 or len(X) != len(turbines):
        raise ValueError("X须为(T,D)，且与turbines等长")
    out = np.empty((len(X), X.shape[1] + 6), dtype=np.float32)
    for name in np.unique(turbines):
        idx = np.where(turbines == name)[0]
        block = X[idx]
        maxc = block.max(axis=1)
        pose = np.mean(np.maximum(0.0, block) ** 2, axis=1)
        series = pd.Series(maxc)
        roll_mean = series.rolling(int(w_feat), min_periods=1).mean().to_numpy()
        roll_max = series.rolling(int(w_feat), min_periods=1).max().to_numpy()
        slope = maxc - series.shift(int(w_feat)).fillna(series.iloc[0]).to_numpy()
        recent = maxc - series.shift(int(k_recent)).fillna(series.iloc[0]).to_numpy()
        out[idx] = np.column_stack(
            [block, maxc, pose, roll_mean, roll_max, slope, recent]
        ).astype(np.float32)
    return out


def sample_valid_indices(turbines: np.ndarray, labels: np.ndarray, *, stride: int = 1) -> np.ndarray:
    """逐机组对非ignore行下采样，避免大型机组压制其他机组。"""
    turbines = np.asarray(turbines).astype(str)
    labels = np.asarray(labels).astype(int)
    stride = max(1, int(stride))
    if len(turbines) != len(labels):
        raise ValueError("turbines/labels 必须等长")
    chunks = []
    for name in np.unique(turbines):
        idx = np.where((turbines == name) & (labels != -1))[0]
        chunks.append(idx[::stride])
    return np.sort(np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64))


def sequence_end_indices(
    turbines: np.ndarray,
    labels: np.ndarray,
    *,
    window: int,
    stride: int = 1,
) -> np.ndarray:
    """返回完整、同机组且历史窗内无ignore的序列窗口末端索引。"""
    turbines = np.asarray(turbines).astype(str)
    labels = np.asarray(labels).astype(int)
    window = int(window)
    stride = max(1, int(stride))
    if window < 1:
        raise ValueError("window 必须>=1")
    if len(turbines) != len(labels):
        raise ValueError("turbines/labels 必须等长")
    chunks = []
    for name in np.unique(turbines):
        idx = np.where(turbines == name)[0]
        if len(idx) < window:
            continue
        # 调用方须已按机组连续排序；防御性拒绝非连续组。
        if len(idx) > 1 and not np.all(np.diff(idx) == 1):
            raise ValueError("同一机组行不连续；请先调用 sort_by_turbine_time")
        valid = labels[idx] != -1
        window_valid = np.convolve(valid.astype(np.int16), np.ones(window, dtype=np.int16), mode="valid")
        local_end = np.where(window_valid == window)[0] + window - 1
        chunks.append(idx[local_end][::stride])
    return np.sort(np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64))


__all__ = [
    "sort_by_turbine_time",
    "per_turbine_causal_features",
    "sample_valid_indices",
    "sequence_end_indices",
]
