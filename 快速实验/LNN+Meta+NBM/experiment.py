# -*- coding: utf-8 -*-
"""LNN + Meta + NBM 的可复现快速实验。

默认只运行缩减规模、单随机种子的 Tier-1 真过温 LOEO 筛选。所有模型只见健康样本；
留出事件不参与阈值选择。完整协议见同目录 ``实验设计.md``。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "SCADA数据集" / "数据预处理"
OUT_ROOT = Path(__file__).resolve().parent / "outputs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tier1_leoo import (  # noqa: E402
    build_episode_labels,
    extract_tier1_episodes,
    fit_mask,
    loeo_folds,
)


STEP_MINUTES = 10
STEP_NS = STEP_MINUTES * 60 * 10**9
LEAD_STEPS = 144
POST_IGNORE_STEPS = 144
GUARD_HOURS = 24.0 * 30
FAR_BUDGET = 0.033
SEED = 20260809


@dataclass(frozen=True)
class ResidualLayout:
    """跨风场按特征名对齐后的残差布局。"""

    names: tuple[str, ...]
    indices: tuple[np.ndarray, ...]
    medians: tuple[np.ndarray, ...]
    iqrs: tuple[np.ndarray, ...]


@dataclass
class TaskArrays:
    name: str
    farm: str
    turbine: str
    medians: np.ndarray
    iqrs: np.ndarray
    X_pretrain: np.ndarray
    y_pretrain: np.ndarray
    X_adapt: np.ndarray
    y_adapt: np.ndarray
    X_calibration: np.ndarray
    y_calibration: np.ndarray
    ts_calibration: np.ndarray
    X_healthy: np.ndarray
    y_healthy: np.ndarray
    ts_healthy: np.ndarray
    X_event: np.ndarray
    y_event: np.ndarray
    ts_event: np.ndarray


@dataclass
class PreparedData:
    tasks: list[TaskArrays]
    episodes: list[dict[str, Any]]
    feature_names: tuple[str, ...]
    audit: dict[str, Any]


def _residual_names(meta: Mapping[str, Any]) -> list[str]:
    return [str(c) for c in meta["cols"] if str(c).endswith("__resid")]


def common_residual_layout(*metas: Mapping[str, Any]) -> ResidualLayout:
    """按名称而非列位置对齐多个风场的原始温度残差通道。"""
    if len(metas) < 2:
        raise ValueError("至少需要两个 meta 才能做跨域对齐")
    common = set(_residual_names(metas[0]))
    for meta in metas[1:]:
        common &= set(_residual_names(meta))
    names = tuple(sorted(common))
    if not names:
        raise ValueError("风场之间没有共同的 __resid 通道")
    all_indices: list[np.ndarray] = []
    all_medians: list[np.ndarray] = []
    all_iqrs: list[np.ndarray] = []
    for meta in metas:
        pos = {str(c): i for i, c in enumerate(meta["cols"])}
        idx = np.asarray([pos[n] for n in names], dtype=np.int64)
        med = np.asarray(meta["scaler"]["medians"], dtype=np.float64)[idx]
        iqr = np.asarray(meta["scaler"]["iqrs"], dtype=np.float64)[idx]
        if np.any(~np.isfinite(iqr)) or np.any(iqr <= 0):
            raise ValueError("残差 IQR 必须为有限正数")
        all_indices.append(idx)
        all_medians.append(med)
        all_iqrs.append(iqr)
    return ResidualLayout(names, tuple(all_indices), tuple(all_medians), tuple(all_iqrs))


def _timestamps_ns(timestamps: np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamps)
    if ts.dtype.kind == "i":
        return ts.astype(np.int64, copy=False)
    return pd.to_datetime(ts, utc=True).tz_convert(None).to_numpy(dtype="datetime64[ns]").astype(np.int64)


def build_sequence_indices(
    timestamps: np.ndarray,
    turbines: np.ndarray,
    eligible: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """构造 `[window 个输入, 1 个 target]` 的严格同机组连续索引。

    任意跨机组、非 10 min 间隔或含不合格行的窗口都会被拒绝。
    """
    ts = _timestamps_ns(timestamps)
    tb = np.asarray(turbines).astype(str)
    ok = np.asarray(eligible, dtype=bool)
    if not (len(ts) == len(tb) == len(ok)):
        raise ValueError("timestamps/turbines/eligible 必须等长")
    if window < 1:
        raise ValueError("window 必须 >= 1")
    parts: list[np.ndarray] = []
    offsets = np.arange(-int(window), 1, dtype=np.int64)
    for turbine in np.unique(tb):
        g = np.flatnonzero(tb == turbine)
        if len(g) <= window:
            continue
        g = g[np.argsort(ts[g], kind="stable")]
        bad_row = (~ok[g]).astype(np.int16)
        bad_gap = (np.diff(ts[g]) != STEP_NS).astype(np.int16)
        row_bad_count = np.convolve(bad_row, np.ones(window + 1, dtype=np.int16), mode="valid")
        gap_bad_count = np.convolve(bad_gap, np.ones(window, dtype=np.int16), mode="valid")
        target_pos = np.flatnonzero((row_bad_count == 0) & (gap_bad_count == 0)) + window
        if target_pos.size:
            parts.append(g[target_pos[:, None] + offsets[None, :]])
    if not parts:
        return np.empty((0, window + 1), dtype=np.int64)
    out = np.concatenate(parts, axis=0)
    order = np.lexsort((ts[out[:, -1]], tb[out[:, -1]]))
    return out[order]


def inverse_residual(z: np.ndarray, medians: np.ndarray, iqrs: np.ndarray) -> np.ndarray:
    """训练集 robust scaling 的逆变换，返回摄氏度 NBM 残差。"""
    return np.asarray(z, dtype=np.float64) * np.asarray(iqrs, dtype=np.float64) + np.asarray(
        medians, dtype=np.float64
    )


def nbm_zero_residual_prediction(shape: Sequence[int]) -> np.ndarray:
    """原始 NBM 不做残差修正，对应摄氏度残差预测恒为 0。"""
    return np.zeros(tuple(int(v) for v in shape), dtype=np.float64)


def forecast_error_summary(true_degc: np.ndarray, pred_degc: np.ndarray) -> dict[str, float]:
    true = np.asarray(true_degc, dtype=np.float64)
    pred = np.asarray(pred_degc, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"true/pred 形状不一致: {true.shape} != {pred.shape}")
    finite = np.isfinite(true) & np.isfinite(pred)
    if not finite.any():
        return {"mve_degc": math.nan, "mae_degc": math.nan, "rmse_degc": math.nan}
    error = pred[finite] - true[finite]
    return {
        "mve_degc": float(error.mean()),
        "mae_degc": float(np.abs(error).mean()),
        "rmse_degc": float(np.sqrt(np.mean(error**2))),
    }


def healthy_error_scale(calibration_error_degc: np.ndarray) -> np.ndarray:
    """仅由健康校准误差估计逐通道 RMS 尺度。"""
    error = np.asarray(calibration_error_degc, dtype=np.float64)
    if error.ndim != 2 or not error.shape[0]:
        raise ValueError("calibration_error_degc 必须是非空二维数组")
    scale = np.sqrt(np.nanmean(error**2, axis=0))
    finite_positive = scale[np.isfinite(scale) & (scale > 1e-8)]
    fallback = float(np.median(finite_positive)) if finite_positive.size else 1.0
    return np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)


def normalized_error_score(error_degc: np.ndarray, channel_scale: np.ndarray) -> np.ndarray:
    """逐点跨通道归一化 RMS 误差，越大越异常。"""
    error = np.asarray(error_degc, dtype=np.float64)
    scale = np.asarray(channel_scale, dtype=np.float64)
    if error.ndim != 2 or error.shape[1] != len(scale):
        raise ValueError("error 的通道维必须与 channel_scale 一致")
    return np.sqrt(np.nanmean((error / scale[None, :]) ** 2, axis=1))


class CfCStyleCell(nn.Module):
    """轻量 CfC-style 闭式连续时间单元，不冒充官方 ncps.CfC。"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        joined = int(input_size) + int(hidden_size)
        self.slow = nn.Linear(joined, hidden_size)
        self.fast = nn.Linear(joined, hidden_size)
        self.rate = nn.Linear(joined, hidden_size)
        self.phase = nn.Parameter(torch.ones(hidden_size))
        nn.init.constant_(self.rate.bias, -3.0)

    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        z = torch.cat((x, h), dim=-1)
        slow = torch.tanh(self.slow(z))
        fast = torch.tanh(self.fast(z))
        positive_rate = F.softplus(self.rate(z)) + 1e-4
        gate = torch.sigmoid(self.phase - positive_rate * float(dt))
        return gate * slow + (1.0 - gate) * fast


class LiquidResidualRegressor(nn.Module):
    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.cell = CfCStyleCell(self.channels, self.hidden)
        self.norm = nn.LayerNorm(self.hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"期望 (B,W,{self.channels})，实际 {tuple(x.shape)}")
        h = x.new_zeros((x.shape[0], self.hidden))
        states = []
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], h)
            states.append(h)
        sequence = self.norm(torch.stack(states, dim=1))
        return self.head(torch.cat((sequence.mean(dim=1), sequence[:, -1]), dim=-1))


class TransformerResidualRegressor(nn.Module):
    def __init__(self, channels: int, window: int, d_model: int = 48, nhead: int = 4):
        super().__init__()
        self.channels = int(channels)
        self.window = int(window)
        self.embed = nn.Linear(channels, d_model)
        self.position = nn.Parameter(torch.zeros(1, window, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, channels)
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) + self.position[:, : x.shape[1]]
        # 整个输入窗都早于 target，因此 encoder 内双向注意力仍保持对 target 的因果性。
        return self.head(self.norm(self.encoder(h)[:, -1]))


@torch.no_grad()
def reptile_update_(base: nn.Module, adapted: nn.Module, *, meta_step: float) -> None:
    """Reptile 外循环：theta <- theta + eps * (phi - theta)。"""
    eps = float(meta_step)
    if not 0.0 <= eps <= 1.0:
        raise ValueError("meta_step 必须位于 [0,1]")
    for p_base, p_adapted in zip(base.parameters(), adapted.parameters(), strict=True):
        p_base.add_(eps * (p_adapted - p_base))


def _alarm_segments(
    pred: np.ndarray,
    timestamps_ns: np.ndarray,
    turbines: np.ndarray,
    *,
    max_gap_steps: int = 3,
) -> list[tuple[str, int, int]]:
    p = np.asarray(pred, dtype=bool)
    ts = _timestamps_ns(timestamps_ns)
    tb = np.asarray(turbines).astype(str)
    gap_limit = (int(max_gap_steps) + 1) * STEP_NS
    segments: list[tuple[str, int, int]] = []
    for turbine in np.unique(tb[p]):
        values = np.sort(ts[p & (tb == turbine)])
        if not len(values):
            continue
        start = previous = int(values[0])
        for value in values[1:]:
            current = int(value)
            if current - previous > gap_limit:
                segments.append((str(turbine), start, previous))
                start = current
            previous = current
        segments.append((str(turbine), start, previous))
    return segments


def strict_event_metrics(
    pred: np.ndarray,
    timestamps_ns: np.ndarray,
    turbines: np.ndarray,
    healthy_mask: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
    *,
    lead_steps: int = LEAD_STEPS,
    max_gap_steps: int = 3,
) -> dict[str, float | int]:
    """事件一对一 TP + 健康报警段 FP 的严格事件指标。"""
    p = np.asarray(pred, dtype=bool)
    ts = _timestamps_ns(timestamps_ns)
    tb = np.asarray(turbines).astype(str)
    healthy = np.asarray(healthy_mask, dtype=bool)
    if not (len(p) == len(ts) == len(tb) == len(healthy)):
        raise ValueError("pred/timestamps/turbines/healthy_mask 必须等长")
    H = int(lead_steps) * STEP_NS
    detected = []
    leads = []
    for episode in episodes:
        start = int(episode["start_ns"])
        turbine = str(episode["turbine"])
        hit_ts = ts[p & (tb == turbine) & (ts >= start - H) & (ts < start)]
        detected.append(bool(hit_ts.size))
        if hit_ts.size:
            leads.append((start - int(hit_ts.min())) / (60 * 10**9))

    false_segments = _alarm_segments(p & healthy, ts, tb, max_gap_steps=max_gap_steps)
    tp = int(np.sum(detected))
    fn = int(len(episodes) - tp)
    fp = int(len(false_segments))
    precision = float(tp / (tp + fp)) if (tp + fp) else (1.0 if not episodes else 0.0)
    recall = float(tp / len(episodes)) if episodes else math.nan
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    healthy_days = float(healthy.sum() * STEP_MINUTES / (60 * 24))
    return {
        "tp_events": tp,
        "fn_events": fn,
        "fp_segments": fp,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "lead_minutes_median": float(np.median(leads)) if leads else math.nan,
        "healthy_turbine_days": healthy_days,
        "false_alarm_segments_per_turbine_day": float(fp / healthy_days) if healthy_days else math.nan,
    }


def strict_event_pr_curve(
    scores: np.ndarray,
    timestamps_ns: np.ndarray,
    turbines: np.ndarray,
    healthy_mask: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
    *,
    lead_steps: int = LEAD_STEPS,
    max_gap_steps: int = 3,
    n_grid: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(score)
    if not finite.any():
        return np.empty(0), np.empty(0), np.empty(0)
    qs = np.linspace(0.0, 1.0, max(3, int(n_grid)))
    thresholds = np.unique(np.quantile(score[finite], qs))[::-1]
    recall: list[float] = []
    precision: list[float] = []
    for threshold in thresholds:
        metrics = strict_event_metrics(
            finite & (score >= threshold),
            timestamps_ns,
            turbines,
            healthy_mask,
            episodes,
            lead_steps=lead_steps,
            max_gap_steps=max_gap_steps,
        )
        recall.append(float(metrics["event_recall"]))
        precision.append(float(metrics["event_precision"]))
    return np.asarray(recall), np.asarray(precision), thresholds


def strict_event_auprc(
    scores: np.ndarray,
    timestamps_ns: np.ndarray,
    turbines: np.ndarray,
    healthy_mask: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> float:
    recall, precision, _ = strict_event_pr_curve(
        scores, timestamps_ns, turbines, healthy_mask, episodes, **kwargs
    )
    if not len(recall):
        return math.nan
    # 同一 recall 取最高 precision；再使用 AP 风格的单调 precision envelope。
    unique_recall = np.unique(recall)
    best_precision = np.asarray([precision[recall == r].max() for r in unique_recall])
    order = np.argsort(unique_recall)
    r = unique_recall[order]
    p = best_precision[order]
    if r[0] > 0:
        r = np.concatenate(([0.0], r))
        p = np.concatenate(([1.0], p))
    envelope = np.maximum.accumulate(p[::-1])[::-1]
    area = float(np.sum(np.diff(r) * envelope[1:])) if len(r) > 1 else 0.0
    return float(np.clip(area, 0.0, 1.0))


def pick_event_threshold(
    scores: np.ndarray,
    timestamps_ns: np.ndarray,
    turbines: np.ndarray,
    healthy_mask: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
    *,
    lead_steps: int = LEAD_STEPS,
    far_budget: float = FAR_BUDGET,
    max_gap_steps: int = 3,
    n_grid: int = 80,
) -> dict[str, Any]:
    """在校准健康块与校准事件上按 FAR 约束选阈值。"""
    score = np.asarray(scores, dtype=np.float64)
    healthy = np.asarray(healthy_mask, dtype=bool)
    finite = np.isfinite(score)
    healthy_scores = score[finite & healthy]
    if not healthy_scores.size:
        raise ValueError("没有有限的健康校准分数")
    all_scores = score[finite]
    candidates = np.quantile(all_scores, np.linspace(0.50, 0.9999, max(8, int(n_grid))))
    candidates = np.unique(
        np.concatenate((candidates, [np.nextafter(float(np.max(healthy_scores)), math.inf)]))
    )
    feasible: list[tuple[tuple[float, float, float], float, dict[str, Any]]] = []
    all_rows: list[tuple[float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = strict_event_metrics(
            finite & (score >= float(threshold)),
            timestamps_ns,
            turbines,
            healthy,
            episodes,
            lead_steps=lead_steps,
            max_gap_steps=max_gap_steps,
        )
        far = float(metrics["false_alarm_segments_per_turbine_day"])
        all_rows.append((float(threshold), metrics))
        if np.isfinite(far) and far <= float(far_budget) + 1e-12:
            key = (float(metrics["tp_events"]), -far, float(threshold))
            feasible.append((key, float(threshold), metrics))
    fallback = False
    if feasible:
        _, threshold, metrics = max(feasible, key=lambda item: item[0])
    else:
        fallback = True
        threshold, metrics = min(
            all_rows,
            key=lambda item: (
                float(item[1]["false_alarm_segments_per_turbine_day"]),
                -float(item[1]["tp_events"]),
                -item[0],
            ),
        )
    return {
        "threshold": float(threshold),
        "n_detected_calibration": int(metrics["tp_events"]),
        "n_events_calibration": int(len(episodes)),
        "far_calibration": float(metrics["false_alarm_segments_per_turbine_day"]),
        "fallback": fallback,
    }


def _stable_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _valid_target_positions(
    timestamps_sorted: np.ndarray,
    eligible_sorted: np.ndarray,
    window: int,
) -> np.ndarray:
    """在单机组已排序轴上返回合法 target 的位置。"""
    ts = _timestamps_ns(timestamps_sorted)
    ok = np.asarray(eligible_sorted, dtype=bool)
    if len(ts) <= int(window):
        return np.empty(0, dtype=np.int64)
    bad_row = (~ok).astype(np.int16)
    bad_gap = (np.diff(ts) != STEP_NS).astype(np.int16)
    row_bad = np.convolve(bad_row, np.ones(window + 1, dtype=np.int16), mode="valid")
    gap_bad = np.convolve(bad_gap, np.ones(window, dtype=np.int16), mode="valid")
    return np.flatnonzero((row_bad == 0) & (gap_bad == 0)).astype(np.int64) + int(window)


def _sequence_rows(group_rows: np.ndarray, target_positions: np.ndarray, window: int) -> np.ndarray:
    pos = np.asarray(target_positions, dtype=np.int64)
    if not len(pos):
        return np.empty((0, int(window) + 1), dtype=np.int64)
    offsets = np.arange(-int(window), 1, dtype=np.int64)
    return np.asarray(group_rows, dtype=np.int64)[pos[:, None] + offsets[None, :]]


def _random_subset(values: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if len(arr) <= int(maximum):
        return arr
    chosen = np.sort(rng.choice(len(arr), size=int(maximum), replace=False))
    return arr[chosen]


def _block_subset(values: np.ndarray, maximum: int, n_blocks: int = 4) -> np.ndarray:
    """取若干连续位置块，保留健康报警段的时间结构。"""
    arr = np.asarray(values, dtype=np.int64)
    maximum = int(maximum)
    if len(arr) <= maximum:
        return arr
    n_blocks = max(1, min(int(n_blocks), maximum))
    width = max(1, maximum // n_blocks)
    starts = np.linspace(0, max(0, len(arr) - width), n_blocks, dtype=int)
    pieces = [arr[s : s + width] for s in starts]
    out = np.unique(np.concatenate(pieces))
    return out[:maximum]


def _materialize_sequences(
    X: np.ndarray,
    sequence_rows: np.ndarray,
    feature_indices: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq = np.asarray(sequence_rows, dtype=np.int64)
    channels = len(feature_indices)
    window = seq.shape[1] - 1 if seq.ndim == 2 and seq.shape[1] else 0
    if not len(seq):
        return (
            np.empty((0, window, channels), dtype=np.float32),
            np.empty((0, channels), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    x = np.asarray(X[seq[:, :-1]][:, :, feature_indices], dtype=np.float32)
    y = np.asarray(X[seq[:, -1]][:, feature_indices], dtype=np.float32)
    ts = _timestamps_ns(timestamps)[seq[:, -1]]
    finite = np.isfinite(x).all(axis=(1, 2)) & np.isfinite(y).all(axis=1)
    x = np.clip(x[finite], -12.0, 12.0)
    return x, y[finite], ts[finite]


def _episode_ns(episode: Mapping[str, Any], farm: str) -> dict[str, Any]:
    start = pd.Timestamp(episode["start"])
    end = pd.Timestamp(episode["end"])
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)
    return {
        "episode_id": f"{farm}::{episode['episode_id']}",
        "farm": farm,
        "turbine": f"{farm}::{episode['turbine']}",
        "local_turbine": str(episode["turbine"]),
        "start_ns": int(start.to_datetime64().astype("datetime64[ns]").astype(np.int64)),
        "end_ns": int(end.to_datetime64().astype("datetime64[ns]").astype(np.int64)),
        "n_raw": int(episode.get("n_raw", 1)),
    }


def prepare_data(
    *,
    window: int = 24,
    max_pretrain_per_task: int = 800,
    max_adapt_per_task: int = 128,
    max_calibration_per_task: int = 600,
    max_healthy_per_task: int = 1200,
    seed: int = SEED,
) -> PreparedData:
    """从两个风场构造健康训练任务与 Tier-1 24 h 评测队列。"""
    farms = ("kelmarsh", "penmanshiel")
    metas = [json.loads((DATA_ROOT / farm / "meta.json").read_text(encoding="utf-8")) for farm in farms]
    layout = common_residual_layout(*metas)
    tasks: list[TaskArrays] = []
    all_episodes: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "schema_version": "lnn-meta-nbm-data-audit-v1",
        "lead_steps": LEAD_STEPS,
        "lead_hours": LEAD_STEPS * STEP_MINUTES / 60,
        "guard_hours": GUARD_HOURS,
        "window_steps": int(window),
        "feature_names": list(layout.names),
        "farms": {},
    }

    for farm_i, (farm, meta) in enumerate(zip(farms, metas, strict=True)):
        folder = DATA_ROOT / farm
        X = np.load(folder / "train_sup.npy", mmap_mode="r")
        timestamps = np.load(folder / "timestamps_train_sup.npy", mmap_mode="r")
        turbines = np.load(folder / "turbines_train_sup.npy", mmap_mode="r").astype(str)
        original_labels = np.load(folder / "train_sup_labels.npy", mmap_mode="r")
        if not (len(X) == len(timestamps) == len(turbines) == len(original_labels)):
            raise RuntimeError(f"{farm}: train_sup 与侧车长度不一致")

        raw_table = pd.read_csv(folder / "event_table.csv")
        episodes_local = extract_tier1_episodes(raw_table)
        if len(episodes_local) < 2:
            raise RuntimeError(f"{farm}: Tier-1 episode 少于 2，无法 LOEO")
        episodes_global = [_episode_ns(ep, farm) for ep in episodes_local]
        all_episodes.extend(episodes_global)

        tier1_labels = build_episode_labels(
            np.asarray(timestamps),
            turbines,
            episodes_local,
            lead_steps=LEAD_STEPS,
            post_ignore_steps=POST_IGNORE_STEPS,
            step_minutes=STEP_MINUTES,
        )
        safe = (np.asarray(original_labels) == 0) & (tier1_labels == 0)
        ts_ns = _timestamps_ns(np.asarray(timestamps))
        guard_ns = int(GUARD_HOURS * 3600 * 10**9)
        for ep in episodes_global:
            local_tb = str(ep["local_turbine"])
            banned = (
                (turbines == local_tb)
                & (ts_ns >= int(ep["start_ns"]) - guard_ns)
                & (ts_ns <= int(ep["end_ns"]) + guard_ns)
            )
            safe[banned] = False
        # train_sup 的历史故障窗会被旧训练标签整体置 -1 以防训练污染；这些行恰是
        # LOEO 的评测对象。事件序列只服从重新构造的 Tier-1 24 h 标签，绝不回流训练。
        event_history_eligible = tier1_labels != -1

        farm_audit: dict[str, Any] = {
            "rows": int(len(X)),
            "turbines": int(len(np.unique(turbines))),
            "tier1_episodes": int(len(episodes_local)),
            "tier1_positive_rows_24h": int((tier1_labels == 1).sum()),
            "safe_rows_after_30d_guards": int(safe.sum()),
            "tasks": {},
            "split_hash": meta.get("split_hash"),
            "cols_hash": meta.get("cols_hash"),
        }

        for turbine in np.unique(turbines):
            group = np.flatnonzero(turbines == turbine)
            group = group[np.argsort(ts_ns[group], kind="stable")]
            safe_pos = _valid_target_positions(ts_ns[group], safe[group], int(window))
            if len(safe_pos) < 400:
                continue
            n = len(safe_pos)
            split_a = max(1, int(0.70 * n))
            split_b = max(split_a + 1, int(0.85 * n))
            train_pos = safe_pos[:split_a]
            cut = max(1, int(0.80 * len(train_pos)))
            pre_pos = train_pos[:cut]
            adapt_pos = train_pos[cut:]
            cal_pos = safe_pos[split_a:split_b]
            healthy_pos = safe_pos[split_b:]
            rng = np.random.default_rng(_stable_seed(seed, f"{farm}:{turbine}"))
            pre_pos = _random_subset(pre_pos, max_pretrain_per_task, rng)
            adapt_pos = _block_subset(adapt_pos, max_adapt_per_task, n_blocks=2)
            cal_pos = _block_subset(cal_pos, max_calibration_per_task, n_blocks=3)
            healthy_pos = _block_subset(healthy_pos, max_healthy_per_task, n_blocks=4)

            event_valid_pos = _valid_target_positions(
                ts_ns[group], event_history_eligible[group], int(window)
            )
            if len(event_valid_pos):
                event_valid_pos = event_valid_pos[tier1_labels[group[event_valid_pos]] == 1]

            idx = layout.indices[farm_i]
            X_pre, y_pre, _ = _materialize_sequences(
                X, _sequence_rows(group, pre_pos, window), idx, np.asarray(timestamps)
            )
            X_adapt, y_adapt, _ = _materialize_sequences(
                X, _sequence_rows(group, adapt_pos, window), idx, np.asarray(timestamps)
            )
            X_cal, y_cal, ts_cal = _materialize_sequences(
                X, _sequence_rows(group, cal_pos, window), idx, np.asarray(timestamps)
            )
            X_healthy, y_healthy, ts_healthy = _materialize_sequences(
                X, _sequence_rows(group, healthy_pos, window), idx, np.asarray(timestamps)
            )
            X_event, y_event, ts_event = _materialize_sequences(
                X, _sequence_rows(group, event_valid_pos, window), idx, np.asarray(timestamps)
            )
            if min(len(X_pre), len(X_adapt), len(X_cal), len(X_healthy)) == 0:
                continue
            task_name = f"{farm}::{turbine}"
            tasks.append(
                TaskArrays(
                    name=task_name,
                    farm=farm,
                    turbine=str(turbine),
                    medians=layout.medians[farm_i].copy(),
                    iqrs=layout.iqrs[farm_i].copy(),
                    X_pretrain=X_pre,
                    y_pretrain=y_pre,
                    X_adapt=X_adapt,
                    y_adapt=y_adapt,
                    X_calibration=X_cal,
                    y_calibration=y_cal,
                    ts_calibration=ts_cal,
                    X_healthy=X_healthy,
                    y_healthy=y_healthy,
                    ts_healthy=ts_healthy,
                    X_event=X_event,
                    y_event=y_event,
                    ts_event=ts_event,
                )
            )
            farm_audit["tasks"][str(turbine)] = {
                "pretrain": int(len(X_pre)),
                "adapt": int(len(X_adapt)),
                "calibration_healthy": int(len(X_cal)),
                "evaluation_healthy": int(len(X_healthy)),
                "event_points_24h": int(len(X_event)),
            }
        audit["farms"][farm] = farm_audit

    task_names = {task.name for task in tasks}
    coverage_episodes = []
    for episode in all_episodes:
        task = next((t for t in tasks if t.name == episode["turbine"]), None)
        if task is None:
            continue
        in_window = (
            (task.ts_event >= int(episode["start_ns"]) - LEAD_STEPS * STEP_NS)
            & (task.ts_event < int(episode["start_ns"]))
        )
        if int(in_window.sum()) > 0:
            ep = dict(episode)
            ep["available_points"] = int(in_window.sum())
            coverage_episodes.append(ep)
    audit["n_tasks"] = int(len(tasks))
    audit["n_tier1_episodes_total"] = int(len(all_episodes))
    audit["n_tier1_episodes_with_sequence_coverage"] = int(len(coverage_episodes))
    audit["task_names"] = sorted(task_names)
    if len(coverage_episodes) < 2:
        raise RuntimeError("有序列覆盖的 Tier-1 episode 少于 2")
    return PreparedData(tasks=tasks, episodes=coverage_episodes, feature_names=layout.names, audit=audit)


def _batch_train(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    label: str,
) -> list[dict[str, Any]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        permutation = rng.permutation(len(X))
        losses = []
        model.train()
        for start in range(0, len(permutation), int(batch_size)):
            idx = permutation[start : start + int(batch_size)]
            xb = torch.from_numpy(X[idx]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(np.clip(y[idx], -12.0, 12.0)).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = F.mse_loss(prediction, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"stage": label, "epoch": epoch + 1, "loss": float(np.mean(losses))}
        history.append(row)
        print(f"[{label}] epoch {epoch + 1}/{epochs} loss={row['loss']:.6f}", flush=True)
    return history


def train_pooled_liquid(
    prepared: PreparedData,
    *,
    device: torch.device,
    hidden: int,
    epochs: int,
    seed: int,
) -> tuple[LiquidResidualRegressor, list[dict[str, Any]]]:
    _seed_everything(seed)
    model = LiquidResidualRegressor(len(prepared.feature_names), hidden=hidden)
    X = np.concatenate([task.X_pretrain for task in prepared.tasks], axis=0)
    y = np.concatenate([task.y_pretrain for task in prepared.tasks], axis=0)
    history = _batch_train(
        model,
        X,
        y,
        device=device,
        epochs=epochs,
        batch_size=256,
        lr=1e-3,
        seed=seed,
        label="LNN-pooled",
    )
    return model, history


def train_reptile_liquid(
    prepared: PreparedData,
    *,
    device: torch.device,
    hidden: int,
    episodes: int,
    seed: int,
) -> tuple[LiquidResidualRegressor, list[dict[str, Any]]]:
    _seed_everything(seed)
    model = LiquidResidualRegressor(len(prepared.feature_names), hidden=hidden).to(device)
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    for episode_i in range(int(episodes)):
        task = prepared.tasks[int(rng.integers(0, len(prepared.tasks)))]
        adapted = copy.deepcopy(model).to(device)
        optimizer = torch.optim.SGD(adapted.parameters(), lr=2e-2, momentum=0.0)
        losses = []
        adapted.train()
        for _ in range(2):
            count = min(64, len(task.X_pretrain))
            idx = rng.choice(len(task.X_pretrain), size=count, replace=False)
            xb = torch.from_numpy(task.X_pretrain[idx]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(np.clip(task.y_pretrain[idx], -12.0, 12.0)).to(
                device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(adapted(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        progress = episode_i / max(1, int(episodes) - 1)
        meta_step = 0.25 * (1.0 - progress) + 0.05 * progress
        reptile_update_(model, adapted, meta_step=meta_step)
        history.append(
            {
                "stage": "LNN-Reptile",
                "episode": episode_i + 1,
                "task": task.name,
                "loss": float(np.mean(losses)),
                "meta_step": float(meta_step),
            }
        )
        if (episode_i + 1) % max(1, int(episodes) // 5) == 0 or episode_i == 0:
            print(
                f"[LNN-Reptile] episode {episode_i + 1}/{episodes} "
                f"loss={np.mean(losses):.6f}",
                flush=True,
            )
    return model, history


def adapt_model(
    initial: nn.Module,
    task: TaskArrays,
    *,
    device: torch.device,
    steps: int,
    seed: int,
) -> nn.Module:
    model = copy.deepcopy(initial).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    rng = np.random.default_rng(_stable_seed(seed, task.name))
    model.train()
    for _ in range(int(steps)):
        count = min(64, len(task.X_adapt))
        idx = rng.choice(len(task.X_adapt), size=count, replace=False)
        xb = torch.from_numpy(task.X_adapt[idx]).to(device=device, dtype=torch.float32)
        yb = torch.from_numpy(np.clip(task.y_adapt[idx], -12.0, 12.0)).to(
            device=device, dtype=torch.float32
        )
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model.eval()


@torch.no_grad()
def _predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    if not len(X):
        return np.empty((0, getattr(model, "channels", 0)), dtype=np.float32)
    model.eval()
    chunks = []
    for start in range(0, len(X), int(batch_size)):
        xb = torch.from_numpy(X[start : start + int(batch_size)]).to(device=device, dtype=torch.float32)
        chunks.append(model(xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _ewma_segmented(scores: np.ndarray, timestamps: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    score = np.asarray(scores, dtype=np.float64)
    ts = _timestamps_ns(timestamps)
    if not len(score):
        return score
    out = np.empty_like(score)
    order = np.argsort(ts, kind="stable")
    previous_time = None
    state = None
    for idx in order:
        current_time = int(ts[idx])
        if previous_time is None or current_time - previous_time > 4 * STEP_NS:
            state = float(score[idx])
        else:
            state = float(alpha * score[idx] + (1.0 - alpha) * state)
        out[idx] = state
        previous_time = current_time
    return out


def _cohort_template() -> dict[str, list[np.ndarray]]:
    return {
        "cal_score": [],
        "cal_ts": [],
        "cal_tb": [],
        "healthy_score": [],
        "healthy_ts": [],
        "healthy_tb": [],
        "event_score": [],
        "event_ts": [],
        "event_tb": [],
        "forecast_true": [],
        "forecast_pred": [],
    }


def _finish_cohort(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, values in parts.items():
        if not values:
            out[key] = np.empty(0)
        else:
            out[key] = np.concatenate(values, axis=0)
    return out


def build_nbm_cohort(prepared: PreparedData) -> dict[str, np.ndarray]:
    parts = _cohort_template()
    for task in prepared.tasks:
        true_cal = inverse_residual(task.y_calibration, task.medians, task.iqrs)
        true_h = inverse_residual(task.y_healthy, task.medians, task.iqrs)
        true_e = inverse_residual(task.y_event, task.medians, task.iqrs)
        pred_cal = nbm_zero_residual_prediction(true_cal.shape)
        pred_h = nbm_zero_residual_prediction(true_h.shape)
        pred_e = nbm_zero_residual_prediction(true_e.shape)
        scale = healthy_error_scale(pred_cal - true_cal)
        parts["cal_score"].append(normalized_error_score(pred_cal - true_cal, scale))
        parts["healthy_score"].append(normalized_error_score(pred_h - true_h, scale))
        parts["event_score"].append(normalized_error_score(pred_e - true_e, scale))
        parts["cal_ts"].append(task.ts_calibration)
        parts["healthy_ts"].append(task.ts_healthy)
        parts["event_ts"].append(task.ts_event)
        parts["cal_tb"].append(np.asarray([task.name] * len(task.ts_calibration)))
        parts["healthy_tb"].append(np.asarray([task.name] * len(task.ts_healthy)))
        parts["event_tb"].append(np.asarray([task.name] * len(task.ts_event)))
        parts["forecast_true"].append(true_h)
        parts["forecast_pred"].append(pred_h)
    return _finish_cohort(parts)


def build_neural_cohort(
    prepared: PreparedData,
    initial: nn.Module,
    *,
    device: torch.device,
    adapt_steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    parts = _cohort_template()
    for task_i, task in enumerate(prepared.tasks):
        model = adapt_model(initial, task, device=device, steps=adapt_steps, seed=seed)
        pred_z_cal = _predict(model, task.X_calibration, device)
        pred_z_h = _predict(model, task.X_healthy, device)
        pred_z_e = _predict(model, task.X_event, device)
        true_cal = inverse_residual(task.y_calibration, task.medians, task.iqrs)
        true_h = inverse_residual(task.y_healthy, task.medians, task.iqrs)
        true_e = inverse_residual(task.y_event, task.medians, task.iqrs)
        pred_cal = inverse_residual(pred_z_cal, task.medians, task.iqrs)
        pred_h = inverse_residual(pred_z_h, task.medians, task.iqrs)
        pred_e = inverse_residual(pred_z_e, task.medians, task.iqrs)
        scale = healthy_error_scale(pred_cal - true_cal)
        parts["cal_score"].append(normalized_error_score(pred_cal - true_cal, scale))
        parts["healthy_score"].append(normalized_error_score(pred_h - true_h, scale))
        parts["event_score"].append(normalized_error_score(pred_e - true_e, scale))
        parts["cal_ts"].append(task.ts_calibration)
        parts["healthy_ts"].append(task.ts_healthy)
        parts["event_ts"].append(task.ts_event)
        parts["cal_tb"].append(np.asarray([task.name] * len(task.ts_calibration)))
        parts["healthy_tb"].append(np.asarray([task.name] * len(task.ts_healthy)))
        parts["event_tb"].append(np.asarray([task.name] * len(task.ts_event)))
        parts["forecast_true"].append(true_h)
        parts["forecast_pred"].append(pred_h)
        if (task_i + 1) % 5 == 0 or task_i + 1 == len(prepared.tasks):
            print(f"[adapt+score] {task_i + 1}/{len(prepared.tasks)} tasks", flush=True)
        del model
    return _finish_cohort(parts)


def build_pca_ewma_cohort(prepared: PreparedData) -> dict[str, np.ndarray]:
    fit = np.concatenate([task.y_pretrain for task in prepared.tasks], axis=0)
    n_components = min(16, fit.shape[1] - 1)
    pca = PCA(n_components=n_components, random_state=SEED).fit(fit)
    parts = _cohort_template()
    for task in prepared.tasks:
        def score(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
            if not len(values):
                return np.empty(0, dtype=np.float64)
            reconstructed = pca.inverse_transform(pca.transform(values))
            raw = np.mean((values - reconstructed) ** 2, axis=1)
            return _ewma_segmented(raw, timestamps, alpha=0.01)

        cal = score(task.y_calibration, task.ts_calibration)
        healthy = score(task.y_healthy, task.ts_healthy)
        event = score(task.y_event, task.ts_event)
        center = float(np.median(cal))
        scale = float(np.median(np.abs(cal - center)) * 1.4826)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(cal) + 1e-12)
        parts["cal_score"].append((cal - center) / scale)
        parts["healthy_score"].append((healthy - center) / scale)
        parts["event_score"].append((event - center) / scale)
        parts["cal_ts"].append(task.ts_calibration)
        parts["healthy_ts"].append(task.ts_healthy)
        parts["event_ts"].append(task.ts_event)
        parts["cal_tb"].append(np.asarray([task.name] * len(task.ts_calibration)))
        parts["healthy_tb"].append(np.asarray([task.name] * len(task.ts_healthy)))
        parts["event_tb"].append(np.asarray([task.name] * len(task.ts_event)))
    return _finish_cohort(parts)


def _rows_for_episodes(
    timestamps: np.ndarray,
    turbines: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
    lead_steps: int = LEAD_STEPS,
) -> np.ndarray:
    ts = _timestamps_ns(timestamps)
    tb = np.asarray(turbines).astype(str)
    mask = np.zeros(len(ts), dtype=bool)
    H = int(lead_steps) * STEP_NS
    for episode in episodes:
        mask |= (
            (tb == str(episode["turbine"]))
            & (ts >= int(episode["start_ns"]) - H)
            & (ts < int(episode["start_ns"]))
        )
    return mask


def evaluate_cohort(
    model_name: str,
    cohort: Mapping[str, np.ndarray],
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    n_folds = len(episodes)
    task_names = sorted(set(np.asarray(cohort["healthy_tb"]).astype(str)))
    task_fold = {name: i % n_folds for i, name in enumerate(task_names)}
    fold_rows: list[dict[str, Any]] = []
    total_tp = total_fp = 0
    total_healthy_days = 0.0
    leads: list[float] = []
    thresholds: list[float] = []

    for fold_i, held_out in enumerate(episodes):
        other = [episode for j, episode in enumerate(episodes) if j != fold_i]
        event_keep = _rows_for_episodes(cohort["event_ts"], cohort["event_tb"], other)
        selection_scores = np.concatenate((cohort["cal_score"], cohort["event_score"][event_keep]))
        selection_ts = np.concatenate((cohort["cal_ts"], cohort["event_ts"][event_keep]))
        selection_tb = np.concatenate((cohort["cal_tb"], cohort["event_tb"][event_keep]))
        selection_healthy = np.concatenate(
            (np.ones(len(cohort["cal_score"]), dtype=bool), np.zeros(int(event_keep.sum()), dtype=bool))
        )
        selected = pick_event_threshold(
            selection_scores,
            selection_ts,
            selection_tb,
            selection_healthy,
            other,
            lead_steps=LEAD_STEPS,
            far_budget=FAR_BUDGET,
        )
        threshold = float(selected["threshold"])
        thresholds.append(threshold)
        held_rows = _rows_for_episodes(
            cohort["event_ts"], cohort["event_tb"], [held_out], lead_steps=LEAD_STEPS
        )
        hit_ts = cohort["event_ts"][held_rows & (cohort["event_score"] >= threshold)]
        detected = int(bool(len(hit_ts)))
        lead = (
            float((int(held_out["start_ns"]) - int(hit_ts.min())) / (60 * 10**9))
            if len(hit_ts)
            else math.nan
        )
        if np.isfinite(lead):
            leads.append(lead)

        assigned_tasks = {name for name, assigned in task_fold.items() if assigned == fold_i}
        healthy_keep = np.isin(cohort["healthy_tb"].astype(str), list(assigned_tasks))
        healthy_pred = cohort["healthy_score"][healthy_keep] >= threshold
        fp = len(
            _alarm_segments(
                healthy_pred,
                cohort["healthy_ts"][healthy_keep],
                cohort["healthy_tb"][healthy_keep],
                max_gap_steps=3,
            )
        )
        healthy_days = float(healthy_keep.sum() * STEP_MINUTES / (60 * 24))
        total_tp += detected
        total_fp += int(fp)
        total_healthy_days += healthy_days
        fold_rows.append(
            {
                "model": model_name,
                "fold": fold_i,
                "held_out_episode": held_out["episode_id"],
                "threshold": threshold,
                "detected": detected,
                "lead_minutes": lead,
                "false_segments": int(fp),
                "healthy_turbine_days": healthy_days,
                **{f"selection_{key}": value for key, value in selected.items() if key != "threshold"},
            }
        )

    recall = float(total_tp / n_folds)
    precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) else 0.0
    event_f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    evaluation_scores = np.concatenate((cohort["event_score"], cohort["healthy_score"]))
    evaluation_ts = np.concatenate((cohort["event_ts"], cohort["healthy_ts"]))
    evaluation_tb = np.concatenate((cohort["event_tb"], cohort["healthy_tb"]))
    evaluation_healthy = np.concatenate(
        (np.zeros(len(cohort["event_score"]), dtype=bool), np.ones(len(cohort["healthy_score"]), dtype=bool))
    )
    labels = np.concatenate(
        (np.ones(len(cohort["event_score"]), dtype=int), np.zeros(len(cohort["healthy_score"]), dtype=int))
    )
    point_auprc = float(average_precision_score(labels, evaluation_scores))
    point_roc_auc = float(roc_auc_score(labels, evaluation_scores))
    event_auprc = strict_event_auprc(
        evaluation_scores,
        evaluation_ts,
        evaluation_tb,
        evaluation_healthy,
        episodes,
        lead_steps=LEAD_STEPS,
        n_grid=101,
    )
    pr_recall, pr_precision, pr_thresholds = strict_event_pr_curve(
        evaluation_scores,
        evaluation_ts,
        evaluation_tb,
        evaluation_healthy,
        episodes,
        lead_steps=LEAD_STEPS,
        n_grid=101,
    )
    forecast = (
        forecast_error_summary(cohort["forecast_true"], cohort["forecast_pred"])
        if np.asarray(cohort["forecast_true"]).ndim == 2 and len(cohort["forecast_true"])
        else {"mve_degc": math.nan, "mae_degc": math.nan, "rmse_degc": math.nan}
    )
    metrics: dict[str, Any] = {
        "model": model_name,
        "n_events": int(n_folds),
        "n_detected": int(total_tp),
        "event_precision_24h": precision,
        "event_recall_24h": recall,
        "event_f1_24h": event_f1,
        "event_auprc_24h": float(event_auprc),
        "point_auprc_matched": point_auprc,
        "point_roc_auc_matched": point_roc_auc,
        "false_alarm_segments": int(total_fp),
        "healthy_turbine_days": total_healthy_days,
        "false_alarm_segments_per_turbine_day": (
            float(total_fp / total_healthy_days) if total_healthy_days else math.nan
        ),
        "lead_hours_median": float(np.median(leads) / 60) if leads else math.nan,
        "threshold_median": float(np.median(thresholds)),
        **forecast,
    }
    curve = {"recall": pr_recall, "precision": pr_precision, "thresholds": pr_thresholds}
    return metrics, fold_rows, curve


def _plot_results(
    metrics_df: pd.DataFrame,
    curves: Mapping[str, Mapping[str, np.ndarray]],
    cohorts: Mapping[str, Mapping[str, np.ndarray]],
    folds_df: pd.DataFrame,
    episodes: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = ["#607D8B", "#F39C12", "#3498DB", "#8E44AD"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, key, title in zip(
        axes,
        ("event_f1_24h", "event_auprc_24h", "event_recall_24h"),
        ("事件 F1（24 h）", "事件 AUPRC（24 h）", "事件召回率（24 h）"),
        strict=True,
    ):
        ax.bar(metrics_df["model"], metrics_df[key], color=colors[: len(metrics_df)])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        for i, value in enumerate(metrics_df[key]):
            ax.text(i, float(value) + 0.02, f"{float(value):.3f}", ha="center", fontsize=8)
    fig.suptitle("Tier-1 真过温快速价值门：同一 24 h LOEO 协议")
    fig.tight_layout()
    fig.savefig(output / "核心事件指标对比.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    forecast = metrics_df[np.isfinite(metrics_df["rmse_degc"].astype(float))].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(forecast))
    width = 0.36
    ax.bar(x - width / 2, forecast["mae_degc"], width, label="MAE (°C)", color="#2E86C1")
    ax.bar(x + width / 2, forecast["rmse_degc"], width, label="RMSE (°C)", color="#AF7AC5")
    ax.set_xticks(x, forecast["model"], rotation=20)
    ax.set_ylabel("温度残差修正误差 (°C)")
    ax.set_title("健康块一步温度预测误差")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "温度预测误差对比.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, curve) in enumerate(curves.items()):
        order = np.argsort(curve["recall"])
        ax.plot(
            curve["recall"][order],
            curve["precision"][order],
            marker="o",
            markersize=2.5,
            linewidth=1.5,
            label=name,
            color=colors[i % len(colors)],
        )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("事件召回率")
    ax.set_ylabel("报警段精确率")
    ax.set_title("严格事件级 PR 曲线（匹配健康块）")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "事件PR曲线.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    episode = episodes[0]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (name, cohort) in enumerate(cohorts.items()):
        rows = _rows_for_episodes(cohort["event_ts"], cohort["event_tb"], [episode])
        if not rows.any():
            continue
        hours = (cohort["event_ts"][rows] - int(episode["start_ns"])) / (3600 * 10**9)
        order = np.argsort(hours)
        ax.plot(hours[order], cohort["event_score"][rows][order], label=name, color=colors[i])
        threshold = folds_df[
            (folds_df["model"] == name) & (folds_df["held_out_episode"] == episode["episode_id"])
        ]["threshold"]
        if len(threshold):
            ax.axhline(float(threshold.iloc[0]), color=colors[i], linestyle="--", alpha=0.45)
    ax.axvline(0, color="black", linestyle=":", label="事件起点")
    ax.set_xlim(-24, 0.3)
    ax.set_xlabel("距事件起点时间 (h)")
    ax.set_ylabel("归一化异常分数")
    ax.set_title(f"案例：{episode['episode_id']} 起点前 24 h")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "事件前24h案例.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _decision(metrics_df: pd.DataFrame) -> tuple[str, dict[str, bool]]:
    rows = metrics_df.set_index("model")
    meta = rows.loc["LNN+Meta+NBM"]
    checks = {
        "rmse_better_than_nbm": bool(meta["rmse_degc"] <= rows.loc["NBM"]["rmse_degc"]),
        "event_f1_better_than_plain_lnn": bool(
            meta["event_f1_24h"] > rows.loc["LNN+NBM"]["event_f1_24h"]
        ),
        "event_auprc_better_than_pca_ewma": bool(
            meta["event_auprc_24h"] > rows.loc["PCA+EWMA+NBM"]["event_auprc_24h"]
        ),
        "far_within_budget": bool(meta["false_alarm_segments_per_turbine_day"] <= FAR_BUDGET),
    }
    passed = sum(checks.values())
    if passed == len(checks):
        decision = "值得继续：通过本次快速价值门，但尚未证明 SOTA。"
    elif passed >= 2 and float(meta["event_recall_24h"]) > 0:
        decision = "有条件继续：出现前兆信号，但优势不稳定，需要完整基线与多种子复核。"
    else:
        decision = "当前不支持继续重投入：该组合未通过快速价值门。"
    return decision, checks


def dataframe_markdown(frame: pd.DataFrame, *, float_digits: int = 4) -> str:
    """无第三方 tabulate 依赖的紧凑 Markdown 表。"""
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{int(float_digits)}f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def _write_quick_report(
    output: Path,
    metrics_df: pd.DataFrame,
    decision: str,
    checks: Mapping[str, bool],
    prepared: PreparedData,
    elapsed: float,
) -> None:
    table = dataframe_markdown(metrics_df[
        [
            "model",
            "mve_degc",
            "mae_degc",
            "rmse_degc",
            "event_f1_24h",
            "event_auprc_24h",
            "event_recall_24h",
            "false_alarm_segments_per_turbine_day",
            "lead_hours_median",
        ]
    ], float_digits=4)
    check_lines = "\n".join(f"- {'通过' if value else '未通过'}：`{key}`" for key, value in checks.items())
    report = f"""# LNN + Meta + NBM 快速价值门结果

## 结论

**{decision}**

这是一轮单种子、缩减样本、{len(prepared.episodes)} 个 Tier-1 真过温事件的 LOEO 快速实验。它可以判断方向是否值得继续，但不能作为论文最终数值，也不能声称超过全球 SOTA。

## 核心对比

{table}

## 预注册检查

{check_lines}

## 如何读结果

- `MVE` 在本实验中明确为平均有符号误差；同时给出更常用的 MAE 与 RMSE，单位均为 °C。
- `event_f1_24h` 使用事件 TP 与健康块假报警段 FP；每个事件最多贡献一个 TP。
- `event_auprc_24h` 是事件召回—报警段精确率曲线面积，不是逐点 PR-AUC。
- PCA+EWMA 直接观察当前残差，不能算因果温度预测，因此其 MVE/MAE/RMSE 留空。
- 匹配健康块改变了异常基率，部署前还必须在完整连续时间轴上复测 FAR 与 PR-AUC。

## 证据边界

1. chronological 2023--2024 测试段没有 Tier-1 真过温事件，故改用项目预先定义的 Tier-1 LOEO。
2. 模型只在健康行上拟合，并排除全部 Tier-1 事件前后 30 d 护带。
3. 阈值在每折只看其余事件；留出事件不参与阈值选择。
4. 本次耗时 {elapsed:.1f} s；随机种子 {SEED}；输入窗 4 h，预警窗 24 h。
5. CfC-style 单元是纯 PyTorch 轻量实现，不冒充官方 `ncps.CfC`。

## 产物

- `metrics.csv` / `metrics.json`：汇总指标。
- `事件逐折.csv`：每个留出事件的阈值、命中、提前量与假报警。
- `核心事件指标对比.png`、`温度预测误差对比.png`、`事件PR曲线.png`、`事件前24h案例.png`：数据和图像对比。
- `data_audit.json`：事件、任务、通道和样本量审计。
"""
    (output / "快速实验报告.md").write_text(report, encoding="utf-8")


def run_quick_value_probe(
    *,
    output: Path,
    device_name: str,
    seed: int,
    max_pretrain_per_task: int,
    pooled_epochs: int,
    meta_episodes: int,
    adapt_steps: int,
) -> dict[str, Any]:
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但 torch.cuda.is_available() 为 False")
    torch.set_float32_matmul_precision("high")
    print("[1/5] 准备严格健康窗口与 Tier-1 事件队列", flush=True)
    prepared = prepare_data(
        window=24,
        max_pretrain_per_task=max_pretrain_per_task,
        max_adapt_per_task=128,
        max_calibration_per_task=600,
        max_healthy_per_task=1200,
        seed=seed,
    )
    _json_dump(output / "data_audit.json", prepared.audit)
    print(
        f"data: tasks={len(prepared.tasks)}, events={len(prepared.episodes)}, "
        f"channels={len(prepared.feature_names)}",
        flush=True,
    )

    print("[2/5] 训练普通 pooled LNN", flush=True)
    pooled, history_pooled = train_pooled_liquid(
        prepared, device=device, hidden=24, epochs=pooled_epochs, seed=seed
    )
    print("[3/5] 训练 Reptile 元学习 LNN", flush=True)
    meta, history_meta = train_reptile_liquid(
        prepared, device=device, hidden=24, episodes=meta_episodes, seed=seed
    )
    torch.save(pooled.state_dict(), output / "LNN_pooled_initialization.pt")
    torch.save(meta.state_dict(), output / "LNN_reptile_initialization.pt")
    pd.DataFrame(history_pooled + history_meta).to_csv(
        output / "training_history.csv", index=False, encoding="utf-8-sig"
    )

    print("[4/5] 构造四个同协议比较器并逐机组适配", flush=True)
    cohorts: dict[str, dict[str, np.ndarray]] = {
        "NBM": build_nbm_cohort(prepared),
        "PCA+EWMA+NBM": build_pca_ewma_cohort(prepared),
        "LNN+NBM": build_neural_cohort(
            prepared, pooled, device=device, adapt_steps=adapt_steps, seed=seed
        ),
        "LNN+Meta+NBM": build_neural_cohort(
            prepared, meta, device=device, adapt_steps=adapt_steps, seed=seed
        ),
    }

    print("[5/5] LOEO 阈值、事件指标和图片", flush=True)
    metrics_rows: list[dict[str, Any]] = []
    all_folds: list[dict[str, Any]] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    for name, cohort in cohorts.items():
        metrics, folds, curve = evaluate_cohort(name, cohort, prepared.episodes)
        metrics_rows.append(metrics)
        all_folds.extend(folds)
        curves[name] = curve
        print(
            f"{name}: eF1={metrics['event_f1_24h']:.3f}, "
            f"eAUPRC={metrics['event_auprc_24h']:.3f}, "
            f"recall={metrics['event_recall_24h']:.3f}, "
            f"FAR={metrics['false_alarm_segments_per_turbine_day']:.4f}",
            flush=True,
        )
    metrics_df = pd.DataFrame(metrics_rows)
    folds_df = pd.DataFrame(all_folds)
    metrics_df.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    folds_df.to_csv(output / "事件逐折.csv", index=False, encoding="utf-8-sig")
    _json_dump(output / "metrics.json", metrics_rows)
    np.savez_compressed(
        output / "scores_and_sidecars.npz",
        **{
            f"{name}__{key}": value
            for name, cohort in cohorts.items()
            for key, value in cohort.items()
            if key in {"cal_score", "healthy_score", "event_score", "cal_ts", "healthy_ts", "event_ts"}
        },
    )
    _plot_results(metrics_df, curves, cohorts, folds_df, prepared.episodes, output)
    decision, checks = _decision(metrics_df)
    elapsed = time.time() - started
    _write_quick_report(output, metrics_df, decision, checks, prepared, elapsed)
    payload = {
        "decision": decision,
        "checks": checks,
        "elapsed_sec": elapsed,
        "seed": seed,
        "device": str(device),
        "metrics": metrics_rows,
    }
    _json_dump(output / "run_manifest.json", payload)
    print(decision, flush=True)
    return payload


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_sanitize(payload),
            ensure_ascii=False,
            indent=2,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    raise TypeError(type(value).__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="只做合成数据接线检查")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUT_ROOT)
    parser.add_argument("--max-pretrain-per-task", type=int, default=800)
    parser.add_argument("--pooled-epochs", type=int, default=2)
    parser.add_argument("--meta-episodes", type=int, default=30)
    parser.add_argument("--adapt-steps", type=int, default=5)
    args = parser.parse_args(argv)
    _seed_everything(args.seed)
    if args.smoke:
        model = LiquidResidualRegressor(channels=3, hidden=8).to(args.device)
        x = torch.randn(4, 6, 3, device=args.device)
        y = model(x)
        print(json.dumps({"smoke": True, "shape": list(y.shape), "device": args.device}, ensure_ascii=False))
        return 0
    run_quick_value_probe(
        output=args.output,
        device_name=args.device,
        seed=args.seed,
        max_pretrain_per_task=args.max_pretrain_per_task,
        pooled_epochs=args.pooled_epochs,
        meta_episodes=args.meta_episodes,
        adapt_steps=args.adapt_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
