# -*- coding: utf-8 -*-
"""_pretrain.py — 预训练权重的共享构建与缓存 (语料 = hill_of_towie 健康数据)。

为什么用 HOT 作预训练语料:
    · 21 台机组 / 650 万行, 是项目最大的未利用资产;
    · 0 真实故障 → 纯"正常"语料, 适合自监督学"正常长什么样";
    · HOT 从不参与 kel/pen 的任何评测 → 预训练阶段【天然零泄漏】。

权重按 task 缓存到 快速实验数据/_pretrained/, 供 60–64 各微调策略共用同一 θ₀,
使"微调策略"成为唯一变量。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from _domain import Encoder, apply_std, pretrain, standardize_fit
from _farmfree import load_farmfree

PRETRAIN_FARM = "hill_of_towie"
# HOT 主目录是旧 A′ 期格式(缺 turbines_base/idx_flat 侧车); v3 格式的健康语料在
# _external_local 变体下(同一份 HOT 数据, 由 准备真实故障数据_v3.py 生成, 侧车齐全)。
PRETRAIN_VARIANT = "real_fault_metrics_v1_external_local"
MAX_ROWS = 300_000        # 预训练子采样上限 (快速实验口径; 全量 650 万行留给主实验)


def _cache_dir() -> Path:
    from _common import DATA
    d = DATA.parent / "_pretrained"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_or_load(task: str = "mask", *, epochs: int = 1, seed: int = 0):
    """返回 (encoder, mu, sd)。已缓存则直接读取, 否则在 HOT 上自监督预训练并缓存。"""
    ck = _cache_dir() / f"enc_{task}_e{epochs}_s{seed}.pt"
    if ck.exists():
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        enc = Encoder()
        enc.load_state_dict(blob["state"])
        return enc, blob["mu"], blob["sd"]
    F = load_farmfree(PRETRAIN_FARM, "train", variant=PRETRAIN_VARIANT)
    if len(F) > MAX_ROWS:
        F = F[np.linspace(0, len(F) - 1, MAX_ROWS).astype(int)]   # 均匀抽样, 保时间覆盖
    mu, sd = standardize_fit(F)
    enc = pretrain(Encoder(), apply_std(F, mu, sd), task=task, epochs=epochs, seed=seed)
    enc_cpu = enc.to("cpu")
    torch.save({"state": enc_cpu.state_dict(), "mu": mu, "sd": sd,
                "task": task, "rows": int(len(F)), "farm": PRETRAIN_FARM}, ck)
    return enc_cpu, mu, sd
