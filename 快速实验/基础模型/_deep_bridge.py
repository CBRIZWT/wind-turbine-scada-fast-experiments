# -*- coding: utf-8 -*-
"""_deep_bridge.py — 把主项目4个深度模型(AT/TranAD/TriTrackNet/wt)桥接进快速实验 metrics-v3。

口径(2026-07-24, 用户选定"统一重构/预测误差变体"):
  · 复用各模型【原生架构 + 原生窗口】, 但训练/打分统一为【1 epoch 最小化重构或预测误差】,
    分数 = 窗末误差。—— 明确【非】论文完整训练(不含 AT 关联差异 / TranAD 对抗 / RevIN 精修),
    结果名后缀 "_recon变体"/"_pred变体", extra 记 variant=error-only, 诚实标注, 不冒充原论文性能。
  · 数据 = 快速实验 v3 seq(X_base turbine-sorted); 按机组边界重建各模型原生窗口(clamp 左填充, 不跨机组);
    分数对齐 idx_seq → report(自动路由 report_v3 seq) 统一事件级评测(event_f1/AUPRC/FAR + 全指标)。
  · seed=0 统一; GPU 优先。train 只用正常窗(y==0, 无监督); val 选阈/极性 + test 只评一次由 report_v3 负责。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_HERE = Path(__file__).resolve().parent               # 快速实验/基础模型
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import DATA, now, report                 # noqa: E402  统一数据目录 + 计时 + 评测入口(VARIANT非空→report_v3)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed(k: int = 0) -> None:
    torch.manual_seed(k)
    np.random.seed(k)


def _load_split(split: str):
    """读 v3 seq 表示: X_base(T,C) turbine-sorted / turbines / idx_seq(窗末索引) / y_seq。"""
    base = np.load(DATA / f"X_base_{split}.npy").astype(np.float32)
    turb = np.load(DATA / f"turbines_base_{split}.npy")
    idx = np.load(DATA / f"idx_seq_{split}.npy").astype(np.int64)
    y = np.load(DATA / f"y_seq_{split}.npy").astype(np.int64)
    return base, turb, idx, y


def _block_start(turb: np.ndarray) -> np.ndarray:
    """每个位置所属机组连续块的首行索引(turbine-sorted 下用于窗口 clamp, 防跨机组)。"""
    T = len(turb)
    change = np.empty(T, dtype=bool)
    change[0] = True
    change[1:] = turb[1:] != turb[:-1]
    starts = np.where(change)[0]
    return np.repeat(starts, np.diff(np.append(starts, T)))


def _win_batch(base: np.ndarray, bstart: np.ndarray, idx_b: np.ndarray, W: int) -> np.ndarray:
    """(B,W,C): 以 idx 为窗末的 W 窗; 越过机组块起点的位置 clamp 到块首行(左填充), 不跨机组。"""
    pos = (idx_b[:, None] - W + 1) + np.arange(W)[None, :]      # (B,W) 窗内绝对行号
    clamp = np.maximum(pos, bstart[idx_b][:, None])             # 低于块首→clamp(=复制块首行, 左pad)
    return base[clamp]                                          # (B,W,C)


def _default_recon_forward(model, x):
    """默认重构前向: 若模型返回 tuple(如 AT 的 enc_out,series,...) 取第0项。"""
    o = model(x)
    return o[0] if isinstance(o, (tuple, list)) else o


def run_recon(name: str, build_model, W: int, *, forward_fn=None, laststep: bool = False,
              epochs: int = 1, lr: float = 1e-3, batch: int = 256,
              score_batch: int = 1024, extra: dict | None = None) -> None:
    """重构式桥接(AT全窗 / TranAD末步): 训正常窗最小化重构MSE; 分数=窗末逐通道MSE均值。

    laststep=False: forward_fn(model,x)->(B,W,C) 全窗重构, 分数取 [:,-1]。
    laststep=True : forward_fn(model,x)->(B,C)   仅重构窗末步(TranAD), 分数=该步MSE。
    """
    _seed(0)
    forward_fn = forward_fn or _default_recon_forward
    splits = {s: _load_split(s) for s in ("train", "val", "test")}
    bst = {s: _block_start(splits[s][1]) for s in splits}
    C = splits["train"][0].shape[1]
    model = build_model(W, C).to(DEV)

    b0, _, i0, y0 = splits["train"]
    train_idx = i0[y0 == 0]                                     # 无监督: 只学正常窗
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    t = now()
    model.train()
    rng = np.random.default_rng(0)
    for _ep in range(epochs):
        perm = rng.permutation(len(train_idx))
        for k in range(0, len(perm), batch):
            ib = train_idx[perm[k:k + batch]]
            x = torch.from_numpy(_win_batch(b0, bst["train"], ib, W)).float().to(DEV)
            loss = lossf(forward_fn(model, x), x[:, -1, :] if laststep else x)
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def _score(split):
        base, _tb, idx, _y = splits[split]
        model.eval()
        out = []
        for k in range(0, len(idx), score_batch):
            ib = idx[k:k + score_batch]
            x = torch.from_numpy(_win_batch(base, bst[split], ib, W)).float().to(DEV)
            xhat = forward_fn(model, x)
            if laststep:
                e = ((x[:, -1, :] - xhat) ** 2).mean(dim=-1)    # (B,) 窗末步逐通道MSE
            else:
                e = ((x - xhat) ** 2).mean(dim=-1)[:, -1]       # (B,) 窗末逐通道MSE均值
            out.append(e.float().cpu().numpy())
        return np.concatenate(out)

    sva, ste = _score("val"), _score("test")
    ex = {"device": str(DEV), "win": W, "epochs": epochs, "variant": "recon-error-only",
          "note": "原生架构+1epoch重构误差变体, 非论文完整训练"}
    ex.update(extra or {})
    report(name, splits["val"][3], sva, splits["test"][3], ste, now() - t, extra=ex)


def run_forecast(name: str, build_model, W: int, *, forward_fn, horizon: int = 1,
                 epochs: int = 1, lr: float = 1e-3, batch: int = 256,
                 score_batch: int = 1024, extra: dict | None = None) -> None:
    """预测式桥接(TriTrackNet/wt): 用窗内历史预测末端 horizon 步; 分数=末端预测误差。

    forward_fn(model, hist)->pred, 其中 hist=(B, W-horizon, C), pred 对齐窗末 horizon 步 (B, horizon, C)。
    """
    _seed(0)
    splits = {s: _load_split(s) for s in ("train", "val", "test")}
    bst = {s: _block_start(splits[s][1]) for s in splits}
    C = splits["train"][0].shape[1]
    H = max(1, int(horizon))
    model = build_model(W - H, C, H).to(DEV)

    b0, _, i0, y0 = splits["train"]
    train_idx = i0[y0 == 0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    t = now()
    model.train()
    rng = np.random.default_rng(0)
    for _ep in range(epochs):
        perm = rng.permutation(len(train_idx))
        for k in range(0, len(perm), batch):
            ib = train_idx[perm[k:k + batch]]
            win = torch.from_numpy(_win_batch(b0, bst["train"], ib, W)).float().to(DEV)
            hist, fut = win[:, :W - H, :], win[:, W - H:, :]
            loss = lossf(forward_fn(model, hist), fut)
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def _score(split):
        base, _tb, idx, _y = splits[split]
        model.eval()
        out = []
        for k in range(0, len(idx), score_batch):
            ib = idx[k:k + score_batch]
            win = torch.from_numpy(_win_batch(base, bst[split], ib, W)).float().to(DEV)
            hist, fut = win[:, :W - H, :], win[:, W - H:, :]
            pred = forward_fn(model, hist)
            e = ((pred - fut) ** 2).mean(dim=-1).mean(dim=-1)   # (B,) 末端 horizon 步预测误差均值
            out.append(e.float().cpu().numpy())
        return np.concatenate(out)

    sva, ste = _score("val"), _score("test")
    ex = {"device": str(DEV), "win": W, "horizon": H, "epochs": epochs, "variant": "pred-error-only",
          "note": "原生架构+1epoch预测误差变体, 非论文完整训练"}
    ex.update(extra or {})
    report(name, splits["val"][3], sva, splits["test"][3], ste, now() - t, extra=ex)
