# -*- coding: utf-8 -*-
"""准备数据.py — 从已验证的 v2 预处理产物生成快速实验数据 (每 farm 一次性运行)

来源: SCADA数据集/数据预处理/<farm>/ (chronological_v2+v2, NBM残差通道, RobustScaler仅train拟合)
  - train = train_sup.npy (未去污, 保留 A′ 正例 → 监督可训)  2016-2021
  - val   = val.npy   2022
  - test  = test.npy  2023-2024
标签 = A′ 过温早期预警 99/3/72: 正例=事件onset前72步(12h提前), 事件期=ignore(-1)剔除。
这是"预测未来", 不是"检测当前"。

用法: python 准备数据.py [--farm kelmarsh|penmanshiel|hill_of_towie]   (默认 kelmarsh)

产出到 快速实验数据/<farm>/:
  扁平 (ML+MLP):  X_flat_{split}.npy (N, D+6) + y_flat_{split}.npy (0/1)
      D+6 = 各farm NBM残差通道数D + 6因果衍生(跨通道max/正残差能量/W=72滚动均值/滚动max/
      72步斜率/6步增量), 与已验证的 监督早警.py 同配方; 特征只用过去+当前, 无未来泄漏。
  序列 (NN):      X_base_{split}.npy (T,D) + idx_seq_{split}.npy + y_seq_{split}.npy
      窗宽 W=36(6h), 窗口末端t的标签=y[t]; 惰性取窗防OOM。
  meta.json       口径与形状记录。

快速化(2026-07-18 多farm扩展): train 下采样步长按 farm 规模自适应 —— 锚定 kelmarsh
  历史口径 (T_train=1,840,547 → FLAT_STRIDE=4, 即目标≈46万 train 采样行; SEQ_STRIDE=3×FLAT):
  FLAT_STRIDE = max(1, round(T_train/460137))。kelmarsh 复现出 4/12 与历史完全一致;
  更大的 farm (如 hill_of_towie) 步长成比例放大, 使各 farm train 规模同量级 →
  各模型脚本内置的降采样保护 (KNN/GP等) 在三 farm 上行为一致。val/test 一律全量。
已知局限(如实): pooled 数组按时间排序交错多机组, 滚动/窗口跨机组混合 —— 与战役中
4个深度模型消费方式一致, 保持可比; 快速实验用于模型排序, 冠军确证需回全量+5seed。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent                      # 快速实验/
FARMS = ("kelmarsh", "penmanshiel", "hill_of_towie")

W_FEAT = 72        # 因果滚动特征窗 (12h, 同 监督早警.py)
K_RECENT = 6       # 近增量步数 (1h)
W_SEQ = 36         # 序列窗宽 (6h)
ANCHOR_ROWS = 460137  # kelmarsh 历史口径锚点: 1840547/4 ≈ 46万 train 采样行


def causal_flat_features(X: np.ndarray) -> np.ndarray:
    """(T,D)→(T,D+6) 逐点因果特征, 与 监督早警.py 同配方 (仅过去+当前)。"""
    maxc = X.max(axis=1)
    pose = np.mean(np.maximum(0.0, X) ** 2, axis=1)
    s = pd.Series(maxc)
    roll_mean = s.rolling(W_FEAT, min_periods=1).mean().to_numpy()
    roll_max = s.rolling(W_FEAT, min_periods=1).max().to_numpy()
    slope = maxc - s.shift(W_FEAT).fillna(s.iloc[0]).to_numpy()
    recent = maxc - s.shift(K_RECENT).fillna(s.iloc[0]).to_numpy()
    return np.column_stack([X, maxc, pose, roll_mean, roll_max, slope, recent]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", choices=FARMS, default="kelmarsh")
    args = ap.parse_args()
    src = HERE.parent / "SCADA数据集" / "数据预处理" / args.farm
    out = HERE / "快速实验数据" / args.farm
    out.mkdir(parents=True, exist_ok=True)

    files = {"train": ("train_sup.npy", "train_sup_labels.npy"),
             "val": ("val.npy", "val_labels.npy"),
             "test": ("test.npy", "test_labels.npy")}

    T_train = np.load(src / "train_sup.npy", mmap_mode="r").shape[0]
    flat_stride = max(1, round(T_train / ANCHOR_ROWS))     # kelmarsh→4 (与历史一致)
    seq_stride = 3 * flat_stride                           # kelmarsh→12 (与历史一致)

    meta = {"farm": args.farm, "split_id": "chronological_v2", "feature_version": "v2",
            "label_rule": "overtemp_A1 99/3/72 (正例=onset前72步早警; 事件期ignore剔除)",
            "W_feat": W_FEAT, "k_recent": K_RECENT, "W_seq": W_SEQ,
            "flat_train_stride": flat_stride, "seq_train_stride": seq_stride,
            "stride_rule": f"max(1, round(T_train/{ANCHOR_ROWS})) 锚定kelmarsh历史口径",
            "source": str(src), "flat_dim": None, "splits": {}}

    D = None
    for split, (xf, yf) in files.items():
        X = np.load(src / xf).astype(np.float32)
        y = np.load(src / yf).astype(np.int64)
        assert len(X) == len(y), (split, X.shape, y.shape)
        D = X.shape[1]

        # ---- 扁平表示: 全序列上算因果特征(保因果), 再剔 ignore, train 再下采样
        F = causal_flat_features(X)
        keep = np.where(y != -1)[0]
        if split == "train":
            keep = keep[::flat_stride]
        np.save(out / f"X_flat_{split}.npy", F[keep])
        np.save(out / f"y_flat_{split}.npy", y[keep].astype(np.int8))
        del F

        # ---- 序列表示: 基座 + 有效窗口末端索引 (剔 ignore, 且窗口完整)
        idx = np.where(y != -1)[0]
        idx = idx[idx >= W_SEQ - 1]
        if split == "train":
            idx = idx[::seq_stride]
        np.save(out / f"X_base_{split}.npy", X)
        np.save(out / f"idx_seq_{split}.npy", idx.astype(np.int64))
        np.save(out / f"y_seq_{split}.npy", y[idx].astype(np.int8))

        pos_flat = float(np.mean(y[keep] == 1))
        pos_seq = float(np.mean(y[idx] == 1))
        meta["splits"][split] = {"T": int(len(X)), "n_flat": int(len(keep)),
                                 "pos_flat": round(pos_flat, 4),
                                 "n_seq": int(len(idx)), "pos_seq": round(pos_seq, 4)}
        print(f"[{args.farm}/{split}] T={len(X):>9,}  flat n={len(keep):>9,} 正例率={pos_flat:.1%}  "
              f"seq n={len(idx):>9,} 正例率={pos_seq:.1%}")
        del X, y

    meta["flat_dim"] = int(D + 6)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成 → {out}  (flat_stride={flat_stride}, seq_stride={seq_stride})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
