# -*- coding: utf-8 -*-
"""性能排行热图.py — 以 AUPRC 为核心指标给模型排名, 画"纵轴=全部指标, 横轴=按AUPRC排名的模型"
的热图 (每 farm 一张 PNG)。方向感知配色: 绿=好/红=差 (越小越好的指标已翻转); 计数/上下文
指标单列灰底块 (无好坏含义, 仅参考)。数据源 = 快速实验结果/全指标汇总.csv (汇总.py 产出)。
2026-07-19。cmd 直接运行: python 性能排行热图.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---- 中文字体 (Windows: 微软雅黑) ----
for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False

RESULT = Path(__file__).resolve().parent / "快速实验结果"
FARMS = ("kelmarsh", "penmanshiel", "hill_of_towie")
CORE = "test_auprc"

# 评价指标 (方向感知; 绿=好). 核心 auprc 置首。
EVAL_HIGHER = ["test_auprc", "test_auc", "test_f1", "test_precision", "test_recall",
               "test_accuracy", "test_balanced_accuracy", "test_mcc", "test_r2",
               "test_range_f1", "test_range_precision", "test_range_recall",
               "test_affiliation_f1", "test_affiliation_precision", "test_affiliation_recall",
               "test_seg_event_f1", "test_seg_event_precision", "test_seg_event_recall",
               "test_seg_lead_rows_median"]
EVAL_LOWER = ["test_loss", "test_mse", "test_mae", "test_rmse", "test_nrmse",
              "test_far_per_day", "test_fp", "test_fn"]                       # 越小越好
APPENDIX = ["test_pa_point_adjust_f1"]                                         # PA附录(虚高, 标注)
CONTEXT = ["test_tp", "test_tn", "test_seg_n_detected", "test_seg_n_events",
           "test_valid_count", "test_threshold"]                              # 计数/上下文(灰, 无好坏)

# 行标签 (去 test_ 前缀 + 中文注释)
LABEL = {
    "test_auprc": "AUPRC ★核心", "test_auc": "ROC-AUC", "test_f1": "F1",
    "test_precision": "Precision", "test_recall": "Recall", "test_accuracy": "Accuracy",
    "test_balanced_accuracy": "Balanced-Acc", "test_mcc": "MCC", "test_r2": "R²(Brier)",
    "test_range_f1": "range-F1(Tatbul)", "test_range_precision": "range-P",
    "test_range_recall": "range-R", "test_affiliation_f1": "affiliation-F1(Huet)",
    "test_affiliation_precision": "affil-P", "test_affiliation_recall": "affil-R",
    "test_seg_event_f1": "seg事件F1(命中即命中)", "test_seg_event_recall": "seg事件召回",
    "test_seg_event_precision": "seg事件精确", "test_seg_lead_rows_median": "seg提前(行)",
    "test_loss": "loss/NLL↓", "test_mse": "MSE(Brier)↓", "test_mae": "MAE↓",
    "test_rmse": "RMSE↓", "test_nrmse": "NRMSE↓", "test_far_per_day": "FAR/天↓",
    "test_fp": "FP↓", "test_fn": "FN↓",
    "test_pa_point_adjust_f1": "PA-F1(附录/虚高)",
    "test_tp": "TP", "test_tn": "TN", "test_seg_n_detected": "检出段数",
    "test_seg_n_events": "事件段数", "test_valid_count": "参评行数", "test_threshold": "阈值",
}


def _row_norm(vals, higher_better=True):
    """行内 min-max 归一到[0,1], 方向感知 (1=好)。全等或全nan → 0.5。"""
    v = np.asarray(vals, dtype=float)
    fin = np.isfinite(v)
    if fin.sum() < 2 or np.nanmax(v[fin]) == np.nanmin(v[fin]):
        z = np.full_like(v, 0.5)
    else:
        lo, hi = np.nanmin(v[fin]), np.nanmax(v[fin])
        z = (v - lo) / (hi - lo)
        if not higher_better:
            z = 1.0 - z
    z[~fin] = np.nan
    return z


def _fmt(x):
    if not np.isfinite(x):
        return ""
    ax = abs(x)
    if ax >= 1000:
        return f"{x:.0f}"
    if ax >= 1:
        return f"{x:.2f}" if ax < 100 else f"{x:.1f}"
    return f"{x:.3f}"


def draw_farm(df_farm, farm):
    g = df_farm.sort_values(CORE, ascending=False).reset_index(drop=True)
    models = [m.split("_", 1)[-1] if "_" in m else m for m in g["model"]]  # 去数字前缀
    nums = [m.split("_", 1)[0] for m in g["model"]]
    xlab = [f"{n}\n{nm}" for n, nm in zip(nums, models)]

    eval_rows = EVAL_HIGHER + EVAL_LOWER + APPENDIX
    lower_set = set(EVAL_LOWER)
    Z_eval = np.vstack([_row_norm(g[k].to_numpy(), higher_better=(k not in lower_set))
                        for k in eval_rows])
    Z_ctx = np.vstack([_row_norm(g[k].to_numpy(), higher_better=True) for k in CONTEXT])
    raw_eval = np.vstack([g[k].to_numpy(dtype=float) for k in eval_rows])
    raw_ctx = np.vstack([g[k].to_numpy(dtype=float) for k in CONTEXT])

    nM = len(models)
    ncol = nM
    fig_w = max(14, 0.42 * ncol + 3)
    fig_h = 0.42 * (len(eval_rows) + len(CONTEXT)) + 2.2
    fig, (axE, axC) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h), dpi=130,
        gridspec_kw={"height_ratios": [len(eval_rows), len(CONTEXT)], "hspace": 0.06})

    # ---- 评价块 (RdYlGn, 绿=好) ----
    axE.imshow(Z_eval, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axE.set_yticks(range(len(eval_rows)))
    axE.set_yticks(range(len(eval_rows)))
    axE.set_yticklabels([LABEL[k] for k in eval_rows], fontsize=8)
    axE.set_xticks([])
    axE.axhline(len(EVAL_HIGHER) - 0.5, color="k", lw=0.6, ls="--", alpha=0.4)
    axE.axhline(len(EVAL_HIGHER) + len(EVAL_LOWER) - 0.5, color="k", lw=0.6, ls="--", alpha=0.4)
    for i in range(len(eval_rows)):
        for j in range(ncol):
            v = raw_eval[i, j]
            if np.isfinite(v):
                axE.text(j, i, _fmt(v), ha="center", va="center", fontsize=4.2, color="black")
    axE.set_title(f"{farm} — 模型性能排行 (横轴=按核心指标 AUPRC 降序; 纵轴=全部指标; "
                  f"绿=好/红=差, ↓项越小越好已翻转)", fontsize=11, pad=8)
    # 排名条 (第一行上方标注名次)
    for j in range(ncol):
        axE.text(j, -0.8, f"#{j+1}", ha="center", va="center", fontsize=5, color="dimgray")

    # ---- 上下文块 (Greys, 无好坏) ----
    axC.imshow(Z_ctx, aspect="auto", cmap="Greys", vmin=0, vmax=1, alpha=0.55)
    axC.set_yticks(range(len(CONTEXT)))
    axC.set_yticklabels([LABEL[k] for k in CONTEXT], fontsize=8)
    axC.set_xticks(range(ncol))
    axC.set_xticklabels(xlab, fontsize=5.5, rotation=90)
    for i in range(len(CONTEXT)):
        for j in range(ncol):
            v = raw_ctx[i, j]
            if np.isfinite(v):
                axC.text(j, i, _fmt(v), ha="center", va="center", fontsize=4.2, color="black")
    axC.set_ylabel("上下文/计数\n(灰,无好坏)", fontsize=7)

    out = RESULT / f"性能排行_{farm}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    top = g.iloc[0]
    print(f"[{farm}] → {out}  冠军(AUPRC)={top['model']} AUPRC={top[CORE]:.4f}")
    return out


def main():
    csv = RESULT / "全指标汇总.csv"
    if not csv.exists():
        print("缺 全指标汇总.csv, 先跑 汇总.py"); return 1
    df = pd.read_csv(csv)
    for farm in FARMS:
        sub = df[df["farm"] == farm]
        if len(sub):
            draw_farm(sub, farm)
    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
