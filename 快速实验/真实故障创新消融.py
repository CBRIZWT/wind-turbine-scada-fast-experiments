# -*- coding: utf-8 -*-
"""真实故障创新消融 (2026-07-11) — 在 val 最优基模型分数上逐项叠加增强, 事件级口径量化 Δ。

增强项 (均来自本仓已复现论文方向, train/val-only 拟合, 杜绝 test 泄漏):
  A. EWMA 平滑        (综述主流早警骨架; 组合战役已证平滑是真实增益来源)
  B. CUSUM 持续性     (累积小偏置比逐点阈值灵敏; 缓慢过热物理先验)
  C. fleet-median     (同场同时刻跨机组中位扣除, 抑共模; Wilms WES 2025 已复现方向)
  D. 共形式健康分位阈 (val 健康段 (1−α) 分位定阈; Conformal Prediction 论文方向)

协议: 基模型 = 快速实验结果_真实故障/<farm>/汇总.csv 中 **val_event_f1 最高**者
      (以 val 选基, 不碰 test); 每变体超参仅 val 网格选 (event_f1), test 各评一次。
预注册达标线: test event_f1 相对基模型 ≥ +0.02 记"真提升" (与历史战役口径一致)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT / "SCADA数据集"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from 事件级评测 import event_level_metrics, select_threshold_event  # noqa: E402
from 真实故障事件级实验 import (load_variant, per_turbine_cusum,  # noqa: E402
                                per_turbine_ewma, rank_normalize)

DELTA_GATE = 0.02  # 预注册: test event_f1 提升 ≥ +0.02 记真提升


def fleet_median_subtract(score: np.ndarray, turbines: np.ndarray,
                          ts_ns: np.ndarray) -> np.ndarray:
    """同时刻跨机组中位扣除 (共模抑制)。秩归一后逐时间组减中位。"""
    s = rank_normalize(score)
    df = pd.DataFrame({"s": s, "t": ts_ns})
    med = df.groupby("t")["s"].transform("median").to_numpy()
    return s - med


def _standardize_by_val_healthy(s_val, s_test, y_val):
    """以 val 健康段 (label==0) 中位/IQR 标准化两段分数 (val-only 拟合)。"""
    base = s_val[(np.asarray(y_val) == 0) & np.isfinite(s_val)]
    med = float(np.median(base))
    iqr = float(np.subtract(*np.percentile(base, [75, 25]))) or 1.0
    return (s_val - med) / iqr, (s_test - med) / iqr


def eval_variant(name, s_val, s_test, data, lead, out_rows, params=None):
    thr, val_f1 = select_threshold_event(
        s_val, data["y_val"], data["ts_val"], data["turb_val"], data["ep_val"], lead_steps=lead)
    pred = np.zeros(len(s_test), dtype=int)
    fin = np.isfinite(s_test)
    pred[fin & (s_test >= thr)] = 1
    m = event_level_metrics(pred, data["y_test"], data["ts_test"], data["turb_test"],
                            data["ep_test"], lead_steps=lead)
    row = {"variant": name, "val_event_f1": round(float(val_f1), 4),
           "test_event_f1": round(float(m["event_f1"]), 4),
           "event_recall": round(float(m["event_recall"]), 4),
           "n_detected": m["n_detected"], "n_events": m["n_events"],
           "lead_min_median": m["lead_minutes_median"],
           "far_per_day": round(float(m["far_per_day"]), 4),
           "alarm_precision": round(float(m["alarm_precision"]), 4),
           "params": json.dumps(params or {}, ensure_ascii=False)}
    out_rows.append(row)
    print(f"  [{name}] val_eF1={val_f1:.3f} test_eF1={m['event_f1']:.3f} "
          f"检出={m['n_detected']}/{m['n_events']} FAR={m['far_per_day']:.3f}/天 {params or ''}",
          flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", default="penmanshiel")
    args = ap.parse_args()

    res_dir = HERE / "快速实验结果_真实故障" / args.farm
    summary = pd.read_csv(res_dir / "汇总.csv")
    base_name = summary.sort_values("val_event_f1", ascending=False).iloc[0]["model"]
    print(f"[{args.farm}] 基模型 (val_event_f1 最高) = {base_name}")

    data = load_variant(args.farm)
    lead = int(data["lead_steps"])
    s_val = np.load(res_dir / base_name / "score_val.npy").astype(float)
    s_test = np.load(res_dir / base_name / "score_test.npy").astype(float)

    rows = []
    base_row = eval_variant(f"base:{base_name}", s_val, s_test, data, lead, rows)
    base_f1 = base_row["test_event_f1"]

    # A. EWMA (alpha 仅 val 选)
    best = (None, -1.0)
    for a in (0.05, 0.1, 0.2):
        ev = per_turbine_ewma(s_val, data["turb_val"], alpha=a)
        _, f = select_threshold_event(ev, data["y_val"], data["ts_val"], data["turb_val"],
                                      data["ep_val"], lead_steps=lead)
        if f > best[1]:
            best = (a, f)
    a = best[0]
    eval_variant("A_EWMA", per_turbine_ewma(s_val, data["turb_val"], alpha=a),
                 per_turbine_ewma(s_test, data["turb_test"], alpha=a),
                 data, lead, rows, {"alpha": a})

    # B. CUSUM (k 仅 val 选; 分数先按 val 健康段标准化)
    zs_val, zs_test = _standardize_by_val_healthy(s_val, s_test, data["y_val"])
    best = (None, -1.0)
    for k in (0.5, 1.0, 2.0):
        cv = per_turbine_cusum(np.nan_to_num(zs_val, nan=0.0), data["turb_val"], k=k)
        _, f = select_threshold_event(cv, data["y_val"], data["ts_val"], data["turb_val"],
                                      data["ep_val"], lead_steps=lead)
        if f > best[1]:
            best = (k, f)
    k = best[0]
    eval_variant("B_CUSUM", per_turbine_cusum(np.nan_to_num(zs_val, nan=0.0), data["turb_val"], k=k),
                 per_turbine_cusum(np.nan_to_num(zs_test, nan=0.0), data["turb_test"], k=k),
                 data, lead, rows, {"k_iqr": k})

    # C. fleet-median 共模扣除
    eval_variant("C_fleet_median",
                 fleet_median_subtract(s_val, data["turb_val"], data["ts_val"]),
                 fleet_median_subtract(s_test, data["turb_test"], data["ts_test"]),
                 data, lead, rows)

    # D. 共形式健康分位阈 (阈值不走 event_f1 网格, 而是 val 健康分位; α 仅 val 选)
    base_h = s_val[(data["y_val"] == 0) & np.isfinite(s_val)]
    best = (None, -1.0)
    for alpha in (0.01, 0.005, 0.001):
        thr = float(np.quantile(base_h, 1 - alpha))
        pred = np.zeros(len(s_val), dtype=int)
        pred[np.isfinite(s_val) & (s_val >= thr)] = 1
        m = event_level_metrics(pred, data["y_val"], data["ts_val"], data["turb_val"],
                                data["ep_val"], lead_steps=lead)
        if m["event_f1"] > best[1]:
            best = (alpha, float(m["event_f1"]))
    alpha = best[0]
    thr = float(np.quantile(base_h, 1 - alpha))
    pred = np.zeros(len(s_test), dtype=int)
    pred[np.isfinite(s_test) & (s_test >= thr)] = 1
    m = event_level_metrics(pred, data["y_test"], data["ts_test"], data["turb_test"],
                            data["ep_test"], lead_steps=lead)
    rows.append({"variant": "D_共形健康分位阈", "val_event_f1": best[1],
                 "test_event_f1": round(float(m["event_f1"]), 4),
                 "event_recall": round(float(m["event_recall"]), 4),
                 "n_detected": m["n_detected"], "n_events": m["n_events"],
                 "lead_min_median": m["lead_minutes_median"],
                 "far_per_day": round(float(m["far_per_day"]), 4),
                 "alarm_precision": round(float(m["alarm_precision"]), 4),
                 "params": json.dumps({"alpha": alpha})})
    print(f"  [D_共形健康分位阈] test_eF1={m['event_f1']:.3f} α={alpha}", flush=True)

    # A+B 最优先叠加 (仅当 A/B 在 val 上均有效)
    ew_val = per_turbine_ewma(s_val, data["turb_val"], alpha=a)
    ew_test = per_turbine_ewma(s_test, data["turb_test"], alpha=a)
    z2_val, z2_test = _standardize_by_val_healthy(ew_val, ew_test, data["y_val"])
    eval_variant("AB_EWMA+CUSUM",
                 per_turbine_cusum(np.nan_to_num(z2_val, nan=0.0), data["turb_val"], k=k),
                 per_turbine_cusum(np.nan_to_num(z2_test, nan=0.0), data["turb_test"], k=k),
                 data, lead, rows, {"alpha": a, "k_iqr": k})

    df = pd.DataFrame(rows)
    df["delta_vs_base"] = (df["test_event_f1"] - base_f1).round(4)
    df["真提升(≥+0.02)"] = df["delta_vs_base"] >= DELTA_GATE
    df.to_csv(res_dir / "创新消融.csv", index=False, encoding="utf-8-sig")
    print("\n==== 消融汇总 ====")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
