# -*- coding: utf-8 -*-
"""汇总真实故障总排序 — 合并全部模型 (基础/组合/论文复现) 的事件级指标, 输出总排序。
主排序口径 (预注册): Penmanshiel test event_f1 (10 事件, 统计上有意义);
辅助视图: 检出/lead/FAR/AUC + Kelmarsh (1 事件, 仅参考)。"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
RES = HERE / "快速实验结果_真实故障"
KEYS = ["model", "family", "event_f1", "event_recall", "n_detected", "n_events",
        "lead_minutes_median", "far_per_day", "alarm_precision", "point_auc_test",
        "val_event_f1"]


def farm_table(farm: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RES / farm / "*/metrics.json")):
        r = json.load(open(f, encoding="utf-8"))
        row = {k: r.get(k) for k in KEYS}
        mm = r.get("model_metrics") or {}
        row["model_metric_note"] = "; ".join(
            f"{k}={round(v,3) if isinstance(v,(int,float)) else v}"
            for k, v in mm.items() if k != "note")[:90]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("event_f1", ascending=False)
    df.to_csv(RES / farm / "汇总.csv", index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    for farm in ("penmanshiel", "kelmarsh"):
        df = farm_table(farm)
        print(f"\n===== {farm}: {len(df)} 模型 (test 事件级, val 选阈) =====")
        print(df[KEYS].to_string(index=False))
