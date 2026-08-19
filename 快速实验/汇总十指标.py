# -*- coding: utf-8 -*-
"""汇总十指标.py — 汇总全部模型的十项实验指标并按异常判别力排序。

十项指标: Accuracy / AUC / Recall / F1 / R² / MAE / RMSE / Precision / LeadTime / MAPE

## 排序依据: 为什么用 AUPRC lift 而不是 F1

本任务基率极低 (kelmarsh 0.0116%, penmanshiel 0.1347%)。该基率下精确率有结构性上限:
即使 ROC-AUC=0.90, 在任何有运营意义的召回上精确率也上不去, 因而 F1 无法反映
"模型对异常的判别能力", 它主要反映基率。

AUPRC 的随机基线恰等于基率, 故

    auprc_lift = AUPRC / base_rate

剥离了基率影响, 是极端不平衡下跨风场可比的判别力口径 (lift=1 即与随机无异)。
主排序用 lift, 次排序用 ROC-AUC。F1/精确率照常输出但不作排序依据。

用法:
    python 汇总十指标.py --result-set 快速实验结果_十指标_20260809
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent

# 十项指标 → v3 结果键名。brier_* 是"报警分数作为 P(y=1) 的回归质量",
# 不是温度回归 (温度回归原义在 NA_METRICS 里声明不可算)。
TEN = [
    ("Accuracy",  "accuracy"),
    ("AUC",       "roc_auc"),
    ("Recall",    "recall"),
    ("F1",        "f1"),
    ("R2",        "r2"),
    ("MAE",       "brier_mae"),
    ("RMSE",      "brier_rmse"),
    ("Precision", "precision"),
    ("LeadTime_h", None),          # 由 lead_minutes_median / 60 得出
    ("MAPE",      "mape_on_positives"),
]


def _num(v):
    if v is None:
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


def collect(result_root: Path, workpoint: str = "balanced") -> pd.DataFrame:
    rows = []
    for mp in sorted(result_root.rglob("metrics.json")):
        try:
            rec = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        test = rec.get("test") or {}
        if not test:
            continue
        row = {"farm": rec.get("farm", mp.parent.parent.name),
               "model": rec.get("model", mp.parent.name)}
        for label, key in TEN:
            if label == "LeadTime_h":
                lm = _num(test.get("lead_minutes_median"))
                row[label] = lm / 60.0 if math.isfinite(lm) else float("nan")
            else:
                row[label] = _num(test.get(key))
        row["AUPRC"] = _num(test.get("auprc"))
        row["base_rate"] = _num(test.get("base_rate"))
        row["auprc_lift"] = _num(test.get("auprc_lift"))
        row["event_f1"] = _num(test.get("event_f1"))
        row["FAR"] = _num(test.get("false_alarm_segments_per_turbine_day"))
        row["n_events"] = _num(test.get("n_events"))
        row["n_detected"] = _num(test.get("n_detected"))
        row["tier1_n"] = _num(test.get("tier1_n"))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-set", default="快速实验结果_十指标_20260809")
    ap.add_argument("--farm", default="")
    a = ap.parse_args()

    root = HERE / a.result_set
    if not root.exists():
        sys.exit(f"结果集不存在: {root}")
    df = collect(root)
    if df.empty:
        sys.exit("未找到任何 metrics.json")
    if a.farm:
        df = df[df["farm"] == a.farm]

    out_dir = HERE.parent / "实验结果" / "十指标排行"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 主排序: 判别力 lift 降序; 次: ROC-AUC 降序
    df = df.sort_values(["auprc_lift", "AUC"], ascending=[False, False], na_position="last")
    df.to_csv(out_dir / "十指标全表.csv", index=False, encoding="utf-8-sig")

    cols = ["model"] + [c for c, _ in TEN] + ["AUPRC", "auprc_lift"]
    for farm, g in df.groupby("farm", sort=True):
        g2 = g.sort_values(["auprc_lift", "AUC"], ascending=[False, False], na_position="last")
        g2.to_csv(out_dir / f"十指标_{farm}.csv", index=False, encoding="utf-8-sig")
        base = g2["base_rate"].dropna()
        print(f"\n{'='*118}")
        print(f"{farm}   模型数={len(g2)}   基率={base.iloc[0]*100:.4f}%   "
              f"tier1_n={int(g2['tier1_n'].iloc[0]) if g2['tier1_n'].notna().any() else '?'}   "
              f"事件数={int(g2['n_events'].iloc[0]) if g2['n_events'].notna().any() else '?'}")
        print(f"按【异常判别力 auprc_lift = AUPRC/基率】排序 (lift=1 即与随机无异)")
        print('='*118)
        show = g2[cols].head(25).copy()
        for c in ["Accuracy", "AUC", "Recall", "F1", "R2", "MAE", "RMSE", "Precision", "MAPE", "AUPRC"]:
            show[c] = show[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
        show["LeadTime_h"] = show["LeadTime_h"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
        show["auprc_lift"] = show["auprc_lift"].map(lambda v: f"{v:.1f}×" if pd.notna(v) else "—")
        print(show.to_string(index=False))

    print(f"\n落盘: {out_dir}")


if __name__ == "__main__":
    main()
