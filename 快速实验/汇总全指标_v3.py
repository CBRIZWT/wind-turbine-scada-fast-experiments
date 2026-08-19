# -*- coding: utf-8 -*-
"""汇总 metrics-v3 快速实验、计算四轴帕累托前沿并生成中文科研报告。"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE / "快速实验结果_真实故障_allmetrics_v3"
OUTPUT_ROOT = HERE.parent / "实验结果" / "统一评价指标_v3" / "快速实验"
PARETO_AXES = (
    "event_recall", "auprc", "lead_minutes_median", "false_alarm_segments_per_turbine_day"
)
MAXIMIZE = {"event_recall", "auprc", "lead_minutes_median"}
MINIMIZE = {"false_alarm_segments_per_turbine_day"}


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def pareto_front(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """返回四轴完整且非支配的记录；缺失轴不插补、不进入前沿。"""
    eligible = [dict(row) for row in rows if all(_finite(row.get(k)) for k in PARETO_AXES)]

    def dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        weak = []
        strict = []
        for key in MAXIMIZE:
            weak.append(float(a[key]) >= float(b[key]))
            strict.append(float(a[key]) > float(b[key]))
        for key in MINIMIZE:
            weak.append(float(a[key]) <= float(b[key]))
            strict.append(float(a[key]) < float(b[key]))
        return all(weak) and any(strict)

    return [row for i, row in enumerate(eligible)
            if not any(i != j and dominates(other, row) for j, other in enumerate(eligible))]


def flatten_record(rec: Dict[str, Any], path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """把单run展开为工作点×split宽表及指标长表，保留每项状态。"""
    base = {
        "farm": rec.get("farm"), "model": rec.get("model"), "seed": rec.get("seed"),
        "label_mode": rec.get("label_mode"), "preprocess_variant": rec.get("preprocess_variant"),
        "external_protocol": rec.get("external_protocol"), "representation": rec.get("representation"),
        "metrics_path": str(path),
    }
    wide: List[Dict[str, Any]] = []
    long: List[Dict[str, Any]] = []
    for workpoint, payload in rec.get("workpoints", {}).items():
        for split in ("test", "val"):
            evaluated = payload.get(split)
            if not isinstance(evaluated, dict):
                continue
            metrics = evaluated.get("metrics", {})
            statuses = evaluated.get("metric_status", {})
            row = {**base, "workpoint": workpoint, "split": split, **metrics}
            selection = payload.get("selection", {})
            row["selection_fallback"] = selection.get("fallback")
            wide.append(row)
            for metric, value in metrics.items():
                long.append({**base, "workpoint": workpoint, "split": split,
                             "metric": metric, "value": value, "status": statuses.get(metric)})
    return wide, long


def discover_records(source_root: Path = SOURCE_ROOT) -> List[Tuple[Path, Dict[str, Any]]]:
    records = []
    for path in sorted(Path(source_root).rglob("metrics.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if rec.get("schema_version") == "metrics-v3":
            records.append((path, rec))
    return records


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        if not fields:
            fh.write("")
            return
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rank_consistency(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fields = ["event_f1", "event_recall", "auprc", "lead_minutes_median",
              "false_alarm_segments_per_turbine_day", "accuracy", "point_adjust_f1_appendix"]
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    for field in fields:
        frame[field] = pd.to_numeric(frame.get(field), errors="coerce")
    # 所有列统一为“越大越好”的秩方向。
    frame["false_alarm_segments_per_turbine_day"] *= -1
    ranks = frame[fields].rank(method="average", ascending=False)
    corr = ranks.corr(method="pearson", min_periods=3)
    return [{"metric_a": a, "metric_b": b,
             "spearman": None if pd.isna(corr.loc[a, b]) else float(corr.loc[a, b])}
            for a in fields for b in fields]


def _best(rows: Sequence[Dict[str, Any]], key: str, *, minimize: bool = False) -> Dict[str, Any] | None:
    candidates = [row for row in rows if _finite(row.get(key))]
    if not candidates:
        return None
    return (min if minimize else max)(candidates, key=lambda row: float(row[key]))


def _fmt(value: Any, digits: int = 4) -> str:
    return "空" if not _finite(value) else f"{float(value):.{digits}f}"


def _report(pen_rows: Sequence[Dict[str, Any]], kel_rows: Sequence[Dict[str, Any]],
            hill_rows: Sequence[Dict[str, Any]], front: Sequence[Dict[str, Any]],
            manifest: Dict[str, Any]) -> str:
    event_counts = sorted({int(float(r["n_events"])) for r in kel_rows if _finite(r.get("n_events"))})
    lines = [
        "# 真实故障快速实验全指标与帕累托分析（metrics-v3）", "",
        "> 本报告由实际 metrics.json 汇总生成。全部快速结果固定 seed=0，只用于探索性筛选；不把单次结果表述为多种子稳定冠军。", "",
        "## 完整性与适用范围", "",
        f"- 发现真标签run：{manifest['true_labeled_runs']} / 94；Hill无标签协议run：{manifest['hill_unlabeled_runs']} / 24。",
        f"- 指标键一致：{'是' if manifest['metric_key_schema_consistent'] else '否'}；完成门禁：{'通过' if manifest['complete'] else '未通过'}。",
        "- Penmanshiel用于主要量化比较；Kelmarsh仅作案例/敏感性证据；Hill没有真实温度故障标签，不进入性能排名。",
    ]
    if event_counts:
        lines.append(f"- Kelmarsh测试真实事件数（由产物读取）：{event_counts}。")
    lines += ["", "## 结论：最合适的项目评价指标", "",
              "项目不应由单一 Accuracy、PA-F1 或 event_f1 决定。最合适的核心评价是四轴组合：事件召回率、AUPRC、预警提前量中位数，以及误报报警段/机组日；event_f1保留为单值参考。推荐候选取四轴帕累托前沿，而不是宣布唯一冠军。", "",
              "## Penmanshiel 帕累托前沿", ""]
    if front:
        lines += ["| 模型 | 工作点 | 事件召回 | AUPRC | 提前量中位数(min) | 误报段/机组日 | event_f1 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for row in sorted(front, key=lambda r: (-float(r["event_recall"]), float(r["false_alarm_segments_per_turbine_day"]))):
            lines.append(f"| {row.get('model')} | {row.get('workpoint')} | {_fmt(row.get('event_recall'))} | {_fmt(row.get('auprc'))} | {_fmt(row.get('lead_minutes_median'), 1)} | {_fmt(row.get('false_alarm_segments_per_turbine_day'))} | {_fmt(row.get('event_f1'))} |")
    else:
        lines.append("当前没有四个帕累托轴均非空的Penmanshiel记录，不能构造前沿。")
    lines += ["", "## 各单项‘数值最好’与是否进入前沿", ""]
    front_ids = {(x.get("model"), x.get("workpoint")) for x in front}
    for label, key, minimize in (
        ("Accuracy", "accuracy", False), ("PA-F1（附录）", "point_adjust_f1_appendix", False),
        ("event_f1（参考）", "event_f1", False), ("事件召回", "event_recall", False),
        ("AUPRC", "auprc", False), ("提前量中位数", "lead_minutes_median", False),
        ("误报段/机组日", "false_alarm_segments_per_turbine_day", True),
    ):
        row = _best(pen_rows, key, minimize=minimize)
        if row:
            on_front = (row.get("model"), row.get("workpoint")) in front_ids
            lines.append(f"- {label}：{row.get('model')} / {row.get('workpoint')} = {_fmt(row.get(key))}；帕累托前沿：{'是' if on_front else '否'}。")
    lines += ["", "## 指标差异的科研解释", "",
              "- 极端类别不平衡时，Accuracy主要受健康点主导；AUPRC更直接反映异常类排序质量。",
              "- PA-F1把事件内命中扩展为整段命中，可能产生明显虚高，因此只放附录。",
              "- 点级误报可能碎片化成多个报警段；运维负担应看误报段/机组日，而不是只看误报点数。",
              "- 三个验证集冻结工作点揭示阈值敏感性：低误报与高召回通常不可同时最优。",
              "- 跨风场特征分布和故障机制不同，零样本迁移可能发生域漂移；Hill无标签结果只能报告分数分布与报警负担，不能报告性能。",
              "- Kelmarsh事件数过少时，单次检出与否会导致事件指标大幅跳变，不能用它单独宣布冠军。", "",
              "## 方法依据（原始论文）", "",
              "- Tatbul et al. (2018), *Precision and Recall for Time Series*, NeurIPS 2018: https://arxiv.org/abs/1803.03639",
              "- Huet et al. (2022), *Local Evaluation of Time Series Anomaly Detection Algorithms*, KDD 2022: https://arxiv.org/abs/2206.13167",
              "- Kim et al. (2022; arXiv 2021), *Towards a Rigorous Evaluation of Time-series Anomaly Detection*, AAAI 2022: https://arxiv.org/abs/2109.05257",
              "- Saito & Rehmsmeier (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, PLOS ONE: https://doi.org/10.1371/journal.pone.0118432",
              "", "## 复现说明", "",
              "阈值和score方向只由验证集决定，测试标签不参与选择。缺失指标保留空值与状态，不写NaN；Hill的所有标签性能状态均为 external_unlabeled。完整明细见同目录宽表、长表、帕累托表、Spearman表和运行清单。", ""]
    return "\n".join(lines)


def build_summary(source_root: Path = SOURCE_ROOT, output_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    found = discover_records(source_root)
    wide: List[Dict[str, Any]] = []
    long: List[Dict[str, Any]] = []
    metric_key_sets = []
    for path, rec in found:
        w, l = flatten_record(rec, path)
        wide.extend(w)
        long.extend(l)
        for payload in rec.get("workpoints", {}).values():
            for split in ("val", "test"):
                if isinstance(payload.get(split), dict):
                    metric_key_sets.append(frozenset(payload[split].get("metrics", {})))
    test_rows = [r for r in wide if r.get("split") == "test"]
    pen = [r for r in test_rows if r.get("farm") == "penmanshiel" and r.get("label_mode") != "external_unlabeled"]
    kel = [r for r in test_rows if r.get("farm") == "kelmarsh" and r.get("label_mode") != "external_unlabeled"]
    hill = [r for r in test_rows if r.get("farm") == "hill_of_towie"]
    front = pareto_front(pen)
    for row in test_rows:
        row["pareto_eligible"] = all(_finite(row.get(k)) for k in PARETO_AXES)
        row["on_penmanshiel_pareto_front"] = (
            row.get("farm") == "penmanshiel"
            and any(row.get("model") == x.get("model") and row.get("workpoint") == x.get("workpoint") for x in front)
        )
    true_runs = len({r[0].parent for r in found if r[1].get("label_mode") != "external_unlabeled"})
    hill_runs = len({r[0].parent for r in found if r[1].get("label_mode") == "external_unlabeled"})
    schema_consistent = bool(metric_key_sets) and len(set(metric_key_sets)) == 1
    hill_status_ok = all(x.get("status") == "external_unlabeled" for x in long
                         if x.get("farm") == "hill_of_towie" and x.get("metric") != "threshold")
    manifest = {
        "schema_version": "fast-summary-manifest-v3", "source_root": str(source_root),
        "expected_true_labeled_runs": 94, "true_labeled_runs": true_runs,
        "expected_hill_unlabeled_runs": 24, "hill_unlabeled_runs": hill_runs,
        "metric_key_schema_consistent": schema_consistent,
        "hill_label_metrics_external_unlabeled": hill_status_ok,
        "n_wide_rows": len(wide), "n_long_rows": len(long), "n_pareto": len(front),
    }
    manifest["complete"] = true_runs == 94 and hill_runs == 24 and schema_consistent and hill_status_ok
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "全指标宽表.csv", wide)
    _write_csv(output_root / "全指标长表.csv", long)
    _write_csv(output_root / "帕累托前沿.csv", front)
    _write_csv(output_root / "Spearman排名一致性.csv", _rank_consistency(pen))
    (output_root / "运行清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_root / "真实故障全指标对比分析.md").write_text(
        _report(pen, kel, hill, front, manifest), encoding="utf-8"
    )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = ap.parse_args()
    manifest = build_summary(args.source_root, args.output_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
