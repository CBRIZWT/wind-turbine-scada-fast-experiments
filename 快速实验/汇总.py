# -*- coding: utf-8 -*-
"""汇总.py — 读 快速实验结果/<farm>/*/metrics.json。
核心指标 = AUPRC (排序键), 其余为辅助指标。2026-07-19 改核心指标为 AUPRC。
产出:
  每farm  汇总.csv          (按 test_auprc 降序, 全指标列)
  根目录  全指标汇总.csv     (farm×模型, 全指标)
  根目录  指标矩阵_<farm>.csv (行=全部指标, 列=按AUPRC排名的模型; 供"纵轴全指标"可视化)
"""
import json                       # 解析各模型的 metrics.json
import sys                        # 重配置标准输出编码
from pathlib import Path          # 路径拼接

if hasattr(sys.stdout, "reconfigure"):                       # 若支持
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # 强制UTF-8防中文乱码

RESULT = Path(__file__).resolve().parent / "快速实验结果"     # 结果根目录/<farm>/<模型>/metrics.json
FARMS = ("kelmarsh", "penmanshiel", "hill_of_towie")
CORE = "auprc"                                                # ★ 核心指标 (排序键)

# 全部指标列 (核心 auprc 置首; 其余辅助); val 只留 auprc/f1 作参考
TEST_KEYS = ("auprc", "auc", "f1", "precision", "recall", "accuracy",
             "balanced_accuracy", "mcc", "loss",
             "mse", "mae", "rmse", "r2", "nrmse",
             "range_precision", "range_recall", "range_f1",
             "affiliation_precision", "affiliation_recall", "affiliation_f1",
             "far_per_day", "pa_point_adjust_f1",
             "seg_event_recall", "seg_event_precision", "seg_event_f1", "seg_lead_rows_median",
             "seg_n_events", "seg_n_detected", "seg_n_alarm_segments", "seg_n_alarm_hit",
             "tp", "tn", "fp", "fn",
             "valid_count", "threshold")

all_rows = []
for farm in FARMS:
    fdir = RESULT / farm
    if not fdir.is_dir():
        continue
    rows = []
    for p in sorted(fdir.glob("*/metrics.json")):           # 遍历该farm所有模型
        d = json.loads(p.read_text(encoding="utf-8"))
        t = d["test"]
        rows.append([d["model"], d["val"].get("auprc"), d["val"].get("f1")] +
                    [t.get(k) for k in TEST_KEYS] + [d.get("elapsed_sec")])
    # ★ 按 test_auprc(核心指标) 降序; nan 沉底
    ci = 3 + TEST_KEYS.index(CORE)
    rows.sort(key=lambda r: -(r[ci] if isinstance(r[ci], (int, float)) and r[ci] == r[ci] else -1))

    K = {k: i + 3 for i, k in enumerate(TEST_KEYS)}          # 键名→行内索引 (0=model,1=val_auprc,2=val_f1)
    print(f"\n===== {farm} — 按核心指标 AUPRC 排名 ({len(rows)} 模型) =====")
    print(f"{'#':>2} {'模型':<20}{'AUPRC':>7}{'AUC':>7}{'F1':>7}{'affF1':>7}"
          f"{'segEvtF1':>9}{'FAR/日':>8}{'PA-F1':>7}")
    for rank, r in enumerate(rows, 1):
        g = lambda i: (f"{r[i]:.4f}" if isinstance(r[i], float) else str(r[i]))
        print(f"{rank:>2} {r[0]:<20}{g(K['auprc']):>7}{g(K['auc']):>7}{g(K['f1']):>7}"
              f"{g(K['affiliation_f1']):>7}{g(K['seg_event_f1']):>9}"
              f"{g(K['far_per_day']):>8}{g(K['pa_point_adjust_f1']):>7}")

    hdr = "rank,model,val_auprc,val_f1," + ",".join("test_" + k for k in TEST_KEYS) + ",elapsed_sec"
    lines = [",".join([str(i + 1)] + [str(x) for x in r]) for i, r in enumerate(rows)]
    (fdir / "汇总.csv").write_text(hdr + "\n" + "\n".join(lines), encoding="utf-8-sig")

    # 指标矩阵: 行=全部指标, 列=按AUPRC排名的模型 (供"纵轴=全部指标"可视化)
    models = [r[0] for r in rows]
    mhdr = "metric," + ",".join(models)
    mlines = []
    for k in TEST_KEYS:
        vals = [r[K[k]] for r in rows]
        mlines.append("test_" + k + "," + ",".join(str(v) for v in vals))
    (RESULT / f"指标矩阵_{farm}.csv").write_text(mhdr + "\n" + "\n".join(mlines), encoding="utf-8-sig")
    print(f"→ {fdir / '汇总.csv'}  +  指标矩阵_{farm}.csv")
    all_rows += [[farm, i + 1] + r for i, r in enumerate(rows)]

if all_rows:
    hdr = "farm,rank,model,val_auprc,val_f1," + ",".join("test_" + k for k in TEST_KEYS) + ",elapsed_sec"
    (RESULT / "全指标汇总.csv").write_text(
        hdr + "\n" + "\n".join(",".join(str(x) for x in r) for r in all_rows), encoding="utf-8-sig")
    print(f"\n共 {len(all_rows)} 行 (farm×模型), 核心指标=AUPRC → {RESULT / '全指标汇总.csv'}")
