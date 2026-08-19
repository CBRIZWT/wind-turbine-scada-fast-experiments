# -*- coding: utf-8 -*-
"""组合达标战役 (2026-07-12) — 预处理标签层优化(v2缓冲带) + 全模型重评 + 组合冲基线。

预注册协议 (防泄漏红线):
  基线   = 11_直方图梯度提升_监督 (v1 全模型排序冠军; 在 v2 口径下重训重评作对照)。
  目标   = 组合模型在 Penmanshiel test 的 5 项指标全面≥基线:
           event_f1↑, event_recall↑(≥), lead_minutes↑(≥), far_per_day↓(≤), point_auc↑(≥)。
  v2 标签 = v1 + 事前缓冲带 G=72 (CARE padding: [s−144, s−72) 步 ignore, 不再把
           更早的真预警惩罚为误报)。事件表/特征/切分/NBM/scaler 一律不变 (上游确定性
           复用, 文档化); 监督模型用 v2 train_sup 标签重训; 无监督分数确定性复用。
  组合   = 贪心前向选择 (仅 val v2 event_f1): 候选=全部模型分数, 融合=秩归一均值+EWMA;
           阈值 = max(val event_f1 网格, 共形健康分位 α∈{0.01,0.005,0.001}) 按 val 选。
  test   = 每配置只评一次; 达标与否如实报告。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT / "SCADA数据集"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import 数据预处理 as prep  # noqa: E402
from 事件级评测 import event_level_metrics, select_threshold_event  # noqa: E402
from 真实故障事件级实验 import (load_variant, per_turbine_causal_features,  # noqa: E402
                                per_turbine_ewma, rank_normalize)

SEED, LEAD, GRACE, POST, MERGE = 0, 72, 72, 144, 72.0
RES = HERE / "快速实验结果_真实故障"
BASELINE = "11_直方图梯度提升_监督"
SUPERVISED = {"01_逻辑回归_监督", "11_直方图梯度提升_监督", "13_LightGBM_监督"}


def rebuild_labels_v2(data, split: str, ts_key: str, turb_key: str) -> np.ndarray:
    """由事件表+sidecar 重建 v2 标签 (含缓冲带); 上游特征不变 → 标签层独立重算。"""
    et = pd.read_csv(data["dir"] / "event_table.csv")
    et = et[et["split"].astype(str) == split]
    ev = pd.DataFrame({
        "Timestamp start": pd.to_datetime(et["start"], utc=True, errors="coerce"),
        "Timestamp end": pd.to_datetime(et["end"], utc=True, errors="coerce"),
        "Message": et["message"].fillna("").astype(str),
        "_turbine": et["turbine"].astype(str),
    })
    idx = pd.DatetimeIndex(pd.to_datetime(data[ts_key], utc=True))
    y = prep.make_real_fault_earlywarning_labels(
        ev, idx, policy="real_fault_wl", lead_steps=LEAD, turbine_col=data[turb_key],
        merge_hours=MERGE, post_ignore_steps=POST, pre_grace_steps=GRACE)
    gap_f = data["dir"] / f"gap_mask_{split if split != 'train' else 'train'}.npy"
    if split in ("val", "test") and gap_f.exists():
        y[np.load(gap_f).astype(bool)] = -1
    return y


def eval_on(name, s_val, s_test, data, yv2_val, yv2_test, conformal_grid=False):
    """v2 口径: val 选阈 (可选共形分位一并入网格, 均 val-only) → test 评一次。"""
    thr, vf1 = select_threshold_event(s_val, yv2_val, data["ts_val"], data["turb_val"],
                                      data["ep_val"], lead_steps=LEAD)
    picked = {"rule": "event_f1_grid", "thr": float(thr), "val_f1": float(vf1)}
    if conformal_grid:
        base_h = s_val[(yv2_val == 0) & np.isfinite(s_val)]
        for a in (0.01, 0.005, 0.001):
            t2 = float(np.quantile(base_h, 1 - a))
            pred = np.zeros(len(s_val), dtype=int)
            pred[np.isfinite(s_val) & (s_val >= t2)] = 1
            f2 = event_level_metrics(pred, yv2_val, data["ts_val"], data["turb_val"],
                                     data["ep_val"], lead_steps=LEAD)["event_f1"]
            if f2 > picked["val_f1"]:
                picked = {"rule": f"conformal_a={a}", "thr": t2, "val_f1": float(f2)}
    thr = picked["thr"]
    pred = np.zeros(len(s_test), dtype=int)
    fin = np.isfinite(s_test)
    pred[fin & (s_test >= thr)] = 1
    m = event_level_metrics(pred, yv2_test, data["ts_test"], data["turb_test"],
                            data["ep_test"], lead_steps=LEAD)
    from sklearn.metrics import roc_auc_score
    ok = fin & (yv2_test != -1)
    m["point_auc_test"] = float(roc_auc_score(yv2_test[ok], s_test[ok])) \
        if len(np.unique(yv2_test[ok])) > 1 else float("nan")
    m["model"], m["threshold_pick"], m["val_event_f1"] = name, picked, picked["val_f1"]
    return m


def retrain_supervised(data, feats, y_sup_v2):
    """监督三模型按 v2 标签重训 → 新分数 (与 真实故障事件级实验 同配方)。"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    out = {}
    keep = np.where(y_sup_v2 != -1)[0][::4]
    for name, clf in [
        ("01_逻辑回归_监督", make_pipeline(StandardScaler(),
                                       LogisticRegression(max_iter=2000, class_weight="balanced"))),
        ("11_直方图梯度提升_监督", HistGradientBoostingClassifier(random_state=SEED, max_iter=300,
                                                          class_weight="balanced")),
        ("13_LightGBM_监督", lgb.LGBMClassifier(random_state=SEED, n_estimators=400,
                                              class_weight="balanced", verbosity=-1)),
    ]:
        clf.fit(feats["F_sup"][keep], y_sup_v2[keep])
        out[name] = (clf.predict_proba(feats["F_val"])[:, 1],
                     clf.predict_proba(feats["F_test"])[:, 1])
    return out


def main(farm: str) -> dict:
    data = load_variant(farm)
    res = RES / farm
    # ---- v2 标签 (缓冲带) ----
    yv2_val = rebuild_labels_v2(data, "val", "ts_val", "turb_val")
    yv2_test = rebuild_labels_v2(data, "test", "ts_test", "turb_test")
    y_sup_v2 = rebuild_labels_v2(data, "train", "ts_sup", "turb_sup")
    np.save(data["dir"] / "val_labels_v2.npy", yv2_val)
    np.save(data["dir"] / "test_labels_v2.npy", yv2_test)
    np.save(data["dir"] / "train_sup_labels_v2.npy", y_sup_v2)
    for tag, y1, y2 in (("val", data["y_val"], yv2_val), ("test", data["y_test"], yv2_test)):
        print(f"[{farm}] {tag}: v1 pos={(y1==1).sum()} ign={(y1==-1).sum()} → "
              f"v2 pos={(y2==1).sum()} ign={(y2==-1).sum()} (缓冲带生效)", flush=True)

    # ---- 全模型 v2 重评 (监督重训; 其余分数确定性复用) ----
    feats = {"F_sup": per_turbine_causal_features(data["X_sup"], data["turb_sup"]),
             "F_val": per_turbine_causal_features(data["X_val"], data["turb_val"]),
             "F_test": per_turbine_causal_features(data["X_test"], data["turb_test"])}
    sup_scores = retrain_supervised(data, feats, y_sup_v2)
    rows, val_scores, test_scores = [], {}, {}
    for mdir in sorted(res.iterdir()):
        if not (mdir / "score_val.npy").exists():
            continue
        name = mdir.name
        if name in sup_scores:
            s_val, s_test = sup_scores[name]
            np.save(mdir / "score_val_v2.npy", s_val.astype(np.float32))
            np.save(mdir / "score_test_v2.npy", s_test.astype(np.float32))
        else:
            s_val = np.load(mdir / "score_val.npy").astype(float)
            s_test = np.load(mdir / "score_test.npy").astype(float)
        m = eval_on(name, np.asarray(s_val, float), np.asarray(s_test, float),
                    data, yv2_val, yv2_test)
        (mdir / "metrics_v2.json").write_text(
            json.dumps({k: v for k, v in m.items() if k != "lead_minutes_all"},
                       ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        rows.append(m)
        val_scores[name], test_scores[name] = np.asarray(s_val, float), np.asarray(s_test, float)

    # ---- 贪心前向组合 (仅 val v2) ----
    def fuse(members, which):
        pool = val_scores if which == "val" else test_scores
        f = np.mean([rank_normalize(pool[m]) for m in members], axis=0)
        return per_turbine_ewma(f, data[f"turb_{which}"], alpha=0.1)

    def val_f1_of(members):
        s = fuse(members, "val")
        _, f = select_threshold_event(s, yv2_val, data["ts_val"], data["turb_val"],
                                      data["ep_val"], lead_steps=LEAD)
        return f

    members, best_f1 = [], -1.0
    cand = sorted(val_scores)
    for _ in range(5):                                   # 最多 5 成员 (防过拟 val)
        pick, pick_f1 = None, best_f1
        for c in cand:
            if c in members:
                continue
            f = val_f1_of(members + [c])
            if f > pick_f1:
                pick, pick_f1 = c, f
        if pick is None:
            break
        members.append(pick); best_f1 = pick_f1
        print(f"[{farm}] 组合+= {pick} → val_eF1={best_f1:.4f}", flush=True)

    combo = eval_on("90_贪心组合_" + "+".join(m.split("_")[0] for m in members),
                    fuse(members, "val"), fuse(members, "test"),
                    data, yv2_val, yv2_test, conformal_grid=True)
    combo["members"] = members
    rows.append(combo)

    # ---- 达标判定 (vs 预注册基线 HGB, 同 v2 口径) ----
    base = next(r for r in rows if r["model"] == BASELINE)
    def geq(a, b):  # lead 允许 NaN 特判
        return (a == a) and (b != b or a >= b)
    verdict = {
        "event_f1": combo["event_f1"] > base["event_f1"],
        "event_recall": combo["event_recall"] >= base["event_recall"],
        "lead_minutes": geq(combo["lead_minutes_median"], base["lead_minutes_median"]),
        "far_per_day": combo["far_per_day"] <= base["far_per_day"],
        "point_auc": combo["point_auc_test"] >= base["point_auc_test"],
    }
    df = pd.DataFrame([{k: r.get(k) for k in
                        ("model", "event_f1", "event_recall", "n_detected", "n_events",
                         "lead_minutes_median", "far_per_day", "alarm_precision",
                         "point_auc_test", "val_event_f1")} for r in rows]
                      ).sort_values("event_f1", ascending=False)
    df.to_csv(res / "汇总_v2.csv", index=False, encoding="utf-8-sig")
    report = {"farm": farm, "baseline": {k: base.get(k) for k in
              ("event_f1", "event_recall", "lead_minutes_median", "far_per_day", "point_auc_test")},
              "combo": {k: combo.get(k) for k in
              ("model", "members", "threshold_pick", "event_f1", "event_recall",
               "lead_minutes_median", "far_per_day", "point_auc_test")},
              "goal_met_per_metric": verdict, "goal_met_all": all(verdict.values())}
    (res / "组合达标报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=1,
                                                default=str), encoding="utf-8")
    print(f"\n[{farm}] 组合={combo['model']} vs 基线={BASELINE}")
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    return report


if __name__ == "__main__":
    for farm in ("penmanshiel", "kelmarsh"):
        main(farm)
