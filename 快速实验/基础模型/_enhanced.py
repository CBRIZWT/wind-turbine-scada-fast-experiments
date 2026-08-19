# -*- coding: utf-8 -*-
"""_enhanced.py — 「基模型 + 后处理增强」通用执行器。

用法: 各增强脚本只需给出基学习器构造函数, 本模块负责:
    ① 用与基模型相同的扁平特征训练;
    ② 在 val 上网格搜 (EWMA span, 持续门控 K), 按【val 事件F1】选优;
    ③ 用选定超参对 val/test 分数施加后处理, 交 report 统一评测。

防泄漏: 超参只在 val 上选; test 分数只做同一变换后评一次, 不参与任何选择。
"""
from __future__ import annotations

import numpy as np

from _common import DATA, load_flat, now, report, standardize
from _postproc import EWMA_SPANS, PERSIST_KS, apply, ewma_smooth, persistence_gate


def _val_event_f1_factory(yva, turb_va):
    """构造 val 事件F1 评估器: 给定(后处理后的)分数, 返回该分数下可达的最佳事件F1。

    做法与 report 的选阈一致 —— 遍历分位阈值, 取事件级 F1 最大者。
    事件段 = 标签连续正例段(与 _common._seg_metrics 同口径)。
    """
    y = np.asarray(yva).astype(int)
    d = np.diff(np.r_[0, (y == 1).astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1

    def best_event_f1(scores):
        s = np.asarray(scores, dtype=float)
        if not np.isfinite(s).any():
            return 0.0
        best = 0.0
        for q in np.linspace(0.90, 0.9995, 40):
            thr = float(np.quantile(s[np.isfinite(s)], q))
            pred = (s >= thr).astype(int)
            if pred.sum() == 0:
                continue
            # 召回: 事件段内有>=1报警
            det = sum(1 for a, b in zip(starts, ends) if pred[a:b + 1].any())
            rec = det / max(len(starts), 1)
            # 精确: 报警段与事件段相交
            dd = np.diff(np.r_[0, pred, 0])
            ps, pe = np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1
            if len(ps) == 0:
                continue
            hit = sum(1 for a, b in zip(ps, pe) if (y[a:b + 1] == 1).any())
            prec = hit / len(ps)
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            best = max(best, f1)
        return best

    return best_event_f1


def run_enhanced(name: str, build, *, needs_scale: bool = False, extra: dict | None = None):
    """训练基学习器 → val 调后处理超参 → 施加 → 统一评测。

    build(Xtr, ytr) -> 已拟合对象, 须有 predict_proba 或 decision_function/score 方法。
    """
    Xtr, ytr, Xva, yva, Xte, yte = load_flat()
    if needs_scale:
        Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)
    turb_va = np.load(DATA / "turbines_flat_val.npy")
    turb_te = np.load(DATA / "turbines_flat_test.npy")

    t0 = now()
    m = build(Xtr, ytr)
    score = lambda X: (m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba")
                       else np.asarray(m.decision_function(X), dtype=float))
    sva_raw, ste_raw = score(Xva), score(Xte)

    # ---- 只在 val 上网格搜后处理超参 ----
    ev = _val_event_f1_factory(yva, turb_va)
    best_span, best_k, best_v = 1, 1, -np.inf
    for sp in EWMA_SPANS:
        sm = ewma_smooth(sva_raw, turb_va, sp)
        for k in PERSIST_KS:
            v = ev(persistence_gate(sm, turb_va, k))
            if v > best_v:
                best_span, best_k, best_v = sp, k, v

    sva = apply(sva_raw, turb_va, best_span, best_k)
    ste = apply(ste_raw, turb_te, best_span, best_k)      # test 只做同一变换, 不参与选择
    ex = {"增强": "EWMA平滑 + 持续性门控", "EWMA_span": best_span, "持续门控K": best_k,
          "val调参依据": f"val事件F1={best_v:.4f}", "针对": "降误报段以提升event_F1/AUPRC",
          "scores_are_probabilities": bool(hasattr(m, "predict_proba"))}
    ex.update(extra or {})
    report(name, yva, sva, yte, ste, now() - t0, extra=ex)
