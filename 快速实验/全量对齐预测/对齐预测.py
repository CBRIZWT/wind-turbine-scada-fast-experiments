# -*- coding: utf-8 -*-
r"""对齐预测.py — 用"全量实验所用预处理数据的一部分"快速预测全量4深度模型的A′指标。

对齐口径(与全量实验逐项一致, 见 SCADA数据集/数据预处理.py 与 实验结果/.../metrics.jsonl):
  · 数据 = 全量深度模型读的【同一份】 train.npy(去污,正例0) / val.npy / test.npy (87通道),
    train 行子集下采样提速 = "全量数据的一部分"; 不加任何额外特征。
  · 范式 = 无监督(train去污无正例, 与AnomalyTransformer/TranAD/TriTrackNet/wt同范式)。
  · 标签 = A′过温早警(99/3/72); 评测 = val选阈(最大F1)+极性 → test只评一次, raw逐点 F1/AUC/AUPRC。
运行(IDE或cmd): E:\ancoda\chuangxin\python.exe 对齐预测.py
输出: 每farm无监督打分器的预测带 vs 全量4深度模型真实指标 + 预测误差。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path(r"E:\创新\SCADA数据集\数据预处理")                # 全量与快速共用的单一真源
SUMMARY = Path(r"E:\创新\实验结果\chronological_v2__v2\matrix_5seed_summary.csv")  # 全量真实结果
FARMS = ["kelmarsh", "penmanshiel", "hill_of_towie"]
TRAIN_SUB = 8          # train 每8行取1 (子集提速; 无监督拟合正常分布用不着全量)
TEST_SUB = 1           # test 全量评测(与全量口径一致)


def load(farm):
    """读【全量深度模型读的同一份】train.npy(去污)/val/test (87通道) + A′标签 + 列名。"""
    d = SRC / farm
    Xtr = np.asarray(np.load(d / "train.npy", mmap_mode="r")[::TRAIN_SUB], np.float32)   # 去污,正例0 → 全为正常行
    Xva = np.asarray(np.load(d / "val.npy"), np.float32)
    yva = np.load(d / "val_labels.npy")
    Xte = np.asarray(np.load(d / "test.npy")[::TEST_SUB], np.float32)
    yte = np.load(d / "test_labels.npy")[::TEST_SUB]
    cols = json.loads((d / "meta.json").read_text(encoding="utf-8"))["cols"]
    base = [i for i, c in enumerate(cols) if c.endswith("__resid")]   # 18个基础残差通道(算物理能量用)
    return Xtr, Xva, yva, Xte, yte, base


def standardize(Xtr, *rest):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8                 # 均值/方差只用train(防泄漏)
    return tuple(((X - mu) / sd).astype(np.float32) for X in (Xtr, *rest))


def eval_scorer(yva, sva, yte, ste):
    """val选阈(最大F1)+极性 → test只评一次; 返回 (f1,auc,auprc)。与全量/快速同口径。"""
    mv, mt = yva != -1, yte != -1                          # 剔除ignore(-1)
    yva, sva, yte, ste = yva[mv], np.asarray(sva)[mv], yte[mt], np.asarray(ste)[mt]
    best = (-1.0, 1, 0.0)                                  # (val F1, 极性, 阈值)
    qs = np.append(np.linspace(0.50, 0.999, 120), 0.0)
    for p in (1, -1):                                      # 极性: 分数正/反向
        s = p * sva
        for q in qs:
            thr = float(np.quantile(s, q))
            fv = f1_score(yva, s >= thr, zero_division=0)
            if fv > best[0]:
                best = (fv, p, thr)
    p, thr = best[1], best[2]
    pred = (p * ste) >= thr
    return (f1_score(yte, pred, zero_division=0),
            roc_auc_score(yte, p * ste), average_precision_score(yte, p * ste))


def scorers(Xtr, Xva, Xte, base):
    """6个轻量无监督打分器(镜像全量深度模型的范式: 重构/密度/隔离/聚类/预测)。"""
    Ztr, Zva, Zte = standardize(Xtr, Xva, Xte)
    out = {}
    # ① 物理残差能量(不训练): 跨18基础残差通道的正能量 —— 对应"温度过热"物理直觉
    en = lambda X: np.mean(np.maximum(0.0, X[:, base]) ** 2, axis=1)
    out["残差能量(不训练)"] = (en(Xva), en(Xte))
    # ② PCA重构误差(≈AnomalyTransformer/TranAD/AE 的重构范式)
    pca = PCA(n_components=16, svd_solver="full").fit(Ztr)
    re = lambda Z: ((Z - pca.inverse_transform(pca.transform(Z))) ** 2).mean(1)
    out["PCA重构"] = (re(Zva), re(Zte))
    # ③ 马氏距离(协方差密度)
    lw = LedoitWolf().fit(Ztr)
    P, c = lw.precision_.astype(np.float32), lw.location_.astype(np.float32)
    maha = lambda Z: np.einsum("ij,jk,ik->i", Z - c, P, Z - c)
    out["马氏距离"] = (maha(Zva), maha(Zte))
    # ④ KMeans最近簇距离(聚类)
    km = MiniBatchKMeans(n_clusters=8, batch_size=4096, n_init=5, random_state=0).fit(Ztr)
    kd = lambda Z: km.transform(Z).min(1)
    out["KMeans距离"] = (kd(Zva), kd(Zte))
    # ⑤ 孤立森林(隔离)
    iso = IsolationForest(n_estimators=200, max_samples=1024, random_state=0, n_jobs=-1).fit(Ztr)
    ic = lambda Z: -iso.score_samples(Z)
    out["孤立森林"] = (ic(Zva), ic(Zte))
    # ⑥ 一步持久性预测误差(≈TriTrackNet/wt-transformer 的预测范式, 不训练)
    pf = lambda X: np.concatenate([[0.0], np.mean(np.abs(np.diff(X[:, base], axis=0)), axis=1)])
    out["一步预测误差"] = (pf(Xva), pf(Xte))
    return out


def load_full_actual():
    """读全量实验真实结果 (matrix_5seed_summary.csv) → {farm: {model: (f1,auc,auprc)}} (baseline_only)。"""
    res = {}
    with open(SUMMARY, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["module"] != "baseline_only":
                continue
            res.setdefault(r["farm"], {})[r["model"]] = (
                float(r["f1_mean"]), float(r["auc_mean"]), float(r["auprc_mean"]))
    return res


def main():
    actual = load_full_actual()
    for farm in FARMS:
        if not (SRC / farm / "train.npy").exists():
            print(f"[跳过] {farm}: 无预处理数据"); continue
        Xtr, Xva, yva, Xte, yte, base = load(farm)
        print(f"\n{'='*78}\n{farm}  (对齐数据: train去污子集 {len(Xtr):,}行 / test {len(Xte):,}行 × 87通道, 无监督)")
        sc = scorers(Xtr, Xva, Xte, base)
        rows = {}
        for name, (sva, ste) in sc.items():
            rows[name] = eval_scorer(yva, sva, yte, ste)
        print(f"  {'无监督打分器':<16}{'F1':>8}{'AUC':>8}{'AUPRC':>8}")
        for name, (f1, auc, ap) in rows.items():
            print(f"  {name:<16}{f1:>8.4f}{auc:>8.4f}{ap:>8.4f}")
        f1s = [v[0] for v in rows.values()]; aucs = [v[1] for v in rows.values()]
        print(f"  → 预测带: F1 [{min(f1s):.3f}, {max(f1s):.3f}]  AUC [{min(aucs):.3f}, {max(aucs):.3f}]")
        # ---- 对照全量真实 ----
        if farm in actual:
            print(f"  {'全量真实(5seed均)':<20}{'F1':>8}{'AUC':>8}{'AUPRC':>8}{'  在预测带内?':>12}")
            for model, (f1, auc, ap) in sorted(actual[farm].items()):
                inband = "✓" if min(f1s) - 0.03 <= f1 <= max(f1s) + 0.03 else "×(偏高:更长窗/预测型)"
                print(f"  {model:<20}{f1:>8.4f}{auc:>8.4f}{ap:>8.4f}{inband:>14}")


if __name__ == "__main__":
    main()
