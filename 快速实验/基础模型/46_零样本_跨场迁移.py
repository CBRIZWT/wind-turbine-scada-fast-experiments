# -*- coding: utf-8 -*-
"""46 零样本·跨场迁移 —— 目标农场(Kelmarsh)零标签:
在 Penmanshiel 的标签上训练, 用6个与农场无关的衍生特征跨场
(两场通道数不同: pen 89 vs kel 87, 原始通道无法对齐);
阈值也在源域(pen val)上选 —— kel 标签全程未参与训练与选阈, 是真零样本。
json里 val=源域pen, test=目标域kel。
"""
import numpy as np                                     # 数组/衍生特征
import pandas as pd                                    # 滚动统计
from sklearn.ensemble import HistGradientBoostingClassifier  # 源域分类器
from sklearn.utils.class_weight import compute_sample_weight  # 样本权重配平

from _common import (DATA, FARM, VARIANT, cross_farm_source, load_flat, now,  # 数据路径/当前farm/加载/计时
                     quick_data_dir, report, report_v3)

W, K = 72, 6                                           # 滚动窗(12h) / 近增量步数(1h), 与准备数据.py同配方


def farm_free_features(X):
    """(T,C)→(T,6) 与通道数无关的因果特征: 跨通道max/正残差能量/滚动统计/趋势"""
    maxc = X.max(axis=1)                               # 每时刻跨通道最大残差
    pose = np.mean(np.maximum(0.0, X) ** 2, axis=1)    # 跨通道正残差能量
    s = pd.Series(maxc)                                # 转Series便于滚动
    return np.column_stack([                           # 6维农场无关特征
        maxc, pose,                                    # 最大残差、正残差能量
        s.rolling(W, min_periods=1).mean().to_numpy(), # 72步滚动均值
        s.rolling(W, min_periods=1).max().to_numpy(),  # 72步滚动最大
        maxc - s.shift(W).fillna(s.iloc[0]).to_numpy(),  # 相对72步前的斜率
        maxc - s.shift(K).fillna(s.iloc[0]).to_numpy()]).astype(np.float32)  # 相对6步前的增量


if VARIANT:
    # metrics-v3: 双真风场互为源域；直接消费逐机组派生的最后6个跨场因果特征。
    SOURCE_FARM = cross_farm_source(FARM)
    SOURCE_DATA = quick_data_dir(SOURCE_FARM)
    Ftr = np.load(SOURCE_DATA / "X_flat_train.npy", mmap_mode="r")[:, -6:].astype(np.float32)
    ytr = np.load(SOURCE_DATA / "y_flat_train.npy").astype(int)
    Fpv = np.load(SOURCE_DATA / "X_flat_val.npy", mmap_mode="r")[:, -6:].astype(np.float32)
    ypv = np.load(SOURCE_DATA / "y_flat_val.npy").astype(int)
    mv = np.ones(len(ypv), dtype=bool)
else:
    # 旧A′兼容路径: 历史实现固定Penmanshiel为源域。
    SOURCE_FARM = "penmanshiel"
    PEN = DATA.parent.parent.parent / "SCADA数据集" / "数据预处理" / SOURCE_FARM
    Xp = np.load(PEN / "train_sup.npy").astype(np.float32)
    yp = np.load(PEN / "train_sup_labels.npy")
    Fp = farm_free_features(Xp)
    mp = yp != -1
    Ftr, ytr = Fp[mp][::4], yp[mp][::4].astype(int)
    Xpv = np.load(PEN / "val.npy").astype(np.float32)
    ypv = np.load(PEN / "val_labels.npy")
    Fpv = farm_free_features(Xpv)
    mv = ypv != -1
    del Xp, Xpv

# ---- 目标域 Kelmarsh: 只取扁平特征的后6列(=同配方衍生特征), 标签只用于test评测
_, _, Xva, yva, Xte, yte = load_flat()                    # 目标域(ytr/yva不参与任何决策)

t0 = now()                                                # 起始计时
m = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08,  # 源域分类器
                                   l2_regularization=1.0, random_state=0)
m.fit(Ftr, ytr, sample_weight=compute_sample_weight("balanced", ytr))  # 用源域标签+配平训练
source_val_score = m.predict_proba(Fpv[mv])[:, 1]
target_test_score = m.predict_proba(Xte[:, -6:])[:, 1]
if VARIANT:
    report_v3(
        "46_零样本_跨场迁移",
        ypv[mv].astype(int),
        source_val_score,
        yte,
        target_test_score,
        now() - t0,
        representation="flat",
        val_sidecars=(np.load(SOURCE_DATA / "timestamps_flat_val.npy"),
                      np.load(SOURCE_DATA / "turbines_flat_val.npy")),
        val_event_table=SOURCE_DATA / "event_table.csv",
        test_event_table=DATA / "event_table.csv",
        extra={"源域": SOURCE_FARM, "目标域": FARM, "特征": "6维农场无关",
               "scores_are_probabilities": True},
    )
else:
    report("46_零样本_跨场迁移", ypv[mv].astype(int), source_val_score,
           yte, target_test_score, now() - t0,
           extra={"源域": SOURCE_FARM, "目标域": FARM, "特征": "6维农场无关",
                  "scores_are_probabilities": True})
