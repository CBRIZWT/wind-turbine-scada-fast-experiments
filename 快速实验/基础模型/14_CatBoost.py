# -*- coding: utf-8 -*-
"""14 CatBoost —— 有序提升+对称树GBDT, 自动类平衡, val早停"""
from catboost import CatBoostClassifier               # CatBoost

from _common import load_flat, now, report            # 统一数据/计时/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
t0 = now()                                            # 起始计时
# [优化 2026-07-26] 早停指标改 PRAUC: 原用默认 Logloss, 在 0.13% 正例率下 logloss 由负类主导,
#   早停点与 AUPRC 最优点不一致。直接以目标指标(PRAUC)早停, 与 12_XGBoost(aucpr)/13_LightGBM
#   (average_precision) 口径统一。早停仍只看 val, 不碰 test。
m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1,        # 500轮/深6/学习率0.1
                       auto_class_weights="Balanced", early_stopping_rounds=50,  # 自动类平衡; val早停50轮
                       eval_metric="PRAUC",                               # 早停指标=PRAUC(即AUPRC), 非默认Logloss
                       random_seed=0, verbose=False)                      # 固定种子, 静默
m.fit(Xtr, ytr, eval_set=(Xva, yva))                  # 训练, 早停只看val
report("14_CatBoost", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类概率评测
