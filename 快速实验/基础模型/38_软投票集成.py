# -*- coding: utf-8 -*-
"""38 软投票集成 —— 集成学习第三类(与Bagging/Boosting并列):
三个异质强模型的概率取平均"""
import lightgbm as lgb                                 # LightGBM(基模型之一)
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,  # HistGBM/随机森林
                              VotingClassifier)         # 投票集成器

from _common import load_flat, now, report            # 统一数据/计时/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
t0 = now()                                            # 起始计时
m = VotingClassifier(voting="soft", estimators=[      # soft=对概率取平均(优于硬投票)
    ("lgbm", lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05,   # 基模型1: LightGBM
                                class_weight="balanced", random_state=0, n_jobs=-1,
                                verbosity=-1)),
    ("hgb", HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08,  # 基模型2: HistGBM
                                           l2_regularization=1.0, random_state=0)),
    ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5,                     # 基模型3: 随机森林
                                  class_weight="balanced_subsample", n_jobs=-1,
                                  random_state=0))]).fit(Xtr, ytr)  # 三基模型各自训练
report("38_软投票集成", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 平均概率评测
