# -*- coding: utf-8 -*-
"""10 AdaBoost —— 决策桩重加权提升, 现代GBDT的前身"""
from sklearn.ensemble import AdaBoostClassifier       # AdaBoost(自适应提升)
from sklearn.utils.class_weight import compute_sample_weight  # 类不平衡配平(AdaBoost无class_weight参数)

from _common import load_flat, now, report            # 统一数据/计时/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
t0 = now()                                            # 起始计时
# [修复 2026-07-26] AdaBoost 无 class_weight 参数, 原实现完全未处理类不平衡:
#   正例率仅 0.13%, 未加权时提升过程几乎只优化负类 → 严重低估该算法。
#   改用 sample_weight='balanced' 初始化样本权重 (与项目其余监督模型同口径)。
m = AdaBoostClassifier(n_estimators=200, random_state=0).fit(
    Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))  # 200个弱桩, 初始权重按类配平
report("10_AdaBoost", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类概率评测
