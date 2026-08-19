# -*- coding: utf-8 -*-
"""08 随机森林 —— Bagging树集成, 表格数据强手"""
from sklearn.ensemble import RandomForestClassifier   # 随机森林(Bagging树集成)

from _common import load_flat, now, report            # 统一数据/计时/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签(树无需标准化)
t0 = now()                                            # 起始计时
m = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,     # 200棵树(边际收益饱和), 叶≥5防过拟合
                           class_weight="balanced_subsample", n_jobs=-1,  # 逐bootstrap包配平不平衡, 多核并行
                           random_state=0).fit(Xtr, ytr)                  # 固定种子, 训练
report("08_随机森林", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类概率评测
