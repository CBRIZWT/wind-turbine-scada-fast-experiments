# -*- coding: utf-8 -*-
"""09 极端随机树 —— 随机森林变体: 分裂点也随机, 方差更低"""
from sklearn.ensemble import ExtraTreesClassifier     # 极端随机树(分裂阈值也随机)

from _common import load_flat, now, report            # 统一数据/计时/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
t0 = now()                                            # 起始计时
m = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5,      # 200棵树, 叶≥5; 分裂阈值随机使树间更去相关
                         class_weight="balanced_subsample", n_jobs=-1,  # 配平不平衡, 多核并行
                         random_state=0).fit(Xtr, ytr)                  # 固定种子, 训练
report("09_ExtraTrees", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类概率评测
