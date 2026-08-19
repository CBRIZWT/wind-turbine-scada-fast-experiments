# -*- coding: utf-8 -*-
"""01 逻辑回归 —— 线性概率基线, 类权重治不平衡"""
from sklearn.linear_model import LogisticRegression   # sklearn逻辑回归

from _common import load_flat, now, report, standardize  # 统一数据/计时/评测/标准化

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平93维特征与0/1标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 线性模型对尺度敏感需标准化(统计量仅取自train防泄漏)
t0 = now()                                            # 起始计时
m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)  # 类权重反比配平不平衡, 训练
report("01_逻辑回归", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类概率评测
