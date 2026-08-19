# -*- coding: utf-8 -*-
"""03 高斯朴素贝叶斯 —— 特征独立假设在87个相关残差上不成立, 作为弱对照"""
from sklearn.naive_bayes import GaussianNB            # 高斯朴素贝叶斯

from _common import load_flat, now, report            # 统一数据/计时/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
t0 = now()                                            # 起始计时
# [修复 2026-07-26] 原实现未设 priors, 类先验由 0.13% 的极端正例率主导, 后验几乎恒判负类。
#   改为等先验 [0.5, 0.5] —— 等价于类配平, 与项目其余监督模型同口径 (决策交由下游 val 选阈)。
m = GaussianNB(priors=[0.5, 0.5]).fit(Xtr, ytr)       # 等先验; 逐特征估计两类高斯密度(假设条件独立)
report("03_朴素贝叶斯", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类后验概率评测
