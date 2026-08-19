# -*- coding: utf-8 -*-
"""04 线性判别分析 —— 共享协方差高斯假设的生成式线性分类"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA

from _common import load_flat, now, report            # 统一数据/计时/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
t0 = now()                                            # 起始计时
# [修复 2026-07-26] 原实现未设 priors, 类先验由 0.13% 的极端正例率主导 → 判别面被推向全判负。
#   改为等先验 [0.5, 0.5], 与项目其余监督模型的 balanced 口径一致。
m = LinearDiscriminantAnalysis(priors=[0.5, 0.5]).fit(Xtr, ytr)  # 等先验; 两类共享协方差→线性决策面
report("04_LDA", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类概率评测
