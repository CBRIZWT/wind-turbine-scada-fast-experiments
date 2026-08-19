# -*- coding: utf-8 -*-
"""05 二次判别分析 —— 逐类协方差二次决策面; 93维近共线特征需加正则"""
from _common import fit_regularized_qda, load_flat, now, report  # 统一数据/正则QDA/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
t0 = now()                                            # 起始计时
m = fit_regularized_qda(Xtr, ytr)                    # eigen+自动收缩支持少数类样本数<特征数
report("05_QDA", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类概率评测
