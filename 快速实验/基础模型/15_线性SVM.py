# -*- coding: utf-8 -*-
"""15 线性SVM —— 最大间隔线性分类; 分数=决策函数(离超平面距离)"""
from sklearn.svm import LinearSVC                      # 线性支持向量机

from _common import load_flat, now, report, standardize  # 统一数据/计时/评测/标准化

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # SVM对尺度敏感需标准化(仅train统计量)
t0 = now()                                            # 起始计时
m = LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=3000).fit(Xtr, ytr)  # C=1软间隔, 配平, 样本>特征用原问题, 训练
report("15_线性SVM", yva, m.decision_function(Xva), yte, m.decision_function(Xte), now() - t0)  # 无概率, 用决策函数(离超平面符号距离)当分数
