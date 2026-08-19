# -*- coding: utf-8 -*-
"""06 K近邻 —— 实例法代表; 93维距离退化+推理贵, train再降到2万控制耗时"""
from sklearn.neighbors import KNeighborsClassifier    # KNN分类器

from _common import load_flat, now, report, standardize  # 统一数据/计时/评测/标准化

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 距离度量需标准化(否则大尺度特征主导)
Xtr, ytr = Xtr[::15], ytr[::15]                       # train每15取1: 29万→2万, 否则对61万test逐点算距离不可行
t0 = now()                                            # 起始计时
m = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1).fit(Xtr, ytr)  # 15近邻按距离加权, 多核
report("06_K近邻", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类概率评测
