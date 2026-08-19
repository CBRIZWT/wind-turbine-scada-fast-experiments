# -*- coding: utf-8 -*-
"""37 KMeans距离 —— 无监督聚类式异常检测:
正常样本聚成8簇, 分数 = 到最近簇中心的距离"""
from sklearn.cluster import KMeans                     # K均值聚类

from _common import load_flat, needs_external_train_scores, now, report, standardize  # 统一数据/计时/评测/标准化

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 距离对尺度敏感需标准化
t0 = now()                                            # 起始计时
Xn = Xtr[ytr == 0]
m = KMeans(n_clusters=8, n_init=3, random_state=0).fit(Xn)  # 正常样本聚8簇(≈工况模式数)
report("37_KMeans距离", yva, m.transform(Xva).min(axis=1),   # 到最近簇中心的距离作异常分数
       yte, m.transform(Xte).min(axis=1), now() - t0,
       train_scores=(m.transform(Xn[::4]).min(axis=1)
                     if needs_external_train_scores() else None))        # test同理评测
