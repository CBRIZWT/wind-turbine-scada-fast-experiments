# -*- coding: utf-8 -*-
"""36 PCA重构误差 —— 无监督线性重构 (自编码器的线性对应物):
用正常样本学主子空间, 分数 = 投影重构误差"""
import numpy as np                                     # (未直接用, 保留兼容)
from sklearn.decomposition import PCA                  # 主成分分析

from _common import (load_flat, needs_external_train_scores, now, pca_component_count,
                     report, standardize)  # 统一数据/计时/评测/标准化

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 标准化
t0 = now()                                            # 起始计时
Xn = Xtr[ytr == 0]
n_components = pca_component_count(len(Xn), Xn.shape[1], requested=16)
m = PCA(n_components=n_components, random_state=0).fit(Xn)  # 低维跨场协议自适应到可辨识维数
err = lambda X: ((X - m.inverse_transform(m.transform(X))) ** 2).mean(axis=1)  # 投影再重构的平方误差=异常分数
report("36_PCA重构误差", yva, err(Xva), yte, err(Xte), now() - t0,
       train_scores=err(Xn[::4]) if needs_external_train_scores() else None)  # 用重构误差评测
