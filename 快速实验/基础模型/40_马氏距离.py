# -*- coding: utf-8 -*-
"""40 马氏距离 —— 无监督协方差法: 正常样本估计均值/协方差(Ledoit-Wolf收缩),
分数 = 到正常分布中心的马氏距离"""
import numpy as np                                     # einsum向量化
from sklearn.covariance import LedoitWolf              # Ledoit-Wolf收缩协方差估计

from _common import load_flat, needs_external_train_scores, now, report, standardize  # 统一数据/计时/评测/标准化

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 标准化
t0 = now()                                            # 起始计时
Xn = Xtr[ytr == 0]
m = LedoitWolf().fit(Xn)                              # 正常样本估计均值与收缩协方差(治近共线)
P = m.precision_.astype(np.float32)                     # 协方差逆矩阵(精度矩阵)
dist = lambda X: np.einsum("ij,jk,ik->i", X - m.location_, P, X - m.location_)  # 马氏距离平方 (x-μ)ᵀΣ⁻¹(x-μ)
report("40_马氏距离", yva, dist(Xva), yte, dist(Xte), now() - t0,
       train_scores=dist(Xn[::4]) if needs_external_train_scores() else None)  # 用马氏距离作异常分数评测
