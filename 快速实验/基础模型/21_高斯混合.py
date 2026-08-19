# -*- coding: utf-8 -*-
"""21 高斯混合 —— 无监督密度模型: 学正常分布, 分数=负对数似然"""
from sklearn.mixture import GaussianMixture           # 高斯混合模型(EM)

from _common import load_flat, needs_external_train_scores, now, report, standardize  # 统一数据/计时/评测/标准化

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 标准化
t0 = now()                                            # 起始计时
m = GaussianMixture(n_components=8, covariance_type="full", random_state=0)  # 8个全协方差高斯(≈工况模式数)
Xref = Xtr[ytr == 0][::3]
m.fit(Xref)                                           # 正常样本下采样3倍(~8万)足够估计密度
report("21_高斯混合", yva, -m.score_samples(Xva), yte, -m.score_samples(Xte), now() - t0,
       train_scores=-m.score_samples(Xref) if needs_external_train_scores() else None)  # 负对数似然作异常分数
