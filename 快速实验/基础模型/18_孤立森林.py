# -*- coding: utf-8 -*-
"""18 孤立森林 —— 无监督: 随机切分隔离异常; 只用train正常样本拟合, 不看标签"""
from sklearn.ensemble import IsolationForest          # 孤立森林

from _common import load_flat, needs_external_train_scores, now, report  # 统一数据/计时/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
t0 = now()                                            # 起始计时
Xn = Xtr[ytr == 0]
m = IsolationForest(n_estimators=200, n_jobs=-1, random_state=0).fit(Xn)  # 200棵隔离树, 仅用train正常样本拟合(无监督)
# 分数取负: score_samples 越小越异常 → 统一成越大越异常
report("18_孤立森林", yva, -m.score_samples(Xva), yte, -m.score_samples(Xte), now() - t0,
       train_scores=-m.score_samples(Xn) if needs_external_train_scores() else None)  # 负分作异常分数评测
