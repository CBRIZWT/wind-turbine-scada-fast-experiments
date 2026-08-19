# -*- coding: utf-8 -*-
"""45 少样本·半监督自训练 —— 100标签 + 5万无标签样本:
基学习器给无标签样本打伪标签、滚动扩充训练集 (经典半监督)"""
import numpy as np                                     # 抽样/拼接
from sklearn.linear_model import LogisticRegression    # 基学习器
from sklearn.semi_supervised import SelfTrainingClassifier  # 自训练包装器

from _common import load_flat, now, report, standardize  # 统一数据/计时/评测/标准化

K = 50                                                 # 每类标签数

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 标准化
rng = np.random.default_rng(0)                         # 固定随机源
lab = np.concatenate([rng.choice(np.where(ytr == 1)[0], K, replace=False),   # 抽K个正例(有标签)
                      rng.choice(np.where(ytr == 0)[0], K, replace=False)])   # 抽K个负例(有标签)
unlab = rng.choice(len(Xtr), 50000, replace=False)     # 随机5万作无标签池
X = np.concatenate([Xtr[lab], Xtr[unlab]])             # 拼接有标签+无标签特征
y = np.concatenate([ytr[lab], np.full(len(unlab), -1)])  # 无标签样本标签设为-1

t0 = now()                                             # 起始计时
m = SelfTrainingClassifier(LogisticRegression(max_iter=1000), threshold=0.9).fit(X, y)  # 高置信(>0.9)才打伪标签, 迭代扩充
report("45_少样本_半监督自训练", yva, m.predict_proba(Xva)[:, 1],   # 取正类概率评测
       yte, m.predict_proba(Xte)[:, 1], now() - t0,
       extra={"标签数": 2 * K, "无标签数": len(unlab)})  # 记录标签/无标签预算
