# -*- coding: utf-8 -*-
"""42 少样本·逻辑回归 —— 只用 50正+50负 共100个标签训练 (标签稀缺场景);
单次抽样(seed=0), 少样本结果对抽样敏感是其本性, 如实呈现"""
import numpy as np                                     # 随机抽样/索引
from sklearn.linear_model import LogisticRegression    # 逻辑回归

from _common import load_flat, now, report, standardize  # 统一数据/计时/评测/标准化

K = 50                                                 # 每类标签数(共100)

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 标准化
rng = np.random.default_rng(0)                         # 固定随机源
ids = np.concatenate([rng.choice(np.where(ytr == 1)[0], K, replace=False),   # 随机抽K个正例
                      rng.choice(np.where(ytr == 0)[0], K, replace=False)])   # 随机抽K个负例

t0 = now()                                             # 起始计时
m = LogisticRegression(max_iter=1000).fit(Xtr[ids], ytr[ids])  # 仅用这100个标签训练
report("42_少样本_逻辑回归", yva, m.predict_proba(Xva)[:, 1],   # 取正类概率评测(val/test全量)
       yte, m.predict_proba(Xte)[:, 1], now() - t0, extra={"标签数": 2 * K})  # 记录标签预算
