# -*- coding: utf-8 -*-
"""43 少样本·梯度提升 —— 同42的100标签设定, 换HistGBM(小配置防过拟合)"""
import numpy as np                                     # 抽样/索引
from sklearn.ensemble import HistGradientBoostingClassifier  # 直方图梯度提升

from _common import load_flat, now, report            # 统一数据/计时/评测

K = 50                                                 # 每类标签数(共100)

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签(树无需标准化)
rng = np.random.default_rng(0)                         # 固定随机源
ids = np.concatenate([rng.choice(np.where(ytr == 1)[0], K, replace=False),   # 抽K个正例
                      rng.choice(np.where(ytr == 0)[0], K, replace=False)])   # 抽K个负例

t0 = now()                                             # 起始计时
m = HistGradientBoostingClassifier(max_iter=50, max_depth=3, learning_rate=0.1,  # 小配置(50轮/深3)防100样本过拟合
                                   random_state=0).fit(Xtr[ids], ytr[ids])       # 仅用100标签训练
report("43_少样本_梯度提升", yva, m.predict_proba(Xva)[:, 1],   # 取正类概率评测(少样本梯队最佳)
       yte, m.predict_proba(Xte)[:, 1], now() - t0, extra={"标签数": 2 * K})  # 记录标签预算
