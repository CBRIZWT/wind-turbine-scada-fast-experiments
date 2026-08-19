# -*- coding: utf-8 -*-
"""44 少样本·原型法 —— 度量学习经典(原型网络的无训练特例):
每类100标签算特征空间原型(均值), 分数 = 到负原型距离 - 到正原型距离"""
import numpy as np                                     # 抽样/范数

from _common import load_flat, now, report, standardize  # 统一数据/计时/评测/标准化

K = 50                                                 # 每类标签数

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 标准化(欧氏距离需尺度一致)
rng = np.random.default_rng(0)                         # 固定随机源
ip = rng.choice(np.where(ytr == 1)[0], K, replace=False)     # 抽K个正例索引
ineg = rng.choice(np.where(ytr == 0)[0], K, replace=False)   # 抽K个负例索引

t0 = now()                                             # 起始计时
proto_pos, proto_neg = Xtr[ip].mean(0), Xtr[ineg].mean(0)  # 两类原型=各自样本均值
score = lambda X: (np.linalg.norm(X - proto_neg, axis=1)   # 到负原型距离
                   - np.linalg.norm(X - proto_pos, axis=1))  # 减到正原型距离: 离正原型越近分越高
report("44_少样本_原型法", yva, score(Xva), yte, score(Xte), now() - t0, extra={"标签数": 2 * K})  # 评测
