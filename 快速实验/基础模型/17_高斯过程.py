# -*- coding: utf-8 -*-
"""17 高斯过程分类 —— 贝叶斯核方法; O(n³)只能3000样本子采样, 完整性对照"""
import numpy as np                                    # 分块拼接用
from sklearn.gaussian_process import GaussianProcessClassifier  # 高斯过程分类器

from _common import load_flat, now, report, standardize  # 统一数据/计时/评测/标准化

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 核方法对尺度敏感需标准化
step = max(1, len(Xtr) // 3000)                       # 计算下采样步长使子集≈3000(GP复杂度O(n³))
Xs, ys = Xtr[::step][:3000], ytr[::step][:3000]       # 等距抽样3000个训练点保时序覆盖

t0 = now()                                            # 起始计时
m = GaussianProcessClassifier(random_state=0).fit(Xs, ys)  # 在3000子集上训练GP分类器
# 分块预测, 防止一次算 61万×3000 的核矩阵爆内存
proba = lambda X: np.concatenate([m.predict_proba(X[i:i + 50000])[:, 1]   # 每5万一块算正类概率
                                  for i in range(0, len(X), 50000)])       # 分块遍历后拼接
report("17_高斯过程", yva, proba(Xva), yte, proba(Xte), now() - t0)  # 取正类概率评测
