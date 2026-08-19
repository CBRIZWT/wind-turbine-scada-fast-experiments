# -*- coding: utf-8 -*-
"""02 感知机 —— 最基础的线性神经元 (历史锚点, 无概率输出用决策函数当分数)"""
from sklearn.linear_model import Perceptron           # sklearn感知机(单线性神经元)

from _common import load_flat, now, report, standardize  # 统一数据/计时/评测/标准化

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 标准化(尺度敏感; 仅用train统计量)
t0 = now()                                            # 起始计时
m = Perceptron(class_weight="balanced", random_state=0).fit(Xtr, ytr)  # 类权重配平, 固定种子, 训练
report("02_感知机", yva, m.decision_function(Xva), yte, m.decision_function(Xte), now() - t0)  # 无概率, 用决策函数(离超平面距离)当分数
