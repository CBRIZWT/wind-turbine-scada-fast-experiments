# -*- coding: utf-8 -*-
"""07 决策树 —— 单棵CART; 限深防记忆树 (不限会过拟合到概率全0/1)"""
from sklearn.tree import DecisionTreeClassifier       # CART决策树

from _common import load_flat, now, report            # 统一数据/计时/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签(树对尺度不敏感, 不标准化)
t0 = now()                                            # 起始计时
m = DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, class_weight="balanced",  # 限深12/叶≥20样本防过拟合
                           random_state=0).fit(Xtr, ytr)                                 # 类权重配平, 固定种子, 训练
report("07_决策树", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类概率评测
