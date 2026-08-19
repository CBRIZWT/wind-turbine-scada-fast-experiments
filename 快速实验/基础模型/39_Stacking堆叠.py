# -*- coding: utf-8 -*-
"""39 Stacking堆叠 —— 集成学习: 基模型的交叉验证概率作特征, 逻辑回归作元学习器"""
import lightgbm as lgb                                 # 基模型之一
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,  # 基模型
                              StackingClassifier)       # 堆叠集成器
from sklearn.linear_model import LogisticRegression    # 元学习器

from _common import load_flat, now, report            # 统一数据/计时/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
t0 = now()                                            # 起始计时
m = StackingClassifier(                                # 堆叠: 基模型折外概率→元学习器
    estimators=[
        ("lgbm", lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05,   # 基模型1
                                    class_weight="balanced", random_state=0, n_jobs=-1,
                                    verbosity=-1)),
        ("hgb", HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08,  # 基模型2
                                               l2_regularization=1.0, random_state=0)),
        ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5,                     # 基模型3
                                      class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=0))],
    final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),  # 元学习器=逻辑回归(学组合权重)
    cv=3, n_jobs=1).fit(Xtr, ytr)  # cv=3: 元特征来自折外预测, 防泄漏
report("39_Stacking堆叠", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类概率评测
