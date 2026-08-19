# -*- coding: utf-8 -*-
"""51 场内·农场无关基线 —— 跨域方法族的【关键控制变量】。

作用: 本场自训 + 本场标签 + 与跨域方法【完全相同】的 26 维农场无关表示。
    有了它, 跨域方法的分数下降才能被拆成两部分:
        (本场原始表示基线) − (本场农场无关基线) = 表示代价
        (本场农场无关基线) − (跨域方法)        = 迁移代价
    没有本脚本, 任何"跨场/跨机效果差"的结论都无法归因。

口径: 与主线一致 —— train 拟合, val 选阈, test 只评一次; 不使用任何 test 信息。
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from _common import FARM, now, report
from _farmfree import N_FEATURES, load_farmfree_xy

Ftr, ytr, Fva, yva, Fte, yte = load_farmfree_xy(FARM)   # 26维农场无关特征 + 本场标签
mtr = ytr != -1                                          # 去掉 ignore 行(事件期不参与训练)

t0 = now()
m = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08,
                                   l2_regularization=1.0, random_state=0)
m.fit(Ftr[mtr], ytr[mtr], sample_weight=compute_sample_weight("balanced", ytr[mtr]))
report("51_场内_农场无关基线",
       yva, m.predict_proba(Fva)[:, 1],
       yte, m.predict_proba(Fte)[:, 1], now() - t0,
       extra={"表示": f"{N_FEATURES}维农场无关", "域": f"{FARM}→{FARM}(场内)",
              "角色": "跨域方法族控制变量", "scores_are_probabilities": True})
