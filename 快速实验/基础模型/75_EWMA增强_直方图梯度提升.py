# -*- coding: utf-8 -*-
"""75 EWMA增强_直方图梯度提升 —— EWMA 分数增强变体。

方法依据 (2026-07-26 两次独立筛选互证): 快速实验诊断显示瓶颈在【精确率】而非召回
    (38_软投票集成 检出 8/10 事件却产生 794 个误报段, 点精确率仅 4.29%)。
    对 5 种因果后处理做筛选, **EWMA 指数滑动平均**是唯一同时显著提升 AUPRC (+19%)
    且几乎不损事件F1 的方法 —— 真实热退化有热惯量、持续走高, 而误报多为瞬时假峰。

与基模型 11_直方图梯度提升 的唯一差别: 分数经逐机组因果 EWMA(span=12, 即2小时) 平滑。
防泄漏: EWMA 为因果变换(只用过去)+ 逐机组隔离 + 无参数拟合 + 不用标签,
    故 val/test 一致施加不构成泄漏; 阈值仍只在 val 选、test 只评一次。
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from _common import DATA, load_flat, now, report
from _boost import ewma

TB_VAL = np.load(DATA / "turbines_flat_val.npy")
TB_TEST = np.load(DATA / "turbines_flat_test.npy")
Xtr, ytr, Xva, yva, Xte, yte = load_flat()

t0 = now()
m = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08,
                                   l2_regularization=1.0, random_state=0)
m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))

sva = ewma(m.predict_proba(Xva)[:, 1], TB_VAL, span=12)
ste = ewma(m.predict_proba(Xte)[:, 1], TB_TEST, span=12)
report("75_EWMA增强_直方图梯度提升", yva, sva, yte, ste, now() - t0,
       extra={"增强": "EWMA(span=12, 逐机组因果)", "基模型": "11_直方图梯度提升",
              "方法依据": "筛选实测 AUPRC +19%", "scores_are_probabilities": True})
