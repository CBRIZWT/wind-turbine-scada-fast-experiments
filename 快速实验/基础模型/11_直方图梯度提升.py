# -*- coding: utf-8 -*-
"""11 直方图梯度提升 —— sklearn HistGBM, 战役已验证配置 (全量bar≈0.559, 主对照点)"""
from sklearn.ensemble import HistGradientBoostingClassifier  # 直方图梯度提升树
from sklearn.utils.class_weight import compute_sample_weight  # 按类频率算样本权重

from _common import load_flat, now, report            # 统一数据/计时/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
t0 = now()                                            # 起始计时
# [优化 2026-07-26] 早停改为"在真正的 val 上按 AUPRC 手动选轮数":
#   原实现用 early_stopping=True + validation_fraction=0.1 —— 从 train 里【随机】切 10% 做早停,
#   既打破了时序(随机切分含未来行), 又用默认 logloss 判据(0.13% 正例率下由负类主导)。
#   现改为: 关内置早停, 用 warm_start 逐段增长树, 每段在 val 上算 AUPRC, 取最优轮数。
#   仍严格只看 val, 不碰 test。
from sklearn.metrics import average_precision_score      # val 早停判据 = AUPRC

_sw = compute_sample_weight("balanced", ytr)              # balanced 样本权重配平不平衡
m = HistGradientBoostingClassifier(max_iter=50, max_depth=6, learning_rate=0.08,
                                   l2_regularization=1.0, early_stopping=False,
                                   warm_start=True, random_state=0)
_best_ap, _best_iter, _bad = -1.0, 50, 0
for _n in range(50, 351, 50):                             # 50→350 轮, 每 50 轮在 val 上评一次
    m.set_params(max_iter=_n)
    m.fit(Xtr, ytr, sample_weight=_sw)
    _ap = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
    if _ap > _best_ap:
        _best_ap, _best_iter, _bad = _ap, _n, 0
    else:
        _bad += 1
        if _bad >= 2:                                     # 连续 2 段未提升 → 早停
            break
if m.get_params()["max_iter"] != _best_iter:               # 回滚到 val AUPRC 最优轮数
    m.set_params(max_iter=_best_iter, warm_start=False)
    m.fit(Xtr, ytr, sample_weight=_sw)
report("11_直方图梯度提升", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类概率评测
