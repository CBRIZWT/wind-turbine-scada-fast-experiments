# -*- coding: utf-8 -*-
"""53 迁移·基于实例 —— KMM 密度比重加权。

迁移学习四大类之【基于实例】: 不改模型也不改表示, 只给源域样本重加权 w(x)=p_t(x)/p_s(x),
    让源域训练分布贴近目标域。目标域标签全程不用 (阈值在源域 val 选) → 仍是零样本。
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from _common import DATA, FARM, VARIANT, cross_farm_source, now, quick_data_dir, report, report_v3
from _domain import kmm_weights
from _farmfree import load_farmfree

SRC_FARM = cross_farm_source(FARM)
SRC = quick_data_dir(SRC_FARM)
Ftr, Fsv = load_farmfree(SRC_FARM, "train"), load_farmfree(SRC_FARM, "val")
ytr = np.load(SRC / "y_flat_train.npy").astype(int)
ysv = np.load(SRC / "y_flat_val.npy").astype(int)
Fte = load_farmfree(FARM, "test")
yte = np.load(DATA / "y_flat_test.npy").astype(int)
# [BUG 修复 2026-07-26] 域适配必须用目标域【train】的无标签特征。此前误用 test 特征
#   拟合权重 → 违反 test_used_for_fit=False 契约(即使不用标签, 也是测试集泄漏)。
Ftgt_adapt = load_farmfree(FARM, "train")
mtr, msv = ytr != -1, ysv != -1

t0 = now()
w_dom = kmm_weights(Ftr[mtr], Ftgt_adapt)                            # 域适配权重(只用目标域train)
w_bal = compute_sample_weight("balanced", ytr[mtr])                  # 类不平衡权重
m = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08,
                                   l2_regularization=1.0, random_state=0)
m.fit(Ftr[mtr], ytr[mtr], sample_weight=w_dom * w_bal)
sv, st = m.predict_proba(Fsv[msv])[:, 1], m.predict_proba(Fte)[:, 1]
_e = {"迁移类型": "基于实例(KMM密度比重加权)", "源域": SRC_FARM, "目标域": FARM,
      "权重范围": f"[{w_dom.min():.2f},{w_dom.max():.2f}]", "scores_are_probabilities": True}
if VARIANT:
    report_v3("53_迁移_实例重加权", ysv[msv], sv, yte, st, now() - t0, representation="flat",
              val_sidecars=(np.load(SRC / "timestamps_flat_val.npy"), np.load(SRC / "turbines_flat_val.npy")),
              val_event_table=SRC / "event_table.csv", test_event_table=DATA / "event_table.csv", extra=_e)
else:
    report("53_迁移_实例重加权", ysv[msv], sv, yte, st, now() - t0, extra=_e)
