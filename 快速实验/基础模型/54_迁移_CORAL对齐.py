# -*- coding: utf-8 -*-
"""54 迁移·基于特征 —— CORAL 二阶统计量对齐。

迁移学习四大类之【基于特征】(无监督域适配): 把源域特征的协方差白化后重着色到目标域协方差,
    再在对齐后的源域上训练。只用目标域【无标签】特征做对齐 → 直推式, 目标标签不参与。
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from _common import DATA, FARM, VARIANT, cross_farm_source, now, quick_data_dir, report, report_v3
from _domain import coral
from _farmfree import load_farmfree

SRC_FARM = cross_farm_source(FARM)
SRC = quick_data_dir(SRC_FARM)
Ftr, Fsv = load_farmfree(SRC_FARM, "train"), load_farmfree(SRC_FARM, "val")
ytr = np.load(SRC / "y_flat_train.npy").astype(int)
ysv = np.load(SRC / "y_flat_val.npy").astype(int)
Fte = load_farmfree(FARM, "test")
yte = np.load(DATA / "y_flat_test.npy").astype(int)
# [BUG 修复 2026-07-26] 对齐目标必须是目标域【train】的无标签特征; 此前误用 test 特征
#   估计协方差 → 违反 test_used_for_fit=False 契约(测试集泄漏)。
Ftgt_adapt = load_farmfree(FARM, "train")
mtr, msv = ytr != -1, ysv != -1

t0 = now()
Ftr_a = coral(Ftr[mtr], Ftgt_adapt)             # 源域→目标域二阶对齐(仅用目标域 train 无标签特征)
Fsv_a = coral(Fsv[msv], Ftgt_adapt)             # 源域 val 同变换, 保证选阈一致
m = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08,
                                   l2_regularization=1.0, random_state=0)
m.fit(Ftr_a, ytr[mtr], sample_weight=compute_sample_weight("balanced", ytr[mtr]))
sv, st = m.predict_proba(Fsv_a)[:, 1], m.predict_proba(Fte)[:, 1]
_e = {"迁移类型": "基于特征(CORAL二阶对齐)", "源域": SRC_FARM, "目标域": FARM,
      "适配": "无监督(仅用目标域无标签特征)", "scores_are_probabilities": True}
if VARIANT:
    report_v3("54_迁移_CORAL对齐", ysv[msv], sv, yte, st, now() - t0, representation="flat",
              val_sidecars=(np.load(SRC / "timestamps_flat_val.npy"), np.load(SRC / "turbines_flat_val.npy")),
              val_event_table=SRC / "event_table.csv", test_event_table=DATA / "event_table.csv", extra=_e)
else:
    report("54_迁移_CORAL对齐", ysv[msv], sv, yte, st, now() - t0, extra=_e)
