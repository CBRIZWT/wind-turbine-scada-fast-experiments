# -*- coding: utf-8 -*-
"""52 零样本·富特征跨场迁移 —— 46 号的受控升级版。

与 46 号唯一差异: 表示 6 维 → 26 维农场无关特征。其余(源域训练、源域选阈、
    目标域零标签、真零样本)完全一致。
目的: 46 号排名 51/51 (AUPRC 0.0046)。对比 52 vs 46 可判定该结果是
    "跨场迁移本身不可行" 还是 "6 维表示太薄"; 对比 52 vs 51 可量化纯迁移代价。

防泄漏: 目标域标签全程不参与训练与选阈 (阈值在源域 val 上选) —— 真零样本。
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from _common import DATA, FARM, VARIANT, cross_farm_source, now, quick_data_dir, report, report_v3
from _farmfree import N_FEATURES, load_farmfree

SOURCE_FARM = cross_farm_source(FARM)
SRC = quick_data_dir(SOURCE_FARM)

# ---- 源域: 训练 + 选阈
Ftr = load_farmfree(SOURCE_FARM, "train")
ytr = np.load(SRC / "y_flat_train.npy").astype(int)
Fsv = load_farmfree(SOURCE_FARM, "val")
ysv = np.load(SRC / "y_flat_val.npy").astype(int)
mtr, msv = ytr != -1, ysv != -1

# ---- 目标域: 只取 test 特征; 标签仅用于最终一次评测
Fte = load_farmfree(FARM, "test")
yte = np.load(DATA / "y_flat_test.npy").astype(int)

t0 = now()
m = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08,
                                   l2_regularization=1.0, random_state=0)
m.fit(Ftr[mtr], ytr[mtr], sample_weight=compute_sample_weight("balanced", ytr[mtr]))
src_val_score = m.predict_proba(Fsv[msv])[:, 1]
tgt_test_score = m.predict_proba(Fte)[:, 1]

_extra = {"源域": SOURCE_FARM, "目标域": FARM, "表示": f"{N_FEATURES}维农场无关",
          "对照": "vs 46(6维) 判表示代价; vs 51(场内) 判迁移代价",
          "scores_are_probabilities": True}
if VARIANT:
    report_v3("52_零样本_富特征跨场", ysv[msv], src_val_score, yte, tgt_test_score, now() - t0,
              representation="flat",
              val_sidecars=(np.load(SRC / "timestamps_flat_val.npy"),
                            np.load(SRC / "turbines_flat_val.npy")),
              val_event_table=SRC / "event_table.csv",
              test_event_table=DATA / "event_table.csv",
              extra=_extra)
else:
    report("52_零样本_富特征跨场", ysv[msv], src_val_score, yte, tgt_test_score, now() - t0,
           extra=_extra)
