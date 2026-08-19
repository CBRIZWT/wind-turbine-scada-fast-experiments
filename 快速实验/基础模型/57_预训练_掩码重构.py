# -*- coding: utf-8 -*-
"""57 预训练·掩码重构(MAE式) —— 仅预训练, 用线性探针评估表示质量。

自监督任务: 随机遮蔽 30% 特征维度, 从 embedding 还原原始向量。
本脚本【只做预训练 + 冻结线性探针】, 不做微调 —— 它衡量的是"预训练表示本身有多好"
(Linear Probing 是表示学习的标准评估协议)。真正的微调对照见 60–64。
语料: hill_of_towie (0 故障, 不参与 kel/pen 评测) → 零泄漏。
"""
import numpy as np

from _common import DATA, FARM, now, report
from _domain import Model, apply_std, predict, train_supervised
from _farmfree import load_farmfree
from _pretrain import MAX_ROWS, PRETRAIN_FARM, build_or_load

TASK = "mask"
enc, mu, sd = build_or_load(TASK)
Ftr, Fva, Fte = (load_farmfree(FARM, s) for s in ("train", "val", "test"))
ytr = np.load(DATA / "y_flat_train.npy").astype(int)
yva = np.load(DATA / "y_flat_val.npy").astype(int)
yte = np.load(DATA / "y_flat_test.npy").astype(int)
mtr = ytr != -1
Tr, Va, Te = (apply_std(F, mu, sd) for F in (Ftr[mtr], Fva, Fte))

t0 = now()
model = Model()
model.enc.load_state_dict(enc.state_dict())                     # 装载预训练骨干
model = train_supervised(model, Tr, ytr[mtr], epochs=1, freeze_encoder=True, seed=0)  # 冻结, 只训头
report("57_预训练_掩码重构", yva, predict(model, Va), yte, predict(model, Te), now() - t0,
       extra={"范式": "预训练(自监督)+线性探针", "自监督任务": "掩码重构(遮蔽30%维度)",
              "预训练语料": f"{PRETRAIN_FARM} ≤{MAX_ROWS}行(0故障, 零泄漏)",
              "骨干": "冻结", "scores_are_probabilities": True})
