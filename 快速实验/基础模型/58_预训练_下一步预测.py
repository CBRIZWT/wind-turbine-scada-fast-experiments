# -*- coding: utf-8 -*-
"""58 预训练·下一步预测(自回归) —— 仅预训练 + 线性探针。

自监督任务: 用 t 时刻的 embedding 预测 t+1 时刻的特征向量。
与 57(掩码重构) 的差别只有自监督代理任务本身 —— 用于对比"哪种预训练目标学到的表示更好"。
"""
import numpy as np

from _common import DATA, FARM, now, report
from _domain import Model, apply_std, predict, train_supervised
from _farmfree import load_farmfree
from _pretrain import MAX_ROWS, PRETRAIN_FARM, build_or_load

TASK = "next"
enc, mu, sd = build_or_load(TASK)
Ftr, Fva, Fte = (load_farmfree(FARM, s) for s in ("train", "val", "test"))
ytr = np.load(DATA / "y_flat_train.npy").astype(int)
yva = np.load(DATA / "y_flat_val.npy").astype(int)
yte = np.load(DATA / "y_flat_test.npy").astype(int)
mtr = ytr != -1
Tr, Va, Te = (apply_std(F, mu, sd) for F in (Ftr[mtr], Fva, Fte))

t0 = now()
model = Model()
model.enc.load_state_dict(enc.state_dict())
model = train_supervised(model, Tr, ytr[mtr], epochs=1, freeze_encoder=True, seed=0)
report("58_预训练_下一步预测", yva, predict(model, Va), yte, predict(model, Te), now() - t0,
       extra={"范式": "预训练(自监督)+线性探针", "自监督任务": "下一步预测(自回归)",
              "预训练语料": f"{PRETRAIN_FARM} ≤{MAX_ROWS}行(0故障, 零泄漏)",
              "骨干": "冻结", "scores_are_probabilities": True})
