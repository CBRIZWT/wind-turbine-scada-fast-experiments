# -*- coding: utf-8 -*-
"""59 预训练·对比学习(SimCLR/NT-Xent式) —— 仅预训练 + 线性探针。

自监督任务: 同一样本经两次噪声增广得到的两个视角互为正对, batch 内其余样本为负对,
    用 InfoNCE 拉近正对、推远负对。学到的是"对扰动不变的判别性表示"。
"""
import numpy as np

from _common import DATA, FARM, now, report
from _domain import Model, apply_std, predict, train_supervised
from _farmfree import load_farmfree
from _pretrain import MAX_ROWS, PRETRAIN_FARM, build_or_load

TASK = "contrast"
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
report("59_预训练_对比学习", yva, predict(model, Va), yte, predict(model, Te), now() - t0,
       extra={"范式": "预训练(自监督)+线性探针", "自监督任务": "对比学习(InfoNCE, 温度0.5)",
              "预训练语料": f"{PRETRAIN_FARM} ≤{MAX_ROWS}行(0故障, 零泄漏)",
              "骨干": "冻结", "scores_are_probabilities": True})
