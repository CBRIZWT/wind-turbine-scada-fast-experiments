# -*- coding: utf-8 -*-
"""61 微调·判别式学习率 —— 骨干小 lr、头大 lr。

微调策略对照组之一: 底层保留通用知识(lr 小 10 倍), 顶层快速适配任务(lr 正常)。
是"全冻结(57)"与"全参同 lr(60)"之间的折中, 常见于 ULMFiT 式实践。
"""
import numpy as np

from _common import DATA, FARM, now, report
from _domain import Model, apply_std, predict, train_supervised
from _farmfree import load_farmfree
from _pretrain import PRETRAIN_FARM, build_or_load

enc, mu, sd = build_or_load("mask")
Ftr, Fva, Fte = (load_farmfree(FARM, s) for s in ("train", "val", "test"))
ytr = np.load(DATA / "y_flat_train.npy").astype(int)
yva = np.load(DATA / "y_flat_val.npy").astype(int)
yte = np.load(DATA / "y_flat_test.npy").astype(int)
mtr = ytr != -1
Tr, Va, Te = (apply_std(F, mu, sd) for F in (Ftr[mtr], Fva, Fte))

t0 = now()
model = Model()
model.enc.load_state_dict(enc.state_dict())
model = train_supervised(model, Tr, ytr[mtr], epochs=1, lr=1e-3, lr_backbone=1e-4, seed=0)
report("61_微调_判别式学习率", yva, predict(model, Va), yte, predict(model, Te), now() - t0,
       extra={"范式": "预训练+微调", "微调策略": "判别式学习率(骨干1e-4 / 头1e-3)",
              "预训练": f"{PRETRAIN_FARM} 掩码重构", "scores_are_probabilities": True})
