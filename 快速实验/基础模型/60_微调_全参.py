# -*- coding: utf-8 -*-
"""60 微调·全参 —— 预训练骨干 + 所有层小学习率更新。

微调策略对照组之一。θ₀ 统一取 57 号的掩码重构预训练权重, 使"微调策略"成为唯一变量。
与 57(冻结) 的差别: 骨干参与更新, 可适配目标域, 但有灾难性遗忘风险。
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
model = train_supervised(model, Tr, ytr[mtr], epochs=1, lr=1e-3, seed=0)   # 全参更新
report("60_微调_全参", yva, predict(model, Va), yte, predict(model, Te), now() - t0,
       extra={"范式": "预训练+微调", "微调策略": "全参(所有层 lr=1e-3)",
              "预训练": f"{PRETRAIN_FARM} 掩码重构", "scores_are_probabilities": True})
