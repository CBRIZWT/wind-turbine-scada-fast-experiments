# -*- coding: utf-8 -*-
"""62 微调·分层解冻 —— 先只训头, 再解冻骨干继续训。

微调策略对照组之一: 两阶段。阶段1 冻结骨干只训头(让头先对齐到合理区域, 避免
    随机初始化的头产生大梯度冲毁预训练骨干); 阶段2 解冻全部小 lr 继续。
    这是对抗"灾难性遗忘"的标准做法。
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
model = train_supervised(model, Tr, ytr[mtr], epochs=1, freeze_encoder=True, seed=0)  # 阶段1: 只训头
for p in model.enc.parameters():                                                      # 解冻骨干
    p.requires_grad = True
model = train_supervised(model, Tr, ytr[mtr], epochs=1, lr=1e-4, seed=0)              # 阶段2: 全参小lr
report("62_微调_分层解冻", yva, predict(model, Va), yte, predict(model, Te), now() - t0,
       extra={"范式": "预训练+微调", "微调策略": "分层解冻(阶段1冻结训头 → 阶段2全参lr=1e-4)",
              "预训练": f"{PRETRAIN_FARM} 掩码重构", "scores_are_probabilities": True})
