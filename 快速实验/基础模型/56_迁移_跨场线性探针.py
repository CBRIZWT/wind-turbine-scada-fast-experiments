# -*- coding: utf-8 -*-
"""56 迁移·基于参数 —— 跨场线性探针(冻结骨干 + 目标域调头)。

迁移学习四大类之【基于参数】(归纳式): 源域训练骨干 → 冻结 → 目标域 train 只训线性头。
    与 52(零样本)差别 = 允许用少量目标域标签调头; 与 61(全参微调)差别 = 骨干完全不动。
    这是衡量"源域表示是否可用"的标准协议 (Linear Probing)。
防泄漏: 只用目标域 train 调头, val 选阈, test 只评一次。
"""
import numpy as np

from _common import DATA, FARM, cross_farm_source, now, quick_data_dir, report
from _domain import Model, apply_std, predict, standardize_fit, train_supervised
from _farmfree import load_farmfree

SRC_FARM = cross_farm_source(FARM)
SRC = quick_data_dir(SRC_FARM)
Fs, ys = load_farmfree(SRC_FARM, "train"), np.load(SRC / "y_flat_train.npy").astype(int)
ms = ys != -1
Ftr, Fva, Fte = (load_farmfree(FARM, s) for s in ("train", "val", "test"))
ytr = np.load(DATA / "y_flat_train.npy").astype(int)
yva = np.load(DATA / "y_flat_val.npy").astype(int)
yte = np.load(DATA / "y_flat_test.npy").astype(int)
mtr = ytr != -1

mu, sd = standardize_fit(Fs[ms])
S = apply_std(Fs[ms], mu, sd)
Tr, Va, Te = (apply_std(F, mu, sd) for F in (Ftr[mtr], Fva, Fte))

t0 = now()
model = train_supervised(Model(), S, ys[ms], epochs=1, seed=0)          # 阶段1: 源域训骨干
model = train_supervised(model, Tr, ytr[mtr], epochs=1,                  # 阶段2: 冻结骨干只调头
                         freeze_encoder=True, seed=0)
report("56_迁移_跨场线性探针", yva, predict(model, Va), yte, predict(model, Te), now() - t0,
       extra={"迁移类型": "基于参数(冻结骨干+线性探针)", "源域": SRC_FARM, "目标域": FARM,
              "可训参数": "仅线性头", "scores_are_probabilities": True})
