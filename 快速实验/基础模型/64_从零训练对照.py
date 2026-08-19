# -*- coding: utf-8 -*-
"""64 从零训练对照 —— 【预训练收益的关键控制变量】。

与 57–63 完全相同的网络结构、相同表示、相同训练预算, 唯一差别: 随机初始化, 不加载任何
    预训练权重。故:
        (57~63 任一) − (64) = 预训练带来的净收益
    没有本脚本, "预训练有效"的结论无法与"这个网络结构本身就好"区分开。
"""
import numpy as np

from _common import DATA, FARM, now, report
from _domain import Model, apply_std, predict, standardize_fit, train_supervised
from _farmfree import load_farmfree

Ftr, Fva, Fte = (load_farmfree(FARM, s) for s in ("train", "val", "test"))
ytr = np.load(DATA / "y_flat_train.npy").astype(int)
yva = np.load(DATA / "y_flat_val.npy").astype(int)
yte = np.load(DATA / "y_flat_test.npy").astype(int)
mtr = ytr != -1
mu, sd = standardize_fit(Ftr[mtr])                       # 标准化统计量只来自本场 train
Tr, Va, Te = (apply_std(F, mu, sd) for F in (Ftr[mtr], Fva, Fte))

t0 = now()
model = train_supervised(Model(), Tr, ytr[mtr], epochs=1, lr=1e-3, seed=0)   # 随机初始化
report("64_从零训练对照", yva, predict(model, Va), yte, predict(model, Te), now() - t0,
       extra={"范式": "无预训练(随机初始化)", "角色": "预训练收益的控制变量",
              "网络": "与57~63完全相同", "scores_are_probabilities": True})
