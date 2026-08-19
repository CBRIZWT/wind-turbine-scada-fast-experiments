# -*- coding: utf-8 -*-
"""55 迁移·基于特征 —— DANN 域对抗(梯度反转)。

迁移学习四大类之【基于特征】(对抗式域适配): 编码器同时被要求
    (a) 在源域上把故障分对, (b) 让域判别器分不出样本来自哪个域(梯度反转实现)。
    结果是【域不变表示】。目标域仅提供无标签特征参与对抗, 标签不参与。
"""
import numpy as np

from _common import DATA, FARM, VARIANT, cross_farm_source, now, quick_data_dir, report, report_v3
from _domain import Model, apply_std, predict, standardize_fit, train_dann
from _farmfree import load_farmfree

SRC_FARM = cross_farm_source(FARM)
SRC = quick_data_dir(SRC_FARM)
Ftr, Fsv = load_farmfree(SRC_FARM, "train"), load_farmfree(SRC_FARM, "val")
ytr = np.load(SRC / "y_flat_train.npy").astype(int)
ysv = np.load(SRC / "y_flat_val.npy").astype(int)
Fte = load_farmfree(FARM, "test")
yte = np.load(DATA / "y_flat_test.npy").astype(int)
mtr, msv = ytr != -1, ysv != -1

mu, sd = standardize_fit(Ftr[mtr])                       # 标准化统计量只来自源域 train
# [BUG 修复 2026-07-26] 对抗训练的目标域样本必须取自目标域【train】; 此前误用 test 特征
#   参与域判别器训练 → 违反 test_used_for_fit=False 契约(测试集泄漏)。
Ftgt_adapt = load_farmfree(FARM, "train")
S, Sv = apply_std(Ftr[mtr], mu, sd), apply_std(Fsv[msv], mu, sd)
Tadapt, T = apply_std(Ftgt_adapt, mu, sd), apply_std(Fte, mu, sd)

t0 = now()
model = train_dann(Model(), S, ytr[mtr], Tadapt, epochs=1, lam=0.3, seed=0)
sv, st = predict(model, Sv), predict(model, T)
_e = {"迁移类型": "基于特征(DANN梯度反转对抗)", "源域": SRC_FARM, "目标域": FARM,
      "lambda": 0.3, "scores_are_probabilities": True}
if VARIANT:
    report_v3("55_迁移_DANN对抗", ysv[msv], sv, yte, st, now() - t0, representation="flat",
              val_sidecars=(np.load(SRC / "timestamps_flat_val.npy"), np.load(SRC / "turbines_flat_val.npy")),
              val_event_table=SRC / "event_table.csv", test_event_table=DATA / "event_table.csv", extra=_e)
else:
    report("55_迁移_DANN对抗", ysv[msv], sv, yte, st, now() - t0, extra=_e)
