# -*- coding: utf-8 -*-
"""70 集中式对照 —— 联邦学习的【性能上界】。

把三个风场的数据【汇集到一处】集中训练同一个网络。这在隐私上不可行(真实场景中
    不同业主的数据不能互传), 但作为上界基线是必需的:
        70 − 65 = 联邦学习为"数据不出域"付出的性能代价
    这是 FL 论文报告结论的标准形式。
"""
import numpy as np

from _common import DATA, FARM, now, report
from _domain import Model, apply_std, predict, standardize_fit, train_supervised
from _farmfree import load_farmfree
from _fed import load_clients

t0 = now()
clients = load_clients(FARM)
Fpool = np.concatenate([F for _f, F, _y in clients])          # 数据汇集(隐私上不可行)
ypool = np.concatenate([y for _f, _F, y in clients])
mu, sd = standardize_fit(Fpool)
model = train_supervised(Model(), apply_std(Fpool, mu, sd), ypool, epochs=1, seed=0)

Fva, Fte = load_farmfree(FARM, "val"), load_farmfree(FARM, "test")
yva = np.load(DATA / "y_flat_val.npy").astype(int)
yte = np.load(DATA / "y_flat_test.npy").astype(int)
report("70_集中式对照", yva, predict(model, apply_std(Fva, mu, sd)),
       yte, predict(model, apply_std(Fte, mu, sd)), now() - t0,
       extra={"范式": "集中式(联邦学习的性能上界)", "训练数据": "三场汇集",
              "角色": "对照 65 量化隐私代价", "样本量": {f: len(y) for f, _F, y in clients},
              "scores_are_probabilities": True})
