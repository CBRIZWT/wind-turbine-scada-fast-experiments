# -*- coding: utf-8 -*-
"""68 联邦·个性化 PFL —— 共享骨干 + 各场本地头。

针对机型异构的最优 FL 变体: 骨干(通用"什么是异常"的表示)全局聚合共享,
    分类头(各场自己的报警阈值/故障模式)留在本地不上传、不聚合。
    这样既享受多场协同, 又不被 Non-IID 拖累。
"""
import numpy as np

from _common import DATA, FARM, now, report
from _domain import Model, apply_std, predict, train_supervised
from _farmfree import load_farmfree
from _fed import eval_global, run_federated

ROUNDS = 3
t0 = now()
g, mu, sd, clients = run_federated(rounds=ROUNDS, target_farm=FARM)   # 全局骨干
model = Model()
model.load_state_dict(g)

# 个性化: 冻结全局骨干, 只在本场 train 上重训本地头
Ftr = load_farmfree(FARM, "train")
ytr = np.load(DATA / "y_flat_train.npy").astype(int)
mtr = ytr != -1
model = train_supervised(model, apply_std(Ftr[mtr], mu, sd), ytr[mtr],
                         epochs=1, freeze_encoder=True, seed=0)

Fva, Fte = load_farmfree(FARM, "val"), load_farmfree(FARM, "test")
yva = np.load(DATA / "y_flat_val.npy").astype(int)
yte = np.load(DATA / "y_flat_test.npy").astype(int)
report("68_联邦_个性化PFL", yva, predict(model, apply_std(Fva, mu, sd)),
       yte, predict(model, apply_std(Fte, mu, sd)), now() - t0,
       extra={"范式": "联邦学习(个性化 PFL)", "算法": "FedAvg共享骨干 + 本地头不聚合",
              "治理问题": "机型异构 Non-IID", "通信轮数": ROUNDS,
              "客户端": {f: n for f, n in clients}, "scores_are_probabilities": True})
