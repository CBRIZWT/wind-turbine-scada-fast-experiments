# -*- coding: utf-8 -*-
"""67 联邦·SCAFFOLD —— Non-IID 增强(控制变量校正)。

与 FedProx 的思路不同: 不惩罚本地更新幅度, 而是显式估计"客户端漂移方向"并在本地更新中
    减去它 (control variate)。理论上比 FedProx 收敛更快且不牺牲本地拟合能力。
"""
from _common import FARM, now, report
from _fed import eval_global, run_federated

ROUNDS = 3
t0 = now()
g, mu, sd, clients = run_federated(rounds=ROUNDS, scaffold=True, target_farm=FARM)
yva, sva, yte, ste, _ = eval_global(g, mu, sd, FARM)
report("67_联邦_SCAFFOLD", yva, sva, yte, ste, now() - t0,
       extra={"范式": "联邦学习(横向)", "算法": "SCAFFOLD(控制变量校正客户端漂移)",
              "治理问题": "Non-IID 客户端漂移", "通信轮数": ROUNDS,
              "客户端": {f: n for f, n in clients}, "scores_are_probabilities": True})
