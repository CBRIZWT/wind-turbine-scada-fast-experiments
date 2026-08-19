# -*- coding: utf-8 -*-
"""66 联邦·FedProx —— Non-IID 增强(近端项)。

三个风场机型异构(Senvion MM82/MM92 与 Siemens), 数据高度 Non-IID, FedAvg 的客户端
    会各自漂离全局最优。FedProx 在本地损失上加 μ/2·‖θ−θ_global‖², 约束本地更新幅度。
"""
from _common import FARM, now, report
from _fed import eval_global, run_federated

ROUNDS, MU = 3, 0.01
t0 = now()
g, mu_, sd, clients = run_federated(rounds=ROUNDS, mu_prox=MU, target_farm=FARM)
yva, sva, yte, ste, _ = eval_global(g, mu_, sd, FARM)
report("66_联邦_FedProx", yva, sva, yte, ste, now() - t0,
       extra={"范式": "联邦学习(横向)", "算法": f"FedProx(近端项 μ={MU})",
              "治理问题": "Non-IID 客户端漂移", "通信轮数": ROUNDS,
              "客户端": {f: n for f, n in clients}, "scores_are_probabilities": True})
