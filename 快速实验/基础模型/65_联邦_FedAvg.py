# -*- coding: utf-8 -*-
"""65 联邦·FedAvg —— 横向联邦学习基础算法。

三个风场 = 三个客户端, 数据不出本地, 每轮只上传权重, 服务器按样本量加权平均后下发。
对照 70(集中式): 70 − 65 = 隐私保护(数据不共享)所付出的性能代价。
"""
from _common import FARM, now, report
from _fed import eval_global, run_federated

ROUNDS = 3
t0 = now()
g, mu, sd, clients = run_federated(rounds=ROUNDS, target_farm=FARM)
yva, sva, yte, ste, _ = eval_global(g, mu, sd, FARM)
report("65_联邦_FedAvg", yva, sva, yte, ste, now() - t0,
       extra={"范式": "联邦学习(横向)", "算法": "FedAvg(按样本量加权平均)",
              "通信轮数": ROUNDS, "客户端": {f: n for f, n in clients},
              "数据是否出域": "否(只传权重)", "scores_are_probabilities": True})
