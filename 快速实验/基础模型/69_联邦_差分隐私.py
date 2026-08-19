# -*- coding: utf-8 -*-
"""69 联邦·差分隐私 —— 梯度裁剪 + 高斯噪声。

即使只传权重, 攻击者仍可能从更新中反推训练数据(梯度泄漏攻击)。DP-FedAvg 在本地
    裁剪梯度范数并向上传权重注入高斯噪声, 提供形式化隐私保证。
对照 65(无DP FedAvg): 65 − 69 = 隐私保证的额外性能代价。
"""
from _common import FARM, now, report
from _fed import eval_global, run_federated

ROUNDS, CLIP, NOISE = 3, 1.0, 0.01
t0 = now()
g, mu, sd, clients = run_federated(rounds=ROUNDS, dp_clip=CLIP, dp_noise=NOISE, target_farm=FARM)
yva, sva, yte, ste, _ = eval_global(g, mu, sd, FARM)
report("69_联邦_差分隐私", yva, sva, yte, ste, now() - t0,
       extra={"范式": "联邦学习(隐私增强)", "算法": f"DP-FedAvg(裁剪{CLIP} + 高斯噪声σ={NOISE})",
              "对照": "vs 65 = 隐私保证的性能代价", "通信轮数": ROUNDS,
              "客户端": {f: n for f, n in clients}, "scores_are_probabilities": True})
