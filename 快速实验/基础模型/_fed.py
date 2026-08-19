# -*- coding: utf-8 -*-
"""_fed.py — 联邦学习共享执行器 (横向 FL, 客户端 = 风场)。

场景映射: 三个风场 = 三个客户端, 各自 SCADA 数据不出本地, 只上传模型权重。
    这对应风电真实痛点 —— 不同业主的运行数据受合规约束不能互传。
    因 kel 87 / pen 89 / hot 53 通道不一致, 客户端统一用 26 维农场无关表示,
    模型结构才能一致 (FedAvg 的前提是参数可加权平均)。

防泄漏: 每个客户端只用自己的 train; 目标场 val 选阈; test 只评一次。
"""
from __future__ import annotations

import numpy as np

from _common import quick_data_dir
from _domain import Model, apply_std, fedavg_aggregate, local_train, predict, standardize_fit
from _farmfree import load_farmfree

CLIENTS = (("kelmarsh", ""), ("penmanshiel", ""), ("hill_of_towie", "real_fault_metrics_v1_external_local"))
DEFAULT_VARIANT = "real_fault_metrics_v1"


def load_clients(target_farm: str):
    """载入各客户端本地训练集 (农场无关表示 + 本地标签)。返回 [(farm, F, y), ...]。"""
    out = []
    for farm, var in CLIENTS:
        v = var if var else DEFAULT_VARIANT
        try:
            F = load_farmfree(farm, "train", variant=v)
            y = np.load(quick_data_dir(farm, v) / "y_flat_train.npy").astype(int)
        except FileNotFoundError:
            continue
        m = y != -1
        if m.sum() < 100:
            continue
        out.append((farm, F[m], y[m]))
    return out


def run_federated(rounds: int = 3, *, local_epochs: int = 1, mu_prox: float = 0.0,
                  dp_clip: float = 0.0, dp_noise: float = 0.0, scaffold: bool = False,
                  target_farm: str = "", seed: int = 0):
    """执行 R 轮 FedAvg 式联邦训练, 返回 (global_state, mu, sd, 客户端信息)。

    mu_prox>0 → FedProx; dp_*>0 → 差分隐私; scaffold=True → 用控制变量校正客户端漂移。
    """
    clients = load_clients(target_farm)
    if not clients:
        raise RuntimeError("没有可用的联邦客户端数据")
    # 标准化统计量: 各客户端本地统计的加权平均 (不汇集原始数据, 符合 FL 约束)
    stats = [(standardize_fit(F), len(F)) for _, F, _ in clients]
    tot = sum(n for _, n in stats)
    mu = sum(s[0][0] * (n / tot) for s, n in stats)
    sd = sum(s[0][1] * (n / tot) for s, n in stats)
    data = [(farm, apply_std(F, mu, sd), y) for farm, F, y in clients]

    g_state, ctrl = None, None
    for r in range(rounds):
        states, sizes = [], []
        for i, (_farm, F, y) in enumerate(data):
            st = local_train(g_state, F, y, epochs=local_epochs, mu_prox=mu_prox,
                             dp_clip=dp_clip, dp_noise=dp_noise, seed=seed + r * 10 + i)
            if scaffold and ctrl is not None:      # SCAFFOLD: 用上一轮全局漂移方向做校正
                for k in st:
                    if st[k].dtype.is_floating_point:
                        st[k] = st[k] - 0.5 * ctrl[k].to(st[k].device)
            states.append(st)
            sizes.append(len(y))
        new_state = fedavg_aggregate(states, sizes)
        if scaffold and g_state is not None:       # 控制变量 = 本轮全局更新方向
            ctrl = {k: (new_state[k].float() - g_state[k].float()) for k in new_state
                    if new_state[k].dtype.is_floating_point}
        g_state = new_state
    return g_state, mu, sd, [(f, len(y)) for f, _F, y in data]


def eval_global(g_state, mu, sd, farm, variant=None):
    """用全局模型在目标场 val/test 上打分。"""
    from _common import DATA
    model = Model()
    model.load_state_dict(g_state)
    Fva, Fte = load_farmfree(farm, "val", variant=variant), load_farmfree(farm, "test", variant=variant)
    yva = np.load(DATA / "y_flat_val.npy").astype(int)
    yte = np.load(DATA / "y_flat_test.npy").astype(int)
    return (yva, predict(model, apply_std(Fva, mu, sd)),
            yte, predict(model, apply_std(Fte, mu, sd)), model)
