# -*- coding: utf-8 -*-
"""00 参照基线: 正残差能量分数 (不训练) —— 一切模型先过这条锚线"""
from _common import load_flat, needs_external_train_scores, now, report  # 复用统一的数据加载/计时/评测汇报

Xtr, ytr, Xva, yva, Xte, yte = load_flat()           # 加载扁平特征(此模型不用train, 只为接口一致)
t0 = now()                                           # 起始计时(本模型无训练, 耗时≈0)
# 倒数第5列 = 跨通道正残差能量 mean(max(0,残差)^2), 物理对齐的现成异常分数 (kel=第88列, 随farm通道数动态)
report("00_基线_残差能量", yva, Xva[:, -5], yte, Xte[:, -5], now() - t0,
       train_scores=Xtr[:, -5] if needs_external_train_scores() else None)  # Hill无标签阈值仅用train
