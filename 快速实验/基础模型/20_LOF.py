# -*- coding: utf-8 -*-
"""20 局部离群因子 —— 无监督密度法; 参考集子采样2万控制推理成本"""
from sklearn.neighbors import LocalOutlierFactor      # 局部离群因子

from _common import load_flat, needs_external_train_scores, now, report, standardize  # 统一数据/计时/评测/标准化

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # 距离/密度对尺度敏感需标准化
t0 = now()                                            # 起始计时
Xref = Xtr[ytr == 0][::12]
m = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(Xref)  # 20近邻密度; novelty模式对新样本打分; 正常参考集下采样12倍
report("20_LOF", yva, -m.score_samples(Xva), yte, -m.score_samples(Xte), now() - t0,
       train_scores=-m.score_samples(Xref) if needs_external_train_scores() else None)  # 负分作异常分数
