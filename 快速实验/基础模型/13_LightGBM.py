# -*- coding: utf-8 -*-
"""13 LightGBM —— 叶子生长GBDT, 大表格上通常最快最强的实现"""
import lightgbm as lgb                                 # LightGBM

from _common import load_flat, now, report            # 统一数据/计时/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
t0 = now()                                            # 起始计时
m = lgb.LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8,  # 600树/63叶(leaf-wise)/学习率0.05/行采样0.8
                       subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",      # 每轮行采样; 列采样0.8; 配平不平衡
                       random_state=0, n_jobs=-1, verbosity=-1)                              # 固定种子, 多核, 静默
m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",   # 训练, 以val AUPRC为早停指标
      callbacks=[lgb.early_stopping(50, verbose=False)])  # 早停50轮只看val(不碰test)
report("13_LightGBM", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类概率评测
