# -*- coding: utf-8 -*-
"""12 XGBoost —— GPU梯度提升树, scale_pos_weight治不平衡, val早停"""
import torch                                           # 仅用于探测是否有GPU
from xgboost import XGBClassifier                       # XGBoost分类器

from _common import load_flat, now, report            # 统一数据/计时/评测

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
t0 = now()                                            # 起始计时
m = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,   # 500树/深6/学习率0.1/行采样0.8
                  colsample_bytree=0.8, scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(),  # 列采样0.8; 正例权重=负/正样本比
                  tree_method="hist", device="cuda" if torch.cuda.is_available() else "cpu",   # 直方图法; 有GPU则用cuda
                  eval_metric="aucpr", early_stopping_rounds=30, random_state=0)     # 以val AUPRC早停30轮, 固定种子
m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)  # 训练, 早停只看val(不碰test)
report("12_XGBoost", yva, m.predict_proba(Xva)[:, 1], yte, m.predict_proba(Xte)[:, 1], now() - t0)  # 取正类概率评测
