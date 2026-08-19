# -*- coding: utf-8 -*-
"""19 一类SVM —— 无监督边界法; 精确OCSVM推理太贵,
用sklearn官方推荐的大规模替代: Nystroem核近似 + SGD一类SVM"""
from sklearn.kernel_approximation import Nystroem     # RBF核近似
from sklearn.linear_model import SGDOneClassSVM        # SGD版一类SVM(大规模)

from _common import load_flat, needs_external_train_scores, now, report, standardize  # 统一数据/计时/评测/标准化

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # RBF对尺度敏感需标准化
t0 = now()                                            # 起始计时
ny = Nystroem(kernel="rbf", n_components=300, random_state=0)  # 300分量RBF核近似
Xn = Xtr[ytr == 0]
m = SGDOneClassSVM(nu=0.05, random_state=0).fit(ny.fit_transform(Xn))  # 仅在正常样本上学包住正常的边界(nu=5%界外)
report("19_一类SVM", yva, -m.decision_function(ny.transform(Xva)),   # 决策函数取负: 界外(异常)分数更高
       yte, -m.decision_function(ny.transform(Xte)), now() - t0,
       train_scores=(-m.decision_function(ny.transform(Xn[::4]))
                     if needs_external_train_scores() else None))    # test同理评测
