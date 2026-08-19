# -*- coding: utf-8 -*-
"""16 核SVM —— RBF核; 精确SVC在29万训练/61万测试上不可行,
用业界标准替代: Nystroem核近似(300分量) + 线性SVM"""
from sklearn.kernel_approximation import Nystroem     # Nystroem核近似(低秩显式特征映射)
from sklearn.svm import LinearSVC                      # 线性SVM(在近似特征上做)

from _common import load_flat, now, report, standardize  # 统一数据/计时/评测/标准化

Xtr, ytr, Xva, yva, Xte, yte = load_flat()            # 加载扁平特征与标签
Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)            # RBF对尺度敏感必须标准化
t0 = now()                                            # 起始计时
ny = Nystroem(kernel="rbf", n_components=300, random_state=0)  # RBF核用300个地标点低秩近似
Ztr = ny.fit_transform(Xtr)                           # 把RBF核映射成300维显式特征
m = LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=3000).fit(Ztr, ytr)  # 在近似特征上训练线性SVM
report("16_核SVM", yva, m.decision_function(ny.transform(Xva)),    # val经同一核映射后取决策函数当分数
       yte, m.decision_function(ny.transform(Xte)), now() - t0)   # test同理; test只评一次
