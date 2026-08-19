# -*- coding: utf-8 -*-
"""63 微调·LoRA(参数高效微调 PEFT) —— 冻结原权重, 只训低秩增量。

微调策略对照组之一: 对骨干每个 Linear 层 W 加旁路 ΔW = B·A (秩 r=4),
    原 W 冻结, 只训 A/B 与头。可训参数量远小于全参微调, 且天然抗灾难性遗忘
    (原权重一字未动)。是当前大模型时代最主流的微调方式。
"""
import numpy as np
import torch
from torch import nn

from _common import DATA, FARM, now, report
from _domain import Model, apply_std, predict, train_supervised
from _farmfree import load_farmfree
from _pretrain import PRETRAIN_FARM, build_or_load

RANK = 4


class LoRALinear(nn.Module):
    """W(冻结) + (B·A)·scale(可训), 秩 r 低秩增量。"""

    def __init__(self, base: nn.Linear, r: int = RANK, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))     # 初始 ΔW=0, 等价原模型
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


enc, mu, sd = build_or_load("mask")
Ftr, Fva, Fte = (load_farmfree(FARM, s) for s in ("train", "val", "test"))
ytr = np.load(DATA / "y_flat_train.npy").astype(int)
yva = np.load(DATA / "y_flat_val.npy").astype(int)
yte = np.load(DATA / "y_flat_test.npy").astype(int)
mtr = ytr != -1
Tr, Va, Te = (apply_std(F, mu, sd) for F in (Ftr[mtr], Fva, Fte))

t0 = now()
model = Model()
model.enc.load_state_dict(enc.state_dict())
for i, layer in enumerate(model.enc.net):                      # 给骨干每个 Linear 挂 LoRA 旁路
    if isinstance(layer, nn.Linear):
        model.enc.net[i] = LoRALinear(layer)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
model = train_supervised(model, Tr, ytr[mtr], epochs=1, lr=1e-3, seed=0)
report("63_微调_LoRA", yva, predict(model, Va), yte, predict(model, Te), now() - t0,
       extra={"范式": "预训练+微调", "微调策略": f"LoRA(秩{RANK}, 原权重冻结)",
              "可训参数占比": f"{n_train}/{n_total} = {100.0 * n_train / n_total:.1f}%",
              "预训练": f"{PRETRAIN_FARM} 掩码重构", "scores_are_probabilities": True})
