# -*- coding: utf-8 -*-
"""_domain.py — 跨域范式共享骨干: 预训练 / 微调 / 迁移学习 / 联邦学习。

统一约定 (让 4 大范式严格可比, 只变"知识来源"这一个变量):
    · 输入 = 26 维农场无关表示 (_farmfree), 因为 kel 87 / pen 89 / hot 53 通道不一致;
    · 骨干 = Encoder(26→64→32) + Head(32→1), backbone/head 可分离 (微调与个性化FL需要);
    · 训练 = BCE + 正例权重; seed=0; 1 epoch 快速口径 (与快速实验其余模型一致);
    · 评测 = 交回 _common.report → report_v3 事件级, 与 51 模型同榜。

防泄漏: 所有 fit/选阈只用 train/val; test 只在 report 内评一次。预训练语料仅用
    hill_of_towie (0 故障, 从不参与 kel/pen 评测) → 预训练阶段天然零泄漏。
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_IN, D_HID, D_EMB = 26, 64, 32


def seed_all(k: int = 0) -> None:
    torch.manual_seed(k)
    np.random.seed(k)


class Encoder(nn.Module):
    """26 → 64 → 32 编码器 (骨干, 可被冻结/迁移/联邦聚合)。"""

    def __init__(self, d_in: int = D_IN, d_hid: int = D_HID, d_emb: int = D_EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.BatchNorm1d(d_hid),
            nn.Linear(d_hid, d_emb), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    """Encoder + 线性头; head 可单独重置/训练 (线性探针 / 个性化FL)。"""

    def __init__(self, d_in: int = D_IN):
        super().__init__()
        self.enc = Encoder(d_in)
        self.head = nn.Linear(D_EMB, 1)

    def forward(self, x):
        return self.head(self.enc(x)).squeeze(-1)


def standardize_fit(F: np.ndarray):
    mu, sd = F.mean(0), F.std(0) + 1e-8
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_std(F, mu, sd):
    return ((np.asarray(F, dtype=np.float32) - mu) / sd).astype(np.float32)


def _batches(n, batch, rng):
    idx = rng.permutation(n)
    for k in range(0, n, batch):
        yield idx[k:k + batch]


def train_supervised(model, F, y, *, epochs=1, lr=1e-3, batch=1024,
                     freeze_encoder=False, lr_backbone=None, seed=0):
    """监督训练 (可冻结骨干 = 线性探针; 可给骨干单独学习率 = 判别式微调)。"""
    seed_all(seed)
    model = model.to(DEV)
    if freeze_encoder:
        for p in model.enc.parameters():
            p.requires_grad = False
    if lr_backbone is None:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
    else:
        opt = torch.optim.Adam([
            {"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": lr_backbone},
            {"params": model.head.parameters(), "lr": lr},
        ])
    pw = torch.tensor(max((y == 0).sum(), 1) / max((y == 1).sum(), 1), dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Ft = torch.from_numpy(np.asarray(F, dtype=np.float32)).to(DEV)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(DEV)
    rng = np.random.default_rng(seed)
    model.train()
    if freeze_encoder:
        # [BUG 修复 2026-07-26] 仅置 requires_grad=False 不够: model.train() 下
        #   BatchNorm 仍持续更新 running_mean/var, 骨干实际仍在适配目标域,
        #   "冻结骨干/线性探针"语义不成立。显式把编码器切 eval 模式才是真冻结。
        model.enc.eval()
    for _ in range(epochs):
        for b in _batches(len(y), batch, rng):
            opt.zero_grad()
            lossf(model(Ft[b]), yt[b]).backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, F, batch=8192):
    model = model.to(DEV)          # 联邦聚合后的 state 可能在 CPU, 统一搬到推理设备
    model.eval()
    out = []
    for k in range(0, len(F), batch):
        x = torch.from_numpy(np.asarray(F[k:k + batch], dtype=np.float32)).to(DEV)
        out.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# 预训练 (自监督, 语料 = hill_of_towie 健康数据; 无标签)
# ----------------------------------------------------------------------------
def pretrain(encoder, F, *, task="mask", epochs=1, lr=1e-3, batch=1024, seed=0):
    """自监督预训练 encoder。task ∈ {mask, next, contrast}。返回预训练好的 encoder。

    mask     — 随机遮蔽 30% 维度, 从 embedding 还原原始向量 (MAE 式);
    next     — 用 t 时刻 embedding 预测 t+1 时刻特征 (自回归, 需 F 按时间有序);
    contrast — 同一样本两次噪声增广互为正对, batch 内其余为负 (SimCLR/NT-Xent)。
    """
    seed_all(seed)
    encoder = encoder.to(DEV)
    dec = nn.Linear(D_EMB, D_IN).to(DEV)
    proj = nn.Sequential(nn.Linear(D_EMB, D_EMB), nn.ReLU(), nn.Linear(D_EMB, D_EMB)).to(DEV)
    params = list(encoder.parameters()) + list(dec.parameters()) + list(proj.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    Ft = torch.from_numpy(np.asarray(F, dtype=np.float32)).to(DEV)
    rng = np.random.default_rng(seed)
    encoder.train()
    for _ in range(epochs):
        for b in _batches(len(F), batch, rng):
            if len(b) < 8:
                continue
            x = Ft[b]
            opt.zero_grad()
            if task == "mask":
                m = (torch.rand_like(x) > 0.3).float()
                loss = ((dec(encoder(x * m)) - x) ** 2).mean()
            elif task == "next":
                bs = np.sort(b)
                nxt = np.clip(bs + 1, 0, len(F) - 1)
                loss = ((dec(encoder(Ft[bs])) - Ft[nxt]) ** 2).mean()
            elif task == "contrast":
                z1 = nn.functional.normalize(proj(encoder(x + 0.1 * torch.randn_like(x))), dim=1)
                z2 = nn.functional.normalize(proj(encoder(x + 0.1 * torch.randn_like(x))), dim=1)
                sim = z1 @ z2.T / 0.5
                loss = nn.functional.cross_entropy(sim, torch.arange(len(x), device=DEV))
            else:
                raise ValueError(f"未知预训练任务: {task}")
            loss.backward()
            opt.step()
    return encoder


# ----------------------------------------------------------------------------
# 迁移学习: 分布对齐 / 实例重加权
# ----------------------------------------------------------------------------
def coral(Fs: np.ndarray, Ft: np.ndarray) -> np.ndarray:
    """CORAL: 把源域特征二阶统计量对齐到目标域 (白化-重着色)。返回变换后的源域特征。"""
    def _sqrtm(C):
        w, V = np.linalg.eigh(C)
        return (V * np.sqrt(np.maximum(w, 1e-8))) @ V.T

    def _isqrtm(C):
        w, V = np.linalg.eigh(C)
        return (V * (1.0 / np.sqrt(np.maximum(w, 1e-8)))) @ V.T

    Fs = np.asarray(Fs, dtype=np.float64)
    Ft = np.asarray(Ft, dtype=np.float64)
    ms, mt = Fs.mean(0), Ft.mean(0)
    Cs = np.cov(Fs - ms, rowvar=False) + np.eye(Fs.shape[1])
    Ct = np.cov(Ft - mt, rowvar=False) + np.eye(Ft.shape[1])
    return (((Fs - ms) @ _isqrtm(Cs)) @ _sqrtm(Ct) + mt).astype(np.float32)


def kmm_weights(Fs: np.ndarray, Ft: np.ndarray, *, n_ref=20000, seed=0) -> np.ndarray:
    """实例重加权: 用高斯核密度比近似 w(x)=p_t(x)/p_s(x), 裁剪到 [0.1, 10]。"""
    rng = np.random.default_rng(seed)
    sub = lambda A: A[rng.choice(len(A), min(n_ref, len(A)), replace=False)]
    S, T = sub(np.asarray(Fs, np.float32)), sub(np.asarray(Ft, np.float32))
    mu, sd = S.mean(0), S.std(0) + 1e-8
    Sz, Tz = (S - mu) / sd, (T - mu) / sd
    sigma = float(np.median(np.linalg.norm(Sz[:500, None] - Sz[None, :500], axis=-1)) + 1e-6)
    Fz = (np.asarray(Fs, np.float32) - mu) / sd
    dens = lambda R: np.exp(-((np.linalg.norm(Fz[:, None, :] - R[None, :256, :], axis=-1) / sigma) ** 2)).mean(1) + 1e-8
    return np.clip(dens(Tz) / dens(Sz), 0.1, 10.0).astype(np.float32)


class GradReverse(torch.autograd.Function):
    """DANN 梯度反转层。"""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def train_dann(model, Fs, ys, Ft, *, epochs=1, lr=1e-3, batch=1024, lam=0.3, seed=0):
    """DANN: 分类损失(源域有标签) + 域判别器对抗(梯度反转) → 学域不变表示。"""
    seed_all(seed)
    model = model.to(DEV)
    disc = nn.Sequential(nn.Linear(D_EMB, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEV)
    opt = torch.optim.Adam(list(model.parameters()) + list(disc.parameters()), lr=lr)
    pw = torch.tensor(max((ys == 0).sum(), 1) / max((ys == 1).sum(), 1), dtype=torch.float32, device=DEV)
    clsf, domf = nn.BCEWithLogitsLoss(pos_weight=pw), nn.BCEWithLogitsLoss()
    S = torch.from_numpy(np.asarray(Fs, np.float32)).to(DEV)
    T = torch.from_numpy(np.asarray(Ft, np.float32)).to(DEV)
    yt = torch.from_numpy(np.asarray(ys, np.float32)).to(DEV)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for b in _batches(len(ys), batch, rng):
            if len(b) < 8:
                continue
            tb = rng.choice(len(T), len(b))
            opt.zero_grad()
            zs, zt = model.enc(S[b]), model.enc(T[tb])
            loss = clsf(model.head(zs).squeeze(-1), yt[b])
            d = disc(GradReverse.apply(torch.cat([zs, zt]), lam)).squeeze(-1)
            dl = torch.cat([torch.ones(len(b), device=DEV), torch.zeros(len(tb), device=DEV)])
            (loss + domf(d, dl)).backward()
            opt.step()
    return model


# ----------------------------------------------------------------------------
# 联邦学习
# ----------------------------------------------------------------------------
def fedavg_aggregate(states, weights):
    """FedAvg: 按样本量加权平均各客户端权重 (只传参数, 不传数据)。"""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = copy.deepcopy(states[0])
    for k in out:
        if out[k].dtype.is_floating_point:
            out[k] = sum(float(w[i]) * states[i][k].float() for i in range(len(states))).to(out[k].dtype)
    return out


def local_train(global_state, F, y, *, epochs=1, lr=1e-3, batch=1024, mu_prox=0.0,
                dp_clip=0.0, dp_noise=0.0, seed=0):
    """客户端本地训练。mu_prox>0 → FedProx 近端项; dp_*>0 → 差分隐私(裁剪+噪声)。"""
    seed_all(seed)
    model = Model().to(DEV)
    if global_state is not None:
        model.load_state_dict(global_state)
    gref = {k: v.detach().clone() for k, v in model.state_dict().items()} if mu_prox > 0 else None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pw = torch.tensor(max((y == 0).sum(), 1) / max((y == 1).sum(), 1), dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Ft = torch.from_numpy(np.asarray(F, np.float32)).to(DEV)
    yt = torch.from_numpy(np.asarray(y, np.float32)).to(DEV)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for b in _batches(len(y), batch, rng):
            opt.zero_grad()
            loss = lossf(model(Ft[b]), yt[b])
            if gref is not None:                       # FedProx: μ/2 · ‖θ − θ_global‖²
                prox = sum(((p - gref[n].to(DEV)) ** 2).sum()
                           for n, p in model.named_parameters() if n in gref)
                loss = loss + 0.5 * mu_prox * prox
            loss.backward()
            if dp_clip > 0:                            # 差分隐私: 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), dp_clip)
            opt.step()
    st = model.state_dict()
    if dp_noise > 0:                                   # 差分隐私: 高斯噪声
        for k in st:
            if st[k].dtype.is_floating_point:
                st[k] = st[k] + dp_noise * torch.randn_like(st[k].float()).to(st[k].dtype)
    return st
