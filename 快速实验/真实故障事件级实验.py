# -*- coding: utf-8 -*-
"""真实故障事件级快速实验 (2026-07-11)。

任务: 预测每一时刻温度是否将偏离正常 (NBM 残差已是"实测−预测温度"), 分数超阈报预警;
以**真实故障事件级口径**(event-recall / lead-time / FAR/天 / event_f1)对比
基础模型 / 论文方法 / 组合模型, 选最优。

数据: SCADA数据集/数据预处理/<farm>__realfault/ (label_mode=real_fault_wl,
      白名单事件 72h 合并 + 事前 H=72 步早警窗 + 事后 24h ignore)。
协议 (防泄漏红线):
  - 监督模型只用 train_sup (2016-21); 无监督只用去污 train; 阈值仅 val (2022) 选
    (最大化 val event_f1); test (2023-24) 每模型评一次。
  - 因果特征逐机组计算 (修正旧版跨机组滚动混合的已知局限)。
  - 单 seed 快速口径, 结论用于模型排序; 不承诺全量数字。
运行: E:\\ancoda\\chuangxin\\python.exe 快速实验/真实故障事件级实验.py --farm penmanshiel
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent            # 快速实验/
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT / "SCADA数据集")):
    if p not in sys.path:
        sys.path.insert(0, p)

from 事件级评测 import (event_level_metrics, load_episodes_from_event_table,  # noqa: E402
                        select_threshold_event)

SEED = 0
W_FEAT = 72      # 因果滚动窗 12h
K_RECENT = 6     # 近增量 1h
TRAIN_STRIDE = 4  # 监督 train 下采样 (快速口径)
UNSUP_MAX_N = 200_000  # 无监督 train 子采样上限
SEQ_SUP_STRIDE = 8     # 窗口级监督 NN 的 train 窗口下采样


class _Transpose(__import__("torch").nn.Module):
    """(B, L, D) → (B, D, L), 供 Conv1d。"""
    def forward(self, x):
        return x.transpose(1, 2)


# ------------------------------------------------------------
# 核心纯函数 (TDD 覆盖: tests/test_realfault_fast_exp.py)
# ------------------------------------------------------------
def per_turbine_causal_features(X: np.ndarray, turbines: np.ndarray,
                                w_feat: int = W_FEAT, k_recent: int = K_RECENT) -> np.ndarray:
    """(T,D)+机组 → (T, D+6) 逐机组因果特征 [X | maxc, pose, roll_mean, roll_max, slope, recent]。

    仅用该机组的过去+当前 (groupby rolling), 杜绝跨机组串扰与未来泄漏。
    """
    X = np.asarray(X, dtype=np.float32)
    turbines = np.asarray(turbines)
    maxc = X.max(axis=1)
    pose = np.mean(np.maximum(0.0, X) ** 2, axis=1)
    s = pd.Series(maxc)
    g = s.groupby(pd.Series(turbines))
    roll_mean = g.transform(lambda v: v.rolling(w_feat, min_periods=1).mean()).to_numpy()
    roll_max = g.transform(lambda v: v.rolling(w_feat, min_periods=1).max()).to_numpy()
    shift_w = g.transform(lambda v: v.shift(w_feat).fillna(v.iloc[0])).to_numpy()
    shift_k = g.transform(lambda v: v.shift(k_recent).fillna(v.iloc[0])).to_numpy()
    slope = maxc - shift_w
    recent = maxc - shift_k
    return np.column_stack([X, maxc, pose, roll_mean, roll_max, slope, recent]).astype(np.float32)


def per_turbine_ewma(x: np.ndarray, turbines: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """逐机组因果 EWMA (adjust=False), 每机组独立状态。"""
    s = pd.Series(np.asarray(x, dtype=float))
    return (s.groupby(pd.Series(np.asarray(turbines)))
             .transform(lambda v: v.ewm(alpha=alpha, adjust=False).mean()).to_numpy())


def per_turbine_cusum(x: np.ndarray, turbines: np.ndarray, k: float = 0.5) -> np.ndarray:
    """逐机组单边 CUSUM: c_t = max(0, c_{t-1} + x_t − k), 每机组从 0 重启。"""
    x = np.asarray(x, dtype=float)
    turbines = np.asarray(turbines)
    out = np.zeros(len(x))
    for t in np.unique(turbines):
        idx = np.where(turbines == t)[0]
        c = 0.0
        for i in idx:
            c = max(0.0, c + x[i] - k)
            out[i] = c
    return out


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """秩归一 [0,1] (NaN→0.5 中性)。跨模型分数可融合。"""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), 0.5)
    fin = np.isfinite(x)
    if fin.sum() > 1:
        r = np.argsort(np.argsort(x[fin]))
        out[fin] = r / (fin.sum() - 1)
    return out


# ------------------------------------------------------------
# 数据装载
# ------------------------------------------------------------
def load_variant(farm: str) -> Dict[str, object]:
    d = ROOT / "SCADA数据集" / "数据预处理" / f"{farm}__realfault"
    need = ["train.npy", "train_sup.npy", "train_sup_labels.npy", "val.npy", "val_labels.npy",
            "test.npy", "test_labels.npy", "timestamps_val.npy", "turbines_val.npy",
            "timestamps_test.npy", "turbines_test.npy", "event_table.csv"]
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"{d} 缺 {missing}; 先跑 realfault 变体预处理")
    out: Dict[str, object] = {"dir": d}
    for split in ("val", "test"):
        out[f"X_{split}"] = np.load(d / f"{split}.npy").astype(np.float32)
        out[f"y_{split}"] = np.load(d / f"{split}_labels.npy").astype(int)
        out[f"ts_{split}"] = np.load(d / f"timestamps_{split}.npy")
        out[f"turb_{split}"] = np.load(d / f"turbines_{split}.npy")
        out[f"ep_{split}"] = load_episodes_from_event_table(d / "event_table.csv", split)
    out["X_train"] = np.load(d / "train.npy").astype(np.float32)
    out["X_sup"] = np.load(d / "train_sup.npy").astype(np.float32)
    out["y_sup"] = np.load(d / "train_sup_labels.npy").astype(int)
    out["ts_sup"] = np.load(d / "timestamps_train_sup.npy")
    out["turb_sup"] = np.load(d / "turbines_train_sup.npy")
    out["turb_train"] = np.load(d / "turbines_train.npy")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    out["lead_steps"] = int(meta["label_rule"]["lead_steps"])
    return out


# ------------------------------------------------------------
# 模型库: name → (family, fit_score) ; fit_score(data, feats) → (score_val, score_test)
# ------------------------------------------------------------
def _sup_fit(clf, F_sup, y_sup, F_val, F_test):
    keep = np.where(y_sup != -1)[0][::TRAIN_STRIDE]
    clf.fit(F_sup[keep], y_sup[keep])
    return (clf.predict_proba(F_val)[:, 1], clf.predict_proba(F_test)[:, 1])


def build_models() -> Dict[str, Tuple[str, Callable]]:
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    rng = np.random.default_rng(SEED)

    def m_resid_energy(data, feats):
        # 00 残差能量 (零训练物理规则): pose 列 = mean(max(0,resid)^2)
        return feats["F_val"][:, feats["D"] + 1], feats["F_test"][:, feats["D"] + 1]

    def m_hgb(data, feats):
        clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=300,
                                             class_weight="balanced")
        return _sup_fit(clf, feats["F_sup"], data["y_sup"], feats["F_val"], feats["F_test"])

    def m_lgbm(data, feats):
        clf = lgb.LGBMClassifier(random_state=SEED, n_estimators=400, class_weight="balanced",
                                 verbosity=-1)
        return _sup_fit(clf, feats["F_sup"], data["y_sup"], feats["F_val"], feats["F_test"])

    def m_logreg(data, feats):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        return _sup_fit(clf, feats["F_sup"], data["y_sup"], feats["F_val"], feats["F_test"])

    def _unsup_train(data):
        Xt = data["X_train"]
        idx = rng.choice(len(Xt), size=min(UNSUP_MAX_N, len(Xt)), replace=False)
        return Xt[idx]

    def m_iforest(data, feats):
        mdl = IsolationForest(random_state=SEED, n_estimators=200, n_jobs=-1)
        mdl.fit(_unsup_train(data))
        return (-mdl.score_samples(data["X_val"]), -mdl.score_samples(data["X_test"]))

    def m_pca(data, feats):
        Xt = _unsup_train(data)
        p = PCA(n_components=0.95, random_state=SEED).fit(Xt)
        def rec_err(X):
            Z = p.transform(X)
            return ((X - p.inverse_transform(Z)) ** 2).mean(axis=1)
        return rec_err(data["X_val"]), rec_err(data["X_test"])

    def m_mahalanobis(data, feats):
        cov = EmpiricalCovariance().fit(_unsup_train(data))
        return cov.mahalanobis(data["X_val"]), cov.mahalanobis(data["X_test"])

    def m_gru_selfsup(data, feats):
        # 41 自监督 GRU 一步预测误差 (无标签; train 去污健康段学正常动态)
        import torch
        import torch.nn as nn
        torch.manual_seed(SEED)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        D = data["X_train"].shape[1]
        W = 36
        class GRU1(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(D, 64, batch_first=True)
                self.head = nn.Linear(64, D)
            def forward(self, x):
                h, _ = self.gru(x)
                return self.head(h[:, -1])
        mdl = GRU1().to(dev)
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        Xt = data["X_train"]
        n_win = len(Xt) - W
        idx = rng.choice(n_win, size=min(60_000, n_win), replace=False)
        bs = 512
        mdl.train()
        for ep in range(2):
            for b in range(0, len(idx), bs):
                bi = idx[b:b + bs]
                xb = torch.tensor(np.stack([Xt[i:i + W] for i in bi])).to(dev)
                yb = torch.tensor(Xt[bi + W]).to(dev)
                opt.zero_grad()
                loss = ((mdl(xb) - yb) ** 2).mean()
                loss.backward(); opt.step()
        mdl.eval()
        def score(X):
            out = np.zeros(len(X))
            with torch.no_grad():
                for b in range(W, len(X), 4096):
                    ee = min(b + 4096, len(X))
                    xb = torch.tensor(np.stack([X[i - W:i] for i in range(b, ee)])).to(dev)
                    pr = mdl(xb).cpu().numpy()
                    out[b:ee] = ((pr - X[b:ee]) ** 2).mean(axis=1)
            out[:W] = np.nan
            return out
        return score(data["X_val"]), score(data["X_test"])

    # ---- 论文复现代表方法 (轻量版, 统一事件级口径; 与 论文复现/ 同方向) ----
    import torch
    import torch.nn as nn

    def _torch_setup():
        torch.manual_seed(SEED)
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _row_ae_scores(data, make_model, loss_kind="mse"):
        """行级自编码器 (AE/VAE): train-healthy 学重构, 分数=重构误差。"""
        dev = _torch_setup()
        Xt = _unsup_train(data).astype(np.float32)
        D = Xt.shape[1]
        mdl = make_model(D).to(dev)
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        bs = 1024
        mdl.train()
        for ep in range(3):
            perm = rng.permutation(len(Xt))
            for b in range(0, len(Xt), bs):
                xb = torch.tensor(Xt[perm[b:b + bs]]).to(dev)
                opt.zero_grad()
                out = mdl(xb)
                if loss_kind == "vae":
                    rec, mu, logvar = out
                    loss = ((rec - xb) ** 2).mean() \
                        + 1e-3 * (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean())
                else:
                    loss = ((out - xb) ** 2).mean()
                loss.backward(); opt.step()
        mdl.eval()
        def score(X):
            out = np.zeros(len(X))
            with torch.no_grad():
                for b in range(0, len(X), 8192):
                    xb = torch.tensor(X[b:b + 8192].astype(np.float32)).to(dev)
                    o = mdl(xb)
                    rec = o[0] if loss_kind == "vae" else o
                    out[b:b + 8192] = ((rec - xb) ** 2).mean(dim=1).cpu().numpy()
            return out
        return score(data["X_val"]), score(data["X_test"])

    def m_paper_ae(data, feats):
        # Wilms et al. WES 2025 (AE-NBM) 方向: 自编码器重构误差
        def make(D):
            return nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(),
                                 nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, D))
        return _row_ae_scores(data, make, "mse")

    def m_paper_vae(data, feats):
        # VAE health-index 论文方向: 变分自编码器重构误差
        class VAE(nn.Module):
            def __init__(self, D):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(D, 64), nn.ReLU())
                self.mu = nn.Linear(64, 16); self.lv = nn.Linear(64, 16)
                self.dec = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, D))
            def forward(self, x):
                h = self.enc(x); mu, lv = self.mu(h), self.lv(h)
                z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
                return self.dec(z), mu, lv
        return _row_ae_scores(data, lambda D: VAE(D), "vae")

    W_SEQ = 36

    def _seq_windows(X, idx):
        return np.stack([X[i - W_SEQ:i] for i in idx]).astype(np.float32)

    def _seq_sup_scores(data, make_model):
        """窗口级监督 NN (1D-CNN / Transformer): train_sup 窗口训练, 分数=P(正)。"""
        dev = _torch_setup()
        Xs, ys = data["X_sup"].astype(np.float32), data["y_sup"]
        D = Xs.shape[1]
        idx = np.where(ys != -1)[0]
        idx = idx[idx >= W_SEQ][::SEQ_SUP_STRIDE]
        pos = idx[ys[idx] == 1]
        idx = np.sort(np.concatenate([idx, np.repeat(pos, 3)]))  # 稀缺正例过采样
        mdl = make_model(D).to(dev)
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        n_pos = max(int((ys[idx] == 1).sum()), 1)
        pw = torch.tensor([max(1.0, (len(idx) - n_pos) / n_pos / 3)], device=dev)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        bs = 256
        mdl.train()
        for ep in range(2):
            perm = rng.permutation(len(idx))
            for b in range(0, len(idx), bs):
                bi = idx[perm[b:b + bs]]
                xb = torch.tensor(_seq_windows(Xs, bi)).to(dev)
                yb = torch.tensor((ys[bi] == 1).astype(np.float32)).to(dev)
                opt.zero_grad()
                loss = lossf(mdl(xb).squeeze(-1), yb)
                loss.backward(); opt.step()
        mdl.eval()
        def score(X):
            out = np.full(len(X), np.nan)
            with torch.no_grad():
                for b in range(W_SEQ, len(X), 2048):
                    ee = min(b + 2048, len(X))
                    xb = torch.tensor(_seq_windows(X, np.arange(b, ee))).to(dev)
                    out[b:ee] = torch.sigmoid(mdl(xb).squeeze(-1)).cpu().numpy()
            return out
        return score(data["X_val"].astype(np.float32)), score(data["X_test"].astype(np.float32))

    def m_paper_cnn(data, feats):
        # Early prediction 1D-CNN + temporal features 论文方向
        def make(D):
            return nn.Sequential(
                _Transpose(), nn.Conv1d(D, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                nn.Flatten(), nn.Linear(32, 1))
        return _seq_sup_scores(data, make)

    def m_paper_transformer(data, feats):
        # A novel transformer network / SLFormer 论文方向 (轻量 encoder)
        class TEnc(nn.Module):
            def __init__(self, D):
                super().__init__()
                self.proj = nn.Linear(D, 64)
                layer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True)
                self.enc = nn.TransformerEncoder(layer, 2)
                self.head = nn.Linear(64, 1)
            def forward(self, x):
                return self.head(self.enc(self.proj(x))[:, -1])
        return _seq_sup_scores(data, lambda D: TEnc(D))

    def m_paper_lstm_ae(data, feats):
        # Feature-alignment LSTM (主轴承温度) 方向: LSTM 自编码器重构误差 (无标签)
        dev = _torch_setup()
        Xt = data["X_train"].astype(np.float32)
        D = Xt.shape[1]
        class LstmAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.LSTM(D, 48, batch_first=True)
                self.dec = nn.LSTM(48, 48, batch_first=True)
                self.out = nn.Linear(48, D)
            def forward(self, x):
                _, (h, _) = self.enc(x)
                z = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
                o, _ = self.dec(z)
                return self.out(o)
        mdl = LstmAE().to(dev)
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        n_win = len(Xt) - W_SEQ
        idx = np.sort(rng.choice(np.arange(W_SEQ, len(Xt)), size=min(40_000, n_win), replace=False))
        bs = 256
        mdl.train()
        for ep in range(2):
            perm = rng.permutation(len(idx))
            for b in range(0, len(idx), bs):
                bi = idx[perm[b:b + bs]]
                xb = torch.tensor(_seq_windows(Xt, bi)).to(dev)
                opt.zero_grad()
                loss = ((mdl(xb) - xb) ** 2).mean()
                loss.backward(); opt.step()
        mdl.eval()
        def score(X):
            X = X.astype(np.float32)
            out = np.full(len(X), np.nan)
            with torch.no_grad():
                for b in range(W_SEQ, len(X), 2048):
                    ee = min(b + 2048, len(X))
                    xb = torch.tensor(_seq_windows(X, np.arange(b, ee))).to(dev)
                    rec = mdl(xb)
                    out[b:ee] = ((rec[:, -1] - xb[:, -1]) ** 2).mean(dim=1).cpu().numpy()
            return out
        return score(data["X_val"]), score(data["X_test"])

    return {
        "00_残差能量_零训练": ("物理规则", m_resid_energy),
        "11_直方图梯度提升_监督": ("监督ML", m_hgb),
        "13_LightGBM_监督": ("监督ML", m_lgbm),
        "01_逻辑回归_监督": ("监督ML", m_logreg),
        "18_孤立森林_无监督": ("无监督ML", m_iforest),
        "36_PCA重构误差_无监督": ("无监督ML", m_pca),
        "40_马氏距离_无监督": ("无监督ML", m_mahalanobis),
        "41_GRU一步预测误差_自监督": ("自监督NN", m_gru_selfsup),
        "60_AE重构_论文AE-NBM": ("论文复现", m_paper_ae),
        "61_VAE重构_论文VAE-HI": ("论文复现", m_paper_vae),
        "62_LSTM自编码_论文FA-LSTM": ("论文复现", m_paper_lstm_ae),
        "63_一维CNN_论文1DCNN": ("论文复现", m_paper_cnn),
        "64_Transformer_论文TransAD": ("论文复现", m_paper_transformer),
    }


# ------------------------------------------------------------
# 评测编排
# ------------------------------------------------------------
def evaluate_scores(name: str, s_val: np.ndarray, s_test: np.ndarray,
                    data: Dict[str, object], out_dir: Path, family: str,
                    extra: Dict[str, object] | None = None) -> Dict[str, object]:
    """val 选阈值(最大 event_f1) → test 评一次; 落盘 metrics.json + 返回摘要。"""
    from sklearn.metrics import roc_auc_score
    lead = int(data["lead_steps"])
    thr, val_f1 = select_threshold_event(
        s_val, data["y_val"], data["ts_val"], data["turb_val"], data["ep_val"], lead_steps=lead)
    pred_test = np.zeros(len(s_test), dtype=int)
    fin = np.isfinite(np.asarray(s_test, dtype=float))
    pred_test[fin & (np.asarray(s_test, dtype=float) >= thr)] = 1
    m = event_level_metrics(pred_test, data["y_test"], data["ts_test"], data["turb_test"],
                            data["ep_test"], lead_steps=lead)
    yv = data["y_test"]
    ok = fin & (yv != -1)
    auc = float(roc_auc_score(yv[ok], np.asarray(s_test, dtype=float)[ok])) \
        if len(np.unique(yv[ok])) > 1 else float("nan")
    # ================= 全指标套件 (2026-07-17 应用户要求扩展) =================
    # 不局限于事件级任务口径, 把异常检测文献常用指标全部算出, 便于横向理解;
    # 主指标仍为事件级 F1 (event_f1), 其余为参考/附录口径。全部在 test 有效行
    # (label≠-1 且分数有限) 上、用同一个 val 选出的阈值计算 → 无额外调参、无泄漏。
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score)
    from 事件指标 import compute_affiliation_metrics, compute_range_metrics, point_adjust_f1
    y_ok = yv[ok].astype(int)
    p_ok = pred_test[ok]
    s_ok = np.asarray(s_test, dtype=float)[ok]
    extra_metrics = {
        # --- 点级阈值指标 (逐 10min 行; 受 0.01-0.5% 基率压制, 解读须对照基率) ---
        "point_f1": float(f1_score(y_ok, p_ok, zero_division=0)),
        "point_precision": float(precision_score(y_ok, p_ok, zero_division=0)),
        "point_recall": float(recall_score(y_ok, p_ok, zero_division=0)),
        "accuracy": float(accuracy_score(y_ok, p_ok)),          # 高基率负类下必然≈1, 仅供对照文献
        "balanced_accuracy": float(balanced_accuracy_score(y_ok, p_ok)),
        "mcc": float(matthews_corrcoef(y_ok, p_ok)) if p_ok.any() else 0.0,
        # --- 阈值无关判别力 ---
        "auprc": float(average_precision_score(y_ok, s_ok)) if y_ok.sum() else float("nan"),
        # --- 事件族指标 (对完整时间轴算, -1 行按事件指标.py 约定处理) ---
        "affiliation_f1": float(compute_affiliation_metrics(
            np.where(yv == 1, 1, 0), pred_test)["affiliation_f1"]),
        "range_f1": float(compute_range_metrics(
            np.where(yv == 1, 1, 0), pred_test)["range_f1"]),
        # --- point-adjust (附录专用! 事件内命中1点→全段算对, 文献高分主要来源,
        #     本项目红线: 不进主表、不参与任何选型, 仅为示证口径膨胀而输出) ---
        "point_adjust_f1_附录": float(point_adjust_f1(yv, pred_test)),
    }
    # =======================================================================
    rec = {"model": name, "family": family, "threshold": float(thr),
           "val_event_f1": float(val_f1), "point_auc_test": auc, **{k: v for k, v in m.items()
           if k != "lead_minutes_all"}, **extra_metrics, "lead_minutes_all": m["lead_minutes_all"]}
    if extra:
        rec.update(extra)
    mdir = out_dir / name
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "metrics.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    np.save(mdir / "score_val.npy", np.asarray(s_val, dtype=np.float32))
    np.save(mdir / "score_test.npy", np.asarray(s_test, dtype=np.float32))
    print(f"  [{name}] test: eF1={rec['event_f1']:.3f} 检出={rec['n_detected']}/{rec['n_events']} "
          f"lead中位={rec['lead_minutes_median']:.0f}min FAR={rec['far_per_day']:.3f}/天 "
          f"AUC={auc:.3f}", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", default="penmanshiel",
                    choices=["kelmarsh", "penmanshiel", "hill_of_towie"])
    ap.add_argument("--models", default="all", help="逗号分隔模型名或 all")
    args = ap.parse_args()

    t0 = time.time()
    data = load_variant(args.farm)
    out_dir = HERE / "快速实验结果_真实故障" / args.farm
    out_dir.mkdir(parents=True, exist_ok=True)
    D = data["X_val"].shape[1]
    print(f"[{args.farm}] val={data['X_val'].shape} test={data['X_test'].shape} "
          f"事件: val={len(data['ep_val'])} test={len(data['ep_test'])} "
          f"(tier1: {int((data['ep_val']['tier']=='tier1').sum())}/"
          f"{int((data['ep_test']['tier']=='tier1').sum())}) lead={data['lead_steps']}步", flush=True)

    feats = {
        "D": D,
        "F_sup": per_turbine_causal_features(data["X_sup"], data["turb_sup"]),
        "F_val": per_turbine_causal_features(data["X_val"], data["turb_val"]),
        "F_test": per_turbine_causal_features(data["X_test"], data["turb_test"]),
    }

    models = build_models()
    wanted = list(models) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    records = []
    scores_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in wanted:
        family, fn = models[name]
        t1 = time.time()
        try:
            s_val, s_test = fn(data, feats)
        except Exception as e:  # 单模型失败不拖垮矩阵, 如实记录
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            records.append({"model": name, "family": family, "error": str(e)})
            continue
        scores_cache[name] = (np.asarray(s_val, dtype=float), np.asarray(s_test, dtype=float))
        rec = evaluate_scores(name, s_val, s_test, data, out_dir, family,
                              extra={"fit_seconds": round(time.time() - t1, 1)})
        records.append(rec)

    # ---- 组合模型: 无监督秩融合 + EWMA (组合战役冠军配方, 迁移到事件级口径)
    unsup = [n for n in ("18_孤立森林_无监督", "36_PCA重构误差_无监督", "40_马氏距离_无监督",
                         "00_残差能量_零训练") if n in scores_cache]
    if len(unsup) >= 2:
        fus_val = np.mean([rank_normalize(scores_cache[n][0]) for n in unsup], axis=0)
        fus_test = np.mean([rank_normalize(scores_cache[n][1]) for n in unsup], axis=0)
        rec = evaluate_scores("50_无监督秩融合", fus_val, fus_test, data, out_dir, "组合",
                              extra={"members": unsup})
        records.append(rec)
        ew_val = per_turbine_ewma(fus_val, data["turb_val"], alpha=0.1)
        ew_test = per_turbine_ewma(fus_test, data["turb_test"], alpha=0.1)
        rec = evaluate_scores("51_秩融合+EWMA", ew_val, ew_test, data, out_dir, "组合",
                              extra={"members": unsup, "ewma_alpha": 0.1})
        records.append(rec)

    # 汇总从磁盘全量合并 (支持增量补跑模型而不丢历史记录)
    disk_recs = []
    for mj in sorted(out_dir.glob("*/metrics.json")):
        try:
            disk_recs.append(json.loads(mj.read_text(encoding="utf-8")))
        except Exception:
            pass
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "lead_minutes_all"}
                       for r in disk_recs])
    if not df.empty:
        df = df.sort_values("event_f1", ascending=False)
        cols = ["model", "family", "event_f1", "event_recall", "n_detected", "n_events",
                "lead_minutes_median", "far_per_day", "alarm_precision", "point_auc_test",
                "val_event_f1", "threshold", "fit_seconds"]
        df[[c for c in cols if c in df.columns]].to_csv(out_dir / "汇总.csv",
                                                        index=False, encoding="utf-8-sig")
        print("\n==== 排名 (test event_f1, val 选阈值) ====")
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f"\n总耗时 {time.time() - t0:.0f}s → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
