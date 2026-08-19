"""Mamba + Meta + NBM — 齿轮油路温度异常快速实验

目的：**快速判定事件级 F1 是否有达到 90 的可能**，而不是把某个数字做大。

做法：同一模型、同一份预测分数，在四种评测口径下各算一遍指标。
四种口径覆盖了文献里实际在用的全部计分方式，因此能直接定位"90 分从哪来"：

  A 逐点原始      —— 无任何修饰，最保守
  B 逐点 point-adjust —— 事件内命中一次即整段判对（多数深度异常检测论文在用）
  C 事件级一对一   —— 每个事件最多贡献一个 TP，禁用 point-adjust
  D 病例-对照配平  —— 事件窗 vs 等量健康窗，正例率人为拉到约 50%

数据：本项目预处理流程 `SCADA数据集/数据预处理/<farm>/`（87 通道 NBM 残差）。
标签：WL 口径 —— 实测 98.5% 为齿轮油路事件（缺油/油压低/油泵过载），真高温仅 1.5%。

模型：NBM 残差 → Mamba 预测下一步残差 → 预测误差作异常分；
      Meta 以 (farm, turbine) 为任务做 Reptile 元训练。
      Mamba/CfC 为纯 PyTorch 手写实现（Windows 无法编译 mamba-ssm），
      已修两个已知坑：dt 初始化跨度、门控跳连（禁恒等直通）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

PRE = Path(r"E:\创新\SCADA数据集\数据预处理")
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS_PER_DAY = 144


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load(farm: str, split: str):
    d = PRE / farm
    X = np.load(d / f"{split}.npy", mmap_mode="r")
    y = np.asarray(np.load(d / f"{split}_labels.npy", mmap_mode="r"))
    ts = np.asarray(np.load(d / f"timestamps_{split}.npy", mmap_mode="r"))
    tb = np.asarray(np.load(d / f"turbines_{split}.npy", mmap_mode="r")).astype(str)
    order = np.lexsort((ts, tb))            # 机组主序，序贯运算前提
    return np.asarray(X)[order], y[order], ts[order], tb[order]


def segments(tb):
    b = np.flatnonzero(tb[1:] != tb[:-1]) + 1
    return list(zip(np.r_[0, b], np.r_[b, len(tb)]))


def events_from(y, seg):
    """连续正例段 = 一个事件的前导窗；返回 (start, end) 索引对。"""
    ev = []
    for s, e in seg:
        pos = np.flatnonzero(y[s:e] == 1)
        if pos.size == 0:
            continue
        brk = np.flatnonzero(np.diff(pos) > 1)
        st = np.r_[pos[0], pos[brk + 1]]
        en = np.r_[pos[brk], pos[-1]]
        for a, b in zip(st, en):
            ev.append((s + a, s + b))
    return ev


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class MambaBlock(nn.Module):
    """最小选择性 SSM。两个已知坑都已修：
       1) dt 初始化落在采样尺度内（朴素 N(0,1) 差两个量级）
       2) 门控跳连而非恒等直通（恒等路径会让 SSM 被完全旁路）"""

    def __init__(self, d_in, d=64, state=16):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.x_proj = nn.Linear(d, state * 2 + 1)
        self.dt_proj = nn.Linear(1, d)
        dt = torch.exp(torch.rand(d) * (np.log(0.1) - np.log(0.001)) + np.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(torch.log(torch.expm1(dt)))
        self.A_log = nn.Parameter(torch.log(
            torch.arange(1, state + 1, dtype=torch.float32).repeat(d, 1)))
        self.D = nn.Parameter(torch.ones(d))
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d_in)

    def forward(self, x):
        b, t, _ = x.shape
        u = self.inp(x)
        A = -torch.exp(self.A_log)
        proj = self.x_proj(u)
        st = self.A_log.shape[1]
        B, C, dtr = proj[..., :st], proj[..., st:2 * st], proj[..., -1:]
        dt = F.softplus(self.dt_proj(dtr))
        h = torch.zeros(b, u.shape[-1], st, device=x.device)
        outs = []
        for i in range(t):
            dA = torch.exp(dt[:, i].unsqueeze(-1) * A.unsqueeze(0))
            dBu = dt[:, i].unsqueeze(-1) * B[:, i].unsqueeze(1) * u[:, i].unsqueeze(-1)
            h = dA * h + dBu
            outs.append((h * C[:, i].unsqueeze(1)).sum(-1) + self.D * u[:, i])
        y = torch.stack(outs, 1) * torch.sigmoid(self.gate(u))
        return self.out(y[:, -1])           # 只取末步 → 预测下一步残差


def make_windows(X, y, seg, W, stride, healthy_only):
    xs, ys, idx = [], [], []
    for s, e in seg:
        n = e - s
        if n <= W:
            continue
        for i in range(s + W, e, stride):
            if healthy_only and y[i] != 0:
                continue
            if y[i] == -1:
                continue
            xs.append((i - W, i))
            ys.append(int(y[i] == 1))
            idx.append(i)
    return np.array(xs), np.array(ys), np.array(idx)


def batch(X, spans, bs, dev):
    for i in range(0, len(spans), bs):
        blk = spans[i:i + bs]
        arr = np.stack([X[a:b] for a, b in blk])
        yield torch.from_numpy(arr).float().to(dev), i, len(blk)


def train_model(model, X, spans, epochs, bs, lr, dev, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.SmoothL1Loss()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(spans))
        tot = 0.0
        for xb, i0, nb in batch(X, spans[perm], bs, dev):
            tgt = torch.from_numpy(
                np.stack([X[b] for a, b in spans[perm][i0:i0 + nb]])).float().to(dev)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb[:, :-1]), tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * nb
        print(f"      {tag} ep{ep+1} loss={tot/max(len(spans),1):.5f}")
    return model


def reptile(model, X, y, seg, tb, W, stride, rounds, inner, bs, lr, meta_lr, dev):
    """Reptile：以 (farm, turbine) 为任务，任务内多步 SGD 后向初始点回拉。"""
    tasks = []
    for s, e in seg:
        sp, _, _ = make_windows(X, y, [(s, e)], W, stride, healthy_only=True)
        if len(sp) > 64:
            tasks.append(sp)
    if not tasks:
        return model
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for r in range(rounds):
        sp = tasks[np.random.randint(len(tasks))]
        model.load_state_dict(base)
        sub = sp[np.random.permutation(len(sp))[:inner * bs]]
        train_model(model, X, sub, 1, bs, lr, dev, tag=f"meta r{r+1}")
        new = model.state_dict()
        base = {k: base[k] + meta_lr * (new[k].float() - base[k]) for k in base}
    model.load_state_dict(base)
    return model


@torch.no_grad()
def score_all(model, X, spans, bs, dev):
    model.eval()
    out = []
    for xb, i0, nb in batch(X, spans, bs, dev):
        tgt = torch.from_numpy(
            np.stack([X[b] for a, b in spans[i0:i0 + nb]])).float().to(dev)
        pred = model(xb[:, :-1])
        out.append(((pred - tgt) ** 2).mean(-1).sqrt().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


# --------------------------------------------------------------------------- #
# 四种评测口径
# --------------------------------------------------------------------------- #
def best_f1_threshold(score, lab):
    qs = np.quantile(score, np.linspace(0.50, 0.9995, 60))
    best = (0.0, qs[0])
    for t in qs:
        f = f1_score(lab, (score > t).astype(int), zero_division=0)
        if f > best[0]:
            best = (f, t)
    return best


def point_adjust(pred, lab):
    """事件内命中一次 → 整段判为命中（多数深度异常检测论文的计分方式）。"""
    pa = pred.copy()
    idx = np.flatnonzero(lab == 1)
    if idx.size:
        brk = np.flatnonzero(np.diff(idx) != 1)
        st = np.r_[idx[0], idx[brk + 1]]
        en = np.r_[idx[brk], idx[-1]]
        for a, b in zip(st, en):
            if pa[a:b + 1].any():
                pa[a:b + 1] = 1
    return pa


def eval_calibers(score, lab, idx_global, y_full, seg, exposure_days):
    res = {}

    # A 逐点原始
    f1a, ta = best_f1_threshold(score, lab)
    pa_ = (score > ta).astype(int)
    res["A_point_raw"] = {
        "F1": f1a, "precision": precision_score(lab, pa_, zero_division=0),
        "recall": recall_score(lab, pa_, zero_division=0),
        "accuracy": float((pa_ == lab).mean()),
        "AUC": roc_auc_score(lab, score) if lab.max() > 0 else float("nan"),
        "AUPRC": average_precision_score(lab, score) if lab.max() > 0 else float("nan"),
        "MCC": matthews_corrcoef(lab, pa_) if len(set(pa_)) > 1 else 0.0,
        "positive_rate": float(lab.mean()),
    }

    # B 逐点 + point-adjust
    best = (0.0, None)
    for t in np.quantile(score, np.linspace(0.50, 0.9995, 60)):
        f = f1_score(lab, point_adjust((score > t).astype(int), lab), zero_division=0)
        if f > best[0]:
            best = (f, t)
    pb = point_adjust((score > best[1]).astype(int), lab)
    res["B_point_adjust"] = {
        "F1": best[0], "precision": precision_score(lab, pb, zero_division=0),
        "recall": recall_score(lab, pb, zero_division=0),
        "accuracy": float((pb == lab).mean()),
        "MCC": matthews_corrcoef(lab, pb) if len(set(pb)) > 1 else 0.0,
    }

    # C 事件级一对一
    ev = events_from(y_full, seg)
    ev = [(a, b) for a, b in ev if b >= idx_global.min() and a <= idx_global.max()]
    best_c = {"F1": 0.0}
    for t in np.quantile(score, np.linspace(0.50, 0.9995, 60)):
        fired = idx_global[score > t]
        if fired.size == 0:
            continue
        keep = [fired[0]]
        for i in fired[1:]:
            if i - keep[-1] >= STEPS_PER_DAY:
                keep.append(i)
        keep = np.array(keep)
        used = np.zeros(len(keep), bool)
        hit = 0
        for a, b in ev:
            m = (~used) & (keep >= a) & (keep <= b)
            if m.any():
                used[np.flatnonzero(m)[0]] = True
                hit += 1
        fp = int((~used).sum())
        prec = hit / max(hit + fp, 1)
        rec = hit / max(len(ev), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_c["F1"]:
            best_c = {"F1": f1, "precision": prec, "recall": rec,
                      "n_events": len(ev), "n_hit": hit, "n_false": fp,
                      "FAR_per_turbine_day": fp / max(exposure_days, 1e-9)}
    res["C_event_one_to_one"] = best_c

    # D 病例-对照配平（正例率拉到约 50%）
    pos = np.flatnonzero(lab == 1)
    neg = np.flatnonzero(lab == 0)
    if len(pos) >= 5 and len(neg) > len(pos):
        rng = np.random.default_rng(0)
        sel = np.r_[pos, rng.choice(neg, len(pos), replace=False)]
        f1d, td = best_f1_threshold(score[sel], lab[sel])
        pd_ = (score[sel] > td).astype(int)
        res["D_case_control"] = {
            "F1": f1d, "precision": precision_score(lab[sel], pd_, zero_division=0),
            "recall": recall_score(lab[sel], pd_, zero_division=0),
            "accuracy": float((pd_ == lab[sel]).mean()),
            "AUC": roc_auc_score(lab[sel], score[sel]),
            "positive_rate": float(lab[sel].mean()),
        }
    return res


# --------------------------------------------------------------------------- #
def run(farm, args):
    t0 = time.time()
    print(f"\n### {farm}")
    Xtr, ytr, _, tbtr = load(farm, "train")
    Xva, yva, _, tbva = load(farm, "val")
    Xte, yte, _, tbte = load(farm, "test")
    seg_tr, seg_va, seg_te = segments(tbtr), segments(tbva), segments(tbte)
    print(f"    train {Xtr.shape} val {Xva.shape} test {Xte.shape}  "
          f"WL 正例 val={int((yva==1).sum())} test={int((yte==1).sum())}")
    print(f"    事件数 val={len(events_from(yva, seg_va))} test={len(events_from(yte, seg_te))}")

    sp_tr, _, _ = make_windows(Xtr, ytr, seg_tr, args.W, args.stride, healthy_only=True)
    if args.max_train and len(sp_tr) > args.max_train:
        sp_tr = sp_tr[np.random.permutation(len(sp_tr))[:args.max_train]]
    sp_te, lab_te, idx_te = make_windows(Xte, yte, seg_te, args.W, 1, healthy_only=False)
    print(f"    训练窗 {len(sp_tr)}  测试窗 {len(sp_te)}（正例 {int(lab_te.sum())}）")

    exposure_days = len(yte) / STEPS_PER_DAY
    out = {}
    for name in args.models.split(","):
        print(f"\n  -- {name}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        mdl = MambaBlock(Xtr.shape[1], d=args.dim).to(DEV)
        if name == "NBM":
            sc = np.sqrt((Xte[idx_te] ** 2).mean(axis=1))     # 残差能量，零训练
        else:
            if "Meta" in name:
                mdl = reptile(mdl, Xtr, ytr, seg_tr, tbtr, args.W, args.stride,
                              args.meta_rounds, args.meta_inner, args.bs,
                              args.lr, args.meta_lr, DEV)
            mdl = train_model(mdl, Xtr, sp_tr, args.epochs, args.bs, args.lr, DEV, tag=name)
            sc = score_all(mdl, Xte, sp_te, args.bs, DEV)
        res = eval_calibers(sc, lab_te, idx_te, yte, seg_te, exposure_days)
        out[name] = res
        for cal, v in res.items():
            print(f"     {cal:20s} F1={v.get('F1', float('nan')):.4f}" +
                  (f"  P={v['precision']:.4f} R={v['recall']:.4f}" if "precision" in v else "") +
                  (f"  AUC={v['AUC']:.4f}" if "AUC" in v else ""))
        del mdl
        torch.cuda.empty_cache()
    out["_meta"] = {"farm": farm, "runtime_s": round(time.time() - t0, 1),
                    "n_test_events": len(events_from(yte, seg_te)),
                    "n_train_windows": int(len(sp_tr)), "device": DEV,
                    "channels": int(Xtr.shape[1]), "config": vars(args)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", default="penmanshiel,kelmarsh")
    ap.add_argument("--models", default="NBM,Mamba+NBM,Mamba+Meta+NBM")
    ap.add_argument("--W", type=int, default=25)
    ap.add_argument("--stride", type=int, default=13)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--meta_rounds", type=int, default=6)
    ap.add_argument("--meta_inner", type=int, default=4)
    ap.add_argument("--meta_lr", type=float, default=0.4)
    ap.add_argument("--max_train", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--tag", default="quick")
    a = ap.parse_args()
    allr = {}
    for farm in a.farms.split(","):
        try:
            allr[farm] = run(farm, a)
        except Exception as exc:
            allr[farm] = {"error": repr(exc)}
            print(f"  !! {farm}: {exc!r}")
        (OUT / f"{a.tag}_metrics.json").write_text(
            json.dumps(allr, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\nwritten -> {OUT / (a.tag + '_metrics.json')}")


if __name__ == "__main__":
    main()
