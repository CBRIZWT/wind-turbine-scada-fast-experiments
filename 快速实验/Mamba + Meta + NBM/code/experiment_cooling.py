"""Mamba + Meta + NBM — 发电机冷却系统热异常快速实验

与 `experiment.py` 完全同构，只换标签来源：冷却告警（`overload generator fan`）
从原始 Status 文本重建后映射到预处理时间轴，其余数据、模型、四种评测口径不变。
两个方向因此可以直接对比——齿轮油路已测（NBM 零训练基线最强），本脚本补上冷却方向。

预警窗与忽略区沿用本项目口径：事件起点前 LEAD 步为正例，事件期与事后 24 h 为 -1（忽略）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment import (MambaBlock, eval_calibers, events_from,  # noqa: E402
                        make_windows, reptile, score_all, segments, train_model)

B = Path(r"E:\创新\SCADA数据集")
PRE = B / "数据预处理"
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "outputs"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS_PER_DAY = 144
LEAD = 144            # 24 h 预警窗
POST_IGNORE = 144     # 事后 24 h 忽略

INC = ["overload generator fan", "generator fan"]
EXC = ["manual operation", "manual stop", "warm-up", "heating enabled", "test"]


def cooling_events(farm: str) -> pd.DataFrame:
    folder = {"kelmarsh": "Kelmarsh Wind Farm Data",
              "penmanshiel": "Penmanshiel Wind Farm Data"}[farm]
    stem = "Kelmarsh" if farm == "kelmarsh" else "Penmanshiel"
    rows = []
    for p in sorted((B / folder).rglob(f"Status_{stem}_*.csv")):
        try:
            head = list(pd.read_csv(p, nrows=0, skiprows=9, encoding="latin-1").columns)
        except Exception:
            continue
        ren = {}
        for c in head:
            lc = c.lower()
            if "timestamp start" in lc:
                ren[c] = "t0"
            elif lc == "duration":
                ren[c] = "dur"
            elif lc == "message":
                ren[c] = "msg"
        if "t0" not in ren.values() or "msg" not in ren.values():
            continue
        try:
            df = pd.read_csv(p, skiprows=9, usecols=list(ren), encoding="latin-1",
                             low_memory=False)
        except Exception:
            continue
        df = df.rename(columns=ren)
        m = re.search(r"_(\d+)_\d{4}-", p.name)
        df["turbine"] = str(int(m.group(1))) if m else "?"
        rows.append(df)
    st = pd.concat(rows, ignore_index=True)
    st["t0"] = pd.to_datetime(st["t0"], utc=True, errors="coerce").dt.tz_convert(None)
    st["hours"] = pd.to_timedelta(st.get("dur"), errors="coerce").dt.total_seconds() / 3600
    st = st.dropna(subset=["t0"])
    low = st["msg"].fillna("").str.lower()
    sub = st[low.apply(lambda s: any(k in s for k in INC))
             & ~low.apply(lambda s: any(k in s for k in EXC))]
    out = []
    for t, g in sub.groupby("turbine"):
        g = g.sort_values("t0")
        cs = ce = None
        for _, r in g.iterrows():
            end = r["t0"] + pd.Timedelta(hours=float(r["hours"])
                                         if r["hours"] == r["hours"] else 0)
            if cs is None:
                cs, ce = r["t0"], end
            elif (r["t0"] - ce).total_seconds() <= 72 * 3600:
                ce = max(ce, end)
            else:
                out.append({"turbine": t, "start": cs, "end": ce})
                cs, ce = r["t0"], end
        if cs is not None:
            out.append({"turbine": t, "start": cs, "end": ce})
    return pd.DataFrame(out)


def load_with_cooling(farm: str, split: str, ev: pd.DataFrame):
    d = PRE / farm
    X = np.asarray(np.load(d / f"{split}.npy", mmap_mode="r"))
    ts = np.asarray(np.load(d / f"timestamps_{split}.npy", mmap_mode="r"))
    tb = np.asarray(np.load(d / f"turbines_{split}.npy", mmap_mode="r")).astype(str)
    tb = np.array([str(int(t)) for t in tb])
    order = np.lexsort((ts, tb))
    X, ts, tb = X[order], ts[order], tb[order]
    tsn = pd.to_datetime(ts, utc=True).tz_convert(None).to_numpy()

    y = np.zeros(len(ts), dtype=np.int64)
    seg = segments(tb)
    for s, e in seg:
        t = tb[s]
        loc = tsn[s:e]
        for _, r in ev[ev["turbine"] == t].iterrows():
            t0 = np.datetime64(r["start"])
            t1 = np.datetime64(r["end"])
            i0 = np.searchsorted(loc, t0)
            # 预警窗：起点前 LEAD 步为正例
            y[s + max(0, i0 - LEAD): s + i0] = 1
            # 事件期 + 事后 24 h 为忽略
            i1 = np.searchsorted(loc, t1)
            y[s + i0: s + min(e - s, i1 + POST_IGNORE)] = -1
    return X, y, tb, seg


def run(farm, args):
    t0 = time.time()
    print(f"\n### {farm} · 冷却方向")
    ev = cooling_events(farm)
    print(f"    冷却 episodes: {len(ev)} / {ev['turbine'].nunique()} 台机组")
    Xtr, ytr, tbtr, seg_tr = load_with_cooling(farm, "train", ev)
    Xte, yte, tbte, seg_te = load_with_cooling(farm, "test", ev)
    ev_te = events_from(yte, seg_te)
    print(f"    train {Xtr.shape} test {Xte.shape}")
    print(f"    正例 train={int((ytr==1).sum())} test={int((yte==1).sum())}  "
          f"测试段事件 {len(ev_te)}")
    if len(ev_te) < 5:
        return {"skipped": "测试段事件不足", "n_test_events": len(ev_te)}

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
        if name == "NBM":
            sc = np.sqrt((Xte[idx_te] ** 2).mean(axis=1))
        else:
            mdl = MambaBlock(Xtr.shape[1], d=args.dim).to(DEV)
            if "Meta" in name:
                mdl = reptile(mdl, Xtr, ytr, seg_tr, tbtr, args.W, args.stride,
                              args.meta_rounds, args.meta_inner, args.bs,
                              args.lr, args.meta_lr, DEV)
            mdl = train_model(mdl, Xtr, sp_tr, args.epochs, args.bs, args.lr, DEV, tag=name)
            sc = score_all(mdl, Xte, sp_te, args.bs, DEV)
            del mdl
            torch.cuda.empty_cache()
        res = eval_calibers(sc, lab_te, idx_te, yte, seg_te, exposure_days)
        out[name] = res
        for cal, v in res.items():
            print(f"     {cal:20s} F1={v.get('F1', float('nan')):.4f}" +
                  (f"  P={v['precision']:.4f} R={v['recall']:.4f}" if "precision" in v else "") +
                  (f"  AUC={v['AUC']:.4f}" if "AUC" in v else ""))
    out["_meta"] = {"farm": farm, "direction": "generator_cooling",
                    "n_episodes": int(len(ev)), "n_test_events": len(ev_te),
                    "runtime_s": round(time.time() - t0, 1), "config": vars(args)}
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
    ap.add_argument("--tag", default="cooling")
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
