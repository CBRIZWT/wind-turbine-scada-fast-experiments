# -*- coding: utf-8 -*-
"""真实故障事件级可视化 (2026-07-11)。

图A 正常 vs 异常残差时序对比 (同机组同通道, 健康窗 vs 事件前窗, 同 y 轴)
图B 实测温度 vs NBM 预测温度 + 报警标记 (事件机组, 事件前后; T̂ = T实测 − 反归一残差)
图C 模型事件级对比 (检出/lead/FAR/AUC 条形图, 来自 汇总.csv)

诚实口径: 图B 的 T 实测来自原始 CSV (未经 Hampel 清洗, 与管线输入略有差异, 已在图注说明);
NBM 预测由管线自身残差反推 (T̂ = T − resid), 不重新拟合任何模型。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT / "SCADA数据集")):
    if p not in sys.path:
        sys.path.insert(0, p)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

from 事件级评测 import load_episodes_from_event_table  # noqa: E402


def _find_raw_csv(farm: str, turbine: str, year: int):
    """定位事件机组该年的 Turbine_Data CSV (仅 Greenbyte 两场)。"""
    base = ROOT / "SCADA数据集"
    if farm == "kelmarsh":
        pats = [f"Kelmarsh Wind Farm Data/Kelmarsh_SCADA_{year}_*/Turbine_Data_Kelmarsh_{turbine}_*.csv"]
    elif farm == "penmanshiel":
        pats = [f"Penmanshiel Wind Farm Data/Penmanshiel_SCADA_{year}*/Turbine_Data_Penmanshiel_{turbine}_*.csv"]
    else:
        return None
    for p in pats:
        hits = sorted(base.glob(p))
        if hits:
            return hits[0]
    return None


def pick_model(res_dir: Path) -> str:
    """展示模型 = test 检出数最多者, 并列取 point_auc_test 高者 (图注如实标注选择规则)。"""
    df = pd.read_csv(res_dir / "汇总.csv")
    df = df.sort_values(["n_detected", "point_auc_test"], ascending=False)
    return str(df.iloc[0]["model"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", default="kelmarsh")
    ap.add_argument("--model", default="", help="默认: test 检出最多/AUC 最高模型")
    ap.add_argument("--channel", default="Gear oil temperature", help="展示通道关键词")
    ap.add_argument("--episode", type=int, default=0, help="test episode 序号")
    ap.add_argument("--skip-raw", action="store_true", help="跳过图B的原始CSV加载")
    args = ap.parse_args()

    d = ROOT / "SCADA数据集" / "数据预处理" / f"{args.farm}__realfault"
    res_dir = HERE / "快速实验结果_真实故障" / args.farm
    fig_dir = res_dir / "图"
    fig_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    cols = meta["cols"]
    med = np.asarray(meta["scaler"]["medians"], dtype=float)
    iqr = np.asarray(meta["scaler"]["iqrs"], dtype=float)
    # 展示通道: 名称含关键词的基础残差列 (前18基通道内)
    cand = [i for i, c in enumerate(cols[:18]) if args.channel.lower() in c.lower()]
    ch = cand[0] if cand else 0
    ch_name = cols[ch].split("__")[0].strip()
    raw_col_key = ch_name.split(" (")[0]  # 原始CSV列名关键词

    X = np.load(d / "test.npy").astype(np.float32)
    y = np.load(d / "test_labels.npy").astype(int)
    ts = pd.to_datetime(np.load(d / "timestamps_test.npy"), utc=True)
    turb = np.load(d / "turbines_test.npy")
    eps = load_episodes_from_event_table(d / "event_table.csv", "test")
    if len(eps) == 0:
        print("test 无事件, 无法出事件图"); return 1
    ep = eps.sort_values("Timestamp start").iloc[args.episode]
    et, es, ee = str(ep["_turbine"]), ep["Timestamp start"], ep["Timestamp end"]
    lead = int(meta["label_rule"]["lead_steps"])
    print(f"[{args.farm}] 事件: T{et} {es} → {ee} tier={ep['tier']} | 通道: {ch_name} (col {ch})")

    model = args.model or pick_model(res_dir)
    mrec = json.loads((res_dir / model / "metrics.json").read_text(encoding="utf-8"))
    score = np.load(res_dir / model / "score_test.npy").astype(float)
    thr = float(mrec["threshold"])
    print(f"展示模型: {model} (阈值={thr:.4g}, 检出={mrec['n_detected']}/{mrec['n_events']})")

    tmask = turb.astype(str) == et
    t_ts, t_z = ts[tmask], X[tmask, ch].astype(float)
    t_y = y[tmask]
    t_alarm = (np.nan_to_num(score[tmask], nan=-np.inf) >= thr)

    # ---- 图A: 正常 vs 异常 残差对比 ----
    w_pre, w_post = pd.Timedelta(days=7), pd.Timedelta(days=2)
    m_evt = (t_ts >= es - w_pre) & (t_ts <= ee + w_post)
    healthy_end = es - pd.Timedelta(days=60)
    m_hlt = (t_ts >= healthy_end - (w_pre + w_post)) & (t_ts <= healthy_end)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharey=True)
    axes[0].plot(t_ts[m_hlt], t_z[m_hlt], lw=0.7, color="tab:green")
    axes[0].set_title(f"正常期 (事件前60天同机组同通道) — {args.farm} T{et} · {ch_name} NBM残差(鲁棒z)")
    axes[1].plot(t_ts[m_evt], t_z[m_evt], lw=0.7, color="tab:red")
    pre_m = m_evt & (t_y == 1)
    axes[1].axvspan(es, ee + pd.Timedelta(hours=1), color="red", alpha=0.25, label="事件期(停机)")
    if pre_m.any():
        axes[1].axvspan(t_ts[pre_m].min(), es, color="orange", alpha=0.2,
                        label=f"早警窗 (事件前{lead}步)")
    axes[1].set_title(f"异常期 (事件 {str(es)[:16]} 前7天→后2天)")
    axes[1].legend(loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_ylabel("残差 z")
    fig.tight_layout()
    fig.savefig(fig_dir / "图A_正常vs异常残差对比.png", dpi=140)
    plt.close(fig)

    # ---- 图B: 实测 vs NBM 预测温度 + 报警 ----
    if not args.skip_raw:
        raw_file = _find_raw_csv(args.farm, et, int(str(es)[:4]))
        rc = []
        if raw_file is not None:
            header = pd.read_csv(raw_file, skiprows=9, nrows=0)
            rc = [c for c in header.columns if raw_col_key.lower() in c.lower()
                  and "min" not in c.lower() and "max" not in c.lower() and "std" not in c.lower()]
        if rc:
            # 定向 usecols 读两列 (时间+温度), 避免整表解析 OOM
            tcol = header.columns[0]
            raw = pd.read_csv(raw_file, skiprows=9, usecols=[tcol, rc[0]], index_col=tcol)
            raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")
            raw = raw[raw.index.notna()].sort_index()
            rW0, rW1 = es - w_pre, ee + w_post
            rs = raw.loc[(raw.index >= rW0) & (raw.index <= rW1), rc[0]].astype(float)
            rs = rs.resample("10min").mean()  # 2023-24 原始文件更高频 → 对齐 10min 网格
            seg = pd.DataFrame({"T": rs}).join(
                pd.DataFrame({"z": t_z, "alarm": t_alarm, "y": t_y}, index=t_ts), how="inner")
            seg["resid"] = seg["z"] * iqr[ch] + med[ch]
            seg["T_pred"] = seg["T"] - seg["resid"]
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(seg.index, seg["T"], lw=0.9, color="tab:red", label="实测温度")
            ax.plot(seg.index, seg["T_pred"], lw=0.9, color="tab:blue", alpha=0.8,
                    label="NBM预测温度(工况归一)")
            al = seg[seg["alarm"]]
            if len(al):
                ax.scatter(al.index, al["T"], s=14, color="black", zorder=5,
                           label=f"模型报警 ({model.split('_')[1] if '_' in model else model})")
            ax.axvspan(es, ee + pd.Timedelta(hours=1), color="red", alpha=0.25, label="事件期")
            pre = seg[seg["y"] == 1]
            if len(pre):
                ax.axvspan(pre.index.min(), es, color="orange", alpha=0.15, label="早警窗")
            ax.set_title(f"{args.farm} T{et} · {ch_name}: 实测 vs NBM预测 (偏差过大→报警) — "
                         f"事件 {str(es)[:16]}")
            ax.set_ylabel("温度 (°C)"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
            fig.text(0.01, 0.01, "注: 实测取原始CSV(未Hampel清洗), NBM预测=实测−管线反归一残差; 零重新拟合",
                     fontsize=8, color="gray")
            fig.tight_layout()
            fig.savefig(fig_dir / "图B_实测vs预测温度_报警.png", dpi=140)
            plt.close(fig)
        else:
            print(f"原始CSV无 {raw_col_key} 列, 跳过图B")

    # ---- 图C: 模型对比 ----
    df = pd.read_csv(res_dir / "汇总.csv").sort_values("point_auc_test", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.45 * len(df) + 2.2))
    names = [m[:16] for m in df["model"]]
    axes[0].barh(names, df["point_auc_test"], color="tab:blue")
    axes[0].axvline(0.5, color="gray", ls="--", lw=0.8); axes[0].set_title("点级 AUC (test)")
    axes[1].barh(names, df["event_recall"].fillna(0), color="tab:orange")
    axes[1].set_title(f"事件检出率 (test, N={int(df['n_events'].iloc[0])})")
    axes[2].barh(names, df["far_per_day"], color="tab:red")
    axes[2].set_title("误报段/天 (test, ↓好)")
    for ax in axes:
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(f"{args.farm} 真实故障事件级快速实验 — 模型对比 (阈值仅val选)")
    fig.tight_layout()
    fig.savefig(fig_dir / "图C_模型对比.png", dpi=140)
    plt.close(fig)
    print(f"图已写入 {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
