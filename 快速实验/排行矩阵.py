# -*- coding: utf-8 -*-
"""排行矩阵.py — 产出用户口径的排行表:
   行 = 按性能(核心 AUPRC)降序的模型;  列 = 按"该指标全体模型最好值"降序的评价指标;
   单元格 = 实验结果数值。 每 farm 一份 CSV + 一个合并的 HTML(artifact 正文, 热力着色)。
数据源 = 快速实验结果/全指标汇总.csv (汇总.py 产出)。2026-07-19 事件级F1改口径后重算。
命令行: E:\\ancoda\\chuangxin\\python.exe 排行矩阵.py
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULT = Path(__file__).resolve().parent / "快速实验结果"
FARMS = ("kelmarsh", "penmanshiel", "hill_of_towie")
SORT_KEY = "test_auprc"   # 行(模型)排序核心指标

# 主矩阵纳入的"越大越好"评价指标(可比, 便于按最好值排列);
# 事件级F1(命中即命中)= test_seg_event_f1, 已在本次改口径。
HIGHER_BETTER = [
    "test_auprc", "test_auc", "test_f1", "test_precision", "test_recall",
    "test_accuracy", "test_balanced_accuracy", "test_mcc", "test_r2",
    "test_range_f1", "test_range_precision", "test_range_recall",
    "test_affiliation_f1", "test_affiliation_precision", "test_affiliation_recall",
    "test_seg_event_f1", "test_seg_event_precision", "test_seg_event_recall",
    "test_pa_point_adjust_f1",
]
LABEL = {
    "test_auprc": "AUPRC", "test_auc": "ROC-AUC", "test_f1": "F1(点级)",
    "test_precision": "精确率", "test_recall": "召回率", "test_accuracy": "准确率",
    "test_balanced_accuracy": "平衡准确率", "test_mcc": "MCC", "test_r2": "R²",
    "test_range_f1": "range-F1", "test_range_precision": "range-P", "test_range_recall": "range-R",
    "test_affiliation_f1": "affil-F1", "test_affiliation_precision": "affil-P",
    "test_affiliation_recall": "affil-R",
    "test_seg_event_f1": "事件F1★(命中即命中)", "test_seg_event_precision": "事件精确",
    "test_seg_event_recall": "事件召回",
    "test_pa_point_adjust_f1": "PA-F1(附录/虚高)",
}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def build_farm(df_farm: pd.DataFrame):
    """返回 (models_sorted, metrics_sorted, value_matrix[np.ndarray], best_val_per_metric)."""
    df = df_farm.copy()
    df[SORT_KEY] = _num(df[SORT_KEY])
    df = df.sort_values(SORT_KEY, ascending=False, na_position="last").reset_index(drop=True)
    cols = [c for c in HIGHER_BETTER if c in df.columns]
    for c in cols:
        df[c] = _num(df[c])
    best = {c: float(np.nanmax(df[c].to_numpy())) if df[c].notna().any() else float("nan")
            for c in cols}
    metrics_sorted = sorted(cols, key=lambda c: (-(best[c] if best[c] == best[c] else -1)))
    models = df["model"].tolist()
    mat = df[metrics_sorted].to_numpy(dtype=float)
    return models, metrics_sorted, mat, best, df


def write_csv(farm, models, metrics_sorted, mat):
    hdr = "rank,model," + ",".join(LABEL.get(m, m) for m in metrics_sorted)
    lines = [hdr]
    for i, mdl in enumerate(models, 1):
        row = ",".join("" if not np.isfinite(v) else f"{v:.4f}" for v in mat[i - 1])
        lines.append(f"{i},{mdl},{row}")
    out = RESULT / f"排行矩阵_{farm}.csv"
    out.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"[{farm}] → {out}  ({len(models)}模型 × {len(metrics_sorted)}指标)")


def _cell_color(v, lo, hi):
    """列内 min-max → 绿(好)到红(差) 的柔和背景色(HSL)。"""
    if not np.isfinite(v):
        return "background:transparent;color:var(--muted)"
    t = 0.5 if hi <= lo else (v - lo) / (hi - lo)
    hue = 8 + t * 122          # 8(红) → 130(绿)
    return f"background:hsl({hue:.0f} 62% 62% / .82);color:#111"


def farm_html(farm, models, metrics_sorted, mat, df):
    ncol = len(metrics_sorted)
    los = np.nanmin(np.where(np.isfinite(mat), mat, np.nan), axis=0)
    his = np.nanmax(np.where(np.isfinite(mat), mat, np.nan), axis=0)
    th = "".join(
        f'<th title="全体最好值 {his[j]:.3f}"><span class="mtop">#{j+1}</span>'
        f'<span class="mlab">{html.escape(LABEL.get(m, m))}</span>'
        f'<span class="mbest">max {his[j]:.3f}</span></th>'
        for j, m in enumerate(metrics_sorted))
    rows = []
    champ = models[0] if models else ""
    for i, mdl in enumerate(models, 1):
        cells = []
        for j in range(ncol):
            v = mat[i - 1, j]
            txt = "" if not np.isfinite(v) else f"{v:.3f}"
            cells.append(f'<td style="{_cell_color(v, los[j], his[j])}">{txt}</td>')
        cls = ' class="champ"' if mdl == champ else ""
        name = html.escape(mdl.split("_", 1)[-1] if "_" in mdl else mdl)
        num = html.escape(mdl.split("_", 1)[0]) if "_" in mdl else ""
        rows.append(f'<tr{cls}><td class="rk">{i}</td>'
                    f'<td class="mdl"><b>{num}</b> {name}</td>{"".join(cells)}</tr>')
    top = df.iloc[0]
    sub = (f"冠军(AUPRC)= <b>{html.escape(str(top['model']))}</b> · "
           f"AUPRC {float(top['test_auprc']):.3f} · "
           f"事件F1(命中即命中) {float(top.get('test_seg_event_f1', float('nan'))):.3f} · "
           f"{len(models)} 模型")
    return (f'<section><h2>{html.escape(farm)}</h2><p class="sub">{sub}</p>'
            f'<div class="scroll"><table><thead><tr>'
            f'<th class="rk">#</th><th class="mdl">模型 (行=按 AUPRC 降序)</th>{th}'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>')


def main():
    csv = RESULT / "全指标汇总.csv"
    if not csv.exists():
        print("缺 全指标汇总.csv, 先跑 汇总.py"); return 1
    df = pd.read_csv(csv)
    sections = []
    for farm in FARMS:
        sub = df[df["farm"] == farm]
        if not len(sub):
            continue
        models, metrics_sorted, mat, best, dff = build_farm(sub)
        write_csv(farm, models, metrics_sorted, mat)
        sections.append(farm_html(farm, models, metrics_sorted, mat, dff))
    body = STYLE + HEADER + "".join(sections) + FOOTER
    out = RESULT / "排行矩阵.html"
    out.write_text(body, encoding="utf-8")
    print(f"\nHTML → {out}")
    return 0


STYLE = """<style>
:root{--bg:#fafafa;--fg:#1a1a1a;--muted:#8a8a8a;--line:#e2e2e2;--card:#fff;--accent:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--fg:#e8e8ea;--muted:#7d818b;--line:#2a2d35;--card:#181b21;--accent:#7aa2ff}}
:root[data-theme=dark]{--bg:#111318;--fg:#e8e8ea;--muted:#7d818b;--line:#2a2d35;--card:#181b21;--accent:#7aa2ff}
:root[data-theme=light]{--bg:#fafafa;--fg:#1a1a1a;--muted:#8a8a8a;--line:#e2e2e2;--card:#fff;--accent:#2563eb}
*{box-sizing:border-box}
body,.wrap{color:var(--fg)}
.wrap{max-width:1500px;margin:0 auto;padding:8px 4px 40px;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif}
h1{font-size:22px;margin:.2em 0 .1em}
h2{font-size:17px;margin:1.4em 0 .1em;color:var(--accent)}
.lede{color:var(--muted);margin:.2em 0 1em;font-size:13px}
.sub{color:var(--muted);font-size:12.5px;margin:.1em 0 .5em}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;font-size:11.5px;min-width:100%}
thead th{position:sticky;top:0;background:var(--card);border-bottom:2px solid var(--line);
  padding:5px 6px;text-align:center;vertical-align:bottom;z-index:2}
th.rk,td.rk{width:30px;text-align:center;color:var(--muted)}
th.mdl,td.mdl{text-align:left;white-space:nowrap;padding-left:8px;position:sticky;left:0;background:var(--card);z-index:1}
td.mdl b{color:var(--accent)}
.mtop{display:block;font-size:9px;color:var(--muted)}
.mlab{display:block;font-weight:600;max-width:78px}
.mbest{display:block;font-size:9px;color:var(--muted);font-weight:400}
tbody td{padding:3px 5px;text-align:center;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
tr.champ td.mdl{box-shadow:inset 3px 0 0 var(--accent)}
tr.champ{font-weight:600}
.note{margin-top:26px;padding:12px 14px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--muted);font-size:12px}
.note b{color:var(--fg)}
code{background:color-mix(in srgb,var(--fg) 8%,transparent);padding:1px 4px;border-radius:4px}
</style>"""

HEADER = """<div class="wrap"><h1>风机 SCADA 温度异常「预测」— 快速实验模型 × 指标排行</h1>
<p class="lede">口径:三风场 kelmarsh / penmanshiel / hill_of_towie(训 2016-21 / 验 2022 / 测 2023-24);
预处理数据集(NBM 残差,严格因果);阈值仅 val 选(最大逐点F1),test 只评一次;单 seed。
<b>行</b> = 模型按核心指标 AUPRC 降序;<b>列</b> = 评价指标按「全体模型最好值」降序;单元格 = test 集实验结果(列内绿=好/红=差)。</p>"""

FOOTER = """<div class="note">
<b>事件级 F1 本次已改口径</b> = 「只要命中一段就相当于命中」的<b>事件检出 F1(段级 P & R)</b>:
真实事件段被任一报警覆盖即算检出(段级召回);报警段与任一真实事件相交即算命中(段级精确);F1=两者调和。
数值与存在性口径 <code>affil-F1</code> 同族。<br>
<b>诚实备注</b>:AUPRC 为核心排序键;<code>PA-F1(附录)</code> 为 point-adjust 口径,已知会虚高,仅列作对照不作主结论;
快速实验 train 下采样提速、单 seed,结论用于<b>模型排序</b>,冠军确证数字应回全量 5-seed 重跑。
</div></div>"""


if __name__ == "__main__":
    raise SystemExit(main())
