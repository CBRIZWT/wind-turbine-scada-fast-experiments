# -*- coding: utf-8 -*-
"""生成逐机组隔离的 metrics-v3 快速实验数据。

默认读取 ``SCADA数据集/数据预处理/<farm>__realfault``，写入新的非空
``快速实验数据/<farm>__real_fault_metrics_v1``，不覆盖A′或旧真实故障结果。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from 数据工具_v3 import (
    per_turbine_causal_features,
    sample_valid_indices,
    sequence_end_indices,
    sort_by_turbine_time,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FARMS = ("kelmarsh", "penmanshiel", "hill_of_towie")
DEFAULT_VARIANT = "real_fault_metrics_v1"
W_FEAT = 72
K_RECENT = 6
W_SEQ = 36
ANCHOR_ROWS = 460_137


def derive_split(
    X: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    turbines: np.ndarray,
    *,
    split: str,
    flat_stride: int,
    seq_stride: int,
    window: int = W_SEQ,
    w_feat: int = W_FEAT,
    k_recent: int = K_RECENT,
) -> Dict[str, np.ndarray]:
    """单split内按机组重排并派生扁平/序列表示及严格对齐侧车。"""
    sx, sy, sts, sturb, order = sort_by_turbine_time(X, labels, timestamps, turbines)
    F = per_turbine_causal_features(sx, sturb, w_feat=w_feat, k_recent=k_recent)
    idx_flat = sample_valid_indices(sturb, sy, stride=flat_stride if split == "train" else 1)
    idx_seq = sequence_end_indices(
        sturb,
        sy,
        window=window,
        stride=seq_stride if split == "train" else 1,
    )
    return {
        "X_base": sx.astype(np.float32, copy=False),
        "X_common_base": F[:, -6:].astype(np.float32, copy=False),
        "labels_base": sy.astype(np.int8, copy=False),
        "timestamps_base": sts.astype(np.int64, copy=False),
        "turbines_base": sturb,
        "source_order": order.astype(np.int64, copy=False),
        "idx_flat": idx_flat.astype(np.int64, copy=False),
        "X_flat": F[idx_flat].astype(np.float32, copy=False),
        "y_flat": sy[idx_flat].astype(np.int8, copy=False),
        "timestamps_flat": sts[idx_flat].astype(np.int64, copy=False),
        "turbines_flat": sturb[idx_flat],
        "idx_seq": idx_seq.astype(np.int64, copy=False),
        "y_seq": sy[idx_seq].astype(np.int8, copy=False),
        "timestamps_seq": sts[idx_seq].astype(np.int64, copy=False),
        "turbines_seq": sturb[idx_seq],
    }


def _source_files(split: str):
    if split == "train":
        return "train_sup.npy", "train_sup_labels.npy", "timestamps_train_sup.npy", "turbines_train_sup.npy"
    return f"{split}.npy", f"{split}_labels.npy", f"timestamps_{split}.npy", f"turbines_{split}.npy"


def _save_split(out: Path, split: str, arrays: Dict[str, np.ndarray]) -> Dict[str, object]:
    names = {
        "X_base": f"X_base_{split}.npy",
        "X_common_base": f"X_common_base_{split}.npy",
        "labels_base": f"labels_base_{split}.npy",
        "timestamps_base": f"timestamps_base_{split}.npy",
        "turbines_base": f"turbines_base_{split}.npy",
        "source_order": f"source_order_{split}.npy",
        "idx_flat": f"idx_flat_{split}.npy",
        "X_flat": f"X_flat_{split}.npy",
        "y_flat": f"y_flat_{split}.npy",
        "timestamps_flat": f"timestamps_flat_{split}.npy",
        "turbines_flat": f"turbines_flat_{split}.npy",
        "idx_seq": f"idx_seq_{split}.npy",
        "y_seq": f"y_seq_{split}.npy",
        "timestamps_seq": f"timestamps_seq_{split}.npy",
        "turbines_seq": f"turbines_seq_{split}.npy",
    }
    for key, filename in names.items():
        np.save(out / filename, arrays[key])
    y_flat = arrays["y_flat"]
    y_seq = arrays["y_seq"]
    return {
        "base_rows": int(len(arrays["X_base"])),
        "flat_rows": int(len(y_flat)),
        "flat_positive": int((y_flat == 1).sum()),
        "flat_positive_rate": float((y_flat == 1).mean()) if len(y_flat) else None,
        "seq_rows": int(len(y_seq)),
        "seq_positive": int((y_seq == 1).sum()),
        "seq_positive_rate": float((y_seq == 1).mean()) if len(y_seq) else None,
        "turbines": int(len(np.unique(arrays["turbines_base"]))),
    }


def build_dataset(
    farm: str,
    *,
    source_variant: str = "realfault",
    output_variant: str = DEFAULT_VARIANT,
    external_unlabeled: bool = False,
    external_protocol: str | None = None,
    dry_run: bool = False,
) -> Path:
    if farm not in FARMS:
        raise ValueError(f"未知farm: {farm}")
    if not output_variant.strip():
        raise ValueError("output_variant 必须非空，防止覆盖既有缓存")
    # 单一真源(2026-07-24): real_fault_wl 主口径产物落 <farm>/ (variant=""); 传空 source-variant 即读它。
    _folder = farm if not str(source_variant).strip() else f"{farm}__{source_variant}"
    src = ROOT / "SCADA数据集" / "数据预处理" / _folder
    out = HERE / "快速实验数据" / f"{farm}__{output_variant}"
    required = [src / name for split in ("train", "val", "test") for name in _source_files(split)]
    required.extend([src / "meta.json", src / "event_table.csv"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少真实故障派生输入:\n" + "\n".join(missing))
    if dry_run:
        print(json.dumps({
            "farm": farm,
            "source": str(src),
            "output": str(out),
            "external_unlabeled": bool(external_unlabeled),
            "external_protocol": external_protocol,
            "required_files": len(required),
            "would_write": False,
        }, ensure_ascii=False, indent=2))
        return out

    out.mkdir(parents=True, exist_ok=True)
    train_rows = int(np.load(src / "train_sup.npy", mmap_mode="r").shape[0])
    flat_stride = max(1, round(train_rows / ANCHOR_ROWS))
    seq_stride = 3 * flat_stride
    meta: Dict[str, object] = {
        "schema_version": "fast-data-v3",
        "farm": farm,
        "source": str(src),
        "source_variant": source_variant,
        "preprocess_variant": output_variant,
        "label_mode": "external_unlabeled" if external_unlabeled else "real_fault_wl",
        "external_unlabeled": bool(external_unlabeled),
        "external_protocol": external_protocol,
        "seed": 0,
        "W_feat": W_FEAT,
        "k_recent": K_RECENT,
        "W_seq": W_SEQ,
        "flat_train_stride": flat_stride,
        "seq_train_stride": seq_stride,
        "turbine_aware": True,
        "sequence_history_excludes_ignore": True,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        xf, yf, tf, uf = _source_files(split)
        X = np.load(src / xf, mmap_mode="r")
        y = np.load(src / yf)
        ts = np.load(src / tf)
        turb = np.load(src / uf)
        arrays = derive_split(
            X,
            y,
            ts,
            turb,
            split=split,
            flat_stride=flat_stride,
            seq_stride=seq_stride,
        )
        meta["splits"][split] = _save_split(out, split, arrays)
        print(f"[{farm}/{split}] {meta['splits'][split]}", flush=True)
        del arrays, X, y, ts, turb

    shutil.copy2(src / "event_table.csv", out / "event_table.csv")
    meta["flat_dim"] = int(np.load(out / "X_flat_train.npy", mmap_mode="r").shape[1])
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成 → {out}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", choices=FARMS, required=True)
    ap.add_argument("--source-variant", default="realfault")
    ap.add_argument("--output-variant", default=DEFAULT_VARIANT)
    ap.add_argument("--external-unlabeled", action="store_true")
    ap.add_argument("--external-protocol", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build_dataset(
        args.farm,
        source_variant=args.source_variant,
        output_variant=args.output_variant,
        external_unlabeled=args.external_unlabeled,
        external_protocol=args.external_protocol,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
