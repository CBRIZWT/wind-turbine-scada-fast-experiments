# -*- coding: utf-8 -*-
"""构建 Hill of Towie 外部无标签的本场拟合与双源域零样本数据。

本场协议由 ``准备真实故障数据_v3.build_dataset`` 生成；源域协议只在
Kelmarsh/Penmanshiel 的健康训练窗上拟合，并把统一六维因果温度残差特征
部署到 Hill 的验证/测试时间段。所有产物使用非空 variant 隔离。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from 准备真实故障数据_v3 import build_dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "快速实验数据"
TRUE_FARMS = ("kelmarsh", "penmanshiel")
TRUE_VARIANT = "real_fault_metrics_v1"
LOCAL_VARIANT = "real_fault_metrics_v1_external_local"
SOURCE_VARIANT = "real_fault_metrics_v1_external_source"
W_SEQ = 36


def prefix_turbines(farm: str, turbines: np.ndarray) -> np.ndarray:
    """给源风场机组加前缀，避免相同机组编号被错误拼接成同一序列。"""
    return np.asarray([f"{farm}:{x}" for x in np.asarray(turbines).astype(str)])


def healthy_sequence_mask(labels_base: np.ndarray, idx: np.ndarray, *, window: int = W_SEQ) -> np.ndarray:
    """仅保留窗口内每一步标签都为0的严格健康序列。"""
    labels = np.asarray(labels_base).astype(int)
    ends = np.asarray(idx, dtype=np.int64)
    starts = ends - int(window) + 1
    bad = (labels != 0).astype(np.int64)
    cumulative = np.concatenate([[0], np.cumsum(bad)])
    mask = starts >= 0
    counts = np.ones(len(ends), dtype=np.int64)
    valid = np.where(mask)[0]
    counts[valid] = cumulative[ends[valid] + 1] - cumulative[starts[valid]]
    return mask & (counts == 0)


def _load(path: Path, name: str, *, mmap: bool = False) -> np.ndarray:
    return np.load(path / name, mmap_mode="r" if mmap else None)


def _save(out: Path, name: str, value: np.ndarray) -> None:
    np.save(out / name, np.asarray(value))


def _copy_hill_split(hill: Path, out: Path, split: str) -> dict:
    """Hill验证/测试侧使用同一六维特征，标签只作占位且绝不进入评测。"""
    common = np.asarray(_load(hill, f"X_common_base_{split}.npy", mmap=True), dtype=np.float32)
    idx_flat = _load(hill, f"idx_flat_{split}.npy")
    idx_seq = _load(hill, f"idx_seq_{split}.npy")
    y_flat = np.zeros(len(idx_flat), dtype=np.int8)
    y_seq = np.zeros(len(idx_seq), dtype=np.int8)
    _save(out, f"X_base_{split}.npy", common)
    _save(out, f"X_common_base_{split}.npy", common)
    _save(out, f"idx_flat_{split}.npy", idx_flat)
    _save(out, f"X_flat_{split}.npy", common[idx_flat])
    _save(out, f"y_flat_{split}.npy", y_flat)
    _save(out, f"timestamps_flat_{split}.npy", _load(hill, f"timestamps_flat_{split}.npy"))
    _save(out, f"turbines_flat_{split}.npy", _load(hill, f"turbines_flat_{split}.npy"))
    _save(out, f"idx_seq_{split}.npy", idx_seq)
    _save(out, f"y_seq_{split}.npy", y_seq)
    _save(out, f"timestamps_seq_{split}.npy", _load(hill, f"timestamps_seq_{split}.npy"))
    _save(out, f"turbines_seq_{split}.npy", _load(hill, f"turbines_seq_{split}.npy"))
    return {"base_rows": int(len(common)), "flat_rows": int(len(idx_flat)), "seq_rows": int(len(idx_seq))}


def _build_source_train(source_dirs: Iterable[tuple[str, Path]], out: Path) -> dict:
    bases = []
    base_turbines = []
    flat_x = []
    flat_y = []
    flat_idx = []
    flat_ts = []
    flat_turb = []
    seq_idx = []
    seq_y = []
    seq_ts = []
    seq_turb = []
    offset = 0
    farm_rows = {}
    for farm, src in source_dirs:
        common = np.asarray(_load(src, "X_common_base_train.npy", mmap=True), dtype=np.float32)
        labels_base = _load(src, "labels_base_train.npy")
        idx = _load(src, "idx_seq_train.npy")
        seq_healthy = healthy_sequence_mask(labels_base, idx, window=W_SEQ)
        idx = idx[seq_healthy]
        raw_flat_y = _load(src, "y_flat_train.npy").astype(np.int8)
        flat_healthy = raw_flat_y == 0
        bases.append(common)
        base_turbines.append(prefix_turbines(farm, _load(src, "turbines_base_train.npy")))
        flat_x.append(np.asarray(_load(src, "X_flat_train.npy", mmap=True)[flat_healthy, -6:], dtype=np.float32))
        flat_y.append(np.zeros(int(flat_healthy.sum()), dtype=np.int8))
        flat_idx.append(_load(src, "idx_flat_train.npy")[flat_healthy].astype(np.int64) + offset)
        flat_ts.append(_load(src, "timestamps_flat_train.npy")[flat_healthy])
        flat_turb.append(prefix_turbines(farm, _load(src, "turbines_flat_train.npy")[flat_healthy]))
        seq_idx.append(idx.astype(np.int64) + offset)
        seq_y.append(np.zeros(len(idx), dtype=np.int8))
        seq_ts.append(_load(src, "timestamps_seq_train.npy")[seq_healthy])
        seq_turb.append(prefix_turbines(farm, _load(src, "turbines_seq_train.npy")[seq_healthy]))
        farm_rows[farm] = {"base_rows": int(len(common)), "flat_rows": int(len(flat_y[-1])), "seq_rows": int(len(seq_y[-1]))}
        offset += len(common)
    base = np.concatenate(bases, axis=0)
    _save(out, "X_base_train.npy", base)
    _save(out, "X_common_base_train.npy", base)
    _save(out, "turbines_base_train.npy", np.concatenate(base_turbines))
    _save(out, "idx_flat_train.npy", np.concatenate(flat_idx))
    _save(out, "X_flat_train.npy", np.concatenate(flat_x, axis=0))
    _save(out, "y_flat_train.npy", np.concatenate(flat_y))
    _save(out, "timestamps_flat_train.npy", np.concatenate(flat_ts))
    _save(out, "turbines_flat_train.npy", np.concatenate(flat_turb))
    _save(out, "idx_seq_train.npy", np.concatenate(seq_idx))
    _save(out, "y_seq_train.npy", np.concatenate(seq_y))
    _save(out, "timestamps_seq_train.npy", np.concatenate(seq_ts))
    _save(out, "turbines_seq_train.npy", np.concatenate(seq_turb))
    return {"farms": farm_rows, "base_rows": int(len(base)), "flat_rows": int(sum(len(x) for x in flat_y)), "seq_rows": int(sum(len(x) for x in seq_y))}


def build_source_dataset(*, true_variant: str = TRUE_VARIANT,
                         hill_variant: str = LOCAL_VARIANT,
                         output_variant: str = SOURCE_VARIANT,
                         dry_run: bool = False) -> Path:
    hill = DATA_ROOT / f"hill_of_towie__{hill_variant}"
    sources = [(farm, DATA_ROOT / f"{farm}__{true_variant}") for farm in TRUE_FARMS]
    required = [hill / "meta.json"]
    required += [path / "meta.json" for _, path in sources]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少Hill双协议输入:\n" + "\n".join(missing))
    out = DATA_ROOT / f"hill_of_towie__{output_variant}"
    if dry_run:
        print(json.dumps({"sources": [str(x[1]) for x in sources], "hill": str(hill),
                          "output": str(out), "would_write": False}, ensure_ascii=False, indent=2))
        return out
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": "fast-data-v3",
        "farm": "hill_of_towie",
        "preprocess_variant": output_variant,
        "label_mode": "external_unlabeled",
        "external_unlabeled": True,
        "external_protocol": "source_zero_shot",
        "source_farms": list(TRUE_FARMS),
        "source_training_rule": "source train split; flat point y==0 and every sequence-window label==0",
        "common_feature_dim": 6,
        "W_seq": W_SEQ,
        "seed": 0,
        "turbine_aware": True,
        "splits": {},
    }
    meta["splits"]["train"] = _build_source_train(sources, out)
    for split in ("val", "test"):
        meta["splits"][split] = _copy_hill_split(hill, out, split)
    (out / "unlabeled.json").write_text(
        json.dumps({"external_unlabeled": True, "labels_used_for_evaluation": False},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成 → {out}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hill-source-variant", default="realfault")
    ap.add_argument("--true-variant", default=TRUE_VARIANT)
    ap.add_argument("--local-variant", default=LOCAL_VARIANT)
    ap.add_argument("--source-variant", default=SOURCE_VARIANT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build_dataset(
        "hill_of_towie",
        source_variant=args.hill_source_variant,
        output_variant=args.local_variant,
        external_unlabeled=True,
        external_protocol="local_unlabeled_fit",
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return 0
    build_source_dataset(true_variant=args.true_variant, hill_variant=args.local_variant,
                         output_variant=args.source_variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
