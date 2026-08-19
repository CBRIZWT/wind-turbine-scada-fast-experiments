# -*- coding: utf-8 -*-
"""可恢复地运行 Hill 12个无标签模型 × 两种协议。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE / "基础模型"
RESULT_SET = "快速实验结果_真实故障_allmetrics_v3"
RESULT_ROOT = HERE / RESULT_SET
LOG_DIR = HERE.parent / "实验结果" / "logs" / "fast_metrics_v3_hill"
HILL_MODEL_IDS = ("00", "18", "19", "20", "21", "29", "30", "31", "36", "37", "40", "41")
PROTOCOL_VARIANTS = {
    "local_unlabeled_fit": "real_fault_metrics_v1_external_local",
    "source_zero_shot": "real_fault_metrics_v1_external_source",
}


def build_hill_matrix(scripts: Sequence[Path]) -> List[Tuple[str, Path]]:
    return [(protocol, Path(script)) for protocol in PROTOCOL_VARIANTS for script in scripts]


def is_complete_external(path: Path) -> bool:
    try:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        rec.get("schema_version") == "metrics-v3"
        and rec.get("label_mode") == "external_unlabeled"
        and set(rec.get("workpoints", {})) == {"q99", "q995", "q999"}
    )


def filter_previous_runs(previous_runs, matrix):
    """恢复时只沿用当前协议×模型子矩阵内的记录。"""
    valid = {f"hill_of_towie/{protocol}/{script.stem}" for protocol, script in matrix}
    return {key: value for key, value in dict(previous_runs or {}).items() if key in valid}


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _scripts(models: str = "") -> list[Path]:
    wanted = {x.strip() for x in models.split(",") if x.strip()} or set(HILL_MODEL_IDS)
    unknown = wanted - set(HILL_MODEL_IDS)
    if unknown:
        raise ValueError(f"Hill仅允许12个无标签模型，收到 {sorted(unknown)}")
    found = []
    for model_id in HILL_MODEL_IDS:
        if model_id not in wanted:
            continue
        matches = list(SCRIPTS_DIR.glob(f"{model_id}_*.py"))
        if len(matches) != 1:
            raise FileNotFoundError(f"模型{model_id}脚本数量应为1，实际{len(matches)}")
        found.append(matches[0])
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--protocols", default=",".join(PROTOCOL_VARIANTS))
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    protocols = tuple(x.strip() for x in args.protocols.split(",") if x.strip())
    unknown = set(protocols) - set(PROTOCOL_VARIANTS)
    if unknown:
        raise ValueError(f"未知Hill协议: {sorted(unknown)}")
    matrix = [(p, s) for p, s in build_hill_matrix(_scripts(args.models)) if p in protocols]
    if args.dry_run:
        print(json.dumps({"python": sys.executable, "n_runs": len(matrix), "would_write": False,
                          "runs": [{"protocol": p, "script": s.name} for p, s in matrix]},
                         ensure_ascii=False, indent=2))
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULT_ROOT / "hill_manifest.json"
    manifest = {
        "schema_version": "hill-fast-run-manifest-v3",
        "started": time.strftime("%F %T"),
        "python": sys.executable,
        "seed": 0,
        "expected_runs": len(matrix),
        "runs": {},
    }
    if args.resume and manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runs"].update(filter_previous_runs(old.get("runs", {}), matrix))
            manifest["resumed_from"] = old.get("started")
        except (OSError, ValueError):
            pass
    for protocol, script in matrix:
        key = f"hill_of_towie/{protocol}/{script.stem}"
        metrics = RESULT_ROOT / "hill_of_towie" / protocol / script.stem / "metrics.json"
        if args.resume and is_complete_external(metrics):
            manifest["runs"][key] = {"status": "skipped_complete", "metrics": str(metrics)}
            _atomic_json(manifest_path, manifest)
            print(f"[SKIP] {key}", flush=True)
            continue
        env = {
            **os.environ,
            "FASTEXP_FARM": "hill_of_towie",
            "FASTEXP_VARIANT": PROTOCOL_VARIANTS[protocol],
            "FASTEXP_RESULT_SET": RESULT_SET,
            "FASTEXP_PROTOCOL": protocol,
            "PYTHONUTF8": "1",
        }
        log = LOG_DIR / f"{protocol}__{script.stem}.log"
        t0 = time.perf_counter()
        print(f"[RUN] {key}", flush=True)
        try:
            proc = subprocess.run([sys.executable, str(script)], cwd=str(SCRIPTS_DIR), env=env,
                                  capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=int(args.timeout))
            elapsed = round(time.perf_counter() - t0, 3)
            log.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
            complete = proc.returncode == 0 and is_complete_external(metrics)
            manifest["runs"][key] = {
                "status": "success" if complete else "failed", "exit_code": int(proc.returncode),
                "elapsed_sec": elapsed, "metrics": str(metrics), "log": str(log),
            }
        except subprocess.TimeoutExpired as exc:
            elapsed = round(time.perf_counter() - t0, 3)
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            log.write_text(stdout + stderr, encoding="utf-8")
            manifest["runs"][key] = {"status": "timeout", "elapsed_sec": elapsed,
                                      "metrics": str(metrics), "log": str(log)}
        _atomic_json(manifest_path, manifest)
        print(f"[{manifest['runs'][key]['status'].upper()}] {key}", flush=True)
    states = [x.get("status") for x in manifest["runs"].values()]
    manifest["finished"] = time.strftime("%F %T")
    manifest["n_success"] = sum(x in {"success", "skipped_complete"} for x in states)
    manifest["n_failed"] = sum(x not in {"success", "skipped_complete"} for x in states)
    manifest["complete"] = manifest["n_success"] == len(matrix) and manifest["n_failed"] == 0
    _atomic_json(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ("expected_runs", "n_success", "n_failed", "complete")},
                     ensure_ascii=False), flush=True)
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
