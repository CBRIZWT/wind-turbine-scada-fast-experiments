# -*- coding: utf-8 -*-
"""可恢复地运行 Kelmarsh/Penmanshiel 各47个 metrics-v3 快速模型。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE / "基础模型"
RESULT_SET = "快速实验结果_真实故障_allmetrics_v3"
RESULT_ROOT = HERE / RESULT_SET
LOG_DIR = HERE.parent / "实验结果" / "logs" / "fast_metrics_v3"
VARIANT = "real_fault_metrics_v1"
TRUE_FAULT_FARMS = ("kelmarsh", "penmanshiel")


def build_run_matrix(farms: Sequence[str], scripts: Sequence[Path]) -> List[Tuple[str, Path]]:
    """笛卡尔积；模型46在两个真标签风场均运行，不设跳过分支。"""
    return [(str(farm), Path(script)) for farm in farms for script in scripts]


def is_complete_metrics(path: Path) -> bool:
    try:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return rec.get("schema_version") == "metrics-v3" and set(rec.get("workpoints", {})) == {
        "balanced", "low_far", "high_recall"
    }


def filter_previous_runs(previous_runs, matrix):
    """恢复时只沿用当前请求矩阵内的记录，避免子矩阵统计混入旧run。"""
    valid = {f"{farm}/{script.stem}" for farm, script in matrix}
    return {key: value for key, value in dict(previous_runs or {}).items() if key in valid}


def manifest_path_for(result_root: Path, farms: Sequence[str]) -> Path:
    """单风场长跑使用隔离清单，双风场复核使用总清单。"""
    farms = tuple(farms)
    return Path(result_root) / (f"manifest__{farms[0]}.json" if len(farms) == 1 else "manifest.json")


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _select_scripts(model_filter: Iterable[str] = ()) -> List[Path]:
    scripts = sorted(SCRIPTS_DIR.glob("[0-9][0-9]_*.py"))
    wanted = {str(x).strip() for x in model_filter if str(x).strip()}
    if wanted:
        scripts = [p for p in scripts if p.stem in wanted or p.stem[:2] in wanted]
    if not scripts:
        raise ValueError("没有匹配的快速模型脚本")
    return scripts


def main() -> int:
    global VARIANT, RESULT_SET, RESULT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", default=",".join(TRUE_FAULT_FARMS))
    ap.add_argument("--models", default="", help="逗号分隔的两位编号或脚本stem；默认全部47个")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--variant", default=VARIANT,
                    help="数据变体 (默认 real_fault_metrics_v1)")
    ap.add_argument("--result-set", default=RESULT_SET, help="结果集目录名")
    args = ap.parse_args()

    VARIANT = args.variant
    RESULT_SET = args.result_set
    RESULT_ROOT = HERE / RESULT_SET

    farms = tuple(x.strip() for x in args.farms.split(",") if x.strip())
    unknown = set(farms) - set(TRUE_FAULT_FARMS)
    if unknown:
        raise ValueError(f"真故障47模型runner只接受 {TRUE_FAULT_FARMS}，收到 {sorted(unknown)}")
    scripts = _select_scripts(args.models.split(","))
    matrix = build_run_matrix(farms, scripts)
    if args.dry_run:
        print(json.dumps({
            "python": sys.executable,
            "variant": VARIANT,
            "result_root": str(RESULT_ROOT),
            "n_runs": len(matrix),
            "runs": [{"farm": f, "script": p.name} for f, p in matrix],
            "would_write": False,
        }, ensure_ascii=False, indent=2))
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path_for(RESULT_ROOT, farms)
    manifest = {
        "schema_version": "fast-run-manifest-v3",
        "started": time.strftime("%F %T"),
        "python": sys.executable,
        "variant": VARIANT,
        "result_set": RESULT_SET,
        "seed": 0,
        "expected_runs": len(matrix),
        "runs": {},
    }
    if args.resume and manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runs"].update(filter_previous_runs(previous.get("runs", {}), matrix))
            manifest["resumed_from"] = previous.get("started")
        except (OSError, ValueError):
            pass

    for farm, script in matrix:
        key = f"{farm}/{script.stem}"
        metrics_path = RESULT_ROOT / farm / script.stem / "metrics.json"
        if args.resume and is_complete_metrics(metrics_path):
            manifest["runs"][key] = {
                "status": "skipped_complete",
                "metrics": str(metrics_path),
            }
            _atomic_json(manifest_path, manifest)
            print(f"[SKIP] {key}", flush=True)
            continue
        env = {
            **os.environ,
            "FASTEXP_FARM": farm,
            "FASTEXP_VARIANT": VARIANT,
            "FASTEXP_RESULT_SET": RESULT_SET,
            "PYTHONUTF8": "1",
        }
        log = LOG_DIR / f"{farm}__{script.stem}.log"
        t0 = time.perf_counter()
        print(f"[RUN] {key}", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(SCRIPTS_DIR),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=int(args.timeout),
            )
            elapsed = round(time.perf_counter() - t0, 3)
            log.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
            complete = proc.returncode == 0 and is_complete_metrics(metrics_path)
            manifest["runs"][key] = {
                "status": "success" if complete else "failed",
                "exit_code": int(proc.returncode),
                "elapsed_sec": elapsed,
                "metrics": str(metrics_path),
                "log": str(log),
            }
        except subprocess.TimeoutExpired as exc:
            elapsed = round(time.perf_counter() - t0, 3)
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            log.write_text(stdout + stderr, encoding="utf-8")
            manifest["runs"][key] = {
                "status": "timeout",
                "elapsed_sec": elapsed,
                "metrics": str(metrics_path),
                "log": str(log),
            }
        _atomic_json(manifest_path, manifest)
        print(f"[{manifest['runs'][key]['status'].upper()}] {key}", flush=True)

    states = [x.get("status") for x in manifest["runs"].values()]
    manifest["finished"] = time.strftime("%F %T")
    manifest["n_success"] = sum(s in {"success", "skipped_complete"} for s in states)
    manifest["n_failed"] = sum(s not in {"success", "skipped_complete"} for s in states)
    manifest["complete"] = manifest["n_success"] == len(matrix) and manifest["n_failed"] == 0
    _atomic_json(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ("expected_runs", "n_success", "n_failed", "complete")},
                     ensure_ascii=False), flush=True)
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
