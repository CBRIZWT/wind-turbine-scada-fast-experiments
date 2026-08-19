# -*- coding: utf-8 -*-
"""跑三farm全部快速实验.py — 依次在 kelmarsh/penmanshiel/hill_of_towie 上跑
基础模型/00-46 全部脚本 (chuangxin 环境, FASTEXP_FARM 注入; 46 在 penmanshiel 跳过:
源域=目标域时零样本口径退化)。逐模型容错, 失败记录不中断; 每farm一份日志,
终态汇总 快速实验结果/运行状态.json。
2026-07-19: 加断点续跑 (RESUME=1 时跳过已存在 metrics.json 的模型, 防 teardown 中断丢进度)。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE / "基础模型"
RESULT_DIR = HERE / "快速实验结果"
LOG_DIR = HERE.parent / "实验结果" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PY = sys.executable
FARMS = ("kelmarsh", "penmanshiel", "hill_of_towie")
RESUME = os.environ.get("RESUME", "0") == "1"                # 断点续跑: 跳过已完成模型

scripts = sorted(p for p in SCRIPTS_DIR.glob("[0-9][0-9]_*.py"))
status: dict = {"started": time.strftime("%F %T"), "python": PY, "resume": RESUME, "farms": {}}

for farm in FARMS:
    log = LOG_DIR / f"fastexp_allmetrics_{farm}.log"
    fs: dict = {}
    with log.open("a" if RESUME else "w", encoding="utf-8") as lf:
        for sp in scripts:
            if farm == "penmanshiel" and sp.name.startswith("46_"):
                fs[sp.name] = {"skip": "源域=目标域, 零样本口径退化"}
                lf.write(f"==== SKIP {sp.name} ({fs[sp.name]['skip']})\n"); lf.flush()
                continue
            model_name = sp.stem                             # 结果子目录名 = 脚本名去扩展
            done = (RESULT_DIR / farm / model_name / "metrics.json").exists()
            if RESUME and done:
                fs[sp.name] = {"exit": 0, "sec": 0.0, "resumed_skip": True}
                lf.write(f"==== RESUME-SKIP {farm}/{sp.name} (已存在 metrics.json)\n"); lf.flush()
                print(f"[{farm}] {sp.name}: RESUME-SKIP", flush=True)
                continue
            env = {**os.environ, "FASTEXP_FARM": farm}
            t0 = time.perf_counter()
            lf.write(f"==== RUN {farm}/{sp.name}  {time.strftime('%T')}\n"); lf.flush()
            r = subprocess.run([PY, str(sp)], cwd=str(SCRIPTS_DIR), env=env,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=7200)
            dt = round(time.perf_counter() - t0, 1)
            lf.write((r.stdout or "") + (r.stderr or "") + f"---- exit={r.returncode} {dt}s\n")
            lf.flush()
            fs[sp.name] = {"exit": r.returncode, "sec": dt}
            print(f"[{farm}] {sp.name}: exit={r.returncode} {dt}s", flush=True)
    n_fail = sum(1 for v in fs.values() if v.get("exit", 0) != 0)
    status["farms"][farm] = {"models": fs, "n_fail": n_fail, "log": str(log)}
    (HERE / "快速实验结果" / "运行状态.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")

status["finished"] = time.strftime("%F %T")
(HERE / "快速实验结果" / "运行状态.json").write_text(
    json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
print("全部完成", flush=True)
