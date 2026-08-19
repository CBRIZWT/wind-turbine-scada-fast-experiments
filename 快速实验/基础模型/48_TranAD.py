# -*- coding: utf-8 -*-
"""48 TranAD(主项目桥接) —— 原生架构, 1epoch【二阶段重构·末步误差变体】。

诚实标注: 用 TranAD 原生网络(seq-first 两阶段 transformer, nhead=通道数), 训练/打分统一为
  最小化【末步】重构MSE(取 x2 相位输出); 分数=末步重构误差。【不含】论文的焦点分数/对抗缩放。
注: TranAD nhead=feats(87), 批次×头数受 cuBLAS 批上限约束 → batch/score_batch 用 128。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]                  # E:\创新
for p in (str(_ROOT), str(_ROOT / "TranAD-main")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models import TranAD                                 # noqa: E402  原生架构
from _deep_bridge import run_recon                            # noqa: E402

WIN = 36   # TranAD 原生窗口 (实验配置.DatasetProtocol.WIN_TRANAD)


def build(W, C):
    return TranAD(C)


def fwd(model, x):
    # x=(B,W,C) → TranAD 需 seq-first: src=(W,B,C), tgt=末步(1,B,C); 取 x2 相位(1,B,C)→(B,C)
    src = x.permute(1, 0, 2).contiguous()
    tgt = src[-1:].contiguous()
    _x1, x2 = model(src, tgt)
    return x2.squeeze(0)


if __name__ == "__main__":
    run_recon("48_TranAD", build, WIN, forward_fn=fwd, laststep=True, batch=128, score_batch=128)
