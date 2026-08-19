# -*- coding: utf-8 -*-
"""50 wt-transformer(主项目桥接) —— 原生Transformer, 1epoch【下一步预测误差变体】。

诚实标注: 用 wt-transformer 原生网络(out_dim=C 预测全通道下一步), 历史 SEQ 步→预测第 SEQ+1 步;
  分数=下一步预测误差。快速实验配置: SEQ=36, blocks=2, heads=4(为提速), extra 记 variant。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]                  # E:\创新
for p in (str(_ROOT), str(_ROOT / "wt-transformer-fault-prediction-main")):
    if p not in sys.path:
        sys.path.insert(0, p)

from Src.model.transformer_torch import WTTransformerTorch    # noqa: E402  原生架构
from _deep_bridge import run_forecast                         # noqa: E402

SEQ = 36
H = 1


def build(seq_len, C, horizon):
    return WTTransformerTorch(input_dim=C, seq_len=seq_len, out_dim=C,
                              num_transformer_blocks=2, num_heads=4)


def fwd(model, hist):
    # hist=(B, seq_len, C) → wt 直接吃 (B,seq,C); 输出 (B,C) 下一步全通道 → (B,1,C)
    out = model(hist)              # (B, C)
    return out.unsqueeze(1)        # (B, 1, C)


if __name__ == "__main__":
    run_forecast("50_wt_transformer", build, SEQ + H, forward_fn=fwd, horizon=H)
