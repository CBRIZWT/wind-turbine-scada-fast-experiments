# -*- coding: utf-8 -*-
"""49 TriTrackNet(主项目桥接) —— 原生架构(RevIN+双通道注意力), 1epoch【预测误差变体】。

诚实标注: 用 TriTrackNet 原生预测网络, 历史 SEQ 步预测末端 H 步; 分数=末端预测误差。
  快速实验配置: SEQ=36, H=12(原生 pred_horizon 默认 96, 为提速缩短), extra 记 variant。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]                  # E:\创新
for p in (str(_ROOT), str(_ROOT / "TriTrackNet-main")):
    if p not in sys.path:
        sys.path.insert(0, p)

from TriTrackNet.TriTrackNet import TriTrackNetArchitecture   # noqa: E402  原生架构
from _deep_bridge import run_forecast                         # noqa: E402

SEQ = 36   # 历史窗
H = 12     # 预测步


def build(seq_len, C, horizon):
    return TriTrackNetArchitecture(num_channels=C, seq_len=seq_len, pred_horizon=horizon, use_revin=True)


def fwd(model, hist):
    # hist=(B, seq_len, C) → TriTrack 需 (B,C,seq_len); flatten 输出 (B,C*H) → (B,C,H) → (B,H,C)
    x_cl = hist.permute(0, 2, 1).contiguous()
    out = model(x_cl, flatten_output=True)          # (B, C*H)
    out = out.view(out.shape[0], x_cl.shape[1], -1)  # (B, C, H)
    return out.permute(0, 2, 1)                      # (B, H, C)


if __name__ == "__main__":
    run_forecast("49_TriTrackNet", build, SEQ + H, forward_fn=fwd, horizon=H)
