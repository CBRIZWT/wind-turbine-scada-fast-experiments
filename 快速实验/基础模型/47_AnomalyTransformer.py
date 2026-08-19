# -*- coding: utf-8 -*-
"""47 AnomalyTransformer(主项目桥接) —— 原生架构 + 原生窗口, 1epoch【重构误差变体】。

诚实标注: 用 AnomalyTransformer 原生网络(output_attention=False→forward返回重构 enc_out),
  训练/打分统一为最小化重构MSE / 分数=窗末重构误差; 【不含】论文的关联差异(association
  discrepancy)min-max 训练。故 extra.variant=recon-error-only, 名义反映"架构"而非"论文完整方法"。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]                 # E:\创新
_AT = str(_ROOT / "Anomaly-Transformer-main")
if _AT not in sys.path:
    sys.path.insert(0, _AT)

from model.AnomalyTransformer import AnomalyTransformer      # noqa: E402  原生架构
from _deep_bridge import run_recon                           # noqa: E402

WIN = 100   # AT 原生窗口 (实验配置.DatasetProtocol.WIN_AT)


def build(W, C):
    # output_attention=True(默认): forward 返回 (enc_out重构, series, prior, sigmas);
    #   桥接器 _default_recon_forward 取 [0]=enc_out 作重构。(False 会触发内注意力解包bug。)
    # 快速实验【轻量配置】: 原生架构+原生窗口(100)不变, 但 d_model 512→128 / e_layers 3→2 / d_ff→128,
    #   为"快速排序"提速(native 512 单场>10min不适合快速实验); extra 已记 variant, 诚实标注非论文满配。
    return AnomalyTransformer(win_size=W, enc_in=C, c_out=C,
                              d_model=128, n_heads=4, e_layers=2, d_ff=128)


if __name__ == "__main__":
    run_recon("47_AnomalyTransformer", build, WIN)
