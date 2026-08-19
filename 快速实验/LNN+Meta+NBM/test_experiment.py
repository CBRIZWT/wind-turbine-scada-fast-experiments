from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

import experiment as exp


def test_common_residual_layout_aligns_by_name_not_position():
    meta_a = {
        "cols": ["B__resid", "A__resid", "A__resid__trend", "x"],
        "scaler": {"medians": [20, 10, 99, 0], "iqrs": [2, 1, 9, 1]},
    }
    meta_b = {
        "cols": ["A__resid", "B__resid", "B__resid__delta"],
        "scaler": {"medians": [11, 21, 88], "iqrs": [1.1, 2.1, 8]},
    }
    layout = exp.common_residual_layout(meta_a, meta_b)
    assert layout.names == ("A__resid", "B__resid")
    assert layout.indices[0].tolist() == [1, 0]
    assert layout.indices[1].tolist() == [0, 1]
    assert layout.medians[0].tolist() == [10, 20]
    assert layout.iqrs[1].tolist() == [1.1, 2.1]


def test_build_sequence_indices_rejects_turbine_crossing_gap_and_unsafe_row():
    step = exp.STEP_NS
    ts = np.array([0, step, 2 * step, 4 * step, 0, step, 2 * step], dtype=np.int64)
    tb = np.array(["A", "A", "A", "A", "B", "B", "B"])
    safe = np.ones(7, dtype=bool)
    safe[5] = False

    got = exp.build_sequence_indices(ts, tb, safe, window=2)

    # A 的 target=2 合法；target=3 前一段有 20 min 缺口。
    # B 的唯一长度足够窗口包含 unsafe 行，因此不合法。
    assert got.shape == (1, 3)
    assert got[0].tolist() == [0, 1, 2]


def test_inverse_residual_and_forecast_errors_are_in_degc():
    z_true = np.array([[0.0, 1.0], [2.0, -1.0]])
    z_pred = np.array([[1.0, 0.0], [1.0, -1.0]])
    med = np.array([10.0, -2.0])
    iqr = np.array([2.0, 4.0])
    true = exp.inverse_residual(z_true, med, iqr)
    pred = exp.inverse_residual(z_pred, med, iqr)
    summary = exp.forecast_error_summary(true, pred)
    np.testing.assert_allclose(true, [[10.0, 2.0], [14.0, -6.0]])
    assert summary["mve_degc"] == pytest.approx(-1.0)
    assert summary["mae_degc"] == pytest.approx(2.0)
    assert summary["rmse_degc"] == pytest.approx(np.sqrt(6.0))


def test_nbm_zero_residual_has_raw_residual_error():
    z_true = np.array([[0.0, 1.0]])
    med = np.array([1.0, -2.0])
    iqr = np.array([2.0, 4.0])
    true = exp.inverse_residual(z_true, med, iqr)
    pred = exp.nbm_zero_residual_prediction(z_true.shape)
    np.testing.assert_allclose(pred - true, [[-1.0, -2.0]])


def test_cfc_style_regressor_is_causal_shaped_and_differentiable():
    torch.manual_seed(3)
    model = exp.LiquidResidualRegressor(channels=4, hidden=8)
    x = torch.randn(5, 6, 4, requires_grad=True)
    y = model(x)
    assert y.shape == (5, 4)
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_reptile_update_moves_initialization_toward_adapted_weights():
    base = torch.nn.Linear(2, 1, bias=False)
    adapted = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        base.weight.fill_(0.0)
        adapted.weight.fill_(2.0)
    exp.reptile_update_(base, adapted, meta_step=0.25)
    np.testing.assert_allclose(base.weight.detach().numpy(), [[0.5, 0.5]])


def test_strict_event_metrics_counts_false_segments_and_one_hit_per_event():
    # 两个事件，各自窗口内可有多个报警段，但每个事件最多贡献一个 TP。
    episodes = [
        {"episode_id": "A@x", "turbine": "A", "start_ns": 10 * exp.STEP_NS},
        {"episode_id": "B@y", "turbine": "B", "start_ns": 10 * exp.STEP_NS},
    ]
    ts = np.array([7, 8, 9, 1, 2, 8], dtype=np.int64) * exp.STEP_NS
    tb = np.array(["A", "A", "A", "A", "A", "B"])
    pred = np.array([1, 0, 1, 1, 0, 0])
    healthy = np.array([False, False, False, True, True, False])
    m = exp.strict_event_metrics(
        pred, ts, tb, healthy, episodes, lead_steps=4, max_gap_steps=0
    )
    assert m["tp_events"] == 1
    assert m["fn_events"] == 1
    assert m["fp_segments"] == 1
    assert m["event_precision"] == pytest.approx(0.5)
    assert m["event_recall"] == pytest.approx(0.5)
    assert m["event_f1"] == pytest.approx(0.5)


def test_event_auprc_is_bounded():
    episodes = [{"episode_id": "A@x", "turbine": "A", "start_ns": 5 * exp.STEP_NS}]
    ts = np.arange(8, dtype=np.int64) * exp.STEP_NS
    tb = np.array(["A"] * 8)
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.1, 0.1, 0.1])
    healthy = np.array([True, True, True, False, False, True, True, True])
    area = exp.strict_event_auprc(
        scores, ts, tb, healthy, episodes, lead_steps=2, n_grid=20
    )
    assert 0.0 <= area <= 1.0


def test_normalized_error_score_uses_healthy_channel_rms_only():
    calibration_error = np.array([[1.0, 2.0], [-1.0, -2.0]])
    scale = exp.healthy_error_scale(calibration_error)
    score = exp.normalized_error_score(np.array([[1.0, 2.0], [2.0, 0.0]]), scale)
    np.testing.assert_allclose(scale, [1.0, 2.0])
    np.testing.assert_allclose(score, [1.0, np.sqrt(2.0)])


def test_pick_threshold_respects_far_budget_and_detects_calibration_event():
    # 健康点只有 t=0 的 0.2；事件窗 t=3,4 的分数高。预算允许零假警时应取 >0.2。
    ts = np.arange(6, dtype=np.int64) * exp.STEP_NS
    tb = np.array(["A"] * 6)
    scores = np.array([0.2, 0.1, 0.1, 0.8, 0.9, 0.1])
    healthy = np.array([True, True, True, False, False, True])
    episodes = [{"episode_id": "A@x", "turbine": "A", "start_ns": 5 * exp.STEP_NS}]
    selected = exp.pick_event_threshold(
        scores,
        ts,
        tb,
        healthy,
        episodes,
        lead_steps=2,
        far_budget=0.0,
        n_grid=50,
    )
    assert selected["threshold"] > 0.2
    assert selected["n_detected_calibration"] == 1
    assert selected["far_calibration"] == 0.0


def test_dataframe_markdown_does_not_require_optional_tabulate():
    frame = pd.DataFrame({"model": ["A"], "score": [0.12345]})
    table = exp.dataframe_markdown(frame, float_digits=3)
    assert "| model | score |" in table
    assert "| A | 0.123 |" in table


def test_json_dump_replaces_nonfinite_with_null(tmp_path):
    path = tmp_path / "strict.json"
    exp._json_dump(path, {"x": float("nan"), "nested": [np.float64(np.inf), 1.0]})
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["x"] is None
