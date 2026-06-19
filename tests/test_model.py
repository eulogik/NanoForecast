import tempfile
import os

import torch
import numpy as np

from nanoforecast.config import NanoForecastConfig
from nanoforecast.model.core import NanoForecast, StreamingState
from nanoforecast.model.utils import InstanceRobustScaler
from nanoforecast.model.heads import MonotonicQuantileHead


def test_config_presets():
    cfg_200 = NanoForecastConfig.nano_200k()
    cfg_500 = NanoForecastConfig.nano_500k()

    assert cfg_200.d_model == 32
    assert cfg_200.num_layers == 4
    assert cfg_500.d_model == 64
    assert cfg_500.num_layers == 8


def test_robust_scaler():
    scaler = InstanceRobustScaler()
    x = torch.randn(2, 1, 100) * 10.0 + 5.0
    x[0, 0, 50] = 1000.0  # outlier

    x_scaled, median, iqr = scaler(x)

    assert x_scaled.shape == x.shape
    assert median.shape == (2, 1, 1)
    assert iqr.shape == (2, 1, 1)

    x_restored = InstanceRobustScaler.inverse_transform(x_scaled, median, iqr)
    assert torch.allclose(x, x_restored, atol=1e-4)


def test_monotonic_quantile_head():
    batch_size = 4
    num_patches = 16
    d_model = 32
    prediction_length = 24

    head = MonotonicQuantileHead(num_patches, d_model, prediction_length)
    dummy_latent = torch.randn(batch_size, num_patches, d_model)

    quantiles = head(dummy_latent)  # [B, 5, prediction_length]
    assert quantiles.shape == (batch_size, 5, prediction_length)

    p10, p25, p50, p75, p90 = (quantiles[:, i, :] for i in range(5))
    assert torch.all(p10 <= p25)
    assert torch.all(p25 <= p50)
    assert torch.all(p50 <= p75)
    assert torch.all(p75 <= p90)


def test_nano_forecast_forward():
    config = NanoForecastConfig(
        context_length=128,
        prediction_length=24,
        d_model=32,
        num_layers=2,
        patch_size=8,
        covariate_dim=4,
    )

    model = NanoForecast(config)
    B, C = 2, 1
    x = torch.randn(B, C, config.context_length)
    freq_ids = torch.randint(0, config.num_frequencies, (B,))
    covariates = torch.randn(B, config.covariate_dim, config.context_length)

    outputs = model(x, freq_ids, covariates)

    for key in ("forecast", "quantiles", "reconstructed", "trend", "seasonal", "residual", "trend_scaled_patches"):
        assert key in outputs, f"missing output: {key}"

    assert outputs["forecast"].shape == (B, C, config.prediction_length)
    assert outputs["quantiles"].shape == (B, C, 5, config.prediction_length)
    assert outputs["reconstructed"].shape == (B, C, config.context_length)
    assert outputs["trend"].shape == (B, C, config.prediction_length)
    assert outputs["seasonal"].shape == (B, C, config.prediction_length)
    assert outputs["residual"].shape == (B, C, config.prediction_length)
    # trend_scaled_patches should be on the patch grid
    num_patches = config.context_length // config.patch_size
    assert outputs["trend_scaled_patches"].shape == (B, C, num_patches)

    # Conservation: trend + seasonal + residual == forecast
    recon = outputs["trend"] + outputs["seasonal"] + outputs["residual"]
    assert torch.allclose(recon, outputs["forecast"], atol=1e-4)


def test_hub_save_load_roundtrip():
    config = NanoForecastConfig(
        context_length=64, prediction_length=12, d_model=16, num_layers=2,
        patch_size=4, covariate_dim=0,
    )
    model = NanoForecast(config)
    model.eval()

    ctx = np.random.randn(config.context_length).astype(np.float32)
    out1 = model.predict(ctx, horizon=config.prediction_length, freq="H")

    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        assert "config.json" in os.listdir(d)
        assert any(f in os.listdir(d) for f in ("model.safetensors", "model.pt"))

        model2 = NanoForecast.from_pretrained(d)
        out2 = model2.predict(ctx, horizon=config.prediction_length, freq="H")

    assert np.allclose(out1["forecast"], out2["forecast"], atol=1e-5)
    assert np.allclose(out1["quantiles"], out2["quantiles"], atol=1e-5)
    p10, p25, p50, p75, p90 = (out2["quantiles"][0, i] for i in range(5))
    assert np.all(p10 <= p25) and np.all(p25 <= p50) and np.all(p50 <= p75) and np.all(p75 <= p90)


def test_streaming_inference():
    config = NanoForecastConfig(
        context_length=64, prediction_length=12, d_model=16, num_layers=2,
        patch_size=4, covariate_dim=0,
    )
    model = NanoForecast(config)
    model.eval()

    ctx = np.sin(np.linspace(0, 4 * np.pi, config.context_length)).astype(np.float32)
    result = model.predict(ctx, horizon=config.prediction_length, freq="H", return_state=True)
    assert "state" in result
    state = result["state"]
    assert isinstance(state, StreamingState)
    assert len(state.delta_states) == config.num_layers

    fore0 = result["forecast"].copy()

    # Stream 8 new values one at a time — forecasts should change smoothly
    forecasts = [fore0]
    for i in range(8):
        new_val = float(np.sin(4 * np.pi + i * 0.1))
        step_result = model.predict_step(new_val, state, horizon=config.prediction_length, freq="H")
        assert "forecast" in step_result
        forecasts.append(step_result["forecast"].copy())

    # Each step should produce a valid forecast (no NaNs, finite)
    for f in forecasts:
        assert np.all(np.isfinite(f))
    # Forecasts should change (consecutive forecasts shouldn't be identical)
    diffs = [np.abs(forecasts[i+1] - forecasts[i]).mean() for i in range(len(forecasts)-1)]
    assert any(d > 1e-6 for d in diffs), "Streaming forecasts should not be identical"
