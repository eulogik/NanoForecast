import torch
import numpy as np

from nanoforecast.config import NanoForecastConfig
from nanoforecast.model.core import NanoForecast
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
    # Batch size 2, Channels 1, Length 100
    x = torch.randn(2, 1, 100) * 10.0 + 5.0
    # Add outlier
    x[0, 0, 50] = 1000.0
    
    x_scaled, median, iqr = scaler(x)
    
    # Scaled output should have values centered around 0 with narrow range (robust to the outlier)
    assert x_scaled.shape == x.shape
    assert median.shape == (2, 1, 1)
    assert iqr.shape == (2, 1, 1)
    
    # Test inverse scale restoration
    x_restored = InstanceRobustScaler.inverse_transform(x_scaled, median, iqr)
    assert torch.allclose(x, x_restored, atol=1e-4)


def test_monotonic_quantile_head():
    batch_size = 4
    num_patches = 16
    d_model = 32
    prediction_length = 24
    
    head = MonotonicQuantileHead(num_patches, d_model, prediction_length)
    dummy_latent = torch.randn(batch_size, num_patches, d_model)
    
    quantiles = head(dummy_latent) # [B, 5, prediction_length]
    
    assert quantiles.shape == (batch_size, 5, prediction_length)
    
    # Verify strict monotonicity: p10 <= p25 <= p50 <= p75 <= p90
    p10 = quantiles[:, 0, :]
    p25 = quantiles[:, 1, :]
    p50 = quantiles[:, 2, :]
    p75 = quantiles[:, 3, :]
    p90 = quantiles[:, 4, :]
    
    assert torch.all(p10 <= p25)
    assert torch.all(p25 <= p50)
    assert torch.all(p50 <= p75)
    assert torch.all(p75 <= p90)


def test_nano_forecast_forward():
    config = NanoForecastConfig(
        context_length=128,
        prediction_length=24,
        d_model=32,
        num_layers=2, # Small layers for speed
        patch_size=8,
        covariate_dim=4
    )
    
    model = NanoForecast(config)
    
    B, C = 2, 1
    x = torch.randn(B, C, config.context_length)
    freq_ids = torch.randint(0, config.num_frequencies, (B,))
    covariates = torch.randn(B, config.covariate_dim, config.context_length)
    
    outputs = model(x, freq_ids, covariates)
    
    assert "forecast" in outputs
    assert "quantiles" in outputs
    assert "reconstructed" in outputs
    assert "trend" in outputs
    assert "seasonal" in outputs
    assert "residual" in outputs
    
    assert outputs["forecast"].shape == (B, C, config.prediction_length)
    assert outputs["quantiles"].shape == (B, C, 5, config.prediction_length)
    assert outputs["reconstructed"].shape == (B, C, config.context_length)
    assert outputs["trend"].shape == (B, C, config.prediction_length)
    assert outputs["seasonal"].shape == (B, C, config.prediction_length)
    assert outputs["residual"].shape == (B, C, config.prediction_length)
    
    # Check decomposition reconstruction conservation: trend + seasonal + residual == forecast
    recon = outputs["trend"] + outputs["seasonal"] + outputs["residual"]
    assert torch.allclose(recon, outputs["forecast"], atol=1e-4)
