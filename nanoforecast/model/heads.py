import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class PointForecastHead(nn.Module):
    """
    Point forecast head that maps latent sequence features to point predictions.
    """
    def __init__(self, num_patches: int, d_model: int, prediction_length: int):
        super().__init__()
        self.prediction_length = prediction_length
        self.projection = nn.Linear(num_patches * d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent representations of shape [BC, num_patches, d_model]
        Returns:
            out: Forecast predictions of shape [BC, prediction_length]
        """
        BC, num_patches, d_model = x.shape
        x_flat = x.view(BC, -1)
        return self.projection(x_flat)


class MonotonicQuantileHead(nn.Module):
    """
    Quantile forecast head predicting p10, p25, p50, p75, p90.
    Enforces monotonicity via softplus cumulative deltas to prevent quantile crossing.
    """
    def __init__(self, num_patches: int, d_model: int, prediction_length: int):
        super().__init__()
        self.prediction_length = prediction_length
        self.num_quantiles = 5
        
        # Predict median (p50)
        self.p50_proj = nn.Linear(num_patches * d_model, prediction_length)
        
        # Predict 4 deltas: [p10_to_p25, p25_to_p50, p50_to_p75, p75_to_p90]
        self.deltas_proj = nn.Linear(num_patches * d_model, 4 * prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent representations of shape [BC, num_patches, d_model]
        Returns:
            quantiles: Monotonic quantiles of shape [BC, 5, prediction_length]
                       representing [p10, p25, p50, p75, p90]
        """
        BC, num_patches, d_model = x.shape
        x_flat = x.view(BC, -1)
        
        p50 = self.p50_proj(x_flat) # [BC, prediction_length]
        
        # Project and apply softplus to force positive delta offsets
        deltas = self.deltas_proj(x_flat) # [BC, 4 * prediction_length]
        deltas = deltas.view(BC, 4, self.prediction_length)
        deltas = F.softplus(deltas) # [BC, 4, prediction_length]
        
        d10_to_25 = deltas[:, 0, :]
        d25_to_50 = deltas[:, 1, :]
        d50_to_75 = deltas[:, 2, :]
        d75_to_90 = deltas[:, 3, :]
        
        # Reconstruct monotonic bounds
        p25 = p50 - d25_to_50
        p10 = p25 - d10_to_25
        p75 = p50 + d50_to_75
        p90 = p75 + d75_to_90
        
        # Stack into [BC, 5, prediction_length]
        return torch.stack([p10, p25, p50, p75, p90], dim=1)


class AnomalyDetectionHead(nn.Module):
    """
    Anomaly detection head that reconstructs the input context sequence.
    Reconstruction error (MSE) is used to compute anomaly scores.
    """
    def __init__(self, num_patches: int, d_model: int, context_length: int):
        super().__init__()
        self.context_length = context_length
        self.projection = nn.Linear(num_patches * d_model, context_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent representations of shape [BC, num_patches, d_model]
        Returns:
            reconstructed: Reconstructed context of shape [BC, context_length]
        """
        BC, num_patches, d_model = x.shape
        x_flat = x.view(BC, -1)
        return self.projection(x_flat)


class DecompositionHead(nn.Module):
    """
    Decomposition head separating Trend and Seasonality components on the patch grid.
    Upsampling to the step grid is performed by the parent model.
    """
    def __init__(self, num_patches: int, d_model: int, num_patches_out: int):
        super().__init__()
        self.num_patches_out = num_patches_out
        self.trend_proj = nn.Linear(num_patches * d_model, num_patches_out)
        self.season_proj = nn.Linear(num_patches * d_model, num_patches_out)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Latent representations of shape [BC, num_patches, d_model]
        Returns:
            trend: Trend prediction of shape [BC, num_patches_out]
            season: Seasonality prediction of shape [BC, num_patches_out]
        """
        BC, num_patches, d_model = x.shape
        x_flat = x.view(BC, -1)

        trend = self.trend_proj(x_flat)
        season = self.season_proj(x_flat)

        return trend, season
