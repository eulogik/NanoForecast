import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for NanoForecast.
    Includes:
      1. Point forecast loss (MSE + MAE)
      2. Quantile loss (Pinball loss) for prediction intervals
      3. Anomaly reconstruction loss (MSE of context window)
      4. Trend smoothness penalty (operates on patch-grid scaled trend)
    """
    def __init__(
        self,
        quantiles: List[float],
        w_point: float = 0.5,
        w_quantile: float = 1.0,
        w_anomaly: float = 0.1,
        w_smooth: float = 0.05,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.w_point = w_point
        self.w_quantile = w_quantile
        self.w_anomaly = w_anomaly
        self.w_smooth = w_smooth

    def pinball_loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes Pinball Loss (quantile loss) for all predicted quantiles.
        Args:
            preds: Quantile forecasts of shape [B, C, num_quantiles, prediction_length]
            target: Actual targets of shape [B, C, prediction_length]
        """
        target_unsqueezed = target.unsqueeze(2)
        diff = target_unsqueezed - preds # [B, C, num_quantiles, prediction_length]

        loss_val = 0.0
        for i, q in enumerate(self.quantiles):
            d = diff[:, :, i, :]
            loss_q = torch.max(q * d, (q - 1) * d)
            loss_val += loss_q.mean()

        return loss_val / len(self.quantiles)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_y: torch.Tensor,
        context_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: Dictionary of predictions from NanoForecast model
            target_y: Ground truth future target of shape [B, C, prediction_length]
            context_x: Raw input context of shape [B, C, context_length] (unscaled)
        Returns:
            total_loss: Aggregated scalar tensor loss
            loss_dict: Dictionary containing break-down values of float losses
        """
        # All losses computed in normalized (scaled) space so datasets with
        # different magnitudes contribute equally.
        median = outputs["median"]
        iqr = outputs["iqr"]
        target_scaled = (target_y - median) / iqr.clamp(min=1e-5)
        context_scaled = (context_x - median) / iqr.clamp(min=1e-5)

        forecast_scaled = outputs["forecast_scaled"]
        quantiles_scaled = outputs["quantiles_scaled"]
        recon_scaled = outputs["recon_scaled"]
        trend_scaled_patches = outputs.get("trend_scaled_patches")

        # 1. Point forecast loss in scaled space
        loss_mse = F.mse_loss(forecast_scaled, target_scaled)
        loss_mae = F.l1_loss(forecast_scaled, target_scaled)
        loss_point = 0.5 * loss_mse + 0.5 * loss_mae

        # 2. Quantile pinball loss in scaled space
        loss_quantile = self.pinball_loss(quantiles_scaled, target_scaled)

        # 3. Anomaly reconstruction loss in scaled space
        loss_anomaly = F.mse_loss(recon_scaled, context_scaled)

        # 4. Trend smoothness on the *patch-grid scaled* trend (unit scale, scale-invariant)
        if trend_scaled_patches is not None and trend_scaled_patches.shape[-1] > 2:
            trend_diff = torch.diff(trend_scaled_patches, dim=-1)
            trend_double_diff = torch.diff(trend_diff, dim=-1)
            loss_smooth = trend_double_diff.pow(2).mean()
        else:
            loss_smooth = torch.tensor(0.0, device=target_y.device)

        total_loss = (
            self.w_point * loss_point
            + self.w_quantile * loss_quantile
            + self.w_anomaly * loss_anomaly
            + self.w_smooth * loss_smooth
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_point": loss_point.item(),
            "loss_quantile": loss_quantile.item(),
            "loss_anomaly": loss_anomaly.item(),
            "loss_smooth": loss_smooth.item(),
        }

        return total_loss, loss_dict
