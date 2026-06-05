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
      4. Decomposition verification (reconstruction error and trend smoothness)
    """
    def __init__(
        self, 
        quantiles: List[float],
        w_point: float = 1.0,
        w_quantile: float = 1.0,
        w_anomaly: float = 0.5,
        w_decomp: float = 0.5,
        w_smooth: float = 0.1
    ):
        super().__init__()
        self.quantiles = quantiles
        self.w_point = w_point
        self.w_quantile = w_quantile
        self.w_anomaly = w_anomaly
        self.w_decomp = w_decomp
        self.w_smooth = w_smooth

    def pinball_loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes Pinball Loss (quantile loss) for all predicted quantiles.
        Args:
            preds: Quantile forecasts of shape [B, C, num_quantiles, prediction_length]
            target: Actual targets of shape [B, C, prediction_length]
        """
        # Align shapes by adding a quantile dimension to target: [B, C, 1, prediction_length]
        target_unsqueezed = target.unsqueeze(2)
        diff = target_unsqueezed - preds # [B, C, num_quantiles, prediction_length]
        
        loss_val = 0.0
        for i, q in enumerate(self.quantiles):
            d = diff[:, :, i, :]
            # Pinball loss formula: max(q * d, (q - 1) * d)
            loss_q = torch.max(q * d, (q - 1) * d)
            loss_val += loss_q.mean()
            
        return loss_val / len(self.quantiles)

    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        target_y: torch.Tensor,
        context_x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: Dictionary of predictions from NanoForecast model
            target_y: Ground truth future target of shape [B, C, prediction_length]
            context_x: Original scaling-normalized input context of shape [B, C, context_length]
        Returns:
            total_loss: Aggregated scalar tensor loss
            loss_dict: Dictionary containing break-down values of float losses
        """
        # 1. Point Forecast Loss (MSE + MAE)
        forecast = outputs["forecast"]
        loss_mse = F.mse_loss(forecast, target_y)
        loss_mae = F.l1_loss(forecast, target_y)
        loss_point = 0.5 * loss_mse + 0.5 * loss_mae
        
        # 2. Quantile Loss
        quantiles = outputs["quantiles"]
        loss_quantile = self.pinball_loss(quantiles, target_y)
        
        # 3. Anomaly Reconstruction Loss
        reconstructed = outputs["reconstructed"]
        loss_anomaly = F.mse_loss(reconstructed, context_x)
        
        # 4. Decomposition Verification Loss
        # (trend + seasonal + residual should reconstruct the forecast target y)
        trend = outputs["trend"]
        seasonal = outputs["seasonal"]
        residual = outputs["residual"]
        recon_decomp = trend + seasonal + residual
        loss_decomp = F.mse_loss(recon_decomp, target_y)
        
        # Trend smoothness penalty (penalize second differences of trend component)
        # Minimize second derivative: (T_t - 2*T_{t-1} + T_{t-2})^2
        if trend.shape[-1] > 2:
            trend_diff = torch.diff(trend, dim=-1)
            trend_double_diff = torch.diff(trend_diff, dim=-1)
            loss_smooth = trend_double_diff.pow(2).mean()
        else:
            loss_smooth = torch.tensor(0.0, device=target_y.device)
            
        # Combine all losses
        total_loss = (
            self.w_point * loss_point +
            self.w_quantile * loss_quantile +
            self.w_anomaly * loss_anomaly +
            self.w_decomp * loss_decomp +
            self.w_smooth * loss_smooth
        )
        
        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_point": loss_point.item(),
            "loss_quantile": loss_quantile.item(),
            "loss_anomaly": loss_anomaly.item(),
            "loss_decomp": loss_decomp.item(),
            "loss_smooth": loss_smooth.item()
        }
        
        return total_loss, loss_dict
