import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class NanoForecastLoss(nn.Module):
    """v0.5 focused loss: next-token (dense) + quantile only.

    Reverso achieves MASE 0.711 by training with a single next-token
    objective (MSE over every future step) plus quantile loss.
    Anomaly/decomposition heads are kept for inference but not supervised.
    """

    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def pinball_loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_unsqueezed = target.unsqueeze(2)
        diff = target_unsqueezed - preds
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
        loc = outputs["loc"]
        scale = outputs["scale"]
        target_scaled = (target_y - loc) / scale.clamp(min=1e-6)

        forecast_scaled = outputs["forecast_scaled"]
        quantiles_scaled = outputs["quantiles_scaled"]

        loss_mse = F.mse_loss(forecast_scaled, target_scaled)
        loss_quantile = self.pinball_loss(quantiles_scaled, target_scaled)

        total_loss = loss_mse + loss_quantile

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_mse": loss_mse.item(),
            "loss_quantile": loss_quantile.item(),
        }
        return total_loss, loss_dict


class MultiTaskLoss(nn.Module):
    """Legacy v0.1-v0.4 multi-task loss — kept for backward compat."""
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
        target_unsqueezed = target.unsqueeze(2)
        diff = target_unsqueezed - preds
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
        median = outputs.get("median", outputs.get("loc"))
        iqr = outputs.get("iqr", outputs.get("scale"))
        target_scaled = (target_y - median) / iqr.clamp(min=0.1)
        context_scaled = (context_x - median) / iqr.clamp(min=0.1)

        forecast_scaled = outputs["forecast_scaled"]
        quantiles_scaled = outputs["quantiles_scaled"]
        recon_scaled = outputs["recon_scaled"]
        trend_scaled_patches = outputs.get("trend_scaled_patches")

        loss_mse = F.mse_loss(forecast_scaled, target_scaled)
        loss_mae = F.l1_loss(forecast_scaled, target_scaled)
        loss_point = 0.5 * loss_mse + 0.5 * loss_mae
        loss_quantile = self.pinball_loss(quantiles_scaled, target_scaled)
        loss_anomaly = F.mse_loss(recon_scaled, context_scaled)

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
