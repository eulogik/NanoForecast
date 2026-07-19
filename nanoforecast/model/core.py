import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from nanoforecast.config import NanoForecastConfig
from nanoforecast.model.utils import DARTNorm, InstanceRobustScaler, ResolutionPrefixEmbedding, AdaptivePatching, PatchPositionalEncoding
from nanoforecast.model.blocks import SequenceMixingBlock, DeltaNetState
from nanoforecast.model.heads import PointForecastHead, MonotonicQuantileHead, AnomalyDetectionHead, DecompositionHead
from nanoforecast.hub import NanoForecastHubMixin


class StreamingState:
    """Mutable state for streaming inference, carried across ``predict_step`` calls.

    Attributes:
        buffer: Rolling window of raw values (length = ``context_length``).
        median, iqr: Scalers computed from the initial context.
        delta_states: Per-layer ``DeltaNetState``.
        num_patches_seen: Counter used to detect when a new patch is ready.
        raw_buffer: Rolling window of *unscaled* values for scaler update.
    """
    def __init__(self, buffer: torch.Tensor, median: torch.Tensor, iqr: torch.Tensor,
                 delta_states: List[DeltaNetState], num_patches_seen: int):
        self.buffer = buffer
        self.median = median
        self.iqr = iqr
        self.delta_states = delta_states
        self.num_patches_seen = num_patches_seen

class NanoForecast(NanoForecastHubMixin, nn.Module):
    """
    NanoForecast Model: The world's smallest time series foundation model.
    Integrates Robust scaling, Adaptive Patching, Resolution Prefixes,
    Gated sequence mixing blocks (Conv, RNN, MLP), and multi-task heads.
    """
    def __init__(self, config: NanoForecastConfig):
        nn.Module.__init__(self)
        self.config = config
        self.patch_size = config.patch_size
        self.d_model = config.d_model

        self.scaler = DARTNorm() if config.use_dart_norm else InstanceRobustScaler()
        self.patcher = AdaptivePatching(self.patch_size, self.d_model)

        self.freq_embedder = ResolutionPrefixEmbedding(config.num_frequencies, self.d_model)

        if config.covariate_dim > 0:
            self.covariate_projection = nn.Linear(config.covariate_dim * self.patch_size, self.d_model)

        rem = config.context_length % self.patch_size
        padded_context_len = config.context_length + (self.patch_size - rem if rem != 0 else 0)
        self.num_patches = padded_context_len // self.patch_size

        self.seq_len_with_prefix = self.num_patches + 1

        self.pos_encoder = PatchPositionalEncoding(self.num_patches, self.d_model)

        self.layers = nn.ModuleList([
            SequenceMixingBlock(
                seq_len=self.seq_len_with_prefix,
                d_model=self.d_model,
                expansion_factor=config.expansion_factor,
                dropout=config.dropout,
                use_router=config.use_gated_router
            ) for _ in range(config.num_layers)
        ])

        self.final_norm = nn.RMSNorm(self.d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(self.d_model)

        self.point_head = PointForecastHead(self.num_patches, self.d_model, config.prediction_length)
        self.quantile_head = MonotonicQuantileHead(self.num_patches, self.d_model, config.prediction_length)
        self.anomaly_head = AnomalyDetectionHead(self.num_patches, self.d_model, config.context_length)
        # Decomposition head now projects onto the *patch* grid (unit-scale, scale-invariant)
        # and the model upsamples to the step grid via a learned linear layer.
        self.decomp_head = DecompositionHead(self.num_patches, self.d_model, self.num_patches)
        self.trend_upsample = nn.Linear(self.num_patches, config.prediction_length)
        self.season_upsample = nn.Linear(self.num_patches, config.prediction_length)

    def forward(
        self,
        x: torch.Tensor,
        freq_ids: torch.Tensor,
        covariates: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input series context of shape [B, C, context_length]
            freq_ids: Frequency IDs of shape [B]
            covariates: Optional exogenous series of shape [B, covariate_dim, context_length]
        Returns:
            Dict containing scaled predictions, quantiles, anomaly scores, and decomposition.
        """
        B, C, L = x.shape
        assert L == self.config.context_length, (
            f"Expected input sequence length {self.config.context_length}, got {L}"
        )

        x_scaled, loc, scale = self.scaler(x)

        patches, padding_len = self.patcher(x_scaled)

        if covariates is not None and self.config.covariate_dim > 0:
            if padding_len > 0:
                covariates = torch.nn.functional.pad(
                    covariates, (0, padding_len), mode="constant", value=0
                )

            cov_dim = covariates.shape[1]
            cov_unfolded = covariates.view(B, cov_dim, self.num_patches, self.patch_size)
            cov_unfolded = cov_unfolded.permute(0, 2, 1, 3).contiguous()
            cov_unfolded = cov_unfolded.view(B, self.num_patches, cov_dim * self.patch_size)
            cov_repeated = cov_unfolded.repeat_interleave(C, dim=0)
            patches = patches + self.covariate_projection(cov_repeated)

        freq_ids_repeated = freq_ids.repeat_interleave(C, dim=0)
        prefix = self.freq_embedder(freq_ids_repeated)

        seq = torch.cat([prefix, patches], dim=1)

        seq = self.pos_encoder(seq)

        for layer in self.layers:
            seq = layer(seq)

        seq = self.final_norm(seq)

        latent_features = seq[:, 1:, :]

        pred_scaled = self.point_head(latent_features)
        quantiles_scaled = self.quantile_head(latent_features)
        recon_scaled = self.anomaly_head(latent_features)

        trend_p, season_p = self.decomp_head(latent_features)
        trend_s = self.trend_upsample(trend_p)
        season_s = self.season_upsample(season_p)
        residual_s = pred_scaled - trend_s - season_s

        pred_scaled = pred_scaled.view(B, C, self.config.prediction_length)
        quantiles_scaled = quantiles_scaled.view(B, C, 5, self.config.prediction_length)
        recon_scaled = recon_scaled.view(B, C, self.config.context_length)
        trend_s = trend_s.view(B, C, self.config.prediction_length)
        season_s = season_s.view(B, C, self.config.prediction_length)
        residual_s = residual_s.view(B, C, self.config.prediction_length)
        trend_scaled_patches = trend_p.view(B, C, self.num_patches)

        InverseT = self.scaler.inverse_transform
        pred = InverseT(pred_scaled, loc, scale)
        quantiles = InverseT(quantiles_scaled, loc, scale)
        reconstructed = InverseT(recon_scaled, loc, scale)

        trend = trend_s * scale + loc
        seasonal = season_s * scale
        residual = residual_s * scale

        return {
            "forecast": pred,
            "forecast_scaled": pred_scaled,
            "quantiles": quantiles,
            "quantiles_scaled": quantiles_scaled,
            "reconstructed": reconstructed,
            "recon_scaled": recon_scaled,
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "trend_scaled_patches": trend_scaled_patches,
            "latent_features": latent_features,
            "loc": loc,
            "scale": scale,
        }

    @torch.no_grad()
    def forward_stream(
        self,
        x: torch.Tensor,
        freq_ids: torch.Tensor,
        delta_states: List[DeltaNetState],
        covariates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Incremental forward pass that preserves DeltaNet states across calls."""
        B, C, L = x.shape
        x_scaled, loc, scale = self.scaler(x)
        patches, padding_len = self.patcher(x_scaled)

        if covariates is not None and self.config.covariate_dim > 0:
            if padding_len > 0:
                covariates = torch.nn.functional.pad(
                    covariates, (0, padding_len), mode="constant", value=0
                )
            cov_dim = covariates.shape[1]
            cov_unfolded = covariates.view(B, cov_dim, self.num_patches, self.patch_size)
            cov_unfolded = cov_unfolded.permute(0, 2, 1, 3).contiguous()
            cov_unfolded = cov_unfolded.view(B, self.num_patches, cov_dim * self.patch_size)
            cov_repeated = cov_unfolded.repeat_interleave(C, dim=0)
            patches = patches + self.covariate_projection(cov_repeated)

        freq_ids_repeated = freq_ids.repeat_interleave(C, dim=0)
        prefix = self.freq_embedder(freq_ids_repeated)
        seq = torch.cat([prefix, patches], dim=1)
        seq = self.pos_encoder(seq)

        batch_prod = B * C
        if len(delta_states) == 0:
            for layer in self.layers:
                ds = DeltaNetState(batch_prod, self.d_model, x.device, x.dtype)
                seq = layer.forward_stream(seq, ds, skip_conv=False)
                delta_states.append(ds)
        else:
            for layer, ds in zip(self.layers, delta_states):
                seq = layer.forward_stream(seq, ds, skip_conv=False)

        seq = self.final_norm(seq)
        latent_features = seq[:, 1:, :]

        pred_scaled = self.point_head(latent_features)
        quantiles_scaled = self.quantile_head(latent_features)
        recon_scaled = self.anomaly_head(latent_features)
        trend_p, season_p = self.decomp_head(latent_features)
        trend_s = self.trend_upsample(trend_p)
        season_s = self.season_upsample(season_p)
        residual_s = pred_scaled - trend_s - season_s

        pred_scaled = pred_scaled.view(B, C, self.config.prediction_length)
        quantiles_scaled = quantiles_scaled.view(B, C, 5, self.config.prediction_length)
        recon_scaled = recon_scaled.view(B, C, self.config.context_length)
        trend_s = trend_s.view(B, C, self.config.prediction_length)
        season_s = season_s.view(B, C, self.config.prediction_length)
        residual_s = residual_s.view(B, C, self.config.prediction_length)

        InverseT = self.scaler.inverse_transform
        pred = InverseT(pred_scaled, loc, scale)
        quantiles = InverseT(quantiles_scaled, loc, scale)
        reconstructed = InverseT(recon_scaled, loc, scale)
        trend = trend_s * scale + loc
        seasonal = season_s * scale
        residual = residual_s * scale

        return {
            "forecast": pred,
            "quantiles": quantiles,
            "reconstructed": reconstructed,
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
        }
