import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from nanoforecast.config import NanoForecastConfig
from nanoforecast.model.utils import InstanceRobustScaler, ResolutionPrefixEmbedding, AdaptivePatching, PatchPositionalEncoding
from nanoforecast.model.blocks import SequenceMixingBlock
from nanoforecast.model.heads import PointForecastHead, MonotonicQuantileHead, AnomalyDetectionHead, DecompositionHead
from nanoforecast.hub import NanoForecastHubMixin

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

        self.scaler = InstanceRobustScaler()
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
        assert L == self.config.context_length, f"Expected input sequence length {self.config.context_length}, got {L}"

        x_scaled, median, iqr = self.scaler(x)

        patches, padding_len = self.patcher(x_scaled)

        if covariates is not None and self.config.covariate_dim > 0:
            if padding_len > 0:
                covariates = torch.nn.functional.pad(covariates, (0, padding_len), mode="constant", value=0)

            cov_dim = covariates.shape[1]
            cov_unfolded = covariates.view(B, cov_dim, self.num_patches, self.patch_size)
            cov_unfolded = cov_unfolded.permute(0, 2, 1, 3).contiguous()
            cov_unfolded = cov_unfolded.view(B, self.num_patches, cov_dim * self.patch_size)
            cov_repeated = cov_unfolded.repeat_interleave(C, dim=0)
            patches = patches + self.covariate_projection(cov_repeated)

        freq_ids_repeated = freq_ids.repeat_interleave(C, dim=0)
        prefix = self.freq_embedder(freq_ids_repeated)

        seq = torch.cat([prefix, patches], dim=1)

        # 4b. Add positional encoding to patch tokens
        seq = self.pos_encoder(seq)

        for layer in self.layers:
            seq = layer(seq)

        seq = self.final_norm(seq)

        latent_features = seq[:, 1:, :]

        pred_scaled = self.point_head(latent_features)            # [BC, prediction_length]
        quantiles_scaled = self.quantile_head(latent_features)    # [BC, 5, prediction_length]
        recon_scaled = self.anomaly_head(latent_features)          # [BC, context_length]

        # Decomposition operates on the patch grid (unit scale)
        trend_p, season_p = self.decomp_head(latent_features)      # [BC, num_patches]
        # Upsample to step grid (learned linear along the patch axis)
        trend_s = self.trend_upsample(trend_p)                     # [BC, prediction_length]
        season_s = self.season_upsample(season_p)                  # [BC, prediction_length]
        residual_s = pred_scaled - trend_s - season_s

        pred_scaled = pred_scaled.view(B, C, self.config.prediction_length)
        quantiles_scaled = quantiles_scaled.view(B, C, 5, self.config.prediction_length)
        recon_scaled = recon_scaled.view(B, C, self.config.context_length)
        trend_s = trend_s.view(B, C, self.config.prediction_length)
        season_s = season_s.view(B, C, self.config.prediction_length)
        residual_s = residual_s.view(B, C, self.config.prediction_length)
        # Patch-grid scaled trend (unit scale) for scale-invariant smoothness loss
        trend_scaled_patches = trend_p.view(B, C, self.num_patches)

        pred = InstanceRobustScaler.inverse_transform(pred_scaled, median, iqr)
        quantiles = InstanceRobustScaler.inverse_transform(quantiles_scaled, median, iqr)
        reconstructed = InstanceRobustScaler.inverse_transform(recon_scaled, median, iqr)

        trend = trend_s * iqr + median
        seasonal = season_s * iqr
        residual = residual_s * iqr

        return {
            "forecast": pred,
            "quantiles": quantiles,
            "reconstructed": reconstructed,
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "trend_scaled_patches": trend_scaled_patches,
        }
