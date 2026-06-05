import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from nanoforecast.config import NanoForecastConfig
from nanoforecast.model.utils import InstanceRobustScaler, ResolutionPrefixEmbedding, AdaptivePatching
from nanoforecast.model.blocks import SequenceMixingBlock
from nanoforecast.model.heads import PointForecastHead, MonotonicQuantileHead, AnomalyDetectionHead, DecompositionHead

class NanoForecast(nn.Module):
    """
    NanoForecast Model: The world's smallest time series foundation model.
    Integrates Robust scaling, Adaptive Patching, Resolution Prefixes,
    Gated sequence mixing blocks (Conv, RNN, MLP), and multi-task heads.
    """
    def __init__(self, config: NanoForecastConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.d_model = config.d_model
        
        # Scaling & Input Processing
        self.scaler = InstanceRobustScaler()
        self.patcher = AdaptivePatching(self.patch_size, self.d_model)
        
        # Resolution Embedding
        self.freq_embedder = ResolutionPrefixEmbedding(config.num_frequencies, self.d_model)
        
        # Exogenous Covariate Projection
        if config.covariate_dim > 0:
            self.covariate_projection = nn.Linear(config.covariate_dim * self.patch_size, self.d_model)
            
        # Calculate number of patches in context sequence
        # We need to account for padding that may be added to context_length
        rem = config.context_length % self.patch_size
        padded_context_len = config.context_length + (self.patch_size - rem if rem != 0 else 0)
        self.num_patches = padded_context_len // self.patch_size
        
        # Sequence Mixing Layers
        # Total sequence length seen by layers is num_patches + 1 (due to prepended resolution prefix)
        self.seq_len_with_prefix = self.num_patches + 1
        self.layers = nn.ModuleList([
            SequenceMixingBlock(
                seq_len=self.seq_len_with_prefix,
                d_model=self.d_model,
                expansion_factor=config.expansion_factor,
                dropout=config.dropout,
                use_router=config.use_gated_router
            ) for _ in range(config.num_layers)
        ])
        
        # Norm before heads
        self.final_norm = nn.RMSNorm(self.d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(self.d_model)
        
        # Output Heads
        self.point_head = PointForecastHead(self.num_patches, self.d_model, config.prediction_length)
        self.quantile_head = MonotonicQuantileHead(self.num_patches, self.d_model, config.prediction_length)
        self.anomaly_head = AnomalyDetectionHead(self.num_patches, self.d_model, config.context_length)
        self.decomp_head = DecompositionHead(self.num_patches, self.d_model, config.prediction_length)

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
        
        # 1. Scale Input Time Series Robustly
        x_scaled, median, iqr = self.scaler(x) # x_scaled shape: [B, C, L]
        
        # 2. Patch Input Time Series
        # patches shape: [B * C, num_patches, d_model]
        patches, padding_len = self.patcher(x_scaled)
        
        # 3. Handle Covariates if provided
        if covariates is not None and self.config.covariate_dim > 0:
            # covariates shape: [B, cov_dim, L]
            # Zero-pad context covariates if target sequence was padded in patcher
            if padding_len > 0:
                covariates = torch.nn.functional.pad(covariates, (0, padding_len), mode="constant", value=0)
            
            # Reshape covariates to match target patches: [B * C, num_patches, cov_dim * patch_size]
            # Since covariates are global (batch-level), repeat them for each channel C
            cov_dim = covariates.shape[1]
            cov_unfolded = covariates.view(B, cov_dim, self.num_patches, self.patch_size)
            cov_unfolded = cov_unfolded.permute(0, 2, 1, 3).contiguous() # [B, num_patches, cov_dim, patch_size]
            cov_unfolded = cov_unfolded.view(B, self.num_patches, cov_dim * self.patch_size)
            
            # Repeat covariates for channels: [B * C, num_patches, cov_dim * patch_size]
            cov_repeated = cov_unfolded.repeat_interleave(C, dim=0)
            
            # Project covariates and add to patches
            patches = patches + self.covariate_projection(cov_repeated)
            
        # 4. Generate Resolution Prefix
        # freq_ids shape: [B] -> repeat for each channel to match BC: [B * C]
        freq_ids_repeated = freq_ids.repeat_interleave(C, dim=0)
        prefix = self.freq_embedder(freq_ids_repeated) # [B * C, 1, d_model]
        
        # Prepend resolution prefix token to sequence: [B * C, num_patches + 1, d_model]
        seq = torch.cat([prefix, patches], dim=1)
        
        # 5. Run sequence mixing layers
        for layer in self.layers:
            seq = layer(seq)
            
        seq = self.final_norm(seq)
        
        # 6. Extract sequence representation tokens (discarding prefix token)
        # shape: [B * C, num_patches, d_model]
        latent_features = seq[:, 1:, :]
        
        # 7. Run Output Heads
        pred_scaled = self.point_head(latent_features)            # [B * C, prediction_length]
        quantiles_scaled = self.quantile_head(latent_features)    # [B * C, 5, prediction_length]
        recon_scaled = self.anomaly_head(latent_features)          # [B * C, context_length]
        trend_s, season_s = self.decomp_head(latent_features)     # [B * C, prediction_length] each
        residual_s = pred_scaled - trend_s - season_s             # Guarantees reconstruction identity
        
        # Reshape to [B, C, ...]
        pred_scaled = pred_scaled.view(B, C, self.config.prediction_length)
        quantiles_scaled = quantiles_scaled.view(B, C, 5, self.config.prediction_length)
        recon_scaled = recon_scaled.view(B, C, self.config.context_length)
        trend_s = trend_s.view(B, C, self.config.prediction_length)
        season_s = season_s.view(B, C, self.config.prediction_length)
        residual_s = residual_s.view(B, C, self.config.prediction_length)
        
        # 8. Inverse scale all outputs to restore original magnitudes
        pred = InstanceRobustScaler.inverse_transform(pred_scaled, median, iqr)
        quantiles = InstanceRobustScaler.inverse_transform(quantiles_scaled, median, iqr)
        reconstructed = InstanceRobustScaler.inverse_transform(recon_scaled, median, iqr)
        
        # Trend inherits scaling & level offset (median), while seasonality & residual are scaled only (zero-mean)
        trend = trend_s * iqr + median
        seasonal = season_s * iqr
        residual = residual_s * iqr
        
        return {
            "forecast": pred,
            "quantiles": quantiles,
            "reconstructed": reconstructed,
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual
        }
