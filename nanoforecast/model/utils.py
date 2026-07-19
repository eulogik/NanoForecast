import torch
import torch.nn as nn
from typing import Tuple, Dict


class DARTNorm(nn.Module):
    """Drift-Aware Rolling Timescale Normalization (Reverso-style).

    Computes mean/std from the *context* window only, so normalization
    statistics never leak future information.  During streaming inference
    the context window is the same as training — no adaptation needed.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [B, C, L] (context window only).
        Returns:
            x_scaled, mean, std — all [B, C, ...] shaped for inverse transform.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        std = torch.clamp(std, min=self.eps)
        x_scaled = (x - mean) / std
        return x_scaled, mean, std

    @staticmethod
    def inverse_transform(
        x_scaled: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Restore original scale. Handles quantile dim (4D scaled, 3D stats)."""
        if x_scaled.dim() == 4 and mean.dim() == 3:
            mean = mean.unsqueeze(2)
            std = std.unsqueeze(2)
        return x_scaled * std + mean


class InstanceRobustScaler(nn.Module):
    """Legacy median/IQR scaler — kept for backward compat with v0.1-v0.3."""
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sorted_x, _ = torch.sort(x, dim=-1)
        seq_len = x.shape[-1]
        idx_50 = seq_len // 2
        median = sorted_x[..., idx_50:idx_50+1]
        idx_25 = int(0.25 * seq_len)
        idx_75 = int(0.75 * seq_len)
        q25 = sorted_x[..., idx_25:idx_25+1]
        q75 = sorted_x[..., idx_75:idx_75+1]
        iqr = q75 - q25
        iqr = torch.clamp(iqr, min=self.eps)
        x_scaled = (x - median) / iqr
        return x_scaled, median, iqr

    @staticmethod
    def inverse_transform(
        x_scaled: torch.Tensor, median: torch.Tensor, iqr: torch.Tensor
    ) -> torch.Tensor:
        if x_scaled.dim() == 4 and median.dim() == 3:
            median = median.unsqueeze(2)
            iqr = iqr.unsqueeze(2)
        return x_scaled * iqr + median


class ResolutionPrefixEmbedding(nn.Module):
    """
    Learned frequency embeddings (Resolution Prefix) to tell the model
    the time scale of the sequence (e.g. hourly, daily, weekly).
    """
    def __init__(self, num_frequencies: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_frequencies, d_model)

    def forward(self, freq_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            freq_ids: Tensor of shape [batch_size] containing frequency IDs
        Returns:
            prefix_embed: Tensor of shape [batch_size, 1, d_model]
        """
        embeds = self.embedding(freq_ids)  # [batch_size, d_model]
        return embeds.unsqueeze(1)         # [batch_size, 1, d_model]


class AdaptivePatching(nn.Module):
    """
    Partitions a time series into patches and projects them into d_model.
    """
    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: Input tensor of shape [batch_size, channels, seq_len]
        Returns:
            patches: Projected patches of shape [batch_size * channels, num_patches, d_model]
            padding_len: Amount of padding added to the input sequence
        """
        batch_size, channels, seq_len = x.shape

        padding_len = 0
        rem = seq_len % self.patch_size
        if rem != 0:
            padding_len = self.patch_size - rem
            x = torch.nn.functional.pad(x, (0, padding_len), mode="replicate")

        new_seq_len = seq_len + padding_len
        num_patches = new_seq_len // self.patch_size

        patches = x.view(batch_size, channels, num_patches, self.patch_size)
        patches = patches.contiguous().view(batch_size * channels, num_patches, self.patch_size)
        projected = self.projection(patches)

        return projected, padding_len


class PatchPositionalEncoding(nn.Module):
    """
    Learned positional embeddings for the patch index axis (0..max_patches-1).
    A separate zero embedding is provided for the prepended resolution prefix token
    so the prefix remains untouched.
    """
    def __init__(self, max_patches: int, d_model: int):
        super().__init__()
        # Index 0 is the prefix slot; patches are 1..max_patches
        self.pe = nn.Embedding(max_patches + 1, d_model)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
        # Zero out the prefix slot
        with torch.no_grad():
            self.pe.weight[0].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: Tensor of shape [B, num_patches + 1, d_model]
        Returns:
            seq with positional encoding added.
        """
        positions = torch.arange(seq.shape[1], device=seq.device, dtype=torch.long)
        return seq + self.pe(positions)
