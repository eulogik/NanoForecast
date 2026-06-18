"""Hugging Face Hub integration and high-level predict API for NanoForecast.

This module owns the artifact layout for pretrained checkpoints and the
one-call ``predict()`` interface that hides tensor reshaping from users.

Artifact layout (a directory):
    config.json
    scaler_stats.json   (optional: per-channel mean/std fallback stats)
    model.safetensors
    model_card.json     (benchmark results + training metadata)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from nanoforecast.config import NanoForecastConfig

if TYPE_CHECKING:
    from nanoforecast.model.core import NanoForecast


_FREQ_ALIASES = {
    "H": 1, "h": 1, "hourly": 1,
    "D": 2, "d": 2, "daily": 2,
    "W": 3, "w": 3, "weekly": 3,
    "M": 4, "m": 4, "monthly": 4,
    "T": 0, "min": 0, "5min": 0, "5-min": 0, "5minutely": 0,
    "15min": 0, "15-min": 0,
}


def _resolve_freq(freq: Union[str, int, None], default: int = 1) -> int:
    if freq is None:
        return default
    if isinstance(freq, int):
        return freq
    if freq in _FREQ_ALIASES:
        return _FREQ_ALIASES[freq]
    raise ValueError(f"Unknown frequency alias: {freq!r}. Use one of {list(_FREQ_ALIASES)} or an int freq_id.")


def _to_tensor(x: Union[np.ndarray, torch.Tensor, List[float]], dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    arr = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(arr).to(dtype)


class NanoForecastHubMixin:
    """Mixin providing Hugging Face-style save/load and predict for NanoForecast."""

    # ---------------- Save / Load ----------------
    def save_pretrained(self, save_directory: str) -> None:
        """Save model weights, config, and a stub model_card.json to ``save_directory``."""
        os.makedirs(save_directory, exist_ok=True)
        config = self.config  # type: ignore[attr-defined]
        with open(os.path.join(save_directory, "config.json"), "w") as fh:
            json.dump(asdict(config), fh, indent=2)
        # Prefer safetensors if available, else fall back to torch.save.
        weights_path = os.path.join(save_directory, "model.safetensors")
        try:
            from safetensors.torch import save_file
            state = {k: v.contiguous() for k, v in self.state_dict().items()}  # type: ignore[attr-defined]
            save_file(state, weights_path)
        except Exception:
            torch.save(self.state_dict(), os.path.join(save_directory, "model.pt"))  # type: ignore[attr-defined]

        # Stub model card with placeholders; users (or pretrain.py) overwrite it.
        card_path = os.path.join(save_directory, "model_card.json")
        if not os.path.exists(card_path):
            with open(card_path, "w") as fh:
                json.dump({
                    "model_name": "NanoForecast",
                    "params": sum(p.numel() for p in self.parameters() if p.requires_grad),
                    "config": asdict(config),
                    "benchmarks": {},
                }, fh, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str,
        map_location: Union[str, torch.device, Dict[str, str]] = "cpu",
    ):
        """Load a NanoForecast model from a local directory or a Hugging Face Hub repo.

        Examples:
            model = NanoForecast.from_pretrained("checkpoints/nanoforecast-200k")
            model = NanoForecast.from_pretrained("eulogik/nanoforecast-200k")
        """
        # Resolve local path or download from the Hub
        if os.path.isdir(repo_or_path):
            load_dir = repo_or_path
        else:
            from huggingface_hub import snapshot_download
            load_dir = snapshot_download(repo_id=repo_or_path)

        with open(os.path.join(load_dir, "config.json"), "r") as fh:
            cfg_dict = json.load(fh)
        config = NanoForecastConfig(**cfg_dict)
        model = cls(config)

        safetensors_path = os.path.join(load_dir, "model.safetensors")
        pt_path = os.path.join(load_dir, "model.pt")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state = load_file(safetensors_path)
        else:
            state = torch.load(pt_path, map_location=map_location, weights_only=False)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    # ---------------- Predict ----------------
    @torch.no_grad()
    def predict(
        self,
        context: Union[np.ndarray, torch.Tensor, List[float]],
        horizon: int = 48,
        freq: Union[str, int, None] = "H",
        return_components: bool = True,
        num_samples: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Run a single-call forecast on a 1-D or 2-D context.

        Args:
            context: 1-D array of length ``context_length``, or 2-D ``(B, context_length)``,
                or 3-D ``(B, C, context_length)``.
            horizon: number of future steps to predict.
            freq: frequency alias (e.g. "H", "D") or integer freq_id.
            return_components: include trend/seasonal/residual arrays.
            num_samples: number of stochastic sample draws (1 = deterministic).

        Returns:
            dict with keys:
                forecast: ``(..., horizon)`` point forecast.
                quantiles: ``(..., Q, horizon)`` quantile forecasts (Q=5 by default).
                trend, seasonal, residual: same shape as forecast (only if ``return_components``).
        """
        device = next(self.parameters()).device
        ctx = _to_tensor(context, dtype=torch.float32)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        elif ctx.ndim == 2:
            ctx = ctx.unsqueeze(1)              # (B, 1, L)
        ctx = ctx.to(device)

        cfg = self.config  # type: ignore[attr-defined]
        if ctx.shape[-1] != cfg.context_length:
            raise ValueError(
                f"Context length {ctx.shape[-1]} does not match model context_length {cfg.context_length}. "
                f"For arbitrary-length inputs, use model.predict_with_overlap() or retrain with the right config."
            )

        freq_id = _resolve_freq(freq, default=1)
        freq_ids = torch.full((ctx.shape[0],), freq_id, dtype=torch.long, device=device)
        covariates = torch.zeros(
            (ctx.shape[0], cfg.covariate_dim, cfg.context_length),
            dtype=torch.float32,
            device=device,
        ) if cfg.covariate_dim > 0 else None

        out = self(ctx, freq_ids, covariates)
        result: Dict[str, np.ndarray] = {
            "forecast": out["forecast"].squeeze(1).cpu().numpy(),
            "quantiles": out["quantiles"].squeeze(1).cpu().numpy(),
        }
        if return_components:
            result["trend"] = out["trend"].squeeze(1).cpu().numpy()
            result["seasonal"] = out["seasonal"].squeeze(1).cpu().numpy()
            result["residual"] = out["residual"].squeeze(1).cpu().numpy()
        return result


def attach_hub_methods(model_cls: type) -> type:
    """Decorator-like helper that mixes Hub methods onto the model class."""
    for name in ("save_pretrained", "from_pretrained", "predict"):
        setattr(model_cls, name, getattr(NanoForecastHubMixin, name))
    return model_cls
