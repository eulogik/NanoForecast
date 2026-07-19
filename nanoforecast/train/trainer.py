import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, List, Optional

from nanoforecast.model.core import NanoForecast
from nanoforecast.train.loss import NanoForecastLoss, MultiTaskLoss


class NanoForecastTrainer:
    """Standard Trainer for NanoForecast models.
    Supports gradient clipping, AMP, checkpoint saving, and validation loops.
    """
    def __init__(
        self,
        model: NanoForecast,
        loss_fn: NanoForecastLoss | MultiTaskLoss,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        clip_grad: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        device: Optional[torch.device] = None
    ):
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.clip_grad = clip_grad
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = None

    def _autocast_settings(self):
        if isinstance(self.device, torch.device):
            if self.device.type == "cuda":
                return "cuda", torch.bfloat16
            if self.device.type == "mps":
                return "cpu", torch.float32
        return "cpu", torch.float32

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.train()
        total_losses = {}
        num_batches = len(dataloader)

        device_type, autocast_dtype = self._autocast_settings()
        use_amp = autocast_dtype in (torch.bfloat16, torch.float16)

        for batch in dataloader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            freq_ids = batch["freq_id"].to(self.device)
            covariates = (
                batch["covariates"].to(self.device)
                if "covariates" in batch else None
            )
            horizons = batch.get("horizon")

            self.optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast(
                    device_type=device_type, dtype=autocast_dtype
                ):
                    outputs = self.model(x, freq_ids, covariates)
                    loss, loss_dict = self._compute_loss(
                        outputs, y, x, horizons
                    )
            else:
                outputs = self.model(x, freq_ids, covariates)
                loss, loss_dict = self._compute_loss(
                    outputs, y, x, horizons
                )

            loss.backward()

            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad
                )

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v

        for k in total_losses:
            total_losses[k] /= num_batches

        return total_losses

    @torch.no_grad()
    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_losses = {}
        num_batches = len(dataloader)

        device_type, autocast_dtype = self._autocast_settings()
        use_amp = autocast_dtype in (torch.bfloat16, torch.float16)

        for batch in dataloader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            freq_ids = batch["freq_id"].to(self.device)
            covariates = (
                batch["covariates"].to(self.device)
                if "covariates" in batch else None
            )
            horizons = batch.get("horizon")

            if use_amp:
                with torch.amp.autocast(
                    device_type=device_type, dtype=autocast_dtype
                ):
                    outputs = self.model(x, freq_ids, covariates)
                    _, loss_dict = self._compute_loss(
                        outputs, y, x, horizons
                    )
            else:
                outputs = self.model(x, freq_ids, covariates)
                _, loss_dict = self._compute_loss(
                    outputs, y, x, horizons
                )

            for k, v in loss_dict.items():
                val_key = f"val_{k}"
                total_losses[val_key] = total_losses.get(val_key, 0.0) + v

        for k in total_losses:
            total_losses[k] /= num_batches

        return total_losses

    def _compute_loss(self, outputs, y, x, horizons=None):
        """Compute loss with per-sample truncation for multi-horizon."""
        if horizons is None:
            # Legacy path: truncate to padded y length
            target_h = y.shape[-1]
            outputs = self._truncate_to_horizon(outputs, target_h)
            return self.loss_fn(outputs, y, x)

        # Multi-horizon: per-sample loss, then average
        B = y.shape[0]
        losses = []
        mses = []
        quantiles = []
        for i in range(B):
            h = int(horizons[i].item())
            out_i = {k: v[i:i+1, ..., :h] if v.dim() >= 3 else v[i:i+1]
                     for k, v in outputs.items()
                     if k not in ("loc", "scale")}
            out_i["loc"] = outputs["loc"][i:i+1]
            out_i["scale"] = outputs["scale"][i:i+1]
            y_i = y[i:i+1, ..., :h]
            x_i = x[i:i+1]
            loss_i, d_i = self.loss_fn(out_i, y_i, x_i)
            losses.append(loss_i)
            mses.append(d_i.get("loss_mse", 0))
            quantiles.append(d_i.get("loss_quantile", 0))

        total = torch.stack(losses).mean()
        return total, {
            "loss_total": total.item(),
            "loss_mse": sum(mses) / len(mses),
            "loss_quantile": sum(quantiles) / len(quantiles),
        }

    @staticmethod
    def _truncate_to_horizon(
        outputs: Dict[str, torch.Tensor], h: int
    ) -> Dict[str, torch.Tensor]:
        """Truncate prediction tensors to actual horizon (multi-horizon)."""
        out = {}
        for k, v in outputs.items():
            if k in (
                "forecast", "forecast_scaled", "quantiles",
                "quantiles_scaled", "trend", "seasonal", "residual",
            ) and v.shape[-1] > h:
                out[k] = v[..., :h]
            else:
                out[k] = v
        return out

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 5
    ) -> List[Dict[str, float]]:
        """Runs the training curriculum loop."""
        steps_per_epoch = len(train_loader)
        base_lr = self.optimizer.param_groups[0]["lr"]
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=min(base_lr * 3, 2e-4),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos"
        )

        history = []
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            history.append(metrics)

            # Handle both loss formats
            vl = metrics.get(
                "val_loss_point", metrics.get("val_loss_mse", 0)
            )
            vq = metrics.get("val_loss_quantile", 0)
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Loss: {metrics['loss_total']:.4f} | "
                f"Val Loss: {metrics['val_loss_total']:.4f} | "
                f"Val MSE: {vl:.4f} | "
                f"Val Q: {vq:.4f}"
            )

            val_loss = metrics["val_loss_total"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, "best_model.pt"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": self.model.config
                }, checkpoint_path)
                print(
                    f"--> Saved best model checkpoint to {checkpoint_path}"
                )

        return history
