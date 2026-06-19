import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, List, Optional, Tuple

from nanoforecast.model.core import NanoForecast
from nanoforecast.train.loss import MultiTaskLoss

class NanoForecastTrainer:
    """
    Standard Trainer for NanoForecast models.
    Supports gradient clipping, AMP (Automatic Mixed Precision),
    checkpoint saving, and validation loops.
    """
    def __init__(
        self,
        model: NanoForecast,
        loss_fn: MultiTaskLoss,
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

        # Optimizer Setup
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = None # Set up dynamically based on epochs & steps in fit()

    def _autocast_settings(self):
        """Return (device_type, autocast_dtype) for the active device."""
        if isinstance(self.device, torch.device):
            if self.device.type == "cuda":
                return "cuda", torch.bfloat16
            if self.device.type == "mps":
                return "cpu", torch.float32  # MPS bf16 autocast is unstable
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
            covariates = batch["covariates"].to(self.device) if "covariates" in batch else None
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision (disabled for float32)
            if use_amp:
                with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype):
                    outputs = self.model(x, freq_ids, covariates)
                    loss, loss_dict = self.loss_fn(outputs, y, x)
            else:
                outputs = self.model(x, freq_ids, covariates)
                loss, loss_dict = self.loss_fn(outputs, y, x)
                
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Accumulate losses
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
                
        # Average losses
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
            covariates = batch["covariates"].to(self.device) if "covariates" in batch else None
            
            if use_amp:
                with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype):
                    outputs = self.model(x, freq_ids, covariates)
                    _, loss_dict = self.loss_fn(outputs, y, x)
            else:
                outputs = self.model(x, freq_ids, covariates)
                _, loss_dict = self.loss_fn(outputs, y, x)
                
            for k, v in loss_dict.items():
                val_key = f"val_{k}"
                total_losses[val_key] = total_losses.get(val_key, 0.0) + v
                
        for k in total_losses:
            total_losses[k] /= num_batches
            
        return total_losses

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 5
    ) -> List[Dict[str, float]]:
        """
        Runs the training curriculum loop for a specified number of epochs.
        """
        # Initialize OneCycleLR scheduler
        steps_per_epoch = len(train_loader)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]["lr"] * 10,
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
            
            # Print epoch summary
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Loss: {metrics['loss_total']:.4f} | "
                f"Val Loss: {metrics['val_loss_total']:.4f} | "
                f"Val Point: {metrics['val_loss_point']:.4f} | "
                f"Val Quant: {metrics['val_loss_quantile']:.4f}"
            )
            
            # Save checkpoint if best validation loss
            val_loss = metrics["val_loss_total"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": self.model.config
                }, checkpoint_path)
                print(f"--> Saved best model checkpoint to {checkpoint_path}")
                
        return history
