"""Train NanoForecast on your own CSV data — the primary UX path.

Usage:
    python3 train_from_csv.py --csv sales.csv --target revenue --horizon 30

Trains a model on the full CSV, generates a forecast, and saves the
checkpoint so you can deploy it or continue streaming.

No GPU required — runs in ~2 minutes on a MacBook Air.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch

from nanoforecast import NanoForecast, NanoForecastConfig
from nanoforecast.data.generator import SyntheticTimeSeriesGenerator
from nanoforecast.data.pipeline import create_dataloader
from nanoforecast.data.real_datasets import WindowSpec, time_based_split
from nanoforecast.train.loss import MultiTaskLoss
from nanoforecast.train.trainer import NanoForecastTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NanoForecast on a CSV file")
    p.add_argument("--csv", required=True, help="Path to CSV file")
    p.add_argument("--target", required=True, help="Name of the target column to forecast")
    p.add_argument("--date-col", default=None, help="Optional date column name (for display)")
    p.add_argument("--horizon", type=int, default=48, help="Forecast horizon (steps)")
    p.add_argument("--context", type=int, default=256, help="Context window length")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs")
    p.add_argument("--output", default="checkpoints/nanoforecast-custom", help="Output directory")
    p.add_argument("--plot", action="store_true", help="Show a matplotlib plot of the forecast")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("NanoForecast — Train on Your Data")
    print("=" * 60)
    print(f"  CSV:     {args.csv}")
    print(f"  Target:  {args.target}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Context: {args.context}")
    print()

    # --- 1. Load CSV ---
    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        print(f"❌ Column '{args.target}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    series = df[args.target].dropna().values.astype(np.float32)
    if len(series) < args.context + args.horizon:
        print(f"❌ Need at least {args.context + args.horizon} rows, got {len(series)}")
        sys.exit(1)

    print(f"  Loaded {len(series)} timesteps")
    print()

    # --- 2. Build sliding-window training records ---
    stride = max(1, args.context // 4)
    records = []
    for start in range(0, len(series) - args.context - args.horizon + 1, stride):
        ctx = series[start : start + args.context]
        pred = series[start + args.context : start + args.context + args.horizon]
        records.append({
            "context": ctx,
            "prediction": pred,
            "freq_id": 1,
            "context_covariates": np.zeros((4, args.context), dtype=np.float32),
            "prediction_covariates": np.zeros((4, args.horizon), dtype=np.float32),
        })

    print(f"  Created {len(records)} training windows")
    print()

    # --- 3. Augment with synthetic data for robustness ---
    gen = SyntheticTimeSeriesGenerator(seed=42)
    syn = gen.generate_dataset(
        num_series=max(200, len(records) // 2),
        context_len=args.context,
        prediction_len=args.horizon,
    )
    all_records = records + syn
    train_records, val_records = time_based_split(all_records, val_fraction=0.15)

    print(f"  Train windows: {len(train_records)} (real + synthetic)")
    print(f"  Val windows:   {len(val_records)}")
    print()

    # --- 4. Create model ---
    config = NanoForecastConfig(
        context_length=args.context,
        prediction_length=args.horizon,
        d_model=64,
        num_layers=6,
        patch_size=8,
        covariate_dim=4,
    )
    model = NanoForecast(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params / 1e3:.1f}K")
    print()

    # --- 5. Train ---
    train_loader = create_dataloader(
        train_records, batch_size=32, augment=True, shuffle=True, drop_last=False
    )
    val_loader = create_dataloader(
        val_records, batch_size=32, augment=False, shuffle=False, drop_last=False
    )

    loss_fn = MultiTaskLoss(
        quantiles=config.quantiles,
        w_point=0.5, w_quantile=1.0, w_anomaly=0.1, w_smooth=0.05,
    )
    trainer = NanoForecastTrainer(
        model=model, loss_fn=loss_fn, lr=3e-4,
        checkpoint_dir=os.path.dirname(args.output) or "checkpoints",
    )

    t0 = time.time()
    trainer.fit(train_loader, val_loader, epochs=args.epochs)
    dt = time.time() - t0
    print(f"\n  Training finished in {dt:.1f}s ({dt/args.epochs:.1f}s/epoch)")

    # --- 6. Save ---
    best_ckpt = os.path.join(trainer.checkpoint_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Best val_loss: {ckpt.get('val_loss', 'N/A'):.4f}")

    model.eval()
    model.save_pretrained(args.output)
    print(f"  Saved model to: {args.output}")
    print()

    # --- 7. Forecast the last segment ---
    ctx = series[-args.context:]
    result = model.predict(ctx, horizon=args.horizon, freq=1, return_components=True)
    fc = result["forecast"][0]

    print("  Forecast (first 10):", np.round(fc[:10], 4).tolist())

    # Write forecast CSV
    fc_path = os.path.join(args.output, "forecast.csv")
    idx_col = args.date_col or "step"
    fc_df = pd.DataFrame({
        idx_col: range(1, args.horizon + 1),
        "forecast": fc,
        "p10": result["quantiles"][0, :, 0],
        "p50": result["quantiles"][0, :, 2],
        "p90": result["quantiles"][0, :, 4],
    })
    fc_df.to_csv(fc_path, index=False)
    print(f"  Forecast saved to: {fc_path}")

    # --- 8. Plot (optional) ---
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(series, color="#1f77b4", alpha=0.6, label="History")
            future_idx = np.arange(len(series), len(series) + args.horizon)
            ax.plot(future_idx, fc, color="#d62728", label="Forecast")
            ax.fill_between(
                future_idx,
                result["quantiles"][0, :, 0],
                result["quantiles"][0, :, 4],
                alpha=0.2, color="#d62728", label="p10–p90"
            )
            ax.axvline(len(series) - 1, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(f"NanoForecast — {args.target}")
            ax.set_xlabel("Step")
            ax.set_ylabel(args.target)
            ax.legend()
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("  (Install matplotlib for --plot)")

    print()
    print("  Next steps:")
    print(f"    model = NanoForecast.from_pretrained('{args.output}')")
    print("    model.predict(my_data, horizon=H)")
    print("    # or stream: model.predict_step(new_val, state)")
    print("  Done.")


if __name__ == "__main__":
    main()
