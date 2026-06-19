"""Pretrain NanoForecast on a mixed real + synthetic corpus and save the artifact.

Usage:
    python3 pretrain.py --datasets ETTh1,exchange_rate --epochs 5 --output checkpoints/nanoforecast-200k

Defaults to a small Nano-200K profile for fast CPU runs. The artifact is
written as a HF-Hub-style directory (config.json + model.safetensors + model_card.json).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import List, Optional

import numpy as np
import torch

from nanoforecast.config import NanoForecastConfig
from nanoforecast.model.core import NanoForecast
from nanoforecast.data.generator import SyntheticTimeSeriesGenerator
from nanoforecast.data.real_datasets import (
    WindowSpec,
    build_mixed_pretraining_corpus,
    time_based_split,
)
from nanoforecast.data.pipeline import create_dataloader
from nanoforecast.train.loss import MultiTaskLoss
from nanoforecast.train.trainer import NanoForecastTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain NanoForecast on real + synthetic data")
    p.add_argument("--datasets", type=str, default="ETTh1,exchange_rate",
                   help="Comma-separated real dataset names to mix in")
    p.add_argument("--synthetic-records", type=int, default=400,
                   help="Number of synthetic series to mix in (0 to disable)")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--prediction-length", type=int, default=48)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--max-channels", type=int, default=4)
    p.add_argument("--output", type=str, default="checkpoints/nanoforecast-200k")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None,
                   help="Device override: 'cpu', 'cuda', 'mps', or 'auto' (default).")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(spec: Optional[str]) -> Optional[torch.device]:
    if spec is None or spec == "auto":
        return None  # let the trainer pick
    spec = spec.lower()
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda")
    if spec == "mps":
        return torch.device("mps")
    raise ValueError(f"Unknown device: {spec!r}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print("NANOFORECAST PRETRAIN")
    print("=" * 70)
    print(f"  datasets        : {args.datasets}")
    print(f"  synthetic recs  : {args.synthetic_records}")
    print(f"  context x patch : {args.context_length} x {args.patch_size}")
    print(f"  d_model, layers : {args.d_model}, {args.num_layers}")
    print(f"  epochs, batch   : {args.epochs}, {args.batch_size}")
    print(f"  output dir      : {args.output}")

    config = NanoForecastConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        patch_size=args.patch_size,
        covariate_dim=4,
    )

    # --- Build corpus ---
    spec = WindowSpec(context_len=args.context_length,
                      prediction_len=args.prediction_length,
                      stride=args.stride)
    real_records: List[dict] = []
    if args.datasets:
        ds_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
        real_records = build_mixed_pretraining_corpus(
            spec, datasets=ds_names, max_channels_per_dataset=args.max_channels,
        )

    syn_records: List[dict] = []
    if args.synthetic_records > 0:
        gen = SyntheticTimeSeriesGenerator(seed=args.seed)
        syn_records = gen.generate_dataset(
            num_series=args.synthetic_records,
            context_len=args.context_length,
            prediction_len=args.prediction_length,
        )

    all_records = real_records + syn_records
    if not all_records:
        raise SystemExit("No training records available. Pass --datasets or --synthetic-records.")

    # Time-based split (each dataset already has its own temporal order; we
    # split the concatenated corpus by index, which keeps channels from
    # being mixed across train/val in a leaked way).
    train_records, val_records = time_based_split(all_records, val_fraction=args.val_fraction)
    print(f"  train records   : {len(train_records)}")
    print(f"  val records     : {len(val_records)}")

    train_loader = create_dataloader(train_records, batch_size=args.batch_size, augment=True, shuffle=True, drop_last=False)
    val_loader = create_dataloader(val_records, batch_size=args.batch_size, augment=False, shuffle=False, drop_last=False)
    print(f"  train batches   : {len(train_loader)}")
    print(f"  val batches     : {len(val_loader)}")

    # --- Model & Loss ---
    model = NanoForecast(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params / 1e3:.2f}K")

    loss_fn = MultiTaskLoss(
        quantiles=config.quantiles,
        w_point=0.5,
        w_quantile=1.0,
        w_anomaly=0.1,
        w_smooth=0.05,
    )

    trainer = NanoForecastTrainer(
        model=model,
        loss_fn=loss_fn,
        lr=args.lr,
        checkpoint_dir=os.path.dirname(args.output) or "checkpoints",
        device=_resolve_device(args.device),
    )
    print(f"  device          : {trainer.device}")

    t0 = time.time()
    trainer.fit(train_loader, val_loader, epochs=args.epochs)
    dt = time.time() - t0
    print(f"\nTraining finished in {dt:.1f}s ({dt / max(1, args.epochs):.1f}s/epoch)")

    # --- Save HF-Hub-style artifact ---
    # We re-attach the best checkpoint first (the trainer already wrote it).
    best_ckpt = os.path.join(trainer.checkpoint_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_epoch = int(ckpt.get("epoch", -1))
        best_val = float(ckpt.get("val_loss", float("nan")))
        print(f"  best epoch      : {best_epoch} (val_loss={best_val:.4f})")
    else:
        best_epoch, best_val = -1, float("nan")

    model.eval()
    model.save_pretrained(args.output)

    card = {
        "model_name": "NanoForecast",
        "profile": "nano-200k" if (config.d_model, config.num_layers) == (32, 4) else f"d{config.d_model}-L{config.num_layers}",
        "params": n_params,
        "config": {
            "context_length": config.context_length,
            "prediction_length": config.prediction_length,
            "patch_size": config.patch_size,
            "d_model": config.d_model,
            "num_layers": config.num_layers,
            "quantiles": list(config.quantiles),
        },
        "training": {
            "datasets": [d.strip() for d in args.datasets.split(",") if d.strip()],
            "synthetic_records": args.synthetic_records,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "wall_time_s": dt,
        },
    }
    with open(os.path.join(args.output, "model_card.json"), "w") as fh:
        json.dump(card, fh, indent=2)
    print(f"\nSaved pretrained artifact to: {args.output}")
    print("  - config.json")
    print("  - model.safetensors  (or model.pt if safetensors missing)")
    print("  - model_card.json")


if __name__ == "__main__":
    main()
