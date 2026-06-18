"""Benchmark a pretrained NanoForecast on real datasets and write JSON results.

Usage:
    python3 benchmark.py --checkpoint checkpoints/nanoforecast-200k \
        --datasets ETTh1,exchange_rate --output results/benchmark.json

Computes per-dataset MASE / sMAPE / MSE / MAE / CRPS and quantile coverage,
then writes a summary JSON to ``--output``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from nanoforecast.model.core import NanoForecast
from nanoforecast.data.real_datasets import (
    WindowSpec,
    build_mixed_pretraining_corpus,
    time_based_split,
)
from nanoforecast.evaluation.benchmark import TimeSeriesEvaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark a pretrained NanoForecast")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Local path or HF repo id for a pretrained NanoForecast")
    p.add_argument("--datasets", type=str, default="ETTh1,exchange_rate")
    p.add_argument("--context-length", type=int, default=None,
                   help="Override config context length (must match model)")
    p.add_argument("--prediction-length", type=int, default=None)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--max-channels", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--max-windows", type=int, default=64,
                   help="Cap windows per dataset for fast CPU benchmarking")
    p.add_argument("--output", type=str, default="results/benchmark.json")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


@torch.no_grad()
def predict_window(model: NanoForecast, context: np.ndarray, freq_id: int) -> Dict[str, np.ndarray]:
    out = model.predict(context=context, horizon=model.config.prediction_length, freq=freq_id)
    return out


def benchmark_dataset(
    model: NanoForecast,
    dataset: str,
    args,
) -> Dict[str, float]:
    spec = WindowSpec(
        context_len=args.context_length or model.config.context_length,
        prediction_len=args.prediction_length or model.config.prediction_length,
        stride=args.stride,
    )
    records = build_mixed_pretraining_corpus(
        spec, datasets=[dataset], max_channels_per_dataset=args.max_channels,
    )
    _, val_records = time_based_split(records, val_fraction=args.val_fraction)
    val_records = val_records[: args.max_windows]

    contexts, targets, forecasts, quantiles = [], [], [], []
    for rec in val_records:
        out = predict_window(model, rec["context"], rec["freq_id"])
        forecasts.append(out["forecast"].squeeze())
        quantiles.append(out["quantiles"].squeeze())  # (Q, H)
        contexts.append(rec["context"])
        targets.append(rec["prediction"])

    evaluator = TimeSeriesEvaluator()
    metrics = evaluator.evaluate_batch(
        contexts=contexts,
        targets=targets,
        forecasts=forecasts,
        quantiles=quantiles,
        quantile_levels=model.config.quantiles,
    )
    return metrics


def main() -> None:
    args = parse_args()

    print(f"Loading model from {args.checkpoint} ...")
    model = NanoForecast.from_pretrained(args.checkpoint, map_location=args.device)
    model.eval()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    summary: Dict[str, Dict] = {
        "checkpoint": args.checkpoint,
        "config": {
            "context_length": model.config.context_length,
            "prediction_length": model.config.prediction_length,
            "patch_size": model.config.patch_size,
            "d_model": model.config.d_model,
            "num_layers": model.config.num_layers,
            "quantiles": list(model.config.quantiles),
        },
        "datasets": {},
    }

    overall_keys = ["mase", "smape", "mse", "mae", "crps"]
    accum = {k: [] for k in overall_keys}

    for ds in datasets:
        print(f"\nBenchmarking on {ds} ...")
        try:
            metrics = benchmark_dataset(model, ds, args)
        except Exception as e:
            print(f"  ! failed: {e}")
            summary["datasets"][ds] = {"error": str(e)}
            continue
        summary["datasets"][ds] = metrics
        for k in overall_keys:
            accum[k].append(metrics[k])
        print(f"  MASE={metrics['mase']:.4f}  sMAPE={metrics['smape']:.2f}%  "
              f"MSE={metrics['mse']:.4f}  MAE={metrics['mae']:.4f}  CRPS={metrics['crps']:.4f}")

    if accum["mase"]:
        summary["overall"] = {k: float(np.mean(v)) for k, v in accum.items() if v}
        print("\nOVERALL:")
        for k, v in summary["overall"].items():
            print(f"  {k.upper():>6}: {v:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote benchmark results to {args.output}")


if __name__ == "__main__":
    main()
