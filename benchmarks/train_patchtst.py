"""Train PatchTST baselines for the NanoForecast paper using the official
thuml/Time-Series-Library implementation (vendored under benchmarks/tsl/).

Hyperparameters follow the PatchTST paper (ICLR 2023) headline config:
  d_model 512, d_ff 512, n_heads 8, e_layers 3, dropout 0.3,
  patch_len 16, stride 8, lookback 512, pred_len 48, lr 1e-4 (Adam),
  MSE loss, train_epochs 100, patience 3.

Inputs match the paper protocol exactly:
  * splits    : ETT 70/20/10, other 70/10/20 (same as benchmark_standard.py)
  * ETT       : oil temperature (OT) only, univariate
  * other     : all channels jointly (multivariate, features=M)
  * scaling   : per-channel z-score from TRAIN stats (TSL StandardScaler)

Output: benchmarks/checkpoints/patchtst/<dataset>.pt (state dict + meta json).
Evaluation is done by benchmark_standard.py --models patchtst.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.tsl.PatchTST import Model as PatchTST  # noqa: E402  (vendored TSL)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark_standard import (  # noqa: E402
    CTX, H, TRAIN_FRAC, TRAIN_VAL_FRAC, load_series,
)

SEASONALITY = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "exchange_rate": 7,
               "electricity": 24, "traffic": 24}
BATCH = {"ETTh1": 128, "ETTh2": 128, "ETTm1": 128, "exchange_rate": 64,
         "electricity": 16, "traffic": 8}
# The large multi-series datasets put every channel through one transformer
# (~666 GFLOP/batch for electricity on CPU), so full stride-1 windowing is
# intractable on a laptop. Train on a strided subsample of windows; the
# architecture and all hyperparameters stay official TSL. Recorded in meta.
WINDOW_STRIDE = {"ETTh1": 1, "ETTh2": 1, "ETTm1": 1, "exchange_rate": 1,
                 "electricity": 1024, "traffic": 1024}
VAL_STRIDE = {"ETTh1": 1, "ETTh2": 1, "ETTm1": 1, "exchange_rate": 1,
              "electricity": 128, "traffic": 128}
DEVICE_BY_DATASET = {"electricity": "cpu", "traffic": "cpu"}
LR = 1e-4
MAX_EPOCHS = 100
PATIENCE = 3


class Configs:
    task_name = "long_term_forecast"
    seq_len = CTX
    pred_len = H
    enc_in = 1
    dec_in = 1
    c_out = 1
    d_model = 512
    n_heads = 8
    e_layers = 3
    d_ff = 512
    dropout = 0.3
    factor = 3
    activation = "gelu"
    output_attention = False
    embed = "timeF"
    freq = "h"


def make_model(n_vars: int) -> PatchTST:
    cfg = Configs()
    cfg.enc_in = n_vars
    cfg.dec_in = n_vars
    cfg.c_out = n_vars
    return PatchTST(cfg, patch_len=16, stride=8)


@dataclass
class SplitData:
    x_train: np.ndarray  # (T, C)
    x_val: np.ndarray
    means: np.ndarray    # (C,)
    stds: np.ndarray


def load_split(dataset: str) -> SplitData:
    series = load_series(dataset)
    if dataset.startswith("ETT"):
        x = np.stack(series, axis=1)          # (T, 1) OT only
    else:
        x = np.stack(series, axis=1)          # (T, C)
    n = len(x)
    t0 = int(n * TRAIN_FRAC[dataset])
    t1 = int(n * TRAIN_VAL_FRAC[dataset])
    train = x[:t0].astype(np.float32)
    val = x[t0:t1].astype(np.float32)
    means = train.mean(axis=0, keepdims=True)
    stds = train.std(axis=0, keepdims=True) + 1e-8
    return SplitData((train - means) / stds, (val - means) / stds,
                     means[0], stds[0])


def windows(x: np.ndarray, seq: int, pred: int):
    """Stride-1 windows, TSL style. Returns number of windows and the data.

    Windows are NOT materialized: gathering them all at once for the large
    multi-series datasets (electricity 12.8 GB, traffic 22.6 GB) exceeds RAM
    and pushes the Mac into swap/disk-full aborts. Callers gather the rows
    they need per batch instead.
    """
    n = x.shape[0]
    total = seq + pred
    return max(0, n - total + 1), x


def train_one(dataset: str, device: str) -> dict:
    data = load_split(dataset)
    C = data.x_train.shape[1]
    n_tr, x_tr = windows(data.x_train, CTX, H)
    n_va, x_va = windows(data.x_val, CTX, H)
    total = CTX + H
    model = make_model(C).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    batch = BATCH[dataset]
    out_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "patchtst")
    os.makedirs(out_dir, exist_ok=True)
    resume_path = os.path.join(out_dir, f"{dataset}_resume.pt")

    start_epoch, best_va, best_state, bad = 0, float("inf"), None, 0
    if os.path.exists(resume_path):
        r = torch.load(resume_path, map_location=device)
        model.load_state_dict(r["model"])
        opt.load_state_dict(r["opt"])
        np.random.set_state(r["rng"])
        start_epoch, best_va, bad = r["epoch"], r["best_va"], r["bad"]
        if r["best_state"] is not None:
            best_state = r["best_state"]
        print(f"[{dataset}] resumed from epoch {start_epoch} "
              f"(best va {best_va:.6f})", flush=True)

    def save_resume(epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "rng": np.random.get_state(), "epoch": epoch + 1,
                    "best_va": best_va, "bad": bad, "best_state": best_state},
                   resume_path)

    def gather(x, sel):
        w = x[sel[:, None] + np.arange(total)]
        return (torch.from_numpy(np.ascontiguousarray(w[:, :CTX, :])).to(device),
                torch.from_numpy(np.ascontiguousarray(w[:, CTX:, :])).to(device))

    def evaluate(starts):
        model.eval()
        total_loss, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, starts.shape[0], batch):
                sel = starts[i:i + batch]
                b, tgt = gather(x_va, sel)
                out = model(b, None, None, None)
                total_loss += (loss_fn(out, tgt).item() * out.shape[0])
                cnt += out.shape[0]
        return total_loss / max(cnt, 1)

    tr_stride = WINDOW_STRIDE[dataset]
    va_starts = np.arange(0, n_va, VAL_STRIDE[dataset])
    t0 = time.time()
    for epoch in range(start_epoch, MAX_EPOCHS):
        if device == "mps":
            torch.mps.empty_cache()
        model.train()
        perm = np.random.permutation(n_tr)[::tr_stride]
        tot, cnt = 0.0, 0
        for i in range(0, perm.shape[0], batch):
            sel = perm[i:i + batch]
            b, tgt = gather(x_tr, sel)
            out = model(b, None, None, None)
            loss = loss_fn(out, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * sel.shape[0]
            cnt += sel.shape[0]
        tr_l = tot / cnt
        va_l = evaluate(va_starts)
        msg = f"epoch {epoch:3d} train {tr_l:.6f} val {va_l:.6f}"
        if va_l < best_va:
            best_va = va_l
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            msg += " *"
        else:
            bad += 1
        print(f"[{dataset}] {msg} ({time.time() - t0:.0f}s)", flush=True)
        save_resume(epoch)
        if bad >= PATIENCE:
            break

    if os.path.exists(resume_path):
        os.remove(resume_path)
    torch.save(best_state, os.path.join(out_dir, f"{dataset}.pt"))
    meta = {"dataset": dataset, "n_vars": C, "means": data.means.tolist(),
            "stds": data.stds.tolist(), "best_val_mse": best_va,
            "epochs": epoch + 1,
            "window_stride": {"train": tr_stride, "val": VAL_STRIDE[dataset]},
            "arch": {"d_model": 512, "d_ff": 512,
            "n_heads": 8, "e_layers": 3, "patch_len": 16, "stride": 8,
            "dropout": 0.3, "seq_len": CTX, "pred_len": H},
            "train_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, f"{dataset}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[{dataset}] done: best val {best_va:.6f} in {epoch + 1} epochs "
          f"({meta['train_seconds']}s) -> {os.path.join(out_dir, dataset + '.pt')}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1,exchange_rate,electricity,traffic")
    args = ap.parse_args()
    print(f"torch={torch.__version__}")
    if torch.backends.mps.is_available():
        torch.set_num_threads(10)
    out_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "patchtst")
    for ds in [d for d in args.datasets.split(",") if d]:
        if os.path.exists(os.path.join(out_dir, f"{ds}.pt")) and not os.path.exists(
                os.path.join(out_dir, f"{ds}_resume.pt")):
            print(f"[{ds}] checkpoint exists, skipping")
            continue
        device = DEVICE_BY_DATASET.get(ds, "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[{ds}] device={device} window_stride={WINDOW_STRIDE[ds]}", flush=True)
        try:
            train_one(ds, device)
        except Exception as e:
            import traceback
            print(f"[{ds}] FAILED: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
