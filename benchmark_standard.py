"""Standardized evaluation harness for the NanoForecast paper.

ONE protocol for every model (NanoForecast v0.3/v0.5, TimesFM, Chronos-T5-large):

  * Datasets : ETTh1, ETTh2, ETTm1, exchange_rate, electricity, traffic
  * Splits   : ETT   -> train 70% / val 20% / test 10%  (standard 12/4/4 months)
               other -> train 70% / val 10% / test 20%  (standard 7:1:2)
  * Horizon  : H = 48, context C = 512
  * Windows  : non-overlapping blocks of H inside the test segment;
               context = the 512 observations immediately preceding the window
  * MASE     : window MAE divided by the per-series seasonal-naive in-sample
               MAE computed on the TRAINING segment only
               (seasonality s: 24 hourly, 96 for 15-min, 7 daily)
  * Aggreg   : mean over windows per series, then mean over series
               (all 321 electricity clients / 862 traffic sensors are used)

Usage:
    python3 benchmark_standard.py --models nanoforecast,timesfm,chronos \
        --output results/standard_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from nanoforecast.data.real_datasets import _load_dataframe

H = 48
CTX = 512
DEVICE = os.environ.get("NF_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")

# HF repo holding trained PatchTST CI checkpoints ({ds}.pt + {ds}.json).
# Used as a fallback so local runs work without manually downloading weights.
PATCHTST_HF_REPO = "eulogik/nanoforecast-patchtst-baselines"

SEASONALITY = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "exchange_rate": 7,
               "electricity": 24, "traffic": 24}
TRAIN_VAL_FRAC = {"ETTh1": 0.90, "ETTh2": 0.90, "ETTm1": 0.90,
                  "exchange_rate": 0.80, "electricity": 0.80, "traffic": 0.80}
TRAIN_FRAC = {"ETTh1": 0.70, "ETTh2": 0.70, "ETTm1": 0.70,
              "exchange_rate": 0.70, "electricity": 0.70, "traffic": 0.70}
FREQ_TSFM = {"ETTh1": "hourly", "ETTh2": "hourly", "ETTm1": "minutely",
             "exchange_rate": "daily", "electricity": "hourly", "traffic": "hourly"}
FREQ_NF = {"ETTh1": 1, "ETTh2": 1, "ETTm1": 0,
           "exchange_rate": 2, "electricity": 1, "traffic": 1}


def load_series(dataset: str) -> List[np.ndarray]:
    """One 1-D float32 series per channel. ETT -> oil temperature (OT) only."""
    df = _load_dataframe(dataset)
    if dataset.startswith("ETT"):
        values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        ot = values[:, -1]
        return [ot]
    values = df.to_numpy(dtype=np.float32)
    return [values[:, c] for c in range(values.shape[1])]


def split_series(series: np.ndarray, dataset: str):
    n = len(series)
    t0 = int(n * TRAIN_FRAC[dataset])
    t1 = int(n * TRAIN_VAL_FRAC[dataset])
    return series[:t0], series[t0:t1], series[t1:]


def seasonal_naive_scale(train: np.ndarray, s: int) -> float:
    if len(train) <= s:
        return float(np.mean(np.abs(train))) or 1e-8
    scale = float(np.mean(np.abs(train[s:] - train[:-s])))
    return max(scale, 1e-8)


def aggregate(results: List[Dict]) -> Dict[str, float]:
    """Mean over windows per series, then mean over series."""
    keys = ["mase", "mae", "mse", "smape", "crps"]
    series_means = {k: [] for k in keys}
    for r in results:
        if not r:
            continue
        for k in keys:
            vals = np.array([w[k] for w in r], dtype=np.float64)
            series_means[k].append(float(np.mean(vals)))
    return {k: float(np.mean(v)) for k, v in series_means.items() if v}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class NanoForecastModel:
    name = "nanoforecast"

    def __init__(self, repo: str):
        from nanoforecast.model.core import NanoForecast
        self.model = NanoForecast.from_pretrained(repo, map_location=DEVICE).to(DEVICE).eval()

    def predict_batch(self, contexts: List[np.ndarray], dataset: str,
                              series_idx: int = 0, starts: List[int] = None):
        batch = np.stack(contexts).astype(np.float32)
        out = self.model.predict(context=batch, horizon=H, freq=FREQ_NF[dataset],
                                 return_components=False)
        return out["forecast"], out["quantiles"]


class TimesFMModel:
    name = "timesfm"

    def __init__(self):
        from transformers import TimesFmModelForPrediction
        path = os.path.join(os.path.dirname(__file__), "benchmarks", "checkpoints",
                            "timesfm-1.0-200m")
        self.model = TimesFmModelForPrediction.from_pretrained(path).to(DEVICE).eval()
        self.freq_idx = {"ETTh1": 1, "ETTh2": 1, "ETTm1": 0,
                         "exchange_rate": 2, "electricity": 1, "traffic": 1}

    def predict_batch(self, contexts: List[np.ndarray], dataset: str,
                      series_idx: int = 0, starts: List[int] = None):
        xs = [torch.from_numpy(c.astype(np.float32)).to(DEVICE) for c in contexts]
        freq = [self.freq_idx[dataset]] * len(xs)
        out = self.model(past_values=xs, freq=freq)
        mean = out.mean_predictions.detach().cpu().numpy()        # (B, 128)
        full = out.full_predictions.detach().cpu().numpy()        # (B, 128, 10)
        q_all = full[:, :H, 1:]                                    # (B, 48, 9)
        q_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        want = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        q5 = np.stack([np.stack([np.interp(want, q_levels, q_all[b, h, :])
                                 for h in range(H)]) for b in range(len(xs))])
        return mean[:, :H], np.moveaxis(q5, 1, 2)


class ChronosModel:
    name = "chronos"

    def __init__(self):
        from chronos import ChronosPipeline
        self.model = ChronosPipeline.from_pretrained("amazon/chronos-t5-large",
                                                     device_map=DEVICE, torch_dtype=torch.float32)

    def predict_batch(self, contexts: List[np.ndarray], dataset: str, series_idx: int = 0,
                      starts: List[int] = None):
        contexts_t = [torch.from_numpy(c.astype(np.float32)).to(DEVICE) for c in contexts]
        samples = self.model.predict(context=contexts_t, prediction_length=H,
                                     num_samples=20)
        samples = samples.detach().cpu().numpy()  # (B, 64, H)
        mean = samples.mean(axis=1)
        q = np.quantile(samples, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1)  # (5, B, H)
        return mean, np.moveaxis(q, 0, 1)


class PatchTSTModel:
    """Trained by benchmarks/train_patchtst.py (official thuml TSL config,
    PatchTST paper hyperparameters, lookback 512, pred_len 48)."""
    name = "patchtst"

    def __init__(self, ckpt_dir: str):
        import json as _json
        from benchmarks.tsl.PatchTST import Model as TSLPatchTST
        self.ckpt_dir = ckpt_dir
        self.cls = TSLPatchTST
        os.makedirs(ckpt_dir, exist_ok=True)
        self.metas = {f[:-5]: _json.load(open(os.path.join(ckpt_dir, f)))
                      for f in sorted(os.listdir(ckpt_dir)) if f.endswith(".json")}
        self.models, self._arrays = {}, {}

    def _ensure_checkpoint(self, ds: str):
        """Download {ds}.pt + {ds}.json from HF if not present locally."""
        if ds in self.models and ds in self.metas:
            return
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise FileNotFoundError(
                f"checkpoint {ds}.pt missing in {self.ckpt_dir}; "
                "install huggingface_hub to auto-fetch from HF "
                f"(repo {PATCHTST_HF_REPO})")
        import json as _json
        for ext in ("json", "pt"):
            p = os.path.join(self.ckpt_dir, f"{ds}.{ext}")
            if not os.path.exists(p):
                try:
                    hf_hub_download(PATCHTST_HF_REPO, f"{ds}.{ext}",
                                    local_dir=self.ckpt_dir)
                except Exception as e:  # repo missing / offline / no file
                    raise FileNotFoundError(
                        f"checkpoint {ds}.{ext} missing in {self.ckpt_dir} and "
                        f"could not be fetched from HF repo {PATCHTST_HF_REPO}: {e}")
                print(f"  [patchtst] fetched {ds}.{ext} from HF")
        if ds not in self.metas:
            self.metas[ds] = _json.load(open(os.path.join(self.ckpt_dir, f"{ds}.json")))

    def _arrays_for(self, ds: str):
        if ds not in self._arrays:
            self._arrays[ds] = np.stack(load_series(ds), axis=1).astype(np.float32)
        return self._arrays[ds]

    def _model_for(self, ds: str, C: int):
        if ds not in self.models:
            self._ensure_checkpoint(ds)
            import types
            cfg = types.SimpleNamespace(
                task_name="long_term_forecast", seq_len=CTX, pred_len=H,
                enc_in=C, dec_in=C, c_out=C, d_model=512, n_heads=8,
                e_layers=3, d_ff=512, dropout=0.3, factor=3,
                activation="gelu", output_attention=False)
            m = self.cls(cfg, patch_len=16, stride=8).to(DEVICE).eval()
            m.load_state_dict(torch.load(os.path.join(self.ckpt_dir, f"{ds}.pt"),
                                         map_location=DEVICE))
            self.models[ds] = m
        return self.models[ds]

    def predict_batch(self, contexts, dataset: str, series_idx: int = 0,
                       starts: List[int] = None):
        x = self._arrays_for(dataset)
        meta = self.metas[dataset]
        C = int(meta["n_vars"])
        model = self._model_for(dataset, 1)  # channel-independent (C=1)
        t1 = int(len(x) * TRAIN_VAL_FRAC[dataset])
        means = np.array(meta["means"]).reshape(-1)
        stds = np.array(meta["stds"]).reshape(-1)
        norm = (x - means) / stds
        fc = []
        with torch.no_grad():
            for s in starts:
                # evaluate channel `series_idx` of this window (channel-independent)
                ci = series_idx % C
                c_ctx = norm[ci][max(0, t1 + s - CTX):t1 + s]
                if c_ctx.shape[0] < CTX:
                    c_ctx = np.concatenate([np.zeros(CTX - c_ctx.shape[0], np.float32), c_ctx])
                cb = torch.from_numpy(c_ctx[None, :, None].astype(np.float32)).to(DEVICE)  # (1, L, 1)
                out = model(cb, None, None, None)[0].detach().cpu().numpy()  # (H, 1)
                fc.append(out[:, 0])
        fc = np.stack(fc)
        fc = fc * stds[series_idx] + means[series_idx]
        return fc, None


def run_model(model, datasets: List[str], batch_size: int) -> Dict:
    results = {}
    for ds in datasets:
        t0 = time.time()
        print(f"  [{model.name}] loading {ds}...", flush=True)
        series_list = load_series(ds)
        all_windows = []
        n_windows = 0
        for si, series in enumerate(series_list):
            train, val, test = split_series(series, ds)
            starts = range(0, len(test) - H + 1, H)
            batches = []
            cur = []
            cur_ctx = []
            for start in starts:
                ctx = test[max(0, start - CTX):start]
                if len(ctx) < CTX:
                    ctx = np.concatenate([train[-(CTX - len(ctx)):], ctx])
                cur.append(start)
                cur_ctx.append(ctx)
                if len(cur) == batch_size:
                    batches.append((cur, cur_ctx))
                    cur, cur_ctx = [], []
            if cur:
                batches.append((cur, cur_ctx))
            print(f"  [{model.name}] {ds} series {si+1}/{len(series_list)}: {len(batches)} batches", flush=True)
            for bi, (starts_b, ctxs_b) in enumerate(batches):
                if bi % 25 == 0:
                    print(f"  [{model.name}] {ds} s{si} batch {bi}/{len(batches)}", flush=True)
                fc, q = model.predict_batch(ctxs_b, ds, si, starts_b)
                for i, start in enumerate(starts_b):
                    target = test[start:start + H]
                    scale = seasonal_naive_scale(train, SEASONALITY[ds])
                    mae = float(np.mean(np.abs(target - fc[i])))
                    mse = float(np.mean((target - fc[i]) ** 2))
                    denom = (np.abs(target) + np.abs(fc[i])) / 2.0
                    m = denom > 1e-5
                    smape = float(100 * np.mean(np.abs(target[m] - fc[i][m]) / denom[m])) if m.any() else 0.0
                    crps = 0.0
                    if q is not None:
                        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
                        for k, ql in enumerate(levels):
                            d = target - q[i][k]
                            crps += float(np.mean(np.maximum(ql * d, (ql - 1.0) * d)))
                        crps = 2.0 * crps / len(levels)
                    all_windows.append({"mase": mae / scale, "mae": mae, "mse": mse,
                                        "smape": smape, "crps": crps})
                    n_windows += 1
        results[ds] = {"metrics": aggregate([all_windows]), "windows": n_windows,
                       "series": len(series_list), "seconds": round(time.time() - t0, 1)}
        print(f"  [{model.name}] {ds}: {results[ds]['metrics']} "
              f"({n_windows} windows, {len(series_list)} series)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="nanoforecast,timesfm,chronos")
    ap.add_argument("--output", default="results/standard_benchmark.json")
    ap.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1,exchange_rate,electricity,traffic")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    models = []
    if "nanoforecast" in args.models:
        for repo in ["eulogik/nanoforecast-v03", "eulogik/nanoforecast-v05"]:
            models.append(("nanoforecast-" + repo.split("-")[-1], NanoForecastModel(repo), 64))
    if "timesfm" in args.models:
        models.append(("timesfm", TimesFMModel(), 16))
    if "chronos" in args.models:
        models.append(("chronos", ChronosModel(), 8))
    if "patchtst" in args.models:
        ckpt = os.path.join(os.path.dirname(__file__), "benchmarks", "checkpoints", "patchtst")
        models.append(("patchtst", PatchTSTModel(ckpt), 32))

    out = {"protocol": {
        "horizon": H, "context": CTX, "windows": "non-overlapping H-blocks in test",
        "mase_scale": "seasonal naive in-sample MAE on train segment",
        "splits": {"ETT": "70/20/10", "other": "70/10/20"},
        "aggregation": "mean over windows per series, then mean over series",
        "device": DEVICE}, "results": {}}
    if os.path.exists(args.output):
        try:
            old = json.load(open(args.output))
            out = {"protocol": old.get("protocol", out["protocol"]),
                   "results": old.get("results", {})}
        except Exception:
            pass

    for name, model, bs in models:
        existing = out["results"].get(name, {})
        todo = [d for d in datasets if d not in existing]
        if not todo:
            print(f"  {name}: all datasets already present, skipping")
            continue
        print(f"\n=== {name} ===  (missing: {', '.join(todo)})")
        fresh = run_model(model, todo, bs)
        existing.update(fresh)
        out["results"][name] = existing
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"  saved -> {args.output}")

    print("\n=== SUMMARY (MASE) ===")
    for name in out["results"]:
        row = "  ".join(
            f"{ds}:{out['results'][name][ds]['metrics']['mase']:.3f}"
            for ds in datasets if ds in out["results"][name])
        print(f"{name:>14}  {row}")


if __name__ == "__main__":
    main()
