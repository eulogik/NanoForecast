"""Real time series dataset loaders for NanoForecast pretraining.

Loads canonical Monash / Informer / Autoformer benchmark datasets from
public mirrors. Supports univariate slicing so the same loader can feed
both the univariate NanoForecast core and the multivariate cases.

Datasets supported:
  - ETTh1, ETTh2, ETTm1 (hourly / 15-min oil temperature, 7 features)
  - exchange_rate (daily FX, 8 channels)
  - electricity (hourly household consumption, 321 channels)
  - traffic (hourly road occupancy, 862 channels)
"""
from __future__ import annotations

import gzip
import io
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

CACHE_DIR = os.path.expanduser("~/.cache/nanoforecast/datasets")
os.makedirs(CACHE_DIR, exist_ok=True)

# freq_id convention from synthetic generator:
#   0 -> 5-min, 1 -> hourly, 2 -> daily, 3 -> weekly, 4 -> monthly, 5+ -> other
FREQ_MAP = {
    "ETTh1": 1,
    "ETTh2": 1,
    "ETTm1": 0,  # 15-min intervals
    "exchange_rate": 2,  # daily
    "electricity": 1,  # hourly
    "traffic": 1,  # hourly
}

DATASET_URLS: Dict[str, str] = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "exchange_rate": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
    "electricity": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz",
    "traffic": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz",
}


@dataclass
class WindowSpec:
    """Sliding-window specification for turning long series into pretraining records."""
    context_len: int = 256
    prediction_len: int = 48
    stride: int = 64


def _cache_path(dataset: str, suffix: str = "") -> str:
    safe = dataset.replace("/", "_")
    return os.path.join(CACHE_DIR, safe + suffix)


def _is_gzip(path: str) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def _download(dataset: str, url: str) -> str:
    path = _cache_path(dataset, suffix=".gz" if url.endswith(".gz") else "")
    if os.path.exists(path):
        if os.path.getsize(path) > 0:
            return path
        print(f"[datasets] removing empty cache for {dataset} ...")
        os.remove(path)
    print(f"[datasets] downloading {dataset} from {url} ...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
    if os.path.getsize(path) == 0:
        raise RuntimeError(f"Downloaded file for {dataset} is empty: {url}")
    return path


def _clear_cache(dataset: str) -> None:
    for suffix in ["", ".gz"]:
        p = _cache_path(dataset, suffix=suffix)
        if os.path.exists(p):
            os.remove(p)

def _load_dataframe(dataset: str) -> pd.DataFrame:
    url = DATASET_URLS[dataset]
    for attempt in range(2):
        path = _download(dataset, url)
        try:
            if _is_gzip(path):
                with gzip.open(path, "rb") as fh:
                    data = fh.read()
                arr = np.loadtxt(io.BytesIO(data), delimiter=",", dtype=np.float32)
                return pd.DataFrame(arr)
            return pd.read_csv(path)
        except (pd.errors.EmptyDataError, OSError, ValueError) as e:
            if attempt == 0:
                print(f"[datasets] corrupt cache for {dataset}, re-downloading ... ({e})")
                _clear_cache(dataset)
                continue
            raise


def list_datasets() -> List[str]:
    return list(DATASET_URLS.keys())


def load_univariate_series(
    dataset: str,
    channels: Optional[List[int]] = None,
    max_channels: int = 4,
) -> Tuple[List[np.ndarray], int]:
    """Load selected channels of a dataset as a list of 1-D float32 arrays.

    Args:
        dataset: one of the keys in DATASET_URLS.
        channels: explicit channel indices (0-based, excluding the date column for ETT*).
            If None, picks up to ``max_channels`` channels spaced evenly across the file.
        max_channels: used when ``channels`` is None.

    Returns:
        series_list: list of 1-D float32 arrays, one per selected channel.
        freq_id: integer frequency identifier.
    """
    df = _load_dataframe(dataset)
    if dataset.startswith("ETT"):
        # First column is the date; remaining columns are features.
        values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    else:
        # Comma-separated dense matrix files (one row per timestep).
        values = df.to_numpy(dtype=np.float32)

    n_channels = values.shape[1]
    if channels is None:
        if n_channels <= max_channels:
            channels = list(range(n_channels))
        else:
            channels = np.linspace(0, n_channels - 1, max_channels).round().astype(int).tolist()
            channels = sorted(set(channels))

    series_list = [values[:, c].astype(np.float32) for c in channels]
    return series_list, FREQ_MAP[dataset]


def build_pretraining_records(
    dataset: str,
    spec: WindowSpec,
    channels: Optional[List[int]] = None,
    max_channels: int = 4,
) -> List[Dict]:
    """Build NanoForecast-format pretraining records from a dataset.

    Each record is a dict with: context, prediction, freq_id,
    context_covariates (zeros, dim=4 to match the model config).
    """
    series_list, freq_id = load_univariate_series(dataset, channels, max_channels)

    L_in = spec.context_len + spec.prediction_len
    records: List[Dict] = []
    for s in series_list:
        s = np.asarray(s, dtype=np.float32)
        n = s.shape[0]
        if n < L_in:
            continue
        for start in range(0, n - L_in + 1, spec.stride):
            window = s[start:start + L_in]
            records.append({
                "context": window[:spec.context_len],
                "prediction": window[spec.context_len:spec.context_len + spec.prediction_len],
                "freq_id": freq_id,
                "context_covariates": np.zeros((4, spec.context_len), dtype=np.float32),
                "source": dataset,
            })
    return records


def build_mixed_pretraining_corpus(
    spec: WindowSpec,
    datasets: Optional[List[str]] = None,
    max_channels_per_dataset: int = 4,
) -> List[Dict]:
    """Build a mixed pretraining corpus across multiple datasets."""
    if datasets is None:
        datasets = ["ETTh1", "exchange_rate"]  # small + reliable default
    all_records: List[Dict] = []
    for ds in datasets:
        recs = build_pretraining_records(ds, spec, max_channels=max_channels_per_dataset)
        print(f"[datasets] {ds}: {len(recs)} windows")
        all_records.extend(recs)
    print(f"[datasets] total: {len(all_records)} windows")
    return all_records


def time_based_split(
    records: List[Dict],
    val_fraction: float = 0.2,
) -> Tuple[List[Dict], List[Dict]]:
    """Time-based train/val split: hold out the last ``val_fraction`` of records.

    This avoids look-ahead leakage in pretraining.
    """
    if not records:
        return [], []
    n = len(records)
    cut = int(n * (1.0 - val_fraction))
    return records[:cut], records[cut:]
