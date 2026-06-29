# 🔮 NanoForecast

**World's most deployable time series transformer**

[![PyPI](https://img.shields.io/pypi/v/nanoforecast)](https://pypi.org/project/nanoforecast/)
[![Downloads](https://img.shields.io/pypi/dm/nanoforecast)](https://pypi.org/project/nanoforecast/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](./pyproject.toml)
[![HF Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/eulogik/nanoforecast)
[![HF 200k](https://img.shields.io/badge/🤗%20Model-200k-blue)](https://huggingface.co/eulogik/nanoforecast-200k)
[![HF 500k](https://img.shields.io/badge/🤗%20Model-500k-blue)](https://huggingface.co/eulogik/nanoforecast-500k)
[![HF v0.3](https://img.shields.io/badge/🤗%20Model-v0.3-blue)](https://huggingface.co/eulogik/nanoforecast-v03)
[![Paper](https://img.shields.io/badge/📄%20Paper-LaTeX-lightgrey)](./deploy/paper.tex)
[![Eulogik](https://img.shields.io/badge/by-Eulogik-purple)](https://eulogik.com)

NanoForecast is the **world's most deployable time series foundation model** — a tiny transformer for zero-shot forecasting, streaming inference, and edge deployment. At just 200K–6.5M parameters, it runs on CPU, Raspberry Pi, and in the browser via ONNX. Unlike large foundation models that require GPUs and terabytes of data, NanoForecast:

- **Trains on your data in 2 minutes** — `python3 train_from_csv.py --csv sales.csv --target revenue`
- **Streams forecasts online** — the only TS model where you can feed one value at a time
- **Runs on a Raspberry Pi** (<50ms inference on ARM)
- **Exports to 1.4 MB ONNX** (Edge/IoT/browser ready)
- **Fully Apache 2.0** — no strings attached

It's **not** a foundation model. It's not going to beat TimesFM on benchmarks. What it does is **actually ship to production**.

---

## Install

```bash
pip install nanoforecast
```

[![GitHub release](https://img.shields.io/github/v/release/eulogik/NanoForecast?include_prereleases)](https://github.com/eulogik/NanoForecast/releases)
[![GitHub stars](https://img.shields.io/github/stars/eulogik/NanoForecast)](https://github.com/eulogik/NanoForecast/stargazers)

From source:
```bash
git clone https://github.com/eulogik/NanoForecast.git
cd NanoForecast
pip install -e .
```

## Quick Start

### Train on your own data (primary path)

```bash
python3 train_from_csv.py --csv my_data.csv --target sales --horizon 48
```

That's it. You get a saved model checkpoint + forecast CSV. No GPU, no cloud.

> Built by [Eulogik](https://eulogik.com) — deployable AI for the real world.

### Or use the pretrained model

```bash
pip install nanoforecast
```

```python
import numpy as np
from nanoforecast import NanoForecast

model = NanoForecast.from_pretrained("eulogik/nanoforecast-500k")

context = np.sin(np.linspace(0, 8*np.pi, 256)) + 0.1 * np.random.randn(256)
result = model.predict(context, horizon=48, freq=1)
forecast = result["forecast"][0]  # shape (48,)
```

## Streaming / Online Inference (unique to NanoForecast)

NanoForecast's DeltaNet RNN architecture maintains a recurrent state across calls — no other TS model does this.

```python
# Initial forecast + state
result = model.predict(context, horizon=48, return_state=True)
state = result.pop("state")

# Stream new observations one at a time
for new_val in incoming_data_stream:
    result = model.predict_step(new_val, state, horizon=48)
    print(result["forecast"][0, :5])  # updated forecast instantly
```

Each call preserves the DeltaNet's memory of all past data. Use it for:
- **Real-time IoT sensor monitoring**
- **Live financial tick data**
- **Interactive dashboards**

## Try It Now

[![Open in HF Spaces](https://img.shields.io/badge/🤗%20Open%20in%20Spaces-blueviolet)](https://huggingface.co/spaces/eulogik/nanoforecast)

Upload a CSV, set your horizon, get a forecast + prediction intervals + decomposition plot. No code required.

## Features

| Feature | Details |
|---|---|---|
| **Architecture** | LongConv + DeltaNet RNN + gated router + MLP blocks |
| **Parameters** | 200K–6.5M (tiny) |
| **Outputs** | Point forecast + 5 quantiles (p10/p25/p50/p75/p90) + trend/seasonal/residual decomposition |
| **Context** | 256–512 timesteps |
| **Horizon** | Any length, forecasts up to 48 steps per call |
| **Frequency** | Hourly / Daily / Weekly / Monthly |
| **Deploy targets** | CPU, ARM, Raspberry Pi, Lambda, iOS, browser (via ONNX.js) |
| **Streaming inference** | Stateful RNN — feed one value at a time, no re-processing |
| **Train on your data** | `train_from_csv.py` — 2 min on a laptop, no GPU |

## Deploy

### FastAPI Server

```bash
pip install nanoforecast fastapi uvicorn python-multipart
python3 deploy/fastapi_server.py
# → http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/predict \
  -d "context=[1.0,2.0,3.0,...]" \
  -d "horizon=48" \
  -d "freq=1"
```

### Docker

```bash
docker build -t nanoforecast -f deploy/Dockerfile .
docker run -p 8000:8000 nanoforecast
```

### ONNX (Edge / IoT)

```bash
pip install "nanoforecast[onnx]"
python3 -m nanoforecast.export.onnx_export \
    --checkpoint <checkpoint-dir> \
    --output nanoforecast.onnx
```

Then load with onnxruntime on any platform:

```python
import onnxruntime as ort
session = ort.InferenceSession("nanoforecast.onnx")
forecast = session.run(None, {"input": context_numpy})
```

---

## Repository layout

```
nanoforecast/
  config.py              # NanoForecastConfig dataclass
  model/                 # core architecture
    blocks.py            # LongConv, DeltaNet, GatedMLP, GatedRouter
    heads.py             # point, quantile (monotonic), anomaly, decomposition
    core.py              # NanoForecast nn.Module
    utils.py             # scaler, patching, freq prefix, positional encoding
  train/
    loss.py              # multi-task loss (point + quantile + anomaly + smooth)
    trainer.py           # OneCycleLR trainer with MPS / CUDA / CPU support
  data/
    generator.py         # synthetic time series generator
    pipeline.py          # dataset + resolution-aware batch sampler
    real_datasets.py     # ETTh1/2, ETTm1, exchange_rate, electricity, traffic loaders
  evaluation/
    benchmark.py         # MASE, sMAPE, MSE, MAE, CRPS, coverage
  export/
    onnx_export.py       # FP32 + dynamic INT8 ONNX export
  hub.py                 # save_pretrained / from_pretrained / predict mixin
gradio_app.py            # Hugging Face Space (upload CSV → forecast plot)
demo.py                  # one-liner demo: python3 demo.py
deploy/                  # FastAPI server + Docker
  fastapi_server.py
  Dockerfile
  requirements.txt
pretrain.py              # real + synthetic pretraining CLI
benchmark.py             # multi-dataset benchmark CLI
push_to_hub.py           # publish a checkpoint to the HF Hub
run_pipeline.py          # synthetic-only smoke pipeline (legacy)
train_from_csv.py        # train on your own CSV (primary user path)
tests/                   # unit + smoke tests
```

---

## Architecture

```
Raw Context  ->  Robust Scaling & Patching  ->  Resolution Prefix Token
                                                  |
                                                  v
                              Sequence Mixing Blocks (x N)
                                - LongConv (global periodicity)
                                - DeltaNet RNN (local dependencies)
                                - Gated Router & MLP (dynamic blend)
                                                  |
                                                  v
                            Multi-Task Heads (single forward pass)
                              - Point forecast
                              - Monotonic quantiles (p10..p90)
                              - Context reconstruction (anomaly)
                              - Trend / Seasonality decomposition
```

| Preset | d_model | Layers | Patch | Parameters | FP32 size |
|---|---:|---:|---:|---:|---:|
| `nano-200k` | 32 | 4 | 8 | ~676K | ~2.7 MB |
| `nano-500k` | 64 | 8 | 8 | ~1.6M | ~6.4 MB |
| `nano-1m` (v0.3) | 96 | 8 | 8 | ~6.5M | ~26 MB |

### Design notes

- **Instance Robust Scaler** (median / IQR) makes the model robust to outliers.
- **Monotonic quantile head** guarantees p10 ≤ p25 ≤ p50 ≤ p75 ≤ p90.
- **Conservation identity**: trend + seasonal + residual ≡ point forecast.
- **ONNX exportable** with a drop-in RMSNorm replacement.

---

## Train Your Own

```bash
python3 pretrain.py \
  --datasets ETTh1,ETTh2,exchange_rate,electricity \
  --epochs 50 \
  --batch-size 64 \
  --device cpu \
  --output checkpoints/nanoforecast-my-data
```

## Benchmarking

```bash
python3 benchmark.py \
  --checkpoint <checkpoint-dir> \
  --datasets ETTh1,ETTh2,ETTm1,exchange_rate \
  --max-windows 64 \
  --output results/benchmark.json
```

## Publishing to HF Hub

```bash
huggingface-cli login
python3 push_to_hub.py \
  --checkpoint <checkpoint-dir> \
  --repo-id your-username/nanoforecast-500k \
  --benchmark-json results/benchmark.json
```

---

## Benchmarks

### v0.3 (d_model=96, 6.5M params, context=512, 200 epochs, Colab T4)

| Dataset | MASE | sMAPE (%) | MAE | CRPS |
|---:|---:|---:|---:|---:|
| ETTh1 | **1.95** | 12.06 | 1.30 | 1.05 |
| ETTh2 | **2.74** | 10.47 | 2.46 | 1.99 |
| ETTm1 | **2.17** | 10.70 | 0.72 | 0.65 |
| exchange_rate | **7.44** | 1.72 | 0.011 | 0.014 |
| electricity | **1.29** | 4.76 | 158.30 | 175.24 |
| traffic | **0.81** | 24.00 | 0.004 | 0.003 |
| **Overall** | **2.73** | 10.62 | 27.13 | 29.83 |

### v0.2 (d_model=64, 1.6M params, context=256, 100 epochs, Mac Mini M4)

| Dataset | MASE | sMAPE (%) | MAE | CRPS |
|---:|---:|---:|---:|---:|
| ETTh1 | **3.34** | 25.13 | 2.40 | 1.80 |
| ETTh2 | **3.71** | 17.65 | 3.21 | 2.52 |
| ETTm1 | **3.58** | 17.22 | 1.17 | 1.00 |
| exchange_rate | **7.31** | 1.63 | 0.010 | 0.009 |
| electricity | **1.54** | 5.65 | 189.75 | 187.26 |
| traffic | **1.25** | 44.80 | 0.006 | 0.006 |
| **Overall** | **3.45** | 18.68 | 32.76 | 32.10 |

### v0.3 vs v0.2 comparison

| Dataset | v0.2 | v0.3 | Improvement |
|---:|---:|---:|---|
| ETTh1 | 3.34 | **1.95** | ↓ 42% |
| ETTh2 | 3.71 | **2.74** | ↓ 26% |
| ETTm1 | 3.58 | **2.17** | ↓ 39% |
| exchange_rate | 7.31 | 7.44 | ↑ 2% |
| electricity | 1.54 | **1.29** | ↓ 16% |
| traffic | 1.25 | **0.81** | ↓ 35% |
| **Overall** | **3.45** | **2.73** | **↓ 21%** |

v0.3 beats v0.2 on 5 of 6 datasets. Larger model (6.5M vs 1.6M) and longer context (512 vs 256) provide significant accuracy gains on ETTh1/ETTh2/ETTm1. Exchange rate (volatile FX) slightly regresses.

---

## Known limitations

| Issue | Status |
|---|---|
| **Accuracy** | Modest vs SOTA (MASE ~2.73 overall for v0.3). Good enough for prototypes, not production forecasting at scale. |
| **Training** | Multi-dataset mixing (v0.3: 6 real + 10K synthetic, 200 epochs). |
| **Context** | Fixed 256 — longer history is truncated. |
| **Channels** | Univariate by default; multivariate support is per-dimension independent. |
| **Edge cases** | NaN values, missing timestamps, irregularly-sampled data not handled automatically. |

This is a **developer tool**, not a research paper. It prioritizes deployability over accuracy.

## Roadmap

| Version | Focus | Timeline |
|---|---|---|
| v0.1 | Deployable MVP — train, predict, export, deploy | ✅ Done |
| v0.2 | Streaming inference + train-from-CSV CLI + multi-dataset training (Mac Mini) | ✅ Done |
| v0.3 | Colab T4 training (larger model, more data) + ONNX.js browser demo | TBD |
| v0.4 | OpenRouter API — $0.001/forecast | TBD |

## Why "NanoForecast"?

Because forecasting models shouldn't require:
- A $30K GPU
- 100 GB of training data
- 12 dependencies that break every release
- A team of PhDs to deploy

You should be able to train a forecasting model on your laptop, deploy it to a Raspberry Pi, and have it running in production before lunch.

## License

Apache 2.0. See [LICENSE](./LICENSE).

---

Built by [Eulogik](https://eulogik.com) — deployable AI for the real world.
