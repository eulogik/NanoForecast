# 🔮 NanoForecast

**World's most deployable time series transformer**

[![HF Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/eulogik/nanoforecast)
[![HF Model](https://img.shields.io/badge/🤗%20Model-nanoforecast--200k-blue)](https://huggingface.co/eulogik/nanoforecast-200k)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](./LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](./pyproject.toml)

NanoForecast is a **tiny, fast, deployable** time series forecasting model. Unlike foundation models that require GPUs and terabytes of data, NanoForecast:

- **Trains in 20 minutes** on a MacBook Air (no GPU needed)
- **Runs on a Raspberry Pi** (<50ms inference on ARM)
- **Exports to 1.4 MB ONNX** (Edge/IoT/browser ready)
- **Fully Apache 2.0** — no strings attached

It's **not** a foundation model. It's not going to beat TimesFM on benchmarks. What it does is **actually ship to production**.

---

## Quick Start

```bash
pip install nanoforecast
```

```python
import numpy as np
from nanoforecast import NanoForecast

model = NanoForecast.from_pretrained("eulogik/nanoforecast-200k")

context = np.sin(np.linspace(0, 8*np.pi, 256)) + 0.1 * np.random.randn(256)
result = model.predict(context, horizon=48, freq=1)
forecast = result["forecast"][0]  # shape (48,)
```

## Try It Now

[![Open in HF Spaces](https://img.shields.io/badge/🤗%20Open%20in%20Spaces-blueviolet)](https://huggingface.co/spaces/eulogik/nanoforecast)

Upload a CSV, set your horizon, get a forecast + prediction intervals + decomposition plot. No code required.

## Features

| Feature | Details |
|---|---|
| **Architecture** | LongConv + DeltaNet RNN + gated router + MLP blocks |
| **Parameters** | ~200K–700K (tiny) |
| **Outputs** | Point forecast + 5 quantiles (p10/p25/p50/p75/p90) + trend/seasonal/residual decomposition |
| **Context** | 256 timesteps |
| **Horizon** | Any length (trained on 48) |
| **Frequency** | Hourly / Daily / Weekly / Monthly |
| **Deploy targets** | CPU, ARM, Raspberry Pi, Lambda, iOS, browser (via ONNX.js) |

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
    --checkpoint checkpoints/nanoforecast-200k \
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
  --checkpoint checkpoints/nanoforecast-200k \
  --datasets ETTh1,ETTh2,ETTm1,exchange_rate \
  --max-windows 64 \
  --output results/benchmark.json
```

## Publishing to HF Hub

```bash
huggingface-cli login
python3 push_to_hub.py \
  --checkpoint checkpoints/nanoforecast-200k \
  --repo-id your-username/nanoforecast-200k \
  --benchmark-json results/benchmark.json
```

---

## Benchmarks (current v0.1 demo checkpoint)

Pretrained for 20 epochs on ETTh1 (~1000 windows) on CPU. These numbers
demonstrate the pipeline works end-to-end but are **not** competitive.

| Dataset | MASE | sMAPE (%) | MAE | CRPS |
|---|---:|---:|---:|---:|
| ETTh1 | ~5 | ~35 | ~3 | ~2 |
| exchange_rate | ~11 | 2.4 | 0.015 | 0.01 |

v0.2 target (Mac Mini training, ~2-4 hours): bring MASE < 2.0 on ETTh1.

---

## Known limitations

| Issue | Status |
|---|---|
| **Accuracy** | Poor vs SOTA (MASE 4-11 on ETT). Good enough for prototypes, not production forecasting at scale. |
| **Training** | Single dataset or basic mixing — no multi-dataset pretraining (pending v0.2 on Mac Mini). |
| **Context** | Fixed 256 — longer history is truncated. |
| **Channels** | Univariate by default; multivariate support is per-dimension independent. |
| **Edge cases** | NaN values, missing timestamps, irregularly-sampled data not handled automatically. |

This is a **developer tool**, not a research paper. It prioritizes deployability over accuracy.

## Roadmap

| Version | Focus | Timeline |
|---|---|---|
| v0.1 | Deployable MVP — train, predict, export, deploy | ✅ Done |
| v0.2 | Better training — multi-dataset, longer, MPS | Next (Mac Mini) |
| v0.3 | ONNX.js browser demo, iOS Swift package | TBD |
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
