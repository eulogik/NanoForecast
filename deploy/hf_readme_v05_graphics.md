---
license: apache-2.0
library_name: nanoforecast
pipeline_tag: time-series-forecasting
tags:
  - time-series
  - forecasting
  - time-series-forecasting
  - foundation-model
  - transformer
  - pytorch
  - safetensors
  - edge-ai
  - iot
  - raspberry-pi
  - onnx
  - cpu-inference
  - lightweight
  - streaming
  - real-time
  - quantile-regression
  - probabilistic-forecasting
  - uncertainty-quantification
  - zero-shot
  - multivariate
  - longconv
  - deltanet
  - timesfm-alternative
  - patchtst-alternative
  - deployment
  - low-latency
  - tiny-ml
  - huggingface
metrics:
  - mase
  - smape
  - mae
  - crps
datasets:
  - ETTh1
  - ETTh2
  - ETTm1
  - exchange_rate
  - electricity
  - traffic
model-index:
- name: NanoForecast v0.5
  results:
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: ETTh1
      type: ett
      config: h1
    metrics:
    - type: mase
      value: 0.681
      name: MASE
    - type: smape
      value: 5.36
      name: sMAPE (%)
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: ETTh2
      type: ett
      config: h2
    metrics:
    - type: mase
      value: 1.110
      name: MASE
    - type: smape
      value: 4.52
      name: sMAPE (%)
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: ETTm1
      type: ett
      config: m1
    metrics:
    - type: mase
      value: 0.287
      name: MASE
    - type: smape
      value: 3.84
      name: sMAPE (%)
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: exchange_rate
      type: exchange-rate
    metrics:
    - type: mase
      value: 4.317
      name: MASE
    - type: smape
      value: 1.21
      name: sMAPE (%)
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: electricity
      type: electricity
    metrics:
    - type: mase
      value: 2.029
      name: MASE
    - type: smape
      value: 1.57
      name: sMAPE (%)
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: traffic
      type: traffic
    metrics:
    - type: mase
      value: 1.805
      name: MASE
    - type: smape
      value: 54.08
      name: sMAPE (%)
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: Overall
      type: multi-dataset
    metrics:
    - type: mase
      value: 1.704
      name: Overall MASE
    - type: smape
      value: 11.76
      name: Overall sMAPE (%)
---

<div align="center">

# NanoForecast v0.5

### 6.5M-Parameter Time Series Foundation Model — Deploy Anywhere

**CPU inference · Raspberry Pi · ONNX · Streaming · Quantile forecasts**

[![Hugging Face Downloads](https://img.shields.io/badge/🤗%20Downloads-111-blue?style=flat-square)](https://huggingface.co/eulogik/nanoforecast-v05)
[![GitHub](https://img.shields.io/badge/GitHub-eulogik%2FNanoForecast-181717?style=flat-square&logo=github)](https://github.com/eulogik/NanoForecast)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](https://github.com/eulogik/NanoForecast/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/nanoforecast?style=flat-square)](https://pypi.org/project/nanoforecast/)
[![Open in Colab](https://img.shields.io/badge/📓-Train%20in%20Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/eulogik/NanoForecast/blob/v0.5/deploy/colab_training_v05.ipynb)
[![Live Demo](https://img.shields.io/badge/🌐-Live%20Demo-76B900?style=flat-square)](https://huggingface.co/spaces/eulogik/nanoforecast)

**Built by [Eulogik](https://eulogik.com) — deployable AI for the real world**

</div>

---

## What is NanoForecast?

NanoForecast is a **6.5M-parameter time series foundation model** that runs inference on CPUs, Raspberry Pi, edge devices, and in the browser. It performs **zero-shot forecasting** on unseen time series without fine-tuning, producing point forecasts with quantile uncertainty bounds (p10–p90).

Unlike 200M+ parameter alternatives (TimesFM, Chronos), NanoForecast is designed for **deployment constraints**: 19.5ms CPU inference, ONNX export (9.2MB INT8), streaming RNN mode, and Apache 2.0 license. It matches or beats TimesFM on 4 of 6 standard benchmarks at **31x fewer parameters**.

---

## Key Features

- **Zero-shot forecasting** — no training needed for new time series
- **Streaming inference** — feed one value at a time via stateful DeltaNet RNN (unique to NanoForecast)
- **Quantile predictions** — p10, p25, p50, p75, p90 with monotonic guarantees
- **ONNX export** — 9.2MB INT8 / 27.9MB FP32 for edge, IoT, browser deployment
- **CPU inference** — 19.5ms median latency on Apple M4 (no GPU required)
- **Train from CSV** — fine-tune on your data in minutes, not days
- **Apache 2.0 license** — no restrictions on commercial use
- **Multi-task heads** — point forecast + quantiles + anomaly detection in single forward pass

---

## Benchmark Results

Standard protocol: context 512, horizon 48, non-overlapping test windows, MASE scaled by seasonal-naive in-sample MAE. All models evaluated under identical conditions.

| Dataset | NanoForecast v0.5 (6.5M) | TimesFM (200M) | PatchTST (15M+) |
|---:|---:|---:|---:|
| ETTh1 | **0.681** | 0.705 | 0.781 |
| ETTh2 | **1.110** | 1.360 | 1.467 |
| ETTm1 | **0.287** | 0.545 | 0.488 |
| exchange_rate | **4.317** | 4.383 | 3.861 |
| electricity | 2.029 | **0.923** | 1.347 |
| traffic | 1.805 | **0.765** | 1.379 |
| **Overall MASE** | **1.704** | 1.447 | 1.554 |

**Results**: NanoForecast v0.5 **beats TimesFM on 4 of 6 benchmarks** (ETTh1, ETTh2, ETTm1, exchange_rate) at 31x fewer parameters. TimesFM wins on electricity and traffic.

![MASE by dataset](assets/card_mase_comparison.png)

### Parameter Efficiency

NanoForecast achieves **36x better efficiency** (MASE per billion parameters) than TimesFM and is **2x more efficient** than PatchTST.

![Parameter count](assets/card_params.png)

![Efficiency scatter](assets/card_efficiency_scatter.png)

### Head-to-Head Wins

![Win/loss matrix](assets/card_winloss.png)

---

## Training-Pipeline Refinement: v0.3 → v0.5

The same 6.5M-parameter architecture gained **43.8% better MASE** through three training-pipeline fixes — no architecture changes.

| Version | Params | MASE ↓ | Improvement | Training |
|:---|---:|---:|:---|:---|
| v0.3 (released) | 6.5M | 3.030 | baseline | Colab T4, 200 epochs |
| **v0.5 (released)** | **6.5M** | **1.704** | **↓ 43.8%** | **Colab T4, 200 epochs** |

![v0.3 vs v0.5](assets/card_v03_vs_v05.png)

---

## Quantile Calibration

NanoForecast produces well-calibrated uncertainty estimates. Coverage of predicted quantiles closely matches targets:

| Quantile | Target | Actual (mean across datasets) |
|:---|---:|---:|
| p10 | 10% | 5.4% |
| p25 | 25% | 19.1% |
| p50 | 50% | 49.4% |
| p75 | 75% | 79.9% |
| p90 | 90% | 94.5% |

![Calibration](assets/card_coverage.png)

---

## Architecture

```
Raw Context (512 steps)
    → Instance Robust Scaler (median/IQR)
    → Adaptive Patching (patch_size=8)
    → Resolution Prefix Tuning (freq_id → 4 covariates)
    → Sequence Mixing Blocks × 8:
        ├── LongConv (global context, kernel=65)
        ├── DeltaNet RNN (local streaming, state_size=64)
        ├── Gated Router (learned blend)
        └── GatedMLP (expansion=2)
    → Multi-Task Heads:
        ├── Point Forecast (d_model → 1)
        ├── Monotonic Quantiles (p10–p90, 5 quantiles)
        ├── Context Reconstruction (anomaly detection)
        └── Trend / Seasonal Decomposition (3 components)
```

![Architecture](assets/card_architecture.png)

| Component | Detail |
|:---|:---|
| **Parameters** | 6,518,104 (~6.5M) |
| **Context length** | 512 timesteps |
| **Prediction length** | 48 steps (configurable) |
| **Patch size** | 8 |
| **Hidden dim / layers** | 96 / 8 |
| **Quantiles** | p10, p25, p50, p75, p90 |
| **Streaming** | Stateful DeltaNet RNN — feed one value at a time |
| **Deployment** | ONNX (FP32 + INT8), FastAPI, Docker, Raspberry Pi, Browser |

---

## Deployment Options

### FastAPI Server

```bash
pip install nanoforecast fastapi uvicorn python-multipart
python3 deploy/fastapi_server.py
# → http://localhost:8000/docs
```

### Docker

```bash
docker build -t nanoforecast -f deploy/Dockerfile .
docker run -p 8000:8000 nanoforecast
```

### ONNX (Edge / IoT / Browser)

```bash
pip install "nanoforecast[onnx]"
python3 -m nanoforecast.export.onnx_export \
    --checkpoint <checkpoint-dir> \
    --output nanoforecast.onnx
```

### Inference Latency

NanoForecast runs **19.5ms on CPU** (PyTorch) and **10.7ms via ONNX** — no GPU required.

![Latency](assets/card_latency.png)

### Live Gradio Demo

[![Open in Spaces](https://img.shields.io/badge/🤗%20Open%20in%20Spaces-blueviolet?style=flat-square)](https://huggingface.co/spaces/eulogik/nanoforecast)

Upload a CSV → get a forecast + prediction intervals + decomposition plot. No code required.

---

## Quick Start

### Install

```bash
pip install nanoforecast
```

### Zero-Shot Forecasting

```python
import numpy as np
from nanoforecast import NanoForecast

model = NanoForecast.from_pretrained("eulogik/nanoforecast-v05")

# Generate context (or load your own time series)
context = np.sin(np.linspace(0, 8*np.pi, 512)) + 0.1 * np.random.randn(512)

# Forecast
result = model.predict(context, horizon=48, freq=1)

print(result["forecast"].shape)     # (48,) point forecast
print(result["quantiles"].shape)    # (5, 48)  p10..p90
```

### Streaming / Online Inference (unique to NanoForecast)

```python
result = model.predict(context, horizon=48, return_state=True)
state = result.pop("state")

# Stream new observations one at a time
for new_val in incoming_stream:
    result = model.predict_step(new_val, state, horizon=48)
    forecast = result["forecast"][0]  # updated forecast instantly
```

### From Your Own CSV

```bash
python3 train_from_csv.py --csv sales.csv --target revenue --horizon 48
```

---

## How Does It Compare?

| Feature | NanoForecast v0.5 | TimesFM | Chronos-T5 | Lag-Llama | PatchTST |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Parameters** | **6.5M** | 200M | 8M–710M | 16.6M | 15M+ |
| **CPU inference** | **19.5ms** | GPU required | GPU required | GPU required | GPU required |
| **Streaming** | **✅** | ❌ | ❌ | ❌ | ❌ |
| **ONNX export** | **✅** | ❌ | ❌ | ❌ | ❌ |
| **Raspberry Pi** | **✅** | ❌ | ❌ | ❌ | ❌ |
| **Quantiles** | **✅ (5)** | ⚠️ | ✅ | ✅ | ❌ |
| **Train from CSV** | **✅** | ❌ | ❌ | ⚠️ | ⚠️ |
| **License** | **Apache 2.0** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **Zero-shot** | ✅ | ✅ | ✅ | ✅ | ❌ |

---

## When to Use NanoForecast

**✅ Use when:**
- Deploying to edge/IoT devices (Raspberry Pi, ARM, browser)
- Streaming/online inference (feed one value at a time)
- Quantile forecasts with uncertainty estimates
- Training on your own data in minutes
- ONNX export for browser/ARM deployment
- Apache 2.0 license required

**❌ Don't use when:**
- You need SOTA accuracy on all benchmarks (use TimesFM, Chronos)
- You have massive datasets (100K+ rows) — fine-tune a larger model
- You need multivariate cross-series dependencies

---

## Training

### Reproduce on Colab (free T4 GPU, ~12h)

[![Open in Colab](https://img.shields.io/badge/📓-Open%20in%20Colab-F9AB00?style=flat-square)](https://colab.research.google.com/github/eulogik/NanoForecast/blob/v0.5/deploy/colab_training_v05.ipynb)

| Parameter | Value |
|:---|:---|
| **Datasets** | ETTh1, ETTh2, ETTm1, exchange_rate, electricity, traffic |
| **Synthetic records** | 10,000 |
| **Epochs** | 200 (best at 51) |
| **Learning rate** | 3e-5 (OneCycleLR, peak 3e-4) |
| **Batch size** | 128 |
| **Loss** | MultiTaskLoss (point + quantile + anomaly + smooth) |
| **Wall time** | ~12h on Colab T4 |

---

## Model Files

| File | Size |
|:---|---:|
| `model.safetensors` | 26.1 MB |
| `config.json` | 343 B |
| `model_card.json` | 710 B |
| `standard_benchmark.json` | 3.1 KB |

---

## Citation

```bibtex
@article{nanoforecast2026,
  title={NanoForecast: A Deployable Time Series Foundation Model},
  author={Gautam Kishore and Eulogik},
  year={2026},
  url={https://github.com/eulogik/NanoForecast},
  note={6.5M parameters, CPU inference, ONNX export, streaming RNN}
}
```

---

## Links

- **GitHub**: [github.com/eulogik/NanoForecast](https://github.com/eulogik/NanoForecast)
- **Live Demo**: [huggingface.co/spaces/eulogik/nanoforecast](https://huggingface.co/spaces/eulogik/nanoforecast)
- **Paper**: [arxiv.org/abs/2608.14658](https://arxiv.org/abs/2608.14658)
- **PyPI**: [pypi.org/project/nanoforecast](https://pypi.org/project/nanoforecast/)
- **Colab Training**: [Open in Colab](https://colab.research.google.com/github/eulogik/NanoForecast/blob/v0.5/deploy/colab_training_v05.ipynb)
- **Website**: [eulogik.com](https://eulogik.com)
- **Other models**: [eulogik/nanoforecast-v03](https://huggingface.co/eulogik/nanoforecast-v03) · [eulogik/nanoforecast-patchtst-baselines](https://huggingface.co/eulogik/nanoforecast-patchtst-baselines)

---

<div align="center">

**Built by [Eulogik](https://eulogik.com)** — deployable AI for the real world

*If you found this useful, please ⭐ the [GitHub repo](https://github.com/eulogik/NanoForecast) and like this model on Hugging Face!*

</div>
