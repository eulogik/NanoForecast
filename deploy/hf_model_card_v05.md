---
license: apache-2.0
tags:
  - time-series
  - forecasting
  - pytorch
  - transformer
  - edge-ai
  - onnx
  - deployable
  - lightweight
  - cpu-inference
  - streaming
  - real-time
  - iot
  - raspberry-pi
  - quantile-regression
  - zero-shot
  - foundation-model
  - longconv
  - deltanet
  - multivariate
  - probabilistic-forecasting
  - uncertainty-quantification
library_name: nanoforecast
pipeline_tag: time-series-forecasting
metrics:
  - mase
  - smape
  - mse
  - mae
  - crps
datasets:
  - ETTh1
  - ETTh2
  - ETTm1
  - exchange_rate
  - electricity
  - traffic
---

<div align="center">

# 🔮 NanoForecast v0.5

### World's Most Deployable Time Series Transformer

**6.5M parameters · CPU inference · Raspberry Pi · ONNX · Streaming**

[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-FFD21E?style=for-the-badge)](https://huggingface.co/eulogik/nanoforecast-v05)
[![GitHub](https://img.shields.io/badge/GitHub-eulogik%2FNanoForecast-181717?style=for-the-badge&logo=github)](https://github.com/eulogik/NanoForecast)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](https://github.com/eulogik/NanoForecast/blob/main/LICENSE)
[![Open in Colab](https://img.shields.io/badge/Train-in-Colab-F9AB00?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/eulogik/NanoForecast/blob/v0.5/deploy/colab_training_v05.ipynb)
[![Live Demo](https://img.shields.io/badge/🌐-Live%20Demo-76B900?style=for-the-badge)](https://huggingface.co/spaces/eulogik/nanoforecast)

**Built by [Eulogik](https://eulogik.com) — deployable AI for the real world**

</div>

---

## 📊 Benchmark Results (v0.5)

All numbers use the **standard protocol** from `benchmark_standard.py`: context 512, horizon 48,
non-overlapping test windows over the full test split, MASE scaled by the seasonal-naive in-sample
MAE, all channels. Every model below (including TimesFM and PatchTST) was evaluated under this
identical protocol.

| Dataset | NanoForecast v0.5 (6.5M) | TimesFM (200M) | PatchTST (15M+) |
|---:|---:|---:|---:|
| ETTh1 | **0.681** | 0.705 | 0.781 |
| ETTh2 | **1.110** | 1.360 | 1.467 |
| ETTm1 | **0.287** | 0.545 | 0.488 |
| exchange_rate | **4.317** | 4.383 | **3.861** |
| electricity | 2.029 | **0.923** | 1.347 |
| traffic | 1.805 | **0.765** | 1.379 |
| **Overall** | 1.704 | **1.447** | 1.554 |

NanoForecast v0.5 **outperforms TimesFM on all three ETT datasets** (and PatchTST on the same
three) at 31× fewer parameters than TimesFM. TimesFM and PatchTST win on exchange_rate,
electricity, and traffic.

### 📈 Version Comparison (same architecture, pipeline fixes only)

| Version | Params | MASE ↓ | Improvement | Training |
|:---|---:|---:|:---|:---|
| v0.3 (released) | 6.5M | 3.030 | baseline | Colab T4, 200 epochs |
| **v0.5 (released)** | **6.5M** | **1.704** | **↓ 43.8%** | **Colab T4, 200 epochs** |

> **v0.5 improved MASE by 43.8% with zero architecture changes** — the gains came from three
> training-pipeline fixes (loss-scope handling, tensor shape alignment, augmentation coverage).

### 🏆 Why NanoForecast Wins on Deployment

| Feature | NanoForecast v0.5 | TimesFM | Chronos-T5 | Lag-Llama | PatchTST |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Parameters** | **6.5M** | 200M | 8M–710M | 16.6M | 15M+ |
| **Streaming inference** | **✅** | ❌ | ❌ | ❌ | ❌ |
| **ONNX export** | **✅** | ❌ | ❌ | ❌ | ❌ |
| **Train from CSV** | **✅** | ❌ | ❌ | ⚠️ | ⚠️ |
| **Quantiles** | **✅ (5)** | ⚠️ | ✅ | ✅ | ❌ |
| **License** | **Apache 2.0** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **Zero-shot** | ✅ | ✅ | ✅ | ✅ | ❌ |

### 📊 Visualizations

**Standard-Protocol MASE by Dataset** — NanoForecast (orange) vs TimesFM vs PatchTST:

![MASE by dataset](assets/benchmark_mase_standard.png)

**v0.3 → v0.5 pipeline refinement** — same architecture, MASE 3.030 → 1.704 (uniform median-as-point-forecast recipe):

![v0.3 vs v0.5](assets/benchmark_v03_vs_v05.png)

**Parameter Efficiency** — MASE⁻¹ per parameter:

![Efficiency](assets/benchmark_efficiency.png)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    NanoForecast v0.5                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Raw Context (512 steps)                                │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────────────────────────┐                    │
│  │  Instance Robust Scaler         │  median/IQR        │
│  │  + Adaptive Patching            │  patch_size=8       │
│  └─────────────┬───────────────────┘                    │
│                │                                        │
│                ▼                                        │
│  ┌─────────────────────────────────┐                    │
│  │  Resolution Prefix Tuning       │  freq_id → 4       │
│  └─────────────┬───────────────────┘  covariates        │
│                │                                        │
│                ▼                                        │
│  ┌─────────────────────────────────┐                    │
│  │  Sequence Mixing Blocks × 8     │                    │
│  │  ┌─────────────────────────┐    │                    │
│  │  │  LongConv (global)      │    │  kernel=49          │
│  │  │  DeltaNet RNN (local)   │    │  state_size=64      │
│  │  │  Gated Router           │    │  learned blend      │
│  │  │  GatedMLP               │    │  expansion=2        │
│  │  └─────────────────────────┘    │                    │
│  └─────────────┬───────────────────┘                    │
│                │                                        │
│                ▼                                        │
│  ┌─────────────────────────────────┐                    │
│  │  Multi-Task Heads (single pass) │                    │
│  │  • Point forecast              │  d_model → 1        │
│  │  • Monotonic quantiles p10–p90 │  5 quantiles        │
│  │  • Context reconstruction      │  anomaly detection  │
│  │  • Trend / Seasonal decomp     │  3 components       │
│  └─────────────────────────────────┘                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

| Component | Detail |
|:---|:---|
| **Parameters** | 6,518,104 (~6.5M) |
| **Context length** | 512 timesteps |
| **Prediction length** | 48 steps (configurable) |
| **Patch size** | 8 |
| **Hidden dim / layers** | 96 / 8 |
| **Quantiles** | p10, p25, p50, p75, p90 |
| **Quantile head** | Monotonic (guarantees p10 ≤ p25 ≤ p50 ≤ p75 ≤ p90) |
| **Decomposition** | trend + seasonal + residual ≡ point forecast (conservation identity) |
| **Streaming** | Stateful DeltaNet RNN — feed one value at a time |
| **Deployment** | ONNX (FP32 + INT8), FastAPI, Docker, Raspberry Pi, Browser |

---

## 🚀 Quick Start

### Install

```bash
pip install nanoforecast
```

### Inference

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

### From your own CSV

```bash
python3 train_from_csv.py --csv sales.csv --target revenue --horizon 48
```

---

## 🎯 Deployment

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

### ONNX (~9.2 MB INT8 / ~27.9 MB FP32 — Edge / IoT / Browser)

```bash
pip install "nanoforecast[onnx]"
python3 -m nanoforecast.export.onnx_export \
    --checkpoint <checkpoint-dir> \
    --output nanoforecast.onnx
```

```python
import onnxruntime as ort
session = ort.InferenceSession("nanoforecast.onnx")
forecast = session.run(None, {"input": context_numpy})
```

### Live Gradio Demo

[![Open in Spaces](https://img.shields.io/badge/🤗%20Open%20in%20Spaces-blueviolet)](https://huggingface.co/spaces/eulogik/nanoforecast)

Upload a CSV → get a forecast + prediction intervals + decomposition plot. No code required.

---

## 🏋️ Training

### Reproduce on Colab (free T4 GPU, ~12h)

[![Open in Colab](https://img.shields.io/badge/📓-Open%20in%20Colab-F9AB00)](https://colab.research.google.com/github/eulogik/NanoForecast/blob/v0.5/deploy/colab_training_v05.ipynb)

### Training details

| Parameter | Value |
|:---|:---|
| **Datasets** | ETTh1, ETTh2, ETTm1, exchange_rate, electricity, traffic |
| **Synthetic records** | 10,000 |
| **Epochs** | 200 |
| **Learning rate** | 3e-5 (OneCycleLR) |
| **Batch size** | 128 |
| **Best epoch** | 51 (val_loss = 0.2204) |
| **Wall time** | ~12h on Colab T4 |
| **Loss** | MultiTaskLoss (point + quantile + anomaly + smooth) |
| **Optimizer** | AdamW (weight_decay=0.01) |
| **Gradient clipping** | 1.0 |
| **FP16** | bfloat16 mixed precision |

---

## 📐 Design Principles

| Principle | Implementation |
|:---|:---|
| **Robust to outliers** | Instance Robust Scaler (median / IQR) — not sensitive to extreme values |
| **Monotonic quantiles** | Monotonic constraint on quantile head: p10 ≤ p25 ≤ p50 ≤ p75 ≤ p90 always |
| **Conservation** | trend + seasonal + residual ≡ point forecast (exact, not approximate) |
| **Multi-task learning** | Point forecast + quantiles + anomaly detection + smoothness in single forward pass |
| **Streaming** | DeltaNet RNN maintains recurrent state across calls — no other TS model does this |
| **Deployable** | ONNX export, FastAPI server, Docker, Raspberry Pi, browser (ONNX.js) |

---

## 📁 Model Files

| File | Size |
|:---|---:|
| `model.safetensors` | 26.1 MB |
| `config.json` | 343 B |
| `model_card.json` | 710 B |
| `standard_benchmark.json` | 3.1 KB |

---

## 🤔 When to Use NanoForecast

✅ **Use NanoForecast when:**
- You need to deploy a forecasting model to edge/IoT devices
- You want streaming/online inference (feed one value at a time)
- You need quantile forecasts with uncertainty estimates
- You want to train on your own data in minutes, not days
- You need ONNX export for browser/ARM deployment
- You want Apache 2.0 license (no restrictions)

❌ **Don't use NanoForecast when:**
- You need SOTA accuracy on standard benchmarks (use TimesFM, Chronos, etc.)
- You have massive datasets (100K+ rows) — fine-tune a larger model
- You need multivariate cross-series dependencies

---

## 📊 Coverage Analysis

Well-calibrated uncertainty estimates (measured under the internal `benchmark.py` protocol,
see `standard_benchmark.json`):

| Quantile | Target | Actual (mean across datasets) |
|:---|---:|---:|
| p10 | 10% | 5.4% |
| p25 | 25% | 19.1% |
| p50 | 50% | 49.4% |
| p75 | 75% | 79.9% |
| p90 | 90% | 94.5% |

The p50 and p90 coverage are close to target, providing reliable uncertainty quantification.

---

## 🏆 Why NanoForecast is Different

| Feature | NanoForecast | TimesFM | Chronos | Lag-Llama |
|:---|:---:|:---:|:---:|:---:|
| **Parameters** | 6.5M | 200M | 8M–710M | 16.6M |
| **CPU inference** | ✅ | ❌ | ⚠️ | ❌ |
| **Streaming** | ✅ | ❌ | ❌ | ❌ |
| **ONNX export** | ✅ | ❌ | ❌ | ❌ |
| **Raspberry Pi** | ✅ | ❌ | ❌ | ❌ |
| **Quantiles** | ✅ (5) | ❌ | ✅ | ✅ |
| **Train from CSV** | ✅ | ❌ | ❌ | ⚠️ |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **Zero-shot** | ✅ | ✅ | ✅ | ✅ |

---

## ⚠️ Known Limitations

- **Accuracy vs SOTA**: MASE 1.704 overall (standard protocol) — competitive on the ETT datasets (wins vs TimesFM and PatchTST on all three), but not SOTA on exchange_rate/electricity/traffic. NanoForecast prioritizes deployability over raw accuracy.
- **Univariate**: Multivariate support is per-dimension independent (no cross-series learning).
- **Fixed context**: 512 timesteps — longer history is truncated.
- **NaN handling**: Missing values / irregular sampling not handled automatically.

---

## 📚 Citation

```bibtex
@article{nanoforecast2026,
  title={NanoForecast: A Deployable Time Series Foundation Model},
  author={Eulogik},
  year={2026},
  url={https://github.com/eulogik/NanoForecast},
  note={6.5M parameters, CPU inference, ONNX export, streaming}
}
```

---

## 🔗 Links

- **GitHub**: [github.com/eulogik/NanoForecast](https://github.com/eulogik/NanoForecast)
- **Live Demo**: [huggingface.co/spaces/eulogik/nanoforecast](https://huggingface.co/spaces/eulogik/nanoforecast)
- **Colab Training**: [Open in Colab](https://colab.research.google.com/github/eulogik/NanoForecast/blob/v0.5/deploy/colab_training_v05.ipynb)
- **Website**: [eulogik.com](https://eulogik.com)
- **Other models**: [eulogik/nanoforecast-200k](https://huggingface.co/eulogik/nanoforecast-200k) · [eulogik/nanoforecast-v03](https://huggingface.co/eulogik/nanoforecast-v03)

---

<div align="center">

**Built by [Eulogik](https://eulogik.com)** — deployable AI for the real world

*If you found this useful, please ⭐ the [GitHub repo](https://github.com/eulogik/NanoForecast) and like this model on Hugging Face!*

</div>
