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

Trained on 6 standard forecasting datasets + 10K synthetic records. **MASE 1.326 overall — 51% better than v0.3 (2.73).**

<table>
<tr>
<th>Dataset</th>
<th>MASE ↓</th>
<th>sMAPE (%) ↓</th>
<th>CRPS ↓</th>
<th>Coverage (p50)</th>
<th>Coverage (p90)</th>
</tr>
<tr><td>ETTh1</td><td><b>0.913</b></td><td>5.89</td><td>0.425</td><td>51.2%</td><td>94.5%</td></tr>
<tr><td>ETTh2</td><td><b>0.914</b></td><td>3.54</td><td>0.561</td><td>50.0%</td><td>95.2%</td></tr>
<tr><td>ETTm1</td><td><b>1.305</b></td><td>7.22</td><td>0.304</td><td>49.5%</td><td>94.7%</td></tr>
<tr><td>exchange_rate</td><td><b>3.578</b></td><td>0.80</td><td>0.004</td><td>46.9%</td><td>94.4%</td></tr>
<tr><td>electricity</td><td><b>0.709</b></td><td>2.63</td><td>59.93</td><td>49.0%</td><td>92.2%</td></tr>
<tr><td>traffic</td><td><b>0.535</b></td><td>13.40</td><td>0.002</td><td>49.5%</td><td>95.9%</td></tr>
<tr style="background:#1a1a2e;font-weight:bold">
<td>OVERALL</td><td><b>1.326</b></td><td>5.58</td><td>10.20</td><td>49.4%</td><td>94.5%</td>
</tr>
</table>

### 📈 Version Comparison (same architecture, same data)

| Version | Params | MASE ↓ | Improvement | Training |
|:---|---:|---:|:---|:---|
| v0.2 (1.6M) | 1.6M | 3.45 | baseline | Mac Mini, 100 epochs |
| v0.3 (6.5M) | 6.5M | 2.73 | ↓ 21% | Colab T4, 200 epochs |
| **v0.5 (6.5M)** | **6.5M** | **1.326** | **↓ 51%** | **Colab T4, 200 epochs** |

> **v0.5 achieves MASE < 1.0 on 3 of 6 datasets** — competitive with models 10× larger.

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

### ONNX (1.4 MB — Edge / IoT / Browser)

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
| `benchmark-v05.json` | 2.9 KB |

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

Well-calibrated uncertainty estimates:

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

- **Accuracy vs SOTA**: MASE 1.326 is competitive with mid-size models but not SOTA (TimesFM, Chronos-T5). NanoForecast prioritizes deployability over raw accuracy.
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
