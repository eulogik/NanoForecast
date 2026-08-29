---
license: apache-2.0
library_name: pytorch
pipeline_tag: time-series-forecasting
tags:
  - time-series
  - forecasting
  - time-series-forecasting
  - pytorch
  - safetensors
  - edge-ai
  - iot
  - raspberry-pi
  - onnx
  - streaming
  - real-time
  - transformer
  - foundation-model
  - zero-shot
  - tiny-ml
  - timesfm-alternative
  - huggingface
  - lightweight
  - cpu-inference
metrics:
  - mase
  - smape
  - mae
  - crps
model-index:
- name: NanoForecast v0.3
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
      value: 1.946
      name: MASE
    - type: smape
      value: 12.06
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
      value: 2.741
      name: MASE
    - type: smape
      value: 10.47
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
      value: 2.174
      name: MASE
    - type: smape
      value: 10.70
      name: sMAPE (%)
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: exchange_rate
      type: exchange-rate
    metrics:
    - type: mase
      value: 7.442
      name: MASE
    - type: smape
      value: 1.72
      name: sMAPE (%)
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: electricity
      type: electricity
    metrics:
    - type: mase
      value: 1.294
      name: MASE
    - type: smape
      value: 4.76
      name: sMAPE (%)
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: traffic
      type: traffic
    metrics:
    - type: mase
      value: 0.807
      name: MASE
    - type: smape
      value: 24.00
      name: sMAPE (%)
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: Overall
      type: multi-dataset
    metrics:
    - type: mase
      value: 2.734
      name: Overall MASE
    - type: smape
      value: 10.62
      name: Overall sMAPE (%)
---

<div align="center">

# NanoForecast v0.3

### 6.5M-Parameter Time Series Foundation Model — Deploy Anywhere

**CPU inference · Raspberry Pi · ONNX · Streaming · Quantile forecasts**

[![Hugging Face Downloads](https://img.shields.io/badge/🤗%20Downloads-17-blue?style=flat-square)](https://huggingface.co/eulogik/nanoforecast-v03)
[![GitHub](https://img.shields.io/badge/GitHub-eulogik%2FNanoForecast-181717?style=flat-square&logo=github)](https://github.com/eulogik/NanoForecast)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](https://github.com/eulogik/NanoForecast/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/nanoforecast?style=flat-square)](https://pypi.org/project/nanoforecast/)
[![Live Demo](https://img.shields.io/badge/🌐-Live%20Demo-76B900?style=flat-square)](https://huggingface.co/spaces/eulogik/nanoforecast)

**Built by [Eulogik](https://eulogik.com) — deployable AI for the real world**

</div>

---

## What is NanoForecast v0.3?

NanoForecast v0.3 is the **initial release** of a 6.5M-parameter time series foundation model designed for deployment on CPUs, Raspberry Pi, edge devices, and in the browser. It performs zero-shot forecasting with quantile uncertainty bounds.

**This is a reference release.** For the latest improvements, see [eulogik/nanoforecast-v05](https://huggingface.co/eulogik/nanoforecast-v05) (43.8% better MASE with same architecture).

---

## Key Features

- **Zero-shot forecasting** — no training needed for new time series
- **Streaming inference** — feed one value at a time via stateful DeltaNet RNN
- **Quantile predictions** — p10, p25, p50, p75, p90 with monotonic guarantees
- **ONNX export** — 9.2MB INT8 for edge, IoT, browser deployment
- **CPU inference** — 19.5ms median latency on Apple M4
- **Train from CSV** — fine-tune on your data in minutes
- **Apache 2.0 license** — no restrictions on commercial use

---

## Benchmark Results

Standard protocol: context 512, horizon 48, non-overlapping test windows, MASE scaled by seasonal-naive in-sample MAE.

| Dataset | NanoForecast v0.3 (6.5M) | TimesFM (200M) | PatchTST (15M+) |
|---:|---:|---:|---:|
| ETTh1 | 1.946 | **0.705** | 0.781 |
| ETTh2 | 2.741 | **1.360** | 1.467 |
| ETTm1 | 2.174 | **0.545** | 0.488 |
| exchange_rate | 7.442 | 4.383 | **3.861** |
| electricity | 1.294 | 0.923 | **1.347** |
| traffic | 0.807 | **0.765** | 1.379 |
| **Overall MASE** | 2.734 | **1.447** | 1.554 |

**v0.3 → v0.5 improvement**: Overall MASE 3.030 → 1.704 (↓43.8%) with same architecture, pipeline fixes only.

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

model = NanoForecast.from_pretrained("eulogik/nanoforecast-v03")

context = np.sin(np.linspace(0, 8*np.pi, 512)) + 0.1 * np.random.randn(512)
result = model.predict(context, horizon=48, freq=1)

print(result["forecast"].shape)     # (48,) point forecast
print(result["quantiles"].shape)    # (5, 48)  p10..p90
```

### Streaming Inference

```python
result = model.predict(context, horizon=48, return_state=True)
state = result.pop("state")

for new_val in incoming_stream:
    result = model.predict_step(new_val, state, horizon=48)
    forecast = result["forecast"][0]
```

---

## Model Files

| File | Size |
|:---|---:|
| `model.safetensors` | 26.1 MB |
| `config.json` | 343 B |
| `model_card.json` | 710 B |

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
- **Latest version**: [eulogik/nanoforecast-v05](https://huggingface.co/eulogik/nanoforecast-v05)
- **Website**: [eulogik.com](https://eulogik.com)

---

<div align="center">

**Built by [Eulogik](https://eulogik.com)** — deployable AI for the real world

*If you found this useful, please ⭐ the [GitHub repo](https://github.com/eulogik/NanoForecast) and like this model on Hugging Face!*

</div>
