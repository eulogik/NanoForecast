---
title: NanoForecast v0.5
emoji: 🔮
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
python_version: 3.11
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - time-series
  - forecasting
  - pytorch
  - edge-ai
  - onnx
  - cpu-inference
  - quantile-regression
  - longconv
  - deltanet
  - probabilistic-forecasting
  - streaming
  - raspberry-pi
library_name: nanoforecast
pipeline_tag: time-series-forecasting
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
---

# 🔮 NanoForecast v0.5 — Time Series Forecasting

The most deployable time series foundation model: **6.5M parameters**, CPU inference,
ONNX export, streaming RNN, and quantile forecasts with prediction intervals.

- **Overall MASE:** 1.704 (standard protocol) — beats TimesFM on **4 of 6** benchmarks at 31× fewer parameters
- **Latency:** 19.5 ms CPU inference (10.7 ms via ONNX)
- **License:** Apache 2.0

## Try it

Upload a CSV, paste CSV text, or hit **Load example** — no GPU required. Max horizon is 48
(the model's native window). Forecast includes p10–p90 prediction intervals and a
trend/seasonal/residual decomposition.

## Links

- GitHub: [github.com/eulogik/NanoForecast](https://github.com/eulogik/NanoForecast)
- Model card: [eulogik/nanoforecast-v05](https://huggingface.co/eulogik/nanoforecast-v05)
- Paper: [arxiv.org/abs/2608.14658](https://arxiv.org/abs/2608.14658)
- Built by [Eulogik](https://eulogik.com)
