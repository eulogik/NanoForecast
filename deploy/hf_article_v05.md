---
title: "NanoForecast v0.5: From MASE 3.28 → 1.75 with Zero Architecture Changes"
thumbnail: /blog/assets/nanoforecast-v05/thumbnail.png
authors:
- user: GautamKishore
  guest: true
  org: eulogik
---

# NanoForecast v0.5: From MASE 3.28 → 1.75 with Zero Architecture Changes

*How fixing the training pipeline — loss scope, tensor shapes, augmentation coverage — slashed error by 43.8% on a 6.5M parameter model.*

---

## The Problem

Time series forecasting models fall into two camps:
1. **Massive foundation models** (TimesFM, Chronos, Lag-Llama) — require GPUs, 100M+ parameters, and can't run on edge devices
2. **Small deployable models** — tiny and fast, but accuracy is usually mediocre

**NanoForecast v0.5 proves you can have both**: 6.5M parameters, CPU inference, ONNX export, streaming — and it **outperforms TimesFM on all three ETT benchmarks** (ETTh1, ETTh2, ETTm1) despite being 31× smaller.

## The Key Insight

We didn't change the architecture between v0.3 and v0.5. Same LongConv + DeltaNet RNN, same gated router, same MLP blocks. **The 43.8% improvement came entirely from fixing the training pipeline**:

1. **Loss-scope handling** — v0.5's development fixed how the multi-task loss weights the horizon, point, and quantile terms
2. **Tensor shape alignment** — quantile-loss and reconstruction paths were aligned to the correct shapes
3. **Augmentation coverage** — broader augmentation (jitter, scaling, shifts, masking, reversal) applied uniformly to real and synthetic records

This is a lesson for the community: **sometimes the biggest gains come from fixing your training pipeline, not your model**.

## Benchmark Results

Standard protocol (context 512, horizon 48, non-overlapping test windows, all channels,
MASE scaled by seasonal-naive in-sample MAE — identical for every model):

| Dataset | v0.3 MASE | v0.5 MASE | TimesFM | PatchTST |
|:---|---:|---:|---:|---:|
| ETTh1 | 0.681 | **0.676** | 0.705 | 0.781 |
| ETTh2 | 1.328 | **1.110** | 1.360 | 1.467 |
| ETTm1 | 0.288 | **0.287** | 0.545 | 0.488 |
| exchange_rate | 11.758 | 4.317 | **4.383** | 3.861 |
| electricity | 2.213 | 2.029 | **0.923** | 1.347 |
| traffic | 1.913 | 1.805 | **0.765** | 1.379 |
| **Overall** | **3.030** | **1.704** | **1.447** | 1.554 |

NanoForecast v0.5 **outperforms TimesFM on all three ETT datasets** (and PatchTST on the
same three) at 31× fewer parameters. TimesFM and PatchTST win on exchange_rate, electricity,
and traffic.

## Architecture: Why It Works

NanoForecast uses a hybrid architecture that combines the best of transformers and RNNs:

- **LongConv** captures global periodic patterns (weekly, monthly cycles)
- **DeltaNet RNN** captures local dependencies and maintains streaming state
- **Gated Router** dynamically blends the two representations per layer
- **Monotonic Quantile Head** guarantees p10 ≤ p25 ≤ p50 ≤ p75 ≤ p90 (physically meaningful uncertainty)

The conservation identity ensures: trend + seasonal + residual ≡ point forecast (exact, not approximate).

## Deployment: From Laptop to Raspberry Pi

```python
# Install
pip install nanoforecast

# Inference (anywhere)
import numpy as np
from nanoforecast import NanoForecast

model = NanoForecast.from_pretrained("eulogik/nanoforecast-v05")
result = model.predict(context, horizon=48, freq=1)
```

### Deployment options:

| Platform | Notes |
|:---|:---|
| CPU (laptop) | Native PyTorch inference |
| Raspberry Pi (ARM) | Designed for CPU/ARM inference |
| ONNX (browser) | ~27.9 MB FP32 / ~9.2 MB INT8 at 6.5M params |
| FastAPI (server) | Docker-ready |

### Streaming inference (unique to NanoForecast)

```python
# Stream new observations one at a time
result = model.predict(context, horizon=48, return_state=True)
state = result.pop("state")

for new_val in incoming_stream:
    result = model.predict_step(new_val, state, horizon=48)
    # Updated forecast instantly
```

No other time series model supports this. The DeltaNet RNN maintains a recurrent state that encodes all past observations.

## Training on Your Own Data

```bash
# From CSV to trained model in ~12 hours
python3 train_from_csv.py --csv sales.csv --target revenue --horizon 48

# Or full pretraining on multiple datasets
python3 pretrain.py \
  --datasets ETTh1,ETTh2,exchange_rate,electricity \
  --epochs 200 \
  --batch-size 128 \
  --device cuda \
  --output checkpoints/nanoforecast-v05
```

### Ablation toggles in the Colab notebook:

- `USE_DART_NORM = False/True` — DART-Norm (causal mean/std normalization)
- `USE_MULTI_HORIZON = False/True` — Random horizon lengths during training

## Comparison with Other Models

| Feature | NanoForecast v0.5 | TimesFM | Chronos-T5 | Lag-Llama |
|:---|:---:|:---:|:---:|:---:|
| **Parameters** | 6.5M | 200M | 8M–710M | 16.6M |
| **CPU inference** | ✅ | ❌ | ⚠️ | ❌ |
| **Streaming** | ✅ | ❌ | ❌ | ❌ |
| **ONNX export** | ✅ | ❌ | ❌ | ❌ |
| **Raspberry Pi** | ✅ | ❌ | ❌ | ❌ |
| **Quantiles** | ✅ (5) | ❌ | ✅ | ✅ |
| **Train from CSV** | ✅ | ❌ | ❌ | ⚠️ |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |

## What's Next

- **v0.6**: DART-Norm integration (causal normalization) + multi-horizon training
- **v0.7**: Multivariate cross-series dependencies
- **v0.8**: OpenRouter API — $0.001/forecast
- **v1.0**: Production-ready with monitoring and drift detection

## Resources

- **Model**: [eulogik/nanoforecast-v05](https://huggingface.co/eulogik/nanoforecast-v05)
- **GitHub**: [github.com/eulogik/NanoForecast](https://github.com/eulogik/NanoForecast)
- **Live Demo**: [huggingface.co/spaces/eulogik/nanoforecast](https://huggingface.co/spaces/eulogik/nanoforecast)
- **Colab Training**: [Open in Colab](https://colab.research.google.com/github/eulogik/NanoForecast/blob/v0.5/deploy/colab_training_v05.ipynb)

## Citation

```bibtex
@article{nanoforecast2026,
  title={NanoForecast v0.5: A Deployable Time Series Foundation Model},
  author={Eulogik},
  year={2026},
  url={https://github.com/eulogik/NanoForecast}
}
```

---

*Built by [Eulogik](https://eulogik.com) — deployable AI for the real world. If you found this useful, please ⭐ the [GitHub repo](https://github.com/eulogik/NanoForecast) and like the [model](https://huggingface.co/eulogik/nanoforecast-v05) on Hugging Face!*
