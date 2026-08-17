---
title: "NanoForecast v0.5: From MASE 2.73 → 1.326 with Zero Architecture Changes"
thumbnail: /blog/assets/nanoforecast-v05/thumbnail.png
authors:
- user: GautamKishore
  guest: true
  org: eulogik
---

# NanoForecast v0.5: From MASE 2.73 → 1.326 with Zero Architecture Changes

*How fixing training data pipelines and loss computation slashed error by 51% on a 8.3M parameter model.*

---

## The Problem

Time series forecasting models fall into two camps:
1. **Massive foundation models** (TimesFM, Chronos, Lag-Llama) — require GPUs, 100M+ parameters, and can't run on edge devices
2. **Small deployable models** — tiny and fast, but accuracy is usually mediocre

**NanoForecast v0.5 proves you can have both**: 8.3M parameters, CPU inference, ONNX export, streaming — AND MASE 1.326 (competitive with models 24× larger).

## The Key Insight

We didn't change the architecture between v0.3 and v0.5. Same LongConv + DeltaNet RNN, same gated router, same MLP blocks. **The 51% improvement came entirely from fixing the training pipeline**:

1. **Corrected loss computation** — v0.3 had a subtle bug where the reconstruction loss was computed on mismatched tensor shapes
2. **Fixed tensor truncation** — multi-horizon training was truncating context tensors that should have been preserved
3. **Better data mixing** — 6 real datasets + 10K synthetic records, properly shuffled with resolution-aware batching
4. **Proper training recipe** — OneCycleLR with 3e-5 peak LR, gradient clipping at 1.0, bfloat16 mixed precision

This is a lesson for the community: **sometimes the biggest gains come from fixing your data pipeline, not your model**.

## Benchmark Results

| Dataset | v0.3 MASE | v0.5 MASE | Improvement |
|:---|---:|---:|:---|
| ETTh1 | 1.95 | **0.913** | ↓ 53% |
| ETTh2 | 2.74 | **0.914** | ↓ 67% |
| ETTm1 | 2.17 | **1.305** | ↓ 40% |
| exchange_rate | 7.44 | **3.578** | ↓ 52% |
| electricity | 1.29 | **0.709** | ↓ 45% |
| traffic | 0.81 | **0.535** | ↓ 34% |
| **Overall** | **2.73** | **1.326** | **↓ 51%** |

The model achieves **MASE < 1.0 on 3 of 6 datasets** — a threshold that previously required models 10-100× larger.

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

| Platform | Latency | Memory |
|:---|---:|---:|
| CPU (laptop) | ~5ms | ~33MB |
| Raspberry Pi (ARM) | ~50ms | ~33MB |
| ONNX (browser) | ~10ms | ~1.4MB |
| FastAPI (server) | ~5ms | ~33MB |

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
| **Parameters** | 8.3M | 200M | 8M–710M | 16.6M |
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
