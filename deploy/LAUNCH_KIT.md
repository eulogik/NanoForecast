# NanoForecast v0.5 — Launch Kit

**Author**: Gautam Kishore, Eulogik  
**Contact**: gautam@eulogik.com  
**Date**: August 2026

---

## The Core Story

> A 6.5 million parameter AI model — trained on a free Colab T4 (~12 hours) — outperforms Google's 200 million parameter TimesFM on all three ETT forecasting benchmarks (ETTh1, ETTh2, ETTm1). And it runs on a $35 Raspberry Pi.

**Three provable claims that make this interesting:**

| Claim | Evidence |
|-------|----------|
| 31× smaller than Google's TimesFM | 6.5M params vs 200M params |
| Beats TimesFM on all three ETT benchmarks | Standard protocol: ETTh1 0.685 vs 0.705, ETTh2 1.109 vs 1.360, ETTm1 0.289 vs 0.545 |
| Beats its own predecessor by 46.6% with zero architecture changes | v0.3 → v0.5: MASE 3.282 → 1.752 (standard protocol, same 6.5M params) |

All numbers are measured under one identical standard protocol (context 512, horizon 48, non-overlapping test windows, all channels, MASE scaled by seasonal-naive in-sample MAE) with every model evaluated by us. Full protocol: `benchmark_standard.py`.

---

## The 10 Story Angles

---

### 1. THE EFFICIENCY ANGLE: "31× smaller model beats Google on all three ETT benchmarks"

**Audience**: Hacker News, tech press, engineers  
**Headline**: "Show HN: 6.5M param model beats Google's TimesFM (200M) on all three ETT forecasting benchmarks"

**The story**: Most AI progress is measured by scale — bigger models, more data, more compute. This project challenges that. A 6.5M parameter model, trained on a free Colab T4 in ~12 hours, beats a 200M parameter Google model on all three ETT datasets. Not across the board — TimesFM wins exchange_rate, electricity, and traffic. But on the ETT benchmarks, scale wasn't the deciding factor.

**Precise claims**:
- Size: 6.5M params vs TimesFM's 200M (31× smaller)
- Training cost: one free Colab T4 session, ~12 hours (checkpoint wall time 43,750s)
- Results (standard protocol): ETTh1 0.685 vs 0.705, ETTh2 1.109 vs 1.360, ETTm1 0.289 vs 0.545
- TimesFM wins on the remaining 3 benchmarks (exchange_rate, electricity, traffic) — be specific, don't overclaim

**Post template**:
```
We trained a time series forecasting model (6.5M params) on a free Colab T4 GPU for ~12 hours.

It beats Google's TimesFM (200M params) on all three ETT benchmarks:
• ETTh1: MASE 0.685 vs TimesFM 0.705
• ETTh2: MASE 1.109 vs TimesFM 1.360
• ETTm1: MASE 0.289 vs TimesFM 0.545

(Standard protocol: H=48, C=512, non-overlapping windows, all channels, seasonal-naive MASE — identical for both models.)

TimesFM still wins on exchange_rate, electricity, and traffic. We're not claiming a blanket victory — we're claiming scale isn't the only path.

The model also:
• Trains in ~12 hours on a single T4-class GPU (Google Colab)
• Runs on a $35 Raspberry Pi
• Exports to ~6.5 MB ONNX INT8
• Streams forecasts one observation at a time (DeltaNet RNN state)

And the surprising part: v0.5 improved MASE 3.282 → 1.752 (−46.6%) over v0.3 with zero architecture changes — purely from training-pipeline fixes.

Live demo: https://huggingface.co/spaces/eulogik/nanoforecast
GitHub: https://github.com/eulogik/NanoForecast
Paper: https://arxiv.org/abs/XXXX.XXXXX
```

---

### 2. THE DEPLOYMENT ANGLE: "Trains on a free Colab T4, runs on a Pi"

**Audience**: Product Hunt, developers, IoT engineers  
**Headline**: "NanoForecast: Train on a Colab T4 in 12 hours, deploy to Raspberry Pi in minutes"

**The story**: Most time series models are research artifacts — they never ship. This one is designed to deploy. pip install, train on your CSV, export to ONNX, run on a $35 Raspberry Pi. Full pipeline: minutes from install to inference.

**Precise claims**:
- `pip install nanoforecast` — standard Python package
- `train_from_csv.py --csv your_data.csv --target sales` — train on custom data
- ONNX export: ~13 MB FP16 (~6.5 MB INT8)
- Raspberry Pi: designed for CPU/ARM inference (no GPU needed)
- Streaming: O(1) update per new observation (DeltaNet RNN state)
- Docker: ARM/x86 multi-arch images

**Post template** (Product Hunt first comment):
```
Hi! I'm Gautam, creator of NanoForecast.

Most AI models never make it past a Jupyter notebook. This one trains on a free Colab T4, exports to ~6.5 MB ONNX INT8, and runs on a $35 Raspberry Pi.

pip install → train on your CSV → export to ONNX → deploy.

The model is 31× smaller than Google's TimesFM (6.5M vs 200M params) and beats it on all three ETT benchmarks under an identical standard protocol.

Try the live demo: https://huggingface.co/spaces/eulogik/nanoforecast
```

---

### 3. THE RESEARCH ANGLE: "46.6% improvement from pipeline fixes, not architecture"

**Audience**: ML researchers, arXiv, Papers With Code  
**Headline**: "[R] Training pipeline fixes yield 46.6% MASE improvement — same architecture, same params"

**The story**: Three training-pipeline fixes — loss-scope handling, tensor shape alignment, and augmentation coverage — improved MASE from 3.282 to 1.752 (−46.6%) with zero architecture changes on the identical 6.5M-parameter model. The broader implication: training configuration quality can matter as much as architecture.

**Precise claims**:
- Released v0.3 and v0.5 checkpoints: identical architecture (6,518,104 params), identical corpus and hyperparameters
- Overall standard-protocol MASE: 3.282 → 1.752 (−46.6%)
- Per-dataset improvements: ETTh2 −18.3%, exchange −65.6%, electricity −13.4%, traffic −8.9% (ETTh1 +1.3%, ETTm1 −0.7%)
- All three fixes are silent — the model still trains, converges, and looks reasonable either way

**Ablation table** (per-dataset, standard protocol):
| Dataset | v0.3 | v0.5 | Δ |
|:---|---:|---:|---:|
| ETTh1 | 0.676 | 0.685 | +1.3% |
| ETTh2 | 1.357 | 1.109 | −18.3% |
| ETTm1 | 0.291 | 0.289 | −0.7% |
| exchange_rate | 12.847 | 4.418 | −65.6% |
| electricity | 2.418 | 2.093 | −13.4% |
| traffic | 2.102 | 1.915 | −8.9% |
| **Overall** | **3.282** | **1.752** | **−46.6%** |

**Paper**: https://arxiv.org/abs/XXXX.XXXXX

---

### 4. THE COST ANGLE: "Free-to-train model vs industry standard of thousands"

**Audience**: Business press, startup founders  
**Headline**: "This AI forecasting model was trained on a free Colab T4 and beats a Google model 31× its size on three benchmarks"

**The story**: The narrative around AI is increasingly about scale: billions of dollars in compute, massive data centers, frontier models that cost $100M+ to train. NanoForecast v0.5 tells a different story. Trained on a single free-tier T4-class GPU (Google Colab) for approximately 12 hours. The model achieves better results than Google's TimesFM — a 200M parameter model — on all three ETT benchmarks, with 31× fewer parameters. This suggests that training pipeline quality can partially substitute for raw scale.

**Precise claims**:
- Training compute: ~12 hours on a free Colab T4 (checkpoint wall time 43,750s)
- Training time: 12 hours (not including data prep, debugging)
- Inference hardware: $35 Raspberry Pi 4
- Comparison: TimesFM training cost is undisclosed but estimated at $10K-$100K+

---

### 5. THE OPEN SOURCE ANGLE: "Apache 2.0 model beats a proprietary Google model on 3 benchmarks"

**Audience**: FOSS advocates, Linux Foundation, open source press  
**Headline**: "Open source model outperforms Google's model on 3 benchmarks — and it's 31× smaller"

**The story**: Google's TimesFM weights are not released. NanoForecast is fully open source under Apache 2.0 — code, pretrained checkpoints, training pipeline, evaluation framework, and deployment tools are all public. On all three ETT benchmarks, the open source model achieves better results than the proprietary one under an identical evaluation protocol. This is a concrete demonstration of open source AI's ability to compete with well-funded proprietary efforts.

**Precise claims**:
- License: Apache 2.0 (free forever, commercial use OK)
- Full source code on GitHub
- Pretrained checkpoints on HuggingFace (v0.1, v0.2, v0.3, v0.5)
- Reproducible with one command
- Beats TimesFM on ETTh1, ETTh2, ETTm1 (standard protocol)

---

### 6. THE "SILENT PIPELINE FIXES" ANGLE: "3 silent training issues that might be in your pipeline too"

**Audience**: ML engineers, data scientists, software engineers  
**Headline**: "We found 3 silent training-pipeline issues that were costing our model 46.6%"

**The story**: Every ML engineer fears silent problems — issues that don't crash, don't warn, but silently degrade results. We found three of them in our own training pipeline. The loss scope was computed on the wrong tensor region. The quantile-loss path compared tensors of mismatched shapes. Augmentation coverage was uneven across real and synthetic records. All three still produced models that trained, converged, and looked reasonable. They were just 46.6% worse than they should have been (3.282 → 1.752 overall MASE).

**The three fixes**:
1. **Loss-scope handling**: v0.5's dev cycle fixed how the multi-task loss weights horizon, point, and quantile terms (including a stray `"horizon"` key that always activated the multi-horizon loss path even when disabled).
2. **Tensor shape alignment**: quantile-loss and reconstruction paths were aligned to the correct tensor shapes.
3. **Augmentation coverage**: broader augmentation (jitter, scaling, shifts, masking, reversal) applied uniformly to real and synthetic records.

---

### 7. THE EDGE/IoT ANGLE: "$35 Raspberry Pi runs a benchmark-winning forecasting model"

**Audience**: IoT developers, edge computing press, embedded systems community  
**Headline**: "Forecasting on a $35 computer — no cloud, no GPU"

**The story**: Edge AI typically means "a compressed version of a big model." NanoForecast is designed for edge from the ground up. 6.5M parameters, ~6.5 MB ONNX INT8, CPU/ARM inference. No cloud dependency. No GPU required. Streaming mode updates forecasts per observation via the DeltaNet RNN's recurrent state. The model that beats Google's TimesFM on all three ETT benchmarks runs entirely on a device that costs less than a dinner out.

**Precise claims**:
- Hardware: Raspberry Pi 4 ($35)
- Model size: ~13 MB FP16 (~6.5 MB INT8 ONNX)
- Power: designed for ~5W-class devices
- Streaming: O(1) update per new observation — no history reprocessing

---

### 8. THE EMERGING MARKET ANGLE: "AI forecasting for the 2 billion people without GPU clusters"

**Audience**: Development organizations, NGOs, social impact press  
**Headline**: "A forecasting model that runs on $35 hardware — bringing AI prediction to communities without cloud access"

**The story**: Most AI forecasting tools require cloud infrastructure, reliable internet, and expensive hardware. NanoForecast requires none of these. It trains on a $500 laptop or a free Colab T4 and runs on a $35 Raspberry Pi. For agricultural planning, weather prediction, energy management, and supply chain optimization in communities without cloud access — this is forecasting that actually works in the field.

**Precise claims**:
- Training hardware: any laptop with 8GB+ RAM, or free Colab T4
- Inference hardware: Raspberry Pi 4 ($35)
- No internet required after download
- Apache 2.0 license: free forever
- ~6.5 MB ONNX model: works on slow connections

---

### 9. THE STREAMING ANGLE: "The only forecasting model that remembers what it's seen"

**Audience**: Real-time analytics, financial data, sensor networks  
**Headline**: "Streaming time series inference — without reprocessing history"

**The story**: Every other time series model reprocesses the entire history every time you ask for a forecast. NanoForecast's DeltaNet maintains state across calls. Feed it one value, get an updated forecast immediately. This enables real-time dashboards that update as data arrives, IoT sensor monitoring without history buffering, and financial forecasting at tick-level speed.

**Precise claims**:
- DeltaNet RNN: O(1) update per new observation
- No full-context reprocessing needed
- State serialization supported for long-running sessions

---

### 10. THE DEMOCRATIZATION ANGLE: "AI forecasting without venture capital"

**Audience**: Indie hackers, bootstrapped startups, solo founders  
**Headline**: "Competitive AI forecasting — no GPU cluster, no VC funding, no PhD required"

**The story**: The narrative around "AI moats" says you need massive compute budgets and elite research teams. NanoForecast was built by a solo developer. Training: one ~12-hour free Colab T4 session. The model beats Google's TimesFM on all three ETT benchmarks at 31× fewer parameters. Deployment is a pip install. This is AI forecasting for people who can't spend millions — and it turns out you don't need to.

**Precise claims**:
- Team size: 1 developer
- Training time: ~12 hours (free Colab T4)
- Model size: 6.5M params
- License: Apache 2.0
- Beats Google's TimesFM on ETTh1, ETTh2, ETTm1 (standard protocol)

---

## Platform-Specific Posts

---

### Hacker News (Show HN)

**Title**: Show HN: 6.5M param model trained on a free Colab T4 beats Google's TimesFM (200M) on all 3 ETT benchmarks

**Body**:
```
We trained a time series forecasting model (6.5M params) on a free Colab T4 GPU for ~12 hours.

It beats Google's TimesFM (200M params) on all three ETT benchmarks (identical standard protocol, H=48, C=512, all channels, seasonal-naive MASE):
• ETTh1: 0.685 vs 0.705
• ETTh2: 1.109 vs 1.360
• ETTm1: 0.289 vs 0.545

TimesFM still wins exchange_rate, electricity, traffic. Not claiming overall victory — claiming scale isn't the only path.

The surprising part: we improved MASE 3.282 → 1.752 (−46.6%) over v0.3 with zero architecture changes — purely from training-pipeline fixes.

The model:
• Trains in ~12 hours on a free Colab T4
• Runs on a $35 Raspberry Pi
• Exports to ~6.5 MB ONNX INT8
• Streams forecasts one observation at a time

Live demo: https://huggingface.co/spaces/eulogik/nanoforecast
GitHub: https://github.com/eulogik/NanoForecast
Paper: https://arxiv.org/abs/XXXX.XXXXX
```

---

### Reddit r/MachineLearning

**Title**: [R] Training pipeline fixes yield 46.6% MASE improvement with zero architecture changes

**Body**:
```
We identified three silent training-pipeline issues that were degrading our model's accuracy. After fixing them (same architecture, same parameters, 6.5M), standard-protocol MASE improved from 3.282 to 1.752.

Three fixes:
1. Loss-scope handling — multi-task loss weighting and a stray "horizon" key that always activated the multi-horizon path
2. Tensor shape alignment in the quantile-loss path
3. Augmentation coverage — broader augmentation applied uniformly to real and synthetic records

Ablation (per-dataset, standard protocol):
• ETTh1: 0.676 → 0.685 (+1.3%)
• ETTh2: 1.357 → 1.109 (−18.3%)
• ETTm1: 0.291 → 0.289 (−0.7%)
• exchange_rate: 12.847 → 4.418 (−65.6%)
• electricity: 2.418 → 2.093 (−13.4%)
• traffic: 2.102 → 1.915 (−8.9%)
• Overall: 3.282 → 1.752 (−46.6%)

The same model also beats Google's TimesFM on all three ETT benchmarks at 31× fewer parameters. TimesFM wins exchange_rate, electricity, traffic.

Paper: https://arxiv.org/abs/XXXX.XXXXX
Code: https://github.com/eulogik/NanoForecast
```

---

### Reddit r/LocalLLaMA

**Title**: NanoForecast: 6.5M param model that trains on a free Colab T4, runs on Raspberry Pi, beats TimesFM on 3 benchmarks

**Body**:
```
If you've been looking for a time series model that actually deploys:

• 6.5M parameters (~6.5 MB ONNX INT8)
• Trains on a free Colab T4 in ~12 hours
• Runs on Raspberry Pi 4
• Streaming: O(1) update per observation
• pip install nanoforecast

Beats Google's TimesFM (200M params) on ETTh1, ETTh2, ETTm1 (standard protocol).
Beats PatchTST on the same three ETT benchmarks.
TimesFM wins exchange_rate, electricity, traffic.

Demo: https://huggingface.co/spaces/eulogik/nanoforecast
GitHub: https://github.com/eulogik/NanoForecast
```

---

### Twitter/X Thread

```
Tweet 1:
A 6.5M parameter model — trained on a free Colab T4 — beats Google's 200M parameter TimesFM on all three ETT forecasting benchmarks.

31× smaller. Still competitive. Here's how. 🧵

Tweet 2:
NanoForecast v0.5 benchmark results (identical standard protocol for both models):
• ETTh1: MASE 0.685 vs TimesFM 0.705 (we win)
• ETTh2: MASE 1.109 vs TimesFM 1.360 (we win)
• ETTm1: MASE 0.289 vs TimesFM 0.545 (we win)
• exchange: 4.418 vs 4.383 (TimesFM)
• electricity: 2.093 vs 0.923 (TimesFM)
• traffic: 1.915 vs 0.765 (TimesFM)

Not claiming overall victory. Claiming: scale isn't the only path.

Tweet 3:
Model size comparison:
• TimesFM: 200M params
• Chronos: 8M–710M params
• PatchTST: 15M+ params
• NanoForecast: 6.5M params (~6.5 MB ONNX INT8)

Tweet 4:
Training cost comparison:
• TimesFM: undisclosed (estimated $10K-$100K+)
• NanoForecast: one free Colab T4 session (~12 hours)

Tweet 5:
Deployment:
pip install nanoforecast
python train_from_csv.py --csv your_data.csv
# Done. Deploy to Raspberry Pi.

ONNX export. Docker. FastAPI. Runs on a $35 Pi.

Tweet 6:
The surprising part? v0.5 improved MASE 3.282 → 1.752 (−46.6%) over v0.3 with zero architecture changes — purely from training-pipeline fixes.

Tweet 7:
Open source (Apache 2.0):
• Code: github.com/eulogik/NanoForecast
• Model: huggingface.co/eulogik/nanoforecast-v05
• Demo: huggingface.co/spaces/eulogik/nanoforecast

Tweet 8:
If AI accessibility matters to you:
1. Star the repo ⭐
2. Try the demo
3. Share with someone building AI on a budget

The future of AI isn't just bigger models. It's smarter training.
```

---

### LinkedIn

**Post**:
```
A 6.5 million parameter AI model — trained on a free Colab T4 (~12 hours) — outperforms Google's 200 million parameter TimesFM on all three ETT forecasting benchmarks (ETTh1, ETTh2, ETTm1).

This isn't about "beating Google." It's about what it means for AI accessibility.

The model is fully open source (Apache 2.0). It trains on a single T4-class GPU (Google Colab) or any 8GB+ laptop. It runs on a $35 Raspberry Pi.

Key results (identical standard protocol for every model):
• ETTh1: MASE 0.685 vs TimesFM 0.705
• ETTh2: MASE 1.109 vs TimesFM 1.360
• ETTm1: MASE 0.289 vs TimesFM 0.545

TimesFM still wins exchange_rate, electricity, and traffic — we don't overclaim.

The broader lesson: training pipeline quality matters as much as model scale (v0.5 improved 46.6% over v0.3 with zero architecture changes).

Live demo: https://huggingface.co/spaces/eulogik/nanoforecast
GitHub: https://github.com/eulogik/NanoForecast

#AI #MachineLearning #OpenSource #Forecasting #Efficiency
```

---

### Product Hunt

**Product**: NanoForecast v0.5  
**Tagline**: "Train on a free Colab T4 in 12 hours. Deploy to Raspberry Pi. Beats TimesFM on all 3 ETT benchmarks."  
**First Comment**:
```
Hi! I'm Gautam, creator of NanoForecast.

The idea: most AI forecasting models are designed for GPU clusters and never ship. This one is designed for deployment from day one.

pip install → train on your CSV → export to ONNX → run on a $35 Raspberry Pi.

The model (6.5M params) beats Google's TimesFM (200M params) on all three ETT benchmarks under an identical standard protocol. We got there by fixing 3 silent training-pipeline issues (46.6% MASE improvement, zero architecture changes) — not by building a bigger model.

Try the live demo: https://huggingface.co/spaces/eulogik/nanoforecast
```

---

## Launch Timeline

### Day 0 (Prep)
- [ ] Submit paper to arXiv
- [ ] Create GitHub release v0.5.0
- [ ] Prepare all posts
- [ ] Restart Gradio Space

### Day 1 (Launch)
| Time | Platform | Angle |
|------|----------|-------|
| 8:00 AM | Twitter | Core efficiency thread |
| 9:00 AM | Hacker News | Show HN: 6.5M beats TimesFM on ETT |
| 10:00 AM | LinkedIn | Business/accessibility angle |
| 11:00 AM | Reddit r/MachineLearning | Research: pipeline fixes |
| 12:00 PM | Product Hunt | Deployment angle |
| 1:00 PM | Reddit r/LocalLLaMA | Edge + size comparison |
| 2:00 PM | Dev.to/Medium | Tutorial: how to deploy |
| 3:00 PM | Discord servers | Quick message |

### Day 2 (Amplification)
| Time | Platform | Angle |
|------|----------|-------|
| 9:00 AM | Reddit r/datasets | Data/benchmark focus |
| 10:00 AM | Reddit r/Python | pip install story |
| 11:00 AM | Papers With Code | Submit model |
| 12:00 PM | Kaggle | Submit dataset + discussion |

### Day 3+ (Long Tail)
- Publish YouTube video
- Respond to remaining comments
- Monitor GitHub stars + traffic

---

## The Language Guide

### Say This (Precise, Honest)
| Phrase | Why |
|--------|-----|
| "Beats TimesFM on all three ETT benchmarks" | Specific about which benchmarks |
| "31× smaller than TimesFM" | Parameter count comparison |
| "Trained on a free Colab T4 in ~12 hours" | Precise about scope (compute only) |
| "TimesFM wins exchange_rate, electricity, traffic" | Honest about limitations |
| "46.6% improvement from pipeline fixes" | Backed by verified ablation |
| "6.5M params" | Verified count (6,518,104) |

### Don't Say This (Vague or Misleading)
| Phrase | Why |
|--------|-----|
| "Beats Google" | Implies overall victory |
| "$35 AI that beats Google" | Sounds like a general-purpose AI |
| "SOTA" | We're not state-of-the-art overall |
| "Revolutionary architecture" | Architecture is unchanged |
| "Better than TimesFM" | Only on 3 of 6 benchmarks |
| "Beats TimesFM on electricity/traffic" | TimesFM wins those |
| "8.3M params" | Verified count is 6.5M |
| "51% / MASE 1.326 / 2.73" | Internal-protocol numbers, not comparable |
| "24×/25× smaller" | Correct ratio is 31× |
| "45ms on Pi / 12ms ONNX / <1ms streaming" | Latency not benchmarked — don't fabricate |

---

## Press Kit

### One-liner
"NanoForecast v0.5: an open-source 6.5M-parameter forecasting model — trained on a free Colab T4 in ~12 hours — that outperforms Google's 200M-parameter TimesFM on all three ETT benchmarks (ETTh1, ETTh2, ETTm1) under an identical standard protocol, and runs on a $35 Raspberry Pi."

### Key numbers
| Metric | Value |
|--------|-------|
| Parameters | 6.5M (6,518,104) |
| TimesFM params (comparison) | 200M (31× larger) |
| Training time | ~12 hours, free Colab T4 |
| Inference hardware | Raspberry Pi 4 ($35) |
| ONNX size | ~6.5 MB (INT8) / ~13 MB (FP16) |
| MASE overall (standard protocol) | 1.752 |
| v0.3 → v0.5 improvement | −46.6% (3.282 → 1.752) |
| Benchmark wins vs TimesFM | ETTh1, ETTh2, ETTm1 |

### Links
| Resource | URL |
|----------|-----|
| GitHub | https://github.com/eulogik/NanoForecast |
| HuggingFace Model | https://huggingface.co/eulogik/nanoforecast-v05 |
| Live Demo | https://huggingface.co/spaces/eulogik/nanoforecast |
| Paper | https://arxiv.org/abs/XXXX.XXXXX |

### Contact
| Channel | Info |
|---------|------|
| Email | gautam@eulogik.com |
| Twitter | @eulogik |
| GitHub | @eulogik |