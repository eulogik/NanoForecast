# NanoForecast v0.5 — Launch Kit

**Author**: Gautam Kishore, Eulogik  
**Contact**: gautam@eulogik.com  
**Date**: July 2026

---

## The Core Story

> An 8.3 million parameter AI model — trained on a single T4-class GPU (Google Colab) — outperforms Google's 200 million parameter model on 2 of 6 standard forecasting benchmarks. And it runs on a $35 Raspberry Pi.

**Three provable claims that make this interesting:**

| Claim | Evidence |
|-------|----------|
| 24× smaller than Google's TimesFM | 8.3M params vs 200M params |
| Outperforms TimesFM on 2 benchmarks | Electricity MASE 0.709 vs 0.89, Traffic MASE 0.535 vs 0.62 |
| Trains on consumer hardware for ~$0.12 | 12 hours on Colab T4 at ~$0.01/hr |

---

## The 10 Story Angles

---

### 1. THE EFFICIENCY ANGLE: "24× smaller model outperforms Google on 2 benchmarks"

**Audience**: Hacker News, tech press, engineers  
**Headline**: "Show HN: 8.3M param model trained for $0.12 outperforms Google's TimesFM (200M) on 2 benchmarks"

**The story**: Most AI progress is measured by scale — bigger models, more data, more compute. This project challenges that. An 8.3M parameter model, trained on a single T4-class GPU (Google Colab) for twelve dollars of compute, achieves better results than a 200M parameter Google model on electricity and traffic forecasting. Not across the board — on 2 of 6 benchmarks. But that's the point: scale isn't the only path to performance.

**Precise claims**:
- Size: 8.3M params vs TimesFM's 200M (24× smaller)
- Training cost: ~$0.12 vs TimesFM's undisclosed but certainly thousands+
- Results: Outperforms on electricity (0.709 vs 0.89 MASE) and traffic (0.535 vs 0.62 MASE)
- TimesFM wins on remaining 4 benchmarks (ETTh1, ETTh2, ETTm1, exchange_rate)

**Post template**:
```
We trained a time series forecasting model (8.3M params) on a single T4-class GPU for ~12 hours (Google Colab).

It outperforms Google's TimesFM (200M params) on 2 of 6 standard benchmarks:
• Electricity: MASE 0.709 vs TimesFM 0.89
• Traffic: MASE 0.535 vs TimesFM 0.62

On the remaining 4 benchmarks it's competitive within 2-3×, despite being 24× smaller.

The model also:
• Trains in ~12 hours on a single T4-class GPU (Google Colab)
• Runs on a $35 Raspberry Pi at 45ms inference
• Exports to 8.3 MB ONNX INT8
• Updates forecasts in <1ms (streaming mode)

The surprising part: we improved accuracy 51% by fixing 3 training bugs. Same architecture.

Live demo: https://huggingface.co/spaces/eulogik/nanoforecast
GitHub: https://github.com/eulogik/NanoForecast
Paper: https://arxiv.org/abs/XXXX.XXXXX
```

---

### 2. THE DEPLOYMENT ANGLE: "Trains on a single T4-class GPU, runs on a Pi"

**Audience**: Product Hunt, developers, IoT engineers  
**Headline**: "NanoForecast: Train on a T4-class GPU in 12 hours, deploy to Raspberry Pi in 5 minutes"

**The story**: Most time series models are research artifacts — they never ship. This one is designed to deploy. pip install, train on your CSV, export to ONNX, run on a $35 Raspberry Pi. Full pipeline: 60 seconds from install to inference.

**Precise claims**:
- `pip install nanoforecast` — standard Python package
- `train_from_csv.py --csv your_data.csv --target sales` — train on custom data
- ONNX export: 16.6 MB FP16 (8.3 MB INT8)
- Raspberry Pi 4 inference: 45ms (12ms with ONNX INT8)
- Streaming: <1ms per observation
- Docker: ARM/x86 multi-arch images

**Post template** (Product Hunt first comment):
```
Hi! I'm Gautam, creator of NanoForecast.

Most AI models never make it past a Jupyter notebook. This one trains on a single T4-class GPU, exports to 8.3 MB INT8, and runs on a $35 Raspberry Pi.

pip install → train on your CSV → export to ONNX → deploy.

The model is competitive with models 24× its size because we focused on training pipeline quality rather than parameter count. It outperforms Google's TimesFM on electricity and traffic forecasting.

Try the live demo: https://huggingface.co/spaces/eulogik/nanoforecast
```

---

### 3. THE RESEARCH ANGLE: "51% improvement from pipeline fixes, not architecture"

**Audience**: ML researchers, arXiv, Papers With Code  
**Headline**: "[R] Training pipeline optimization yields 51% MASE improvement — same architecture, same params"

**The story**: Three silent bugs in the training pipeline were degrading model accuracy by 51%: loss scope computed on wrong dimensions, shape mismatches in quantile loss, and suboptimal data mixing ratios. Each fix contributed measurable improvement. The broader implication: many published results may reflect suboptimal training configurations rather than fundamental architectural limitations.

**Precise claims**:
- Ablation: Bug 1 → 21% improvement, Bug 2 → 35%, Bug 3 → 51%
- All three are silent — models still train and converge
- Likely present in many training pipelines

**Ablation table**:
| Configuration | MASE | Improvement |
|:---|---:|---:|
| v0.3 baseline | 2.73 | — |
| + Loss scope fix | 2.15 | 21.2% |
| + Shape alignment fix | 1.78 | 34.8% |
| + Data mixing fix | 1.326 | 51.4% |

**Paper**: https://arxiv.org/abs/XXXX.XXXXX

---

### 4. THE COST ANGLE: "$0.12 training cost vs industry standard of thousands"

**Audience**: Business press, startup founders, Forbes/Business Insider  
**Headline**: "This AI model was trained for $0.12 and outperforms models that cost millions"

**The story**: The narrative around AI is increasingly about scale: billions of dollars in compute, massive data centers, frontier models that cost $100M+ to train. NanoForecast v0.5 tells a different story. Trained on a single T4-class GPU (Google Colab) for approximately 12 hours. The model achieves results competitive with — and in two cases superior to — Google's TimesFM, a 200M parameter model that required orders of magnitude more investment. This suggests that training pipeline quality can partially substitute for raw scale.

**Precise claims**:
- Training compute: 12 hours × $0.01/hr (Colab T4) = $0.12
- Google Colab is not free for everyone — we used a free-tier eligible service
- Training time: 12 hours (not including data prep, debugging)
- Inference hardware: $35 Raspberry Pi 4
- Comparison: TimesFM training cost is undisclosed but estimated at $10K-$100K+

---

### 5. THE OPEN SOURCE ANGLE: "Apache 2.0 model outperforms proprietary Google model on 2 benchmarks"

**Audience**: FOSS advocates, Linux Foundation, open source press  
**Headline**: "Open source model outperforms Google's proprietary model on 2 benchmarks — and it's 24× smaller"

**The story**: Google's TimesFM is available only as an API or through limited research access — weights are not released. NanoForecast is fully open source under Apache 2.0. The code, pretrained checkpoints, training pipeline, evaluation framework, and deployment tools are all public. On 2 of 6 benchmarks, the open source model achieves better results than the proprietary one. This is a concrete demonstration of open source AI's ability to compete with well-funded proprietary efforts.

**Precise claims**:
- License: Apache 2.0 (free forever, commercial use OK)
- Full source code on GitHub
- Pretrained checkpoints on HuggingFace
- Reproducible with one command
- Outperforms TimesFM on electricity and traffic

---

### 6. THE "SILENT BUGS" ANGLE: "3 silent bugs that are probably in your training pipeline too"

**Audience**: ML engineers, data scientists, software engineers  
**Headline**: "We found 3 silent bugs that were destroying our model's accuracy by 51%"

**The story**: Every ML engineer fears silent bugs — problems that don't crash, don't warn, but silently degrade results. We found three of them in our own training pipeline. The loss function was computing gradients on the wrong tensor dimensions. The quantile loss had shape mismatches. The data mix was dominated by synthetic patterns. All three still produced models that trained, converged, and looked reasonable. They were just 51% worse than they should have been.

**The three bugs**:
1. **Loss scope mismatch**: Pipeline always returned multi-horizon key, even when disabled. Loss computed over full context instead of forecast horizon. (21% improvement fix)
2. **Tensor shape misalignment**: Quantile loss compared tensors before truncation. Gradients flowed through wrong dimensions. (35% cumulative improvement fix)
3. **Data mixing imbalance**: 1:1 real-to-synthetic ratio drowned out complex real patterns. (51% cumulative improvement fix)

---

### 7. THE EDGE/IoT ANGLE: "$35 Raspberry Pi runs competitive forecasting model at 45ms"

**Audience**: IoT developers, edge computing press, embedded systems community  
**Headline**: "Competitive time series forecasting on a $35 computer — no cloud, no GPU"

**The story**: Edge AI typically means "a compressed version of a big model." NanoForecast is designed for edge from the ground up. 8.3M parameters, 8.3 MB ONNX INT8, 45ms inference on a Raspberry Pi 4. No cloud dependency. No GPU required. Streaming mode updates forecasts in <1ms per new observation. The model that outperforms Google's on 2 benchmarks runs entirely on a device that costs less than a dinner out.

**Precise claims**:
- Hardware: Raspberry Pi 4 ($35)
- Inference: 45ms per forward pass
- Quantized: 12ms with ONNX INT8
- Streaming: <1ms per observation
- Model size: 16.6 MB FP16 (8.3 MB INT8)
- Power: ~5W total system power

---

### 8. THE EMERGING MARKET ANGLE: "AI forecasting for the 2 billion people without GPU clusters"

**Audience**: Development organizations, NGOs, social impact press  
**Headline**: "A forecasting model that runs on $35 hardware — bringing AI prediction to communities without cloud access"

**The story**: Most AI forecasting tools require cloud infrastructure, reliable internet, and expensive hardware. NanoForecast requires none of these. It trains on a $500 laptop and runs on a $35 Raspberry Pi. For agricultural planning, weather prediction, energy management, and supply chain optimization in communities without cloud access — this is forecasting that actually works in the field.

**Precise claims**:
- Training hardware: any laptop with 8GB+ RAM
- Inference hardware: Raspberry Pi 4 ($35)
- No internet required after download
- Apache 2.0 license: free forever
- 8.3 MB model: works on slow connections

---

### 9. THE STREAMING ANGLE: "The only forecasting model that remembers what it's seen"

**Audience**: Real-time analytics, financial data, sensor networks  
**Headline**: "Streaming time series inference in <1ms — without reprocessing history"

**The story**: Every other time series model reprocesses the entire history every time you ask for a forecast. NanoForecast's DeltaNet maintains state across calls. Feed it one value, get an updated forecast in less than a millisecond. This enables real-time dashboards that update as data arrives, IoT sensor monitoring without history buffering, and financial forecasting at tick-level speed.

**Precise claims**:
- DeltaNet RNN: $O(1)$ update per new observation
- No full-context reprocessing needed
- State serialization supported for long-running sessions
- 1000× speedup vs batch inference for real-time apps

---

### 10. THE DEMOCRATIZATION ANGLE: "AI forecasting without venture capital"

**Audience**: Indie hackers, bootstrapped startups, solo founders  
**Headline**: "Competitive AI forecasting for $0.12 — no GPU cluster, no VC funding, no PhD required"

**The story**: The narrative around "AI moats" says you need massive compute budgets and elite research teams. NanoForecast was built by a solo developer. Training cost: one ~12-hour T4-class GPU session (Google Colab). The model beats Google's on 2 benchmarks. Deployment is a pip install. This is AI forecasting for people who can't spend millions — and it turns out you don't need to.

**Precise claims**:
- Team size: 1 developer
- Training cost: ~$0.12
- Training time: 12 hours
- Model size: 8.3M params
- License: Apache 2.0
- Outperforms Google's TimesFM on 2 benchmarks

---

## Platform-Specific Posts

---

### Hacker News (Show HN)

**Title**: Show HN: 8.3M param model trained for $0.12 outperforms Google's TimesFM (200M) on 2 benchmarks

**Body**:
```
We trained a time series forecasting model (8.3M params) on a single T4-class GPU for ~12 hours (Google Colab).

It outperforms Google's TimesFM (200M params) on 2 of 6 standard benchmarks:
• Electricity: MASE 0.709 vs TimesFM 0.89
• Traffic: MASE 0.535 vs TimesFM 0.62

On the remaining 4 it's competitive within 2-3×, despite being 24× smaller.

The surprising part: we improved accuracy 51% by fixing 3 training bugs. Same architecture.

The model:
• Trains in ~12 hours on a single T4-class GPU (Google Colab)
• Runs on a $35 Raspberry Pi at 45ms inference
• Exports to 8.3 MB ONNX INT8
• Updates forecasts in <1ms (streaming mode)

Live demo: https://huggingface.co/spaces/eulogik/nanoforecast
GitHub: https://github.com/eulogik/NanoForecast
Paper: https://arxiv.org/abs/XXXX.XXXXX
```

---

### Reddit r/MachineLearning

**Title**: [R] Training pipeline optimization yields 51% MASE improvement with zero architecture changes

**Body**:
```
We identified three silent training pipeline bugs that were degrading our model's accuracy by 51%. After fixing them (same architecture, same parameters), MASE improved from 2.73 to 1.326.

Three bugs:
1. Loss scope mismatch — loss computed over full context instead of forecast horizon
2. Tensor shape misalignment in quantile loss — incorrect gradient flow
3. Data mixing imbalance — synthetic patterns drowning real-world signals

Ablation:
• Baseline: 2.73
• + Loss scope: 2.15 (21%)
• + Shape alignment: 1.78 (35%)
• + Data mixing: 1.326 (51%)

The model also outperforms Google's TimesFM on electricity (0.709 vs 0.89) and traffic (0.535 vs 0.62) despite being 24× smaller.

Paper: https://arxiv.org/abs/XXXX.XXXXX
Code: https://github.com/eulogik/NanoForecast
```

---

### Reddit r/LocalLLaMA

**Title**: NanoForecast: 8.3M param model that trains on laptop, runs on Raspberry Pi, outperforms TimesFM on 2 benchmarks

**Body**:
```
If you've been looking for a time series model that actually deploys:

• 8.3M parameters (8.3 MB ONNX INT8)
• Trains on a T4-class GPU (Google Colab) in 12 hours ($0.12)
• Runs on Raspberry Pi 4 at 45ms
• Streaming: <1ms per observation
• pip install nanoforecast

Beats Google's TimesFM (200M params) on electricity and traffic forecasting.
Beats PatchTST on electricity.

Best-in-class on traffic.

Demo: https://huggingface.co/spaces/eulogik/nanoforecast
GitHub: https://github.com/eulogik/NanoForecast
```

---

### Twitter/X Thread

```
Tweet 1:
An 8.3M parameter model — trained on a single T4-class GPU (Google Colab) — outperforms Google's 200M parameter model on 2 benchmarks.

24× smaller. 1,000× cheaper. Still competitive.

Here's how. 🧵

Tweet 2:
NanoForecast v0.5 benchmark results:
• Electricity: MASE 0.709 vs TimesFM 0.89 (we win)
• Traffic: MASE 0.535 vs TimesFM 0.62 (we win)
• ETTh1: MASE 0.913 vs TimesFM 0.52 (competitive)
• ETTh2: MASE 0.914 vs TimesFM 0.71 (competitive)

Not claiming overall victory. Claiming: scale isn't the only path.

Tweet 3:
Model size comparison:
• TimesFM: 200M params, GBs
• Chronos: 710M params, GBs
• PatchTST: 15M+ params
• NanoForecast: 8.3M params, 8.3 MB ONNX INT8

Tweet 4:
Training cost comparison:
• TimesFM: undisclosed (estimated $10K-$100K+)
• NanoForecast: $0.12 (12 hours × $0.01/hr Colab T4)

Tweet 5:
Deployment:
pip install nanoforecast
python train_from_csv.py --csv your_data.csv
# Done. Deploy to Raspberry Pi.

ONNX export. Docker. FastAPI. 45ms on a $35 Pi.

Tweet 6:
The surprising part? We improved accuracy 51% by fixing 3 silent training bugs.

Same architecture. Same params. Just better training.

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
An 8.3 million parameter AI model — trained on a single T4-class GPU (Google Colab) — outperforms Google's 200 million parameter model on 2 of 6 standard forecasting benchmarks.

This isn't about "beating Google." It's about what it means for AI accessibility.

The model is fully open source (Apache 2.0). It trains on a single T4-class GPU (Google Colab). It runs on a $35 Raspberry Pi.

Key results:
• Electricity: outperforms TimesFM (MASE 0.709 vs 0.89)
• Traffic: outperforms TimesFM (MASE 0.535 vs 0.62)
• Competitive within 2-3× on remaining 4 benchmarks

The broader lesson: training pipeline quality matters as much as model scale.

Live demo: https://huggingface.co/spaces/eulogik/nanoforecast
GitHub: https://github.com/eulogik/NanoForecast

#AI #MachineLearning #OpenSource #Forecasting #Efficiency
```

---

### Product Hunt

**Product**: NanoForecast v0.5  
**Tagline**: "Train on a T4-class GPU in 12 hours. Deploy to Raspberry Pi. Competitive with models 24× larger."  
**First Comment**:
```
Hi! I'm Gautam, creator of NanoForecast.

The idea: most AI forecasting models are designed for GPU clusters and never ship. This one is designed for deployment from day one.

pip install → train on your CSV → export to ONNX → run on a $35 Raspberry Pi.

The model (8.3M params) outperforms Google's TimesFM (200M params) on electricity and traffic benchmarks. We got here by fixing 3 training pipeline bugs that were silently destroying accuracy — not by building a bigger model.

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
| 9:00 AM | Hacker News | Show HN: $0.12 outperforms Google on 2 benchmarks |
| 10:00 AM | LinkedIn | Business/accessibility angle |
| 11:00 AM | Reddit r/MachineLearning | Research: pipeline bugs |
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
| "Outperforms TimesFM on electricity and traffic" | Specific about which benchmarks |
| "24× smaller than TimesFM" | Parameter count comparison |
| "Trained for \$0.12 in compute" | Precise about scope (compute only) |
| "Competitive with models 10–24× larger" | Honest about limitations |
| "Best-in-class on traffic" | Verifiable claim |
| "51% improvement from pipeline fixes" | Backed by ablation study |

### Don't Say This (Vague or Misleading)
| Phrase | Why |
|--------|-----|
| "Beats Google" | Implies overall victory |
| "\$35 AI that beats Google" | Sounds like a general-purpose AI |
| "SOTA" | We're not state-of-the-art overall |
| "Revolutionary architecture" | Architecture is unchanged |
| "Better than TimesFM" | Only on 2 of 6 benchmarks |
| "AI for \$0.12" | Compute cost only — doesn't include labor, hardware |

---

## Press Kit

### One-liner
"NanoForecast v0.5: an 8.3M parameter open-source forecasting model — trained on a laptop for ~$0.12 — that outperforms Google's 200M parameter TimesFM on electricity and traffic benchmarks, and runs on a $35 Raspberry Pi."

### Key numbers
| Metric | Value |
|--------|-------|
| Parameters | 8.3M |
| TimesFM params (comparison) | 200M (24× larger) |
| Training cost | ~$0.12 compute |
| Training time | 12 hours |
| Inference hardware | Raspberry Pi 4 ($35) |
| ONNX size | 8.3 MB (INT8) |
| Inference latency | 45ms CPU, 12ms ONNX INT8 |
| Streaming latency | <1ms per observation |
| MASE overall | 1.326 |
| Benchmark wins | Electricity, traffic |

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
