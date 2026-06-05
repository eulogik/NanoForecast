
# NanoForecast: The World's Smallest Time Series Foundation Model
## Project Document — v1.0 | June 2026

---

## 1. Executive Summary

**NanoForecast** is a 100K–500K parameter time series foundation model for **zero-shot forecasting** that beats 500M parameter models like TimesFM-2.5 and Chronos-Bolt-Base. It is not a domain-specific forecaster. It is a universal time series brain that runs on a microcontroller.

**Why this matters:**
- **Reverso proved** 200K–550K parameters can match 500M–1.5B models on zero-shot forecasting (Gift-Eval MASE 0.711–0.760 vs TimesFM-2.5 at 0.705) — but Reverso has limited adoption and no ecosystem
- **HuggingFace has almost NO popular time series models** — the category is wide open for a "default choice"
- Time series is **everywhere**: finance, IoT, sales, energy, DevOps, weather, healthcare, supply chain
- Current solutions are **huge**: TimesFM-2.5 (500M), Chronos-Bolt-Base (205M), Xihe-Max (1.5B) — all require GPUs
- **IBM TTM-R2** (805K) exists but zero-shot MASE is 1.02 (unusable) and only achieves 0.756 after fine-tuning

**Target:** Become the "all-MiniLM of time series" — the default foundation model for any developer building forecasting into their application. Dominate HuggingFace downloads and OpenRouter usage in the time series category within 6 months.

---

## 2. Market Analysis & Competitive Landscape

### 2.1 Current Time Series Foundation Models

| Model | Params | Gift-Eval MASE | TIME MASE | Architecture | License | Size Class |
|-------|--------|----------------|-----------|------------|---------|------------|
| **Chronos-2** | 200M | — | **0.660** | T5 encoder-decoder | Apache 2.0 | Large |
| **TimesFM-2.5** | 500M | **0.705** | 0.667 | Decoder-only Transformer | Apache 2.0 | Large |
| **Xihe-Max** | 1.5B | 0.711 | — | Hierarchical block attention | Proprietary | Huge |
| **Chronos-Bolt-Base** | 205M | — | 0.731 | Distilled T5 | Apache 2.0 | Medium |
| **Moirai-2** | 311M | — | 0.701 | Encoder-decoder (Universal TS) | CC BY-NC 4.0 | Medium |
| **Reverso** | 2.6M | **0.711** | — | Long conv + DeltaNet | Apache 2.0 | Small |
| **Reverso-Small** | 550K | **0.726** | — | Long conv + DeltaNet | Apache 2.0 | Tiny |
| **Reverso-Nano** | 200K | **0.760** | — | Long conv + DeltaNet | Apache 2.0 | Tiny |
| **TTM-R2** | 805K | 1.02 (ZS) / 0.756 (FT) | — | MLP-Mixer | Apache 2.0 | Tiny |
| **Lag-Llama** | 2.5M | — | — | Llama decoder | Apache 2.0 | Small |
| **Tiny-Time Mixers** | 1M | — | — | MLP-Mixer | — | Tiny |

**Key Insight:** The **sub-1M parameter space is nearly empty** and underexplored. Reverso proved the concept but has limited adoption, no HuggingFace ecosystem, and no easy-to-use SDK. TTM-R2 underperforms in zero-shot. **Nobody has claimed the "tiny time series" crown on HuggingFace.**

### 2.2 Market Demand

Time series forecasting is the **#1 statistical ML task** in industry:

| Domain | Use Cases | Current Pain Point |
|--------|-----------|-------------------|
| **Finance** | Stock prices, portfolio risk, trading signals | ARIMA models break, LSTMs overfit, large models too expensive |
| **IoT / DevOps** | CPU/memory metrics, sensor readings, anomaly detection | Thousands of streams = thousands of models to maintain |
| **Retail / E-commerce** | Demand forecasting, inventory optimization, pricing | Seasonal patterns change, models need constant retraining |
| **Energy** | Load forecasting, renewable generation, grid optimization | High-frequency data, need real-time inference |
| **Healthcare** | Patient admissions, ICU capacity, disease spread | Privacy requirements = local inference mandatory |
| **Supply Chain** | Shipping times, warehouse levels, procurement | Multivariate dependencies, external shocks |
| **Weather / Climate** | Temperature, precipitation, extreme events | Long horizons, complex seasonality |

**Current Solutions:**
- **Classical** (ARIMA, Prophet, ETS): Require per-series tuning, don't scale to thousands of series
- **Deep Learning** (DeepAR, TFT, N-BEATS): Require training on each domain, don't generalize
- **Foundation Models** (TimesFM, Chronos, Moirai): Require GPUs, expensive APIs, large memory footprint

**NanoForecast fills the gap:** One tiny model, zero-shot, runs on CPU/Raspberry Pi, no training required.

### 2.3 Benchmark Targets

| Benchmark | Current SOTA (tiny) | NanoForecast Target | Notes |
|-----------|---------------------|-------------------|-------|
| **Gift-Eval MASE** | 0.726 (Reverso-Small, 550K) | **<0.65** | 23 datasets, 97 tasks, 7 domains |
| **Gift-Eval MASE (Nano)** | 0.760 (Reverso-Nano, 200K) | **<0.70** | Sub-200K parameter target |
| **TIME Benchmark MASE** | 0.731 (Chronos-Bolt-Base, 205M) | **<0.70** | 50 datasets, 98 tasks |
| **ETT Long Horizon** | Varies by model | **Best in class** | 96/192/336/720 step forecasting |
| **Electricity** | 0.393 (TimesFM-2.5) | **<0.35** | Hourly power consumption |
| **Weather** | 0.379 (TimesFM-2.5) | **<0.35** | Multi-variate meteorological |
| **Traffic (PEMS)** | 0.046 (Chronos-Bolt) | **<0.05** | Short-term traffic forecasting |
| **M4 Competition** | Varies | **Top-3** | Mixed-frequency competition |
| **Inference Speed** | — | **>10,000 series/sec** | CPU, batch processing |
| **Model Size** | 200K (Reverso-Nano) | **<500K** | Target: 200K–500K range |
| **Memory Footprint** | — | **<50MB** | Quantized, CPU-only |

---

## 3. Architecture Design

### 3.1 Core Philosophy

**"Efficiency through hybridization."**

Reverso proved that combining **long convolutions** (for global pattern capture) with **linear RNN layers** (for sequential dependencies) and **lightweight attention** (for output generation) beats 100× larger transformer models. NanoForecast builds on this insight but optimizes for:

1. **Even smaller size** (target 200K–500K parameters)
2. **Better zero-shot generalization** (broader pretraining data)
3. **Multivariate support** (not just univariate)
4. **Probabilistic forecasting** (prediction intervals, not just point forecasts)
5. **Exogenous variable support** (covariates, holidays, promotions)

### 3.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    NanoForecast Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│  Input: Time Series + Optional Covariates                      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Input Processing Layer                                  │   │
│  │  • Normalization (mean/std or robust median/IQR)       │   │
│  │  • Resolution Prefix (frequency encoding)              │   │
│  │  • Channel Embedding (multivariate variate IDs)        │   │
│  │  • Exogenous Encoding (covariate projection)           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Sequence Mixing Blocks (×4–8 layers)                    │   │
│  │                                                          │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │   │
│  │  │ Long Conv   │    │ Linear RNN  │    │ Gated MLP   │  │   │
│  │  │ (FlashFFT)  │◄──►│ (DeltaNet)  │◄──►│ (Channel)   │  │   │
│  │  │ Global      │    │ Local       │    │ Feature     │  │   │
│  │  │ patterns    │    │ sequential  │    │ mixing      │  │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘  │   │
│  │                                                          │   │
│  │  • Alternating long conv + DeltaNet per layer            │   │
│  │  • Gated skip connections (residual + gating)            │   │
│  │  • LayerNorm + Dropout for stability                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Output Heads (Multi-task)                               │   │
│  │                                                          │   │
│  │  • Point Forecast Head (mean prediction)                 │   │
│  │  • Quantile Head (p10, p50, p90 for intervals)           │   │
│  │  • Anomaly Detection Head (deviation scoring)            │   │
│  │  • Trend/Seasonality Head (decomposition)                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  Output: Forecasts + Intervals + Anomaly Scores + Components   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Component Details

#### A. Input Processing Layer

**Problem:** Time series come in wildly different scales, frequencies, and variate counts.

**Solution:** Adaptive preprocessing:

| Component | Specification | Rationale |
|-----------|--------------|-----------|
| **Normalization** | Robust scaler (median/IQR) or mean/std | Handles outliers better than standard normalization |
| **Resolution Prefix** | Learned frequency embeddings (hourly, daily, weekly, etc.) | Model knows the natural frequency of the data |
| **Channel Embedding** | Learned variate IDs for multivariate series | Distinguishes between temperature, humidity, CPU, etc. |
| **Exogenous Projection** | Small MLP for covariates (promotions, holidays, weather) | Enables external factor inclusion |
| **Context Window** | Up to 512–2048 time steps (configurable) | Long enough for seasonality, short enough for speed |
| **Patching** | Optional adaptive patching (2, 4, 8, 16) | Reduces sequence length for very long contexts |

**Key Innovation:** **Resolution Prefix Tuning** — learned embeddings that tell the model "this is hourly retail data" or "this is daily stock data." This dramatically improves zero-shot performance by priming the model with frequency-specific priors.

#### B. Sequence Mixing Blocks (~300K–400K parameters total)

**Base:** Hybrid architecture inspired by Reverso but optimized for even smaller size.

| Feature | Nano-200K | Nano-500K | Rationale |
|---------|-----------|-----------|-----------|
| **Layers** | 4 | 8 | Depth vs speed tradeoff |
| **d_model** | 32 | 64 | Hidden dimension |
| **Long Conv** | FlashFFTConv, kernel size = context length | FlashFFTConv, kernel size = context length | Global pattern capture via FFT |
| **DeltaNet** | 2 layers per block | 4 layers per block | Linear RNN for sequential deps |
| **Gated MLP** | Expansion factor 2 | Expansion factor 2 | Channel mixing with gating |
| **Skip Connections** | Gated residual (like TSMixer) | Gated residual | Stable gradient flow |
| **Normalization** | RMSNorm | RMSNorm | Better than LayerNorm for small models |

**Key Innovation:** **Adaptive Block Allocation**

Instead of fixed alternating blocks, use a **gating mechanism** to dynamically weight the contribution of long conv vs DeltaNet vs MLP based on the input series characteristics:
- High-frequency/noisy data → more DeltaNet (local smoothing)
- Strong seasonality → more Long Conv (global periodic patterns)
- Multivariate data → more MLP (channel interactions)

This is implemented as a tiny learned router (<<1K parameters) per layer.

#### C. Output Heads (~50K–100K parameters)

Instead of a single point forecast, NanoForecast has multiple lightweight heads:

1. **Point Forecast Head:**
   - Predicts mean value for each horizon step
   - Autoregressive rollout: predicts 48 steps at a time, feeds back as input for longer horizons

2. **Quantile Head:**
   - Predicts p10, p25, p50, p75, p90 simultaneously
   - Enables prediction intervals without Monte Carlo sampling
   - Critical for business decision-making (inventory, staffing)

3. **Anomaly Detection Head:**
   - Computes deviation score for each forecast step
   - Flags unexpected patterns in real-time
   - Useful for IoT/DevOps monitoring

4. **Decomposition Head:**
   - Separates trend, seasonality, and residual components
   - Useful for interpretability and downstream analysis

**Output Format:**
```json
{
  "forecast": [12.3, 12.5, 12.8, ...],
  "intervals": {
    "p10": [11.5, 11.7, ...],
    "p90": [13.1, 13.3, ...]
  },
  "anomaly_scores": [0.1, 0.2, 0.8, ...],
  "components": {
    "trend": [12.0, 12.1, ...],
    "seasonal": [0.3, 0.4, ...],
    "residual": [-0.0, 0.0, ...]
  }
}
```

---

## 4. Training Strategy

### 4.1 Data Pipeline

#### Stage 1: Pretraining Data (10M+ time series)

**Data Sources:**

| Source | Count | Description |
|--------|-------|-------------|
| **Monash Time Series Repository** | 100K+ | 30+ datasets, diverse domains, frequencies |
| **GIFT-Eval Pretrain** | 23 datasets | Salesforce benchmark pretraining split |
| **LOTSA** | 1M+ | Large-scale time series archive (TimesFM training data) |
| **M4 Competition** | 100K | Mixed-frequency economic data |
| **Synthetic Series** | 5M+ | Generated with diverse patterns (trend, seasonality, noise, changepoints) |
| **IoT/Server Metrics** | 1M+ | Prometheus/Grafana-style monitoring data |
| **Financial OHLCV** | 500K | Stock/crypto price series |
| **Energy Data** | 500K | Smart meter, grid, renewable generation |
| **Weather Data** | 500K | Temperature, precipitation, wind |
| **Healthcare** | 200K | Patient vitals, admissions, epidemiology |

**Synthetic Data Generation:**

To ensure the model sees every possible pattern:

```
Base Patterns:
├── Trend: linear, exponential, logistic, piecewise
├── Seasonality: daily, weekly, monthly, yearly, multiplicative
├── Noise: Gaussian, Poisson, Student-T, heteroscedastic
├── Changepoints: level shifts, trend breaks, variance changes
├── Outliers: isolated spikes, level shifts, contextual anomalies
├── Cycles: business cycles, irregular periodicities
└── Multivariate: correlated, cointegrated, lagged relationships

For each synthetic series:
1. Randomly sample 2-4 base patterns
2. Combine with random weights
3. Add noise and outliers
4. Vary length (128 to 4096 steps)
5. Vary scale (0.01 to 1M)
6. Vary frequency (5min to yearly)
```

**Data Augmentation:**
- **Scaling:** Multiply by random factor (0.1–10)
- **Shifting:** Add random offset
- **Windowing:** Extract random subseries
- **Frequency resampling:** Upsample/downsample
- **Noise injection:** Add Gaussian/Student-T noise
- **Missing values:** Randomly mask 5–20% of values
- **Outlier injection:** Add synthetic spikes

#### Stage 2: Domain-Specific Fine-tuning (1M+ series)

While pretraining teaches general patterns, domain fine-tuning teaches specific behaviors:

| Domain | Data | Task |
|--------|------|------|
| **Retail** | Walmart, Rossmann, M5 | Demand forecasting with promotions |
| **Finance** | S&P 500, crypto, forex | Volatility forecasting, returns |
| **Energy** | UCI Electricity, solar/wind | Load forecasting, renewable |
| **IoT** | Server metrics, sensor data | Anomaly detection, capacity planning |
| **Healthcare** | ICU admissions, ER visits | Patient flow, resource planning |
| **Weather** | NOAA, ERA5 | Temperature, precipitation |

#### Stage 3: Instruction Tuning (100K+ tasks)

**Task Format:**
```json
{
  "context": [10.2, 11.5, 12.1, ...],
  "frequency": "hourly",
  "horizon": 96,
  "covariates": {
    "holiday": [0, 0, 1, ...],
    "promotion": [0.0, 0.2, 0.0, ...]
  },
  "instruction": "Forecast the next 96 hours with 90% prediction intervals.",
  "output": {
    "forecast": [...],
    "p10": [...],
    "p90": [...]
  }
}
```

**Instruction Types:**
1. **Point forecast:** "Predict the next N steps"
2. **Interval forecast:** "Predict with 90% confidence intervals"
3. **Anomaly detection:** "Identify unusual patterns in this series"
4. **Decomposition:** "Separate trend and seasonality"
5. **Comparison:** "Will next week be higher than this week?"
6. **What-if:** "Forecast assuming a 20% promotion"
7. **Imputation:** "Fill in the missing values"

### 4.2 Training Curriculum

```
Stage 1: General Pretraining (3 weeks)
├── Data: 10M+ diverse time series (synthetic + real)
├── Task: Next-step prediction + masked imputation
├── Loss: MSE (primary) + MAE (robust) + Quantile loss
├── Objective: Learn general time series patterns
├── Curriculum: Easy (clean, short) → Hard (noisy, long, multivariate)
└── Milestone: MASE <1.0 on Monash benchmark (naive baseline)

Stage 2: Domain Adaptation (2 weeks)
├── Data: 1M+ domain-specific series
├── Task: Domain-specific forecasting with covariates
├── Loss: Domain-weighted MSE + quantile loss
├── Objective: Learn domain-specific patterns
└── Milestone: MASE <0.8 on domain benchmarks

Stage 3: Instruction Tuning (1 week)
├── Data: 100K+ instruction tasks
├── Task: Multi-task forecasting (point, interval, anomaly, decomposition)
├── Loss: Task-specific loss + consistency regularization
├── Objective: Learn to follow forecasting instructions
└── Milestone: Human-evaluated quality >4.0/5.0
```

### 4.3 Training Infrastructure

| Specification | Detail |
|--------------|--------|
| **Compute** | 4× A100 80GB (or 8× RTX 4090) |
| **Framework** | PyTorch 2.6 + FlashFFTConv + flash-linear-attention |
| **Optimizer** | AdamW (β1=0.9, β2=0.999, eps=1e-8) |
| **Learning Rate** | 5×10⁻⁴ → 1×10⁻⁵ (WSD scheduler) |
| **Batch Size** | 512 (global) = 128 per GPU × 4 GPUs |
| **Precision** | BF16 mixed precision |
| **Gradient Clipping** | 1.0 |
| **Warmup** | 5% of total steps |
| **Regularization** | Weight decay 0.1, Dropout 0.1 |
| **Total Training Time** | ~6 weeks |
| **Cost Estimate** | ~$8,000 (cloud) or ~$3,000 (own hardware) |

**Key Training Innovation:** **Resolution-Aware Batching**

Group training samples by frequency (hourly, daily, weekly) and apply frequency-specific augmentation. This prevents the model from confusing daily seasonality with hourly patterns.

---

## 5. Evaluation & Benchmarking

### 5.1 Standard Benchmarks

| Benchmark | Metric | Target | Validation Strategy |
|-----------|--------|--------|---------------------|
| **Gift-Eval** | MASE | <0.65 (500K) / <0.70 (200K) | Official eval (23 datasets, 97 tasks) |
| **TIME Benchmark** | MASE | <0.70 | Official eval (50 datasets, 98 tasks) |
| **ETTh1 (96-step)** | MSE/MAE | <0.35 / <0.38 | Standard long-horizon benchmark |
| **ETTh2 (96-step)** | MSE/MAE | <0.30 / <0.35 | Standard long-horizon benchmark |
| **ETTm1 (96-step)** | MSE/MAE | <0.30 / <0.35 | High-frequency benchmark |
| **ETTm2 (96-step)** | MSE/MAE | <0.20 / <0.30 | High-frequency benchmark |
| **Electricity (96-step)** | MSE/MAE | <0.15 / <0.25 | Multivariate power consumption |
| **Weather (96-step)** | MSE/MAE | <0.20 / <0.25 | Multivariate meteorological |
| **Traffic (PEMS04)** | MAPE | <0.05 | Short-term traffic |
| **M4 (Yearly)** | OWA | <0.85 | Competition metric |
| **M4 (Quarterly)** | OWA | <0.85 | Competition metric |
| **M4 (Monthly)** | OWA | <0.85 | Competition metric |
| **M4 (Weekly)** | OWA | <0.90 | Competition metric |
| **M4 (Daily)** | OWA | <0.90 | Competition metric |
| **M4 (Hourly)** | OWA | <0.90 | Competition metric |

### 5.2 Custom Benchmarks

**NanoForecast-Bench (proprietary):**
- 1,000 real-world series from 10 domains
- 5,000 forecasting tasks (various horizons, frequencies)
- Multivariate + exogenous variable tests
- Edge device performance tests (Raspberry Pi, Jetson, CPU)
- Inference speed benchmarks (series/second)

**Edge Performance Benchmarks:**

| Device | Target Latency | Memory |
|--------|---------------|--------|
| Raspberry Pi 5 | <100ms per series | <500MB RAM |
| iPhone 15 | <50ms per series | <200MB RAM |
| MacBook Air M2 | <20ms per series | <300MB RAM |
| NVIDIA Jetson Nano | <50ms per series | <1GB RAM |
| CPU (Intel i5) | <30ms per series | <300MB RAM |
| AWS Lambda (512MB) | <200ms cold start | <512MB RAM |

### 5.3 Ablation Studies

Critical experiments to run:

1. **Architecture comparison:** Long conv + DeltaNet vs pure Transformer vs pure MLP vs pure RNN
2. **Size scaling:** 50K vs 100K vs 200K vs 500K vs 1M parameters
3. **Patching impact:** No patch vs 2 vs 4 vs 8 vs 16
4. **Context length:** 128 vs 256 vs 512 vs 1024 vs 2048
5. **Quantile loss weighting:** MSE-only vs MSE+MAE vs MSE+quantile
6. **Multivariate vs univariate:** Channel independence vs channel mixing
7. **Exogenous variables:** With vs without covariates
8. **Autoregressive rollout:** 1-step vs 48-step vs 96-step prediction chunks

---

## 6. Distribution & Go-to-Market

### 6.1 HuggingFace Strategy

**Model Card (Critical for Adoption):**
- Extensive benchmark results with comparison tables (vs TimesFM, Chronos, TTM, Reverso)
- Training details: data sources, mixture proportions, architecture decisions
- Memory usage charts (RAM vs batch size vs context length)
- Inference code examples (Python, JavaScript, cURL)
- Fine-tuning guide with LoRA/QLoRA configs for domain adaptation
- Limitations and bias analysis (when does it fail?)

**Spaces Demo:**
- Interactive Gradio demo: upload CSV → get forecast + intervals + anomaly detection
- Pre-loaded examples: stock prices, server metrics, weather, sales, energy
- Side-by-side comparison with naive baseline and seasonal naive
- Real-time processing (show latency)
- Export results (CSV, JSON, plot)
- Adjustable parameters: horizon, context length, confidence level

**Integration:**
- `transformers` native support (custom model class)
- `sktime` integration (the standard Python time series library)
- `darts` integration (another popular TS library)
- `pandas` extension (`df.forecast(horizon=24)`)
- `Grafana` plugin for real-time monitoring dashboards
- `Prometheus` exporter for DevOps forecasting
- `ONNX` export for cross-platform inference
- `llama.cpp` / `ollama` style local deployment

### 6.2 OpenRouter Strategy

**Model Listing:**
- Name: `nanoforecast-200k` / `nanoforecast-500k`
- Description: "The world's smallest time series foundation model. Zero-shot forecasting for any time series in <100ms on CPU."
- Pricing: **Free tier** (1,000 series/day) + **$0.001 per series** (cheapest forecasting API)
- Endpoints:
  - `/forecast` — point + interval forecasts
  - `/anomaly` — anomaly detection
  - `/decompose` — trend/seasonality/residual decomposition
  - `/compare` — compare multiple scenarios

**Tool Calling:**
- Native function calling for agentic forecasting workflows
- Pre-built tools: `forecast_series`, `detect_anomalies`, `decompose_series`, `compare_forecasts`
- JSON schema output with validation

### 6.3 Community Building

| Phase | Actions |
|-------|---------|
| **Week 1-2** | Release model + demo + Python SDK (`pip install nanoforecast`). Post on Hacker News, Reddit (r/MachineLearning, r/datascience), Twitter. YouTube demo video. |
| **Week 3-4** | Kaggle competition: "Zero-shot forecasting challenge." Partner with sktime/darts maintainers. Release domain-specific fine-tunes (retail, finance, IoT). |
| **Month 2-3** | Enterprise case studies (with permission). Conference talks (NeurIPS, ICML, KDD). Academic paper submission. Grafana plugin launch. |
| **Month 4-6** | Mobile SDK (iOS/Android). Browser extension. Excel/Google Sheets add-on. AWS Marketplace listing. |

---

## 7. Moat & Defensibility

### 7.1 Data Moat

- **10M+ diverse time series** with curated domain labels — 2 months to build pipeline
- **Proprietary synthetic generation engine** — can generate infinite realistic series
- **Curated real-world benchmark** (NanoForecast-Bench) with human-verified ground truth
- **Continuous data collection** from community uploads (opt-in, anonymized)
- **Domain-specific datasets** from industry partnerships (retail, energy, healthcare)

### 7.2 Architecture Moat

- **Hybrid architecture** (long conv + DeltaNet + MLP) — proven but requires specific expertise
- **Adaptive block allocation** — dynamic routing based on input characteristics
- **Resolution prefix tuning** — frequency-aware embeddings (not used by competitors)
- **Multi-task output heads** — point + quantile + anomaly + decomposition in one forward pass
- **Multivariate + exogenous support** — most tiny models are univariate-only

### 7.3 Ecosystem Moat

- **Python SDK** (`pip install nanoforecast`) with one-liner forecasting
- **sktime integration** — the default choice for sktime users (massive user base)
- **darts integration** — another major TS library
- **Grafana plugin** — instant adoption by DevOps/SRE teams
- **Pandas extension** — `df.forecast(horizon=24)` syntax
- **API playground** — test without code
- **Excel/Google Sheets add-on** — business users (massive market)

### 7.4 Community Moat

- **Apache 2.0 license** for maximum adoption
- **Active Discord/Slack community** for support and feature requests
- **Fine-tuning guides** for domain-specific variants
- **Bounty program** for bug reports, improvements, and new benchmarks
- **Monthly model updates** with community feedback
- **Academic collaborations** for credibility and research citations

---

## 8. Risk Analysis & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Synthetic data quality issues** | Medium | High | Human validation on subset + real data fallback + adversarial testing |
| **Benchmark overfitting** | Medium | High | Hold-out test set + external evaluation + real-world customer validation |
| **Reverso/TimesFM releases smaller model** | Medium | High | Speed to market + ecosystem lock-in + continuous improvement + community |
| **Multivariate performance gaps** | Medium | Medium | Channel independence fallback + channel mixing experiments + user feedback |
| **Long-horizon degradation** | Medium | High | Autoregressive rollout optimization + multi-horizon training + ensembling |
| **Adoption slower than expected** | Medium | High | Free tier + aggressive marketing + integration partnerships + Kaggle competitions |
| **Quantile calibration issues** | Medium | Medium | Platt scaling + conformal prediction + empirical calibration |
| **Competitor price war** | Low | Medium | Cost advantage is structural (tiny model = tiny compute) — can't be undercut |

---

## 9. Timeline & Milestones

```
Month 1: Architecture & Data Pipeline
├── Week 1-2: Finalize architecture, implement FlashFFTConv + DeltaNet blocks
├── Week 3-4: Build synthetic time series generation pipeline (target: 2M series)
└── Milestone: Generate 2M synthetic series, validate realism vs real data

Month 2: Pretraining
├── Week 1-2: Stage 1 training (general pretraining on 10M series)
├── Week 3-4: Continue pretraining, start domain data curation
└── Milestone: MASE <1.0 on Monash benchmark (beat naive baseline)

Month 3: Domain Adaptation & Evaluation
├── Week 1-2: Stage 2 training (domain adaptation)
├── Week 3-4: Stage 3 training (instruction tuning), comprehensive evaluation
└── Milestone: Gift-Eval MASE <0.70, TIME MASE <0.75 on validation

Month 4: Optimization & Packaging
├── Week 1-2: Quantization (INT8, INT4), ONNX export, edge optimization
├── Week 3-4: Build SDK, API, demo, Grafana plugin, documentation
└── Milestone: Model runs on Raspberry Pi in <100ms per series

Month 5: Launch & Iteration
├── Week 1-2: HuggingFace release, OpenRouter integration, marketing blitz
├── Week 3-4: Community feedback, bug fixes, first domain fine-tunes
└── Milestone: 10K+ downloads on HF, 1K+ API users

Month 6: Scale & Dominate
├── Week 1-2: Industry-specific fine-tunes (retail, finance, IoT, energy)
├── Week 3-4: Academic paper, conference talks, enterprise pilots
└── Milestone: #1 trending time series model on HF, 100K+ downloads
```

---

## 10. Resource Requirements

### 10.1 Compute

| Phase | Duration | Hardware | Cost |
|-------|----------|----------|------|
| Data generation | 2 weeks | 4× A100 (synthetic generation) | $1,500 |
| Pretraining | 3 weeks | 4× A100 80GB | $4,500 |
| Domain adaptation | 2 weeks | 4× A100 80GB | $3,000 |
| Instruction tuning | 1 week | 4× A100 80GB | $1,500 |
| Optimization | 1 week | 2× A100 + CPU servers | $1,000 |
| **Total** | **9 weeks** | | **~$11,500** |

**Alternative:** Own hardware (4× RTX 4090) = ~$8,000 upfront, reusable for future models.

### 10.2 Team

| Role | Time | Skills |
|------|------|--------|
| **ML Engineer (Lead)** | Full-time | Time series, signal processing, PyTorch, FlashFFTConv |
| **ML Engineer (Data)** | Full-time | Synthetic data generation, time series augmentation, data pipelines |
| **ML Engineer (Training)** | Full-time | Distributed training, hyperparameter optimization, benchmarking |
| **Software Engineer** | Half-time | SDK, API, Grafana plugin, integrations |
| **DevRel/Community** | Half-time | Documentation, tutorials, community building, partnerships |
| **Designer/UX** | Contract | Demo UI, marketing materials, Grafana plugin design |

### 10.3 Budget Summary

| Category | Cost |
|----------|------|
| Compute (cloud) | $11,500 |
| Hardware (if buying) | $8,000 |
| Team salaries (3 months, 4 FTE) | $70,000 |
| Marketing & community | $5,000 |
| Infrastructure (API, hosting) | $2,000 |
| **Total (3 months)** | **~$88,500** |
| **Total (if own hardware)** | **~$85,000** |

---

## 11. Success Metrics

### 11.1 Technical Metrics

| Metric | 1 Month | 3 Months | 6 Months |
|--------|---------|----------|----------|
| Gift-Eval MASE (500K) | <0.70 | <0.65 | <0.60 |
| Gift-Eval MASE (200K) | <0.75 | <0.70 | <0.65 |
| TIME Benchmark MASE | <0.75 | <0.70 | <0.65 |
| ETT Long Horizon (96-step) | <0.35 MSE | <0.30 MSE | <0.25 MSE |
| Electricity (96-step) | <0.20 MSE | <0.15 MSE | <0.12 MSE |
| Inference speed (CPU) | <100ms | <50ms | <20ms |
| Memory footprint | <100MB | <50MB | <20MB |

### 11.2 Adoption Metrics

| Metric | 1 Month | 3 Months | 6 Months |
|--------|---------|----------|----------|
| HF Downloads | 5K | 50K | 300K |
| HF Likes | 300 | 1.5K | 8K |
| OpenRouter API calls | 500/day | 5K/day | 50K/day |
| GitHub stars (SDK) | 300 | 1.5K | 7K |
| Community Discord members | 100 | 500 | 3K |
| Grafana plugin installs | 50 | 500 | 3K |
| Enterprise pilots | 1 | 5 | 20 |

### 11.3 Business Metrics

| Metric | 6 Months | 12 Months |
|--------|----------|-----------|
| API revenue | $500/month | $5K/month |
| Enterprise licenses | 0 | 3 |
| Sponsorship/grants | $5K | $30K |
| Consulting/fine-tuning | $2K | $15K/month |

---

## 12. Conclusion

NanoForecast represents a **massive blue-ocean opportunity** in the tiny model space:

1. **Proven technical feasibility** — Reverso proved 200K–550K parameters can match 500M models
2. **Empty competitive landscape** — No dominant tiny time series model on HuggingFace
3. **Universal demand** — Time series is the most common data type in industry
4. **Clear differentiation** — Zero-shot, probabilistic, multivariate, exogenous, edge-first
5. **Strong moat** — Data + architecture + ecosystem + community

**The bet:** A 200K–500K parameter model, trained on 10M+ diverse time series with hybrid long-conv/DeltaNet architecture and resolution prefix tuning, can achieve **Gift-Eval MASE <0.65** while running on a Raspberry Pi in **<100ms per series**.

**If successful:** NanoForecast becomes the **"all-MiniLM of time series"** — the default foundation model that every data scientist, developer, and DevOps engineer reaches for first when they need forecasting.

**The time series foundation model space is waiting for its tiny champion. Build it now.**

---

*Document prepared June 2026.*
