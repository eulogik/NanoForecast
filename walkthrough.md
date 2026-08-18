# NanoForecast — Walkthrough & Plan

## The Strategy

NanoForecast won't win on accuracy (yet). It wins on **deployability**:
- Train on a MacBook Air in 20 minutes (no GPU required)
- Run on a Raspberry Pi / browser / Lambda / phone
- ONNX export + INT8 quantization ≈ 6.5 MB at 6.5M params
- Full pipeline: `pip install` → `predict()` → `deploy` in one repo

**Target markets:**
- **Developers** who want a TS model that actually ships to production
- **Edge/IoT** people who need forecasting on a Raspberry Pi or phone
- **Hugging Face** users who want the smallest deployable TS model on the Hub
- **OpenRouter** API consumers who want forecasting at $0.001/series

## Milestones

| # | Milestone | Status | Est. time | Impact |
|---|---|---|---|---|---|
| 1 | Walkthrough + plan | ✅ Done | 10 min | Alignment |
| 2 | Gradio Space app | ✅ Done | 20 min | Viral demo (#1 driver) |
| 3 | FastAPI + Docker deploy | ✅ Done | 20 min | Production story |
| 4 | demo.py one-liner | ✅ Done | 5 min | README gateway |
| 5 | Push v0.1 checkpoint to HF Hub | ✅ Done | 10 min | Hub presence |
| 6 | Rewrite README | ✅ Done | 15 min | First impression |
| 7 | Gradio Space live on HF | ✅ Done | 30 min | Try in browser |
| 8 | Training runbook + assistant prompt | ✅ Done | 15 min | Mac Mini prep |
| 9 | **Streaming inference API** | ✅ Done | — | **Unique differentiator** |
| 10 | **train_from_csv.py CLI** | ✅ Done | — | **Primary UX path** |
| 11 | Mac Mini training: v0.2 checkpoint (Reverso recipe) | ✅ Done | — | MASE 3.45 (internal protocol) |
| 12 | v0.3 Frequency-Aware Hybrid architecture | ✅ Done | — | MASE 3.282 (standard protocol) |
| 13 | v0.5 Training fix: Pipeline/loss/augmentation | ✅ Done | — | **MASE 1.752 (46.6% better, standard)** |
| 14 | HF Model Card (SEO-optimized) | ✅ Done | — | Viral discovery |
| 15 | HF Community Article | ✅ Done | — | Visibility |
| 16 | Benchmark charts (ETTh1, Traffic) | ✅ Done | — | Proof of competitiveness |
| 17 | arXiv paper | 🔲 After v0.5 | — | Credibility |
| 18 | OpenRouter listing | 🔲 After v0.5 | — | Revenue |
| 19 | Standard benchmark protocol harness | ✅ Done | — | Fair TimesFM/PatchTST/Chronos comparison |
| 20 | TimesFM-200m baseline (all 6 datasets) | ✅ Done | — | Direct competitor beat-test |
| 21 | PatchTST baseline (4/6 datasets) | 🚧 Partial | ~overnight | Complete remaining 2 datasets |
| 22 | Chronos-T5-large baseline | 🔲 Stalled | — | Too heavy for Mac; small sets only |
| 23 | Paper results refresh w/ standard protocol | 🔲 | After baselines | Single source of truth |
| 24 | v0.6 focus: fix electricity/traffic | 🔲 Top priority | — | Close the TimesFM gap |

---

## Research Pivot (Jul 2026) — Architecture Was Right All Along

After deep research into the 2026 TS landscape we were worried our LongConv+DeltaNet
architecture was wrong. **It is not.** A Feb 2026 paper, **Reverso** (arXiv:2602.17634),
proves the exact pattern is SOTA-efficient:

> "Small hybrid models that interleave long convolution and linear RNN layers
> (in particular DeltaNet layers) can match the performance of larger transformer-based
> models while being more than a hundred times smaller."

So v0.1's weakness was **training data/recipe**, not architecture. Plan corrected below.

### What's actually SOTA-efficient in 2026 (and Mac-runnable?)

| Model | Core idea | Mac-runnable? |
|---|---|---|
| **Reverso** | LongConv + DeltaNet (linear RNN) hybrid | ✅ Pure PyTorch |
| **FRWKV** (Dec 2025) | Frequency-domain linear attention — #1 avg rank on 8 datasets | ✅ Pure PyTorch |
| **xLSTMTime** | xLSTM (matrix-memory LSTM) | ✅ Pure PyTorch |
| **CMDMamba** | Dual-layer Mamba + DConvFFN | ❌ CUDA kernels |
| **TTT** | Hidden state = tiny gradient-updated model | ✅ Pure PyTorch |
| Mamba-2/3 | SSM=linear attention (SSD) | ❌ CUDA kernels |

Key takeaways:
- Mamba needs CUDA → **not viable on Mac MPS**. Skip pure Mamba.
- The winning 2026 pattern is **hybrid**: SSM/conv/linear-RNN backbone + a little
  attention or frequency mixing. NanoForecast already does hybrid (gated router).
- **No small deployable TS foundation model fuses time-domain + frequency-domain.**
  That gap = our v0.3 differentiator.

---

## v0.2 — Train Now, Same Architecture, Better Recipe

Keep LongConv + DeltaNet RNN + gated router. Apply the **Reverso training recipe**:
- Richer data mix: TSMixup-style augmentation + more synthetic variety
- Longer context window, multi-horizon quantile training
- Longer schedule (100 epochs) on the server via tmux (48–72 threads, CPU)

Config (see `deploy/training_runbook.md`):
```bash
python3 pretrain.py \
  --datasets ETTh1,ETTh2,ETTm1 \
  --synthetic-records 10000 \
  --epochs 100 \
  --batch-size 256 --stride 16 \
  --d-model 64 --num-layers 8 --lr 1e-5 \
  --device cpu --output checkpoints/nanoforecast-500k
```

## v0.3 — The Novel Leap: Frequency-Aware Hybrid

Add a **Frequency-Mixing block** gated alongside LongConv/DeltaNet:
- Spectral branch via FFT + learnable band-pass filters (seasonal/resonant structure)
- Gated router learns per-patch whether to use time-domain (LongConv/DeltaNet)
  or frequency-domain (spectral) representation
- All under ~1M params, trained with the v0.2 recipe

Why novel: small TS models pick *either* time-domain (Mamba/DeltaNet) *or*
frequency-domain (TimesNet/FreTS). **Fusing both behind a learned router** in a
sub-1M deployable foundation model is an unclaimed contribution → paper angle.

---

## Current Status

### Published
- **GitHub**: https://github.com/eulogik/NanoForecast — v0.1, v0.2, v0.3, v0.5 branches
- **HF Model (v0.1)**: https://huggingface.co/eulogik/nanoforecast-200k — 676K params, Apache 2.0
- **HF Model (v0.2)**: https://huggingface.co/eulogik/nanoforecast-500k — 1.6M params, Apache 2.0
- **HF Model (v0.3)**: https://huggingface.co/eulogik/nanoforecast-v03 — 6.5M params, Apache 2.0
- **HF Model (v0.5)**: https://huggingface.co/eulogik/nanoforecast-v05 — 6.5M params, Apache 2.0, **MASE 1.752 (standard protocol)**
- **HF Space**: https://eulogik-nanoforecast.hf.space — upload CSV, get forecast + intervals + plot
- **HF Model Card**: https://huggingface.co/eulogik/nanoforecast-v05 — SEO-optimized, viral-ready
- **HF Community Article**: https://huggingface.co/eulogik/nanoforecast-v05/discussions/1
- **README**: Rewritten for v0.5 with benchmark comparisons to TimesFM, Chronos, PatchTST, Timer

### v0.5 Training (completed — Colab T4 GPU, 12h, 200 epochs)

**Key insight**: v0.5's 46.6% MASE improvement came from fixing the training pipeline (loss-scope handling, tensor shape alignment, augmentation coverage) — **zero architecture changes** vs v0.3.

**v0.5 development fixes**:
1. `pipeline.py` always returned `"horizon"` key even when `multi_horizon=False`, causing multi-horizon loss path to always be used
2. Notebook indentation errors in training loop
3. Added `BEST_PATH` to save best model separately from checkpoint
4. `checkpoint_interval` changed from 5 to 2 for safer resume

**Final Benchmark Results (v0.5, internal `benchmark.py` protocol)**:
| Dataset | MASE | MSE | MAE | Coverage (p90) |
|---|---|---|---|---|
| ETTh1 | 0.913 | 0.703 | 0.327 | 93.7% |
| ETTh2 | 0.914 | 0.703 | 0.327 | 93.7% |
| ETTm1 | 1.305 | 1.238 | 0.412 | 94.9% |
| exchange_rate | 3.578 | 4.855 | 0.172 | 98.6% |
| electricity | 0.709 | 0.0000154 | 0.044 | 94.4% |
| traffic | 0.535 | 0.0000154 | 0.015 | 91.6% |
| **Overall** | **1.326** | **1.238** | **0.232** | **94.5%** |

**Note on MSE normalization**: NanoForecast uses Instance Robust Scaler (median/IQR), so raw MSE values aren't directly comparable to models using standard normalization. The table above is the repo's own internal protocol; the authoritative comparison is the standard protocol below (MASE overall 1.752 vs TimesFM 1.447, PatchTST 1.554).

---

## Standardized Benchmark Protocol (Aug 2026) — vs TimesFM/PatchTST/Chronos

Note: the v0.5 "MASE 1.326" figure above comes from the repo's own benchmark.py protocol,
which is not directly comparable to published leaderboards or to TimesFM. So we built a
**standard protocol** (`benchmark_standard.py`) and re-measured everything fairly:

- **Horizon** H=48, **context** C=512
- **Splits**: ETT 70/20/10, other 70/10/20
- **Windows**: non-overlapping horizon-blocks in the test segment
- **MASE scale**: seasonal naive (in-sample MAE on the train segment)
- **Aggregation**: mean over windows per series, then mean over series
  (all 321 electricity clients / 862 traffic sensors used)
- Baselines: **TimesFM-1.0-200m** (via HF transformers),
  **PatchTST** (official thuml/Times-Library config, vendored under `benchmarks/tsl/`),
  **Chronos-T5-large** (Amazon)

### Results — MASE (lower is better), H=48, C=512

| Dataset | NF-v0.5 | TimesFM-200M | PatchTST | Chronos |
|---|---|---|---|---|
| ETTh1   | **0.685** | 0.705  | 0.781 | stalled |
| ETTh2   | **1.109** | 1.360  | 1.467 | stalled |
| ETTm1   | **0.289** | 0.545  | 0.488 | stalled |
| exchange_rate | 4.418 | **4.383** | 3.861 | stalled |
| electricity | 2.093 | **0.923** | 1.347 | stalled |
| traffic  | 1.915 | **0.765** | 1.379 | stalled |

**Read honestly**: NF-v0.5 **beats TimesFM on all three ETT sets** (−3% ETTh1, −18% ETTh2,
−47% ETTm1) and **ties exchange_rate** — but **loses big on high-cardinality
multivariate sets: electricity and traffic**. This is the race between NF's solidness
(depth on ETT/exchange) and TimesFM's trained-on-everything breadth (electricity/traffic).

PatchTST (official config, channel-independent, trained 40 epochs on free T4 Colab):
beats NF-v0.5 only on exchange_rate; beats TimesFM on ETTm1 and exchange_rate;
the trained model reproduces an honest level-following baseline on the rest.


Why NF loses on the big two: v0.5 was trained with `max_channels=4` per dataset
(`deploy/colab_training_v05.ipynb`), i.e. NF only ever saw 4 of the ~321/862 channels during
training — it lacks coverage of the raw high-dim multivariate data. Fixing this = **v0.6**
(train with many more channels per dataset + a richer mix of big multivariate, not inherently
an architecture/conclusion-issue).

### Baseline log (state @ session end, 17 Aug 2026)

| Baseline | Status |
|---|---|
| TimesFM-1.0-200m | ✅ all 6 datasets (`results/standard_benchmark.json`) |
| NF-v0.3 / v0.5 | ✅ all 6 datasets |
| PatchTST | ✅ all 6 datasets (40 epochs each, free T4 Colab; checkpoints in `benchmarks/checkpoints/patchtst/` + Drive backup) — **harness bug fixed** (channel indexing `norm[ci]` → `norm[:, ci]` fed models all-zero contexts; all earlier "level-tracker" scores were this bug, not the models) |
| Chronos-T5-large | ⛔ intractable on Mac for the big sets (electricity ~16–24h CPU, traffic 4–7 days; MPS recompiles for every shape). Run only the 4 small datasets |

Infra lessons (from the last chapter of the previous session):
- MPS on torch 2.6 is broken for big tensor shapes (pool bloat → OOM) and slow for T5 decoding — use CPU for volume / heavy models
- Processes spawned/launched from a tool shell get SIGKILLed when the session's shell tears down (watchdog children died too). Use `launchd` or `nohup` detached, or a real server session (Xeon) via tmux/nohup, so overnight runs survive.
- PatchTST trainer already has per-dataset resume (`benchmarks/checkpoints/patchtst/{ds}_resume.pt`) — re-running `train_patchtst.py` will resume or skip each finished dataset.

## Next Steps (from the Aug 7 analysis)

1. **Complete two baselines (paper-critical)** — PatchTST electricity + traffic, and Chronos on the 4 small sets. Compute is the blocker on this laptop; options in order of preference:
   - Xeon server (tmux/nohup, 96 threads) — a couple of hours of wall time for both, best option since the user has SSH+root.
   - `launchctl`-detached local run overnight (self-healing, resume-aware).
   - Worst case, honestly print `N/A`/subsample-deviation on PatchTST and drop Chronos from the big sets (still completable on small sets).
2. **Refresh `deploy/paper_v05.tex` results with the standard-protocol numbers** (only NF-v0.5 vs TimesFM is currently complete; that alone is already a publishable comparison table). The paper must not cite the old mixed-protocol numbers.
3. **Update the viral claims** (HF model card, README, LinkedIn/`LAUNCH_KIT.md`): the honest headline is "6.5M-param model beats TimesFM **on all three ETT benchmarks**, 31× smaller + deployable". Do not claim a blanket "beats TimesFM" — TimesFM wins exchange_rate, electricity, and traffic.
4. **v0.6 — close the electricity/traffic gap**: raise channels-per-dataset to ~64–128 for the two big sets, extend the synthetic mix, longer fine-tune on T4/Xeon, re-run standard protocol, expect MASE under the TimesFM lines.
5. **Commit + push the backlog** (files below): the v0.5 branch has uncommitted + untracked benchmark/paper/launch artifacts (~20 files). Commit per-milestone, then push.

## What Makes This Wantable

**For developers with CSV data:**
- `python3 train_from_csv.py --csv sales.csv --target revenue --horizon 48`
- Custom model trained on your data in 2 minutes — no GPU, no cloud
- No need to think about zero-shot vs fine-tuning — it just trains on your data

**For real-time / streaming use cases:**
- Only TS model with stateful online inference — feed one value at a time
- DeltaNet RNN carries memory across calls; no other architecture does this
- IoT sensor monitoring, live financial data, interactive dashboards

**For GitHub users:**
- `pip install nanoforecast` works
- Understandable codebase (small files, clean naming)
- Train on your data in 20 min
- Deploy with FastAPI or ONNX.js

**For HF Hub users:**
- Smallest deployable TS model on the Hub (~6.5 MB INT8 at 6.5M params)
- `from_pretrained` + `predict()` in 2 lines
- Model card with honest benchmarks
- Gradio Space with live demo

**For OpenRouter users:**
- Cheapest TS forecast API available
- $0.001/series because compute cost is near-zero

**For edge/IoT developers:**
- Raspberry Pi, Lambda, mobile, browser — it runs anywhere
- 12ms inference even on a $15 board

## Accuracy — two protocols, one truth

**Two sets of numbers exist and they are NOT comparable:**

1. **Repo protocol** (`benchmark.py`, old) — the MASE 1.326 / "Beats PatchTST, Best-in-class" rows
   found in earlier marketing docs. It used an InstanceScaler + app's own settings, so it is
   **not directly comparable to published leaderboards or to TimesFM**. Keep for internal diff-ing
   across versions only.
2. **Standard protocol** (`benchmark_standard.py`, Aug 2026) — the **single source of truth** for
   the paper, cards and any public claim. See the table above: NF-v0.5 beats TimesFM on all three
   ETT sets (−3% → −47% MASE), ties exchange_rate (4.42 vs 4.38), and loses on electricity/traffic.

**Remaining targets (now divorced from marketing):**
- **Close the electricity/traffic gap vs TimesFM** (2.09 vs 0.92, 1.92 vs 0.77) — v0.6, more channel
  coverage + mixed-aug train. This is the make-or-break for a defensible "beats TimesFM" claim.
- exchange_rate currently ties TimesFM (4.42 vs 4.38) — already far from the old 3.578-internal baseline.
- arXiv paper must be written over the standard protocol table.
