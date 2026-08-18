# We didn't shrink a big model. We designed a small one.

*By Gautam Kishore — part of the NanoForecast launch series*

---

## The shortcut everyone takes

Every "tiny AI model" story you've read this year goes something like this:

> "We took a giant model and squeezed it."

Prune this layer. Quantize these weights. Distill the knowledge. Ship the leftovers.

That approach works — up to a point. You end up with a model that's small, sure. But it's a *shrunken copy* of something designed for a data center. It still thinks like a big model: it wants long contexts, big batches, and a GPU breathing down its neck.

We went the other way.

We designed a model from scratch that was never supposed to be big. No LLM backbone. No tokenizer. No decoder stack pretending it's a language model when all we want it to do is predict tomorrow's numbers.

**NanoForecast is 6.5 million parameters.**

Google's TimesFM is 200 million. Amazon's Chronos-T5 is 710 million.

And on all three ETT benchmarks — ETTh1, ETTh2, ETTm1 — our 6.5M model *beats* Google's 200M one under an identical standard protocol.

Not "comes close." Beats. Here's the actual table:

| Model | Params | ETTh1 MASE | ETTh2 MASE | ETTm1 MASE |
|---|---|---|---|---|
| NanoForecast v0.5 | **6.5M** | **0.685** | **1.109** | **0.289** |
| TimesFM (Google) | 200M | 0.705 | 1.360 | 0.545 |
| PatchTST | 15M+ | 0.781 | 1.467 | 0.488 |

We also beat PatchTST — a model designed specifically for the forecasting task, and a name you've probably seen in every time-series paper since 2023 — on the same three ETT datasets.

To be fair about it: TimesFM and PatchTST win on exchange_rate, electricity, and traffic. We're not beating them everywhere. But a 6.5M model beating a 200M model on three benchmarks — that's the story.

---

## What "designed small" actually means

Here's the part I'm proudest of, because it's boring in the best way.

When you build a model from the ground up for one job — forecasting — you stop paying for all the machinery you don't need. Large language models are Swiss Army knives with 40 attachments. We only needed the corkscrew.

Our architecture, in plain words:

**Input:** The model reads the last 512 time steps of your series — say, 512 days of electricity demand — and chops them into patches of 8.

**Encoder:** A hybrid of two classic ideas. *Long convolution* (think: a very long memory that never forgets the distant past) and a *DeltaNet-style RNN* (think: a memory that updates efficiently step by step). A learned gate blends both per layer — getting the strengths of transformers (long-range memory) with the strengths of RNNs (cheap, sequential, edge-friendly inference).

**Output:** Instead of one prediction, it outputs five quantiles — p10, p25, p50, p75, p90 — so you get a forecast *with a confidence band* (the 10th–90th percentile range), not just a single guess. That's not a luxury feature; it's how every serious forecasting system in production actually works.

**One pass, four jobs:** A single forward pass produces the point forecast, the quantiles, an anomaly score, and a smoothness signal. It's multi-task learning on purpose: forecasting, uncertainty, and anomaly detection for free in the same forward pass.

**The loss:** We train on a mixture of point-accuracy, quantile-correctness, anomaly, and smoothness losses — a reward function that cares about *real forecasting skill*, not about imitating the next token.

The result? A model that fits in the palm of your hand:

- **6.5M parameters** — that's a sheet of paper compared to TimesFM's book
- **31x smaller than TimesFM, 109x smaller than Chronos-T5**
- **Runs on a $35 Raspberry Pi 4** — no cloud, no GPU, no API bill
- **Trains in ~12 hours on a single T4-class GPU (Google Colab)**

---

## The plot twist: architecture wasn't the real win

Here's the part that made us rethink everything.

Between v0.3 and v0.5, we did **not change a single layer** of the architecture. Same LongConv + DeltaNet hybrid. Same gated router. Same MLP blocks. Same 6.5M parameters.

And we still improved MASE from 3.282 → 1.752 (standard protocol).

**A 46.6% improvement. From training pipeline fixes alone.**

Three fixes, essentially:

1. **Loss-scope handling** — the multi-task loss was weighted and scoped incorrectly (including a stray "horizon" key that always activated the multi-horizon loss path even when disabled).
2. **Tensor shape alignment** — the quantile-loss path compared tensors of mismatched shapes, quietly sabotaging the quantile and anomaly heads.
3. **Augmentation coverage** — broader augmentation (jitter, scaling, shifts, masking, reversal) applied uniformly to real and synthetic records.

Fix those three things, don't touch the model, and watch it get 46.6% better.

That's the quiet lesson of this project: **in 2026, the biggest wins in applied ML aren't new architectures — they're fixing the pipeline.** The architecture was already good. We were grading it wrong.

---

## Why this matters to you

Two questions decide whether an ML model ever ships: *how accurate is it?* and *how much does it cost to run?*

Benchmark tables only ever answer the first one. So here's the second, answered honestly:

- **Training:** ~12 hours on a free T4-class GPU (Google Colab)
- **Inference:** runs on a **$35 Raspberry Pi 4** (CPU/ARM, no GPU)
- **Deployment:** ONNX (~9.2 MB INT8), FastAPI, Docker, browser — or literally no server at all

For sensor networks, retail shelves, energy meters, or any place with thousands of time series and no GPU budget — edge forecasting just stopped being a compromise.

And yes, to be fair about it: we're not beating TimesFM *everywhere*. On exchange_rate, electricity, and traffic their 200M parameters do win. Scale still means something. But the story of this project is that **model scale alone doesn't decide benchmarks — and it certainly doesn't decide deployment.**

---

## The take

Most "small model" projects are big models with their belts tightened.

Ours is small because it was born that way — designed for one job, trained properly, shipped to the edge.

**What's your take: distilled large models or purpose-built small ones?** I'd genuinely love to hear from people who've shipped either — especially the horror stories about serving costs.

The model, weights, Colab notebook, and paper are all open source:
**https://huggingface.co/eulogik/nanoforecast-v05**

*More details in the next post of this series — including the story behind the three pipeline fixes that were silently costing us 46.6%.*
