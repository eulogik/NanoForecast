# Paper v0.5 — Pre-Submission Review (6 Aug 2026)

Full read of `paper_v05.tex` + cross-check of every claim against the cited papers,
the actual model code, checkpoints, and deploy assets. Goal: zero rejection risk.

## Verified GROUND TRUTH (from the actual repo, not the docs)

| Item | Truth | Source |
|---|---|---|
| v0.5 params | **8,294,104 (8.3M)** — d_model=96, L=8, patch=8, ctx=512 | `NanoForecastConfig.nano_v05()` + parameter count |
| 16 MB ONNX | = **FP16** (16.59 MB), not FP32 | 8.294M × 2 B |
| INT8 size | **8.29 MB**, not 1.4 MB | 8.294M × 1 B |
| Training | Colab **T4** (cloud GPU), not laptop GPU | setup section + notebooks |
| d64-L8 variant | 5.37M params (likely source of the bogus "6.5M") | parameter count |

---

## A. BLOCKERS — guaranteed rejection if left unaddressed

### A1. Baseline MASE values in Table 1 are not traceable to the cited papers
The paper says "Published results are taken from the respective papers," but:

- **TimesFM (ICML 2024)** publishes **MAE on ETT datasets**, not per-dataset MASE on
  ETT/Exchange/Electricity/Traffic. MASE 0.52/0.71/0.48/1.12/0.89/0.62 does not exist
  in the TimesFM paper.
- **Chronos (TMLR 2024)** publishes MASE only for Benchmark II (zero-shot) — which
  does **not** include ETT datasets or hourly Electricity/Traffic in its tables —
  and uses a different protocol (horizon 5.12×input, seasonal-naive in-sample
  scaling, average over all series). Chronos values 0.61/0.83/0.55/1.34/0.95/0.71
  do not match anything in Chronos Appendix Table 10.
- **PatchTST (ICLR 2023)** reports MSE/MAE, **no MASE anywhere in the paper**.

Any reviewer who opens the cited papers cannot reproduce Table 1. This is the
single highest rejection risk — it looks like fabricated baselines.

**Fix options (pick one):**
1. **Recompute all baselines** with official checkpoints under the paper's own
   protocol (correct but heavy: TimesFM 200M inference on 6 datasets × full series).
2. **Restrict Table 1 to numbers that are verifiable** (e.g., only datasets where
   published MASE exists and protocol is stated to differ), and re-frame claims.
3. **Replace point-comparisons with re-evaluated numbers**, even if the "beats
   TimesFM on electricity/traffic" headline shrinks.

### A2. First-series-only evaluation vs. all-series published baselines
- Electricity = 321 clients, Traffic = 862 sensors. Published MASE is averaged
  over **all** series. The paper evaluates **only the first series** ("first client
  as target").
- Comparing a single-series MASE to all-series published MASE is apples-to-oranges.
- This invalidates the headline claims: electricity 0.709 vs TimesFM 0.89, traffic
  0.535 vs 0.62, and "best-in-class on traffic."
- **Fix:** either evaluate all series (cheap for an 8.3M model — the whole point of
  the paper), or re-run baselines on the same first-series protocol and say so.

### A3. Non-standard MASE denominator
- Paper defines MASE with in-sample naive error computed on the **context window**
  of each test sample. Standard MASE (Hyndman & Koehler 2006, used by Chronos,
  TimesFM, etc.) uses the in-sample naive MAE over the **training data**.
- Different denominator ⇒ every MASE in the paper is not the MASE in the baseline
  papers, even when the numerator is right.
- **Fix:** adopt the standard definition (or rename the metric, e.g. "context-scaled
  error") and state exactly how baselines were scaled for comparison.

### A4. "1.4 MB INT8" is physically impossible
- 8.294M params ⇒ INT8 = 8.29 MB, FP16 = 16.6 MB, FP32 = 33.2 MB. The ONNX export
  numbers in Contribution 4, abstract, Table 1, and all deploy assets are wrong.
- "16 MB" only works as FP16. "1.4 MB" matches no precision for this model.
- **Fix:** state "~17 MB FP16, ~8.3 MB INT8" — or export and measure the real files.
- Also fix the model card: it says 6,518,104 params (6.5M) everywhere — contradicts
  the paper's 8.3M. Pick one number and propagate.

---

## B. HIGH RISK — reviewer-visible, will be caught by an expert

### B1. Reverso is mischaracterized — and it is essentially the same architecture
- Paper cites Reverso as "time-reversed copies as a data augmentation strategy."
  Reverso is an **architecture** paper: small hybrid models interleaving
  **long convolution + DeltaNet (linear RNN) layers** — exactly NanoForecast's
  architecture (blocks.py: LongConvolution + DeltaNetBlock + gated MLP + router).
  Its "flip equivariance" is an inference-time strategy, not training augmentation.
- Reverso: 2.6M params, Gift-Eval MASE **0.711** — i.e., a 3× smaller model of the
  same architecture achieves a better MASE on the harder GIFT-Eval benchmark.
- A reviewer will conclude: (a) the citation is misread, and (b) the core
  architecture contribution is Reverso's, with NanoForecast's novelty reduced to
  the training fixes — which is the actual paper. It must be cited honestly as the
  closest prior architecture work, differentiated explicitly, and (ideally) compared
  against on GIFT-Eval or equivalent.
- Related: "applied Reverso augmentation (time-reversed copies) to real series" in
  Issue 3 must be re-described — that is not Reverso's method.

### B2. Lag-Llama venue is wrong
- Cited as "Advances in Neural Information Processing Systems (NeurIPS), 2023."
  Lag-Llama was a NeurIPS 2023 **workshop** paper (R0-FoMo workshop). Fix citation.

### B3. Train/val/test splits are wrong for ETT
- Paper claims 70/10/20 for all six datasets. Standard protocol: ETT = 12/4/4 months
  = **70/20/10**; Electricity/Traffic = 7:1:2 = **70/10/20**; Exchange = 7:1:2.
- **Fix:** 70/20/10 for ETT×3, 70/10/20 for Electricity/Traffic/Exchange.

### B4. TimesFM training-corpus wording
- "trained on 100B time series tokens" → TimesFM trained on ~100B time **points**
  (synthetic + real), not tokens. Minor but a TimesFM author will notice.

### B5. Size-multiplier inconsistencies (24× vs 25× vs 10–200×)
- 200/8.294 = **24.1×**; 710/8.294 = **85.6×**.
- Abstract says "25× smaller"; Fig. 1 caption says "24×"; intro/contribution say
  "10–200×" and "25–200×" in different spots. Pick one canonical set:
  **24× (TimesFM)** and **~86× (Chronos)**.

### B6. "Competitive within 2–3×" is overstated
- Actual ratios vs TimesFM: ETTh1 0.913/0.52 = 1.76×, ETTh2 0.914/0.71 = 1.29×,
  ETTm1 1.305/0.48 = 2.7×. Say "within ~2× on most ETT" or give per-dataset ratios.

### B7. Figure 1 arrow labels are wrong
- Arrows claim "incremental improvement" but show **cumulative** percentages
  (21.2% → 34.8% → 51.4%). Incremental deltas are 21.2% / 17.4% / 25.5%.
- Math error: (2.15 − 1.78)/2.15 = **17.2%**, not 17.4%. Fix both.

### B8. Loss-scope bug description contradicts Algorithm 1
- Algorithm 1 truncates to H when `multi_horizon=True`; text says the bug occurred
  when `multi_horizon=False` used the full-context loss path. As written, the fix
  and the bug are about opposite flags. Rewrite Issue 1 to be unambiguous
  (e.g., "loss was computed over the full reconstructed context (B,C) instead of
  the forecast horizon (B,H) under the dense next-token schedule").

---

## C. MEDIUM RISK — undermines credibility, may draw reviewer comments

1. **Cost claim $0.12 for 12h T4** — that's $0.01/hr; implausible for any provider.
   Either verify the actual bill/spot price or say "≈$0.12 under preemptible
   pricing" and show the arithmetic. Also "trained on a laptop GPU" (abstract) vs
   "Colab T4" (setup) — a T4 is a datacenter GPU, not a laptop GPU. Align wording:
   "single cloud GPU (T4-class)… cheap enough to be run on any laptop-class setup"
   or just say T4 consistently.
2. **Quantile heads are never evaluated** — architecture section sells quantile
   heads + anomaly head; results contain no CRPS/WQL/PICP. Either add a small
   probabilistic evaluation or explicitly scope it out with a sentence.
3. **Streaming speedup "256–512×" vs patching** — with patch_size=8, per-observation
   reprocessing cost is O(C/P)=64 tokens, so the honest claim is ~64× vs full
   reprocessing, not 512×. Adjust the claim.
4. **Related work is stale for a 2026 submission** — no Moirai, Chronos-Bolt,
   TimesFM 2.x, Timer-XL, GIFT-Eval benchmark, or the 2025-26 "foundation models
   don't beat simple baselines" literature. Add at least: GIFT-Eval (2025), Moirai
   (ICLR 2024), Chronos-Bolt (2025), TimesFM 2.5/2.0, Toto (ICML 2025).
5. **"~12 hours / 200 epochs" vs "trains in 2 minutes"** — model card claims
   "trains on your laptop in 2 minutes"; paper says 12 hours for 200 epochs. Align
   the docs; paper is fine, card is not.
6. **`pip install nanoforecast`** — verify this package exists on PyPI before
   claiming it; otherwise use the GitHub/HF installation path.
7. **Table 1 efficiency ratio** — arithmetic is internally consistent (verified
   0.091 / 0.007 / 0.0017; 13.1× / 53.5×), but the whole table inherits A1–A3.

---

## D. Minor / cosmetic
- Model card param count (6.5M) everywhere vs paper (8.3M) — propagate the fix.
- "45% fewer parameters than PatchTST": 8.294/15 = 55.3% ⇒ 44.7% fewer ✓ (fine).
- Exchange "8 currencies 1990–2016" ✓; Electricity "321 clients 2012–2014" ✓;
  Traffic "862 sensors 2015–2016" ✓; ETT "oil temperature" ✓.
- Citations verified real: FrAug (2302.09292), Timer (ICML 2024), SAMformer,
  iTransformer, DLinear, N-BEATS, Informer, LSTNet (Lai SIGIR 2018), UCI
  Electricity (Trindade 2015), Hyndman-Koehler 2006, Shazeer 2020, Douaioui 2024 ✓.
- Reverso (2602.17634) is real but context is wrong → B1.

---

## Suggested fix order
1. Decide the **baseline strategy** (A1) — everything else hinges on it.
2. Fix A2 (all-series evaluation for Electricity/Traffic — cheap for this model).
3. Fix A3 (standard MASE denominator) + rerun metrics.
4. Fix A4 (ONNX sizes; model-card param count).
5. Fix B1 (Reverso framing + add architecture differentiation + GIFT-Eval mention).
6. Apply B2–B8, C1–C4 mechanical fixes.
7. Recompile; regenerate figures; update LAUNCH_KIT/LINKEDIN assets to match the
   corrected numbers (the 1.4 MB and 6.5M/25× claims are in the launch materials).
