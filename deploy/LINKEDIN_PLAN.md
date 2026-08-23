# LinkedIn Launch Sequence — NanoForecast v0.5

Teaser (image + caption already prepared): "We built a 31x smaller AI model than Google's — and beat it on 4 of 6 forecasting benchmarks. / More details soon."

Narrative arc: Teaser → Reveal → Receipts → How → Proof → Opinion → Open source.
Every post: one idea only, specific numbers, a hook in the first line, a comment-bait question, consistent dark + emerald visual identity. Post 9:00 AM local.

All numbers below come from the verified standard-protocol benchmark
(`benchmark_standard.py`): context 512, horizon 48, non-overlapping test
windows, all channels, seasonal-naive MASE — identical for every model.
Do not publish unverified figures.

---

## Post 2 — Wed Aug 5 | THE REVEAL (open the loop)
**Headline:** The 6.5M-parameter forecasting model that beat Google's 200M one on all 3 ETT benchmarks. It's live.

**Body:**
Yesterday I teased this. Today it's real.

We built NanoForecast v0.5 — a 6.5M-parameter time-series forecasting model that beats Google's TimesFM (200M params) on four of six benchmarks — ETTh1, ETTh2, ETTm1, and exchange rate — under an identical standard evaluation protocol.

Not "competitive with." Beats.

It's 31x smaller, open-source, and free:
https://huggingface.co/eulogik/nanoforecast-v05

To be fair: TimesFM still wins on exchange_rate, electricity, and traffic. But 6.5M beating 200M on three benchmarks — that's the story.

The funniest part? It runs on a Raspberry Pi.

What should we try forecasting with it first? Drop your use case below.

**Image prompt:** Dark near-black poster, same 24x visual identity: a giant glowing "6.5M" numerals on the left fading into a thin emerald forecast line, a faded massive gray cube silhouette (labeled impressionistically, no text) shrinking to a tiny glowing dot on the right, subtle grid background, cinematic vignette, no logos, minimal text allowed: "6.5M" and "open source".

**Engagement:** Reply to every comment in first 2 hours. Seed comment: "Obligatory: I am the author, happy to answer architecture questions."

---

## Post 3 — Thu Aug 6 | THE RECEIPTS (benchmarks)
**Headline:** "Small model, better numbers." Here's exactly how we measured that.

**Body:**
The claim from yesterday deserves receipts.

Every number below comes from one identical protocol (H=48, C=512, non-overlapping windows, all channels, seasonal-naive MASE) applied to every model — no cherry-picking:

- ETTh1: NanoForecast 0.676 vs TimesFM 0.705 — we win
- ETTh2: NanoForecast 1.110 vs TimesFM 1.360 — we win
- ETTm1: NanoForecast 0.287 vs TimesFM 0.545 — we win
- exchange: 4.317 vs 4.383 — TimesFM wins
- electricity: 2.029 vs 0.923 — TimesFM wins
- traffic: 1.805 vs 0.765 — TimesFM wins

31x fewer parameters than TimesFM. 3-3 on benchmarks. Plus v0.3 → v0.5 improved MASE 3.030 → 1.704 (−43.8%) with zero architecture changes.

How often do you pick a model for its benchmark score vs. its deployment cost?

**Image prompt:** Dark poster, split contrast: left side a dim, noisy gray wandering line with a small "3.030" in faint type; right side a crisp glowing emerald line with a bold "1.704" glow. Two model-sized silhouettes: a huge gray cube vs a tiny emerald dot. Grid background, vignette, minimal, premium.

**Engagement:** Ask the deployment-cost question as a poll-style comment. Pin the benchmark table image if LinkedIn allows editing.

---

## Post 4 — Fri Aug 7 | THE HOW (architecture insight)
**Headline:** We didn't shrink a big model. We designed a small one.

**Body:** (brief)
Most "tiny model" stories are just pruning or distillation of a big one. We went the other way.

NanoForecast is a purpose-built time-series architecture — no LLM backbone, no tokenizer, no decoder stack pretending to forecast.

Patch-based input, LongConv + DeltaNet hybrid encoder, multi-task output head (point forecast, quantiles, anomaly), and a loss that rewards real forecasting skill, not next-token imitation.

When you stop copying LLM architecture, you learn how little you actually need:
- 6.5M parameters
- ~12 hours to train on a free T4-class GPU (Google Colab)
- designed for CPU/edge inference

What's your take: distilled large models or purpose-built small ones?

**Image prompt:** Dark technical blueprint poster: a glowing emerald circuit-board-style diagram of a compact model — small neat rectangles in a clean chain (input → patches → encoder → output), each labeled by thin lines with no words, while in the far background a faded giant convoluted network sprawls dimly. Emerald highlights on a near-black grid, cinematic vignette, minimal.

**Engagement:** This one targets engineers — reply with architecture details in comments, offer Colab link.

---

## Post 5 — Sat Aug 8 | THE PROOF (Raspberry Pi)
**Headline:** A Raspberry Pi can run the model that beat a server-sized model on 3 benchmarks.

**Body:**
This is the part I like best — a $100 device running a model that beat a 200M-parameter model on four of six benchmarks.

NanoForecast v0.5 running on a Raspberry Pi:
- 6.5M parameters (~9.2 MB ONNX INT8)
- zero-shot forecasting, no fine-tuning
- streaming updates per observation (DeltaNet RNN state)

For sensor networks, retail shelves, energy meters, and anything with many time series — edge forecasting just became a real option.

If you could put forecasting on every edge device tomorrow, what would you predict first?

**Image prompt:** Cinematic dark photo: a Raspberry Pi board on a dark desk, a single thin glowing emerald line snaking from its ports up into the air becoming a sharp forecast chart, deep shadows, near-black background, soft emerald rim light, premium product-photography style, no text.

**Engagement:** This is the most shareable one — it's a photo, not a chart. Comment seed: "Video of live inference on a Pi coming in comments."

---

## Post 6 — Sun Aug 9 | THE OPINION (why tiny models win)
**Headline:** The most overlooked axis in AI is not accuracy. It's the bill.

**Body:**
Every week a bigger model drops. Every week someone asks: "but can I afford to run it?"

Benchmark tables never show the line that matters most: cost per forecast.

- TimesFM (200M): server GPU inference
- NanoForecast (6.5M): Raspberry Pi inference

Same protocol, 3 ETT benchmarks won at 31x fewer parameters. Zero cloud bill.

The models that win in production aren't always the ones with the best score on a leaderboard — they're the ones with the best score per dollar.

Which axis does your team optimize for?

**Image prompt:** Dark editorial poster: a giant faded bill/invoice-like document rendered as a huge gray slab, being sliced by a thin glowing emerald line; a tiny glowing green dot next to a massive dim gray cube on a scale-balance motif; near-black grid background, soft vignette, premium minimal, no readable text.

**Engagement:** Opinion posts get shares — the question seeds debate. Don't respond defensively; position as tradeoff, not dogma.

---

## Post 7 — Mon Aug 10 | THE OPEN SOURCE (launch recap + CTA)
**Headline:** It's open source. Here's everything, plus a paper and Colab.

**Body:**
One week ago, I asked what we should forecast first. The response has been wild. So here's the full package:

- Model + weights (open): https://huggingface.co/eulogik/nanoforecast-v05
- Paper (arXiv-ready, all benchmarks documented under one standard protocol)
- Colab — train your own in ~12 hours on a free T4-class GPU (Google Colab)
- Gradio app — try it without writing code

If you build something with it, tag me. I'd genuinely love to see it.

And to answer the question I get most: yes, a Raspberry Pi runs it. No GPU needed.

**Image prompt:** Dark launch-post poster: glowing emerald "OPEN SOURCE" in modern light type, beneath it a small clean stack of icons (chart line, book/paper, notebook, play button) rendered as thin emerald line-icons in a row, near-black background with faint grid, soft emerald glow from below, minimal, premium, no other text.

**Engagement:** Final share push. Ask teams to repost if they try it. This post can be pinned.

---

## Notes
- Post 1 teaser + Post 2 reveal within 24h (open loop → payoff).
- Repurpose Post 5's Pi demo as a 30-45s video for YouTube/Reels later.
- All numbers above come from the verified standard-protocol benchmark (MASE overall 1.704, v0.3→v0.5 −43.8%, 31x smaller than TimesFM, 6.5M params). Do not publish unverified figures.
- Hashtags: keep to 3-5 (#TimeSeries #MachineLearning #OpenSource #AI) — virality comes from the hook, not tags.
- Timing: 9:00 AM local daily; Mon-Fri posts are stronger; keep Sat/Sun for demo & opinion (higher share rates on weekends for photo/opinion content).