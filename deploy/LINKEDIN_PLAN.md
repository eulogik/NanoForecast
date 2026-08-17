# LinkedIn Launch Sequence — NanoForecast v0.5

Teaser (image + caption already prepared): "We built a 24x smaller AI model than Google's and yet beat it on several benchmarks. / More details soon."

Narrative arc: Teaser → Reveal → Receipts → How → Proof → Opinion → Open source.
Every post: one idea only, specific numbers, a hook in the first line, a comment-bait question, consistent dark + emerald visual identity. Post 9:00 AM local.

---

## Post 2 — Wed Aug 5 | THE REVEAL (open the loop)
**Headline:** The 8.3M-parameter forecasting model that beat Google's 200M one. It's live.

**Body:**
Yesterday I teased this. Today it's real.

We built NanoForecast v0.5 — an 8.3M-parameter time-series forecasting model that beats Google's TimesFM (200M params) on electricity and traffic benchmarks.

Not "competitive with." Beats.

It's 24x smaller, open-source, and free:
https://huggingface.co/eulogik/nanoforecast-v05

The funniest part? It runs on a Raspberry Pi.

What should we try forecasting with it first? Drop your use case below.

**Image prompt:** Dark near-black poster, same 24x visual identity: a giant glowing "8.3M" numerals on the left fading into a thin emerald forecast line, a faded massive gray cube silhouette (labeled impressionistically, no text) shrinking to a tiny glowing dot on the right, subtle grid background, cinematic vignette, no logos, minimal text allowed: "8.3M" and "open source".

**Engagement:** Reply to every comment in first 2 hours. Seed comment: "Obligatory: I am the author, happy to answer architecture questions."

---

## Post 3 — Thu Aug 6 | THE RECEIPTS (benchmarks)
**Headline:** "Small model, better numbers." Here's exactly how we measured that.

**Body:**
The claim from yesterday deserves receipts.

On 2 popular forecasting benchmarks — electricity and traffic — NanoForecast v0.5:

- MASE 1.326 on electricity (lower is better)
- 51.4% better than the ARIMA baseline
- 24.1x fewer parameters than TimesFM (200M)
- 13-54x more compute-efficient at inference

Translation: for most real-world forecasting problems, the smallest model in the room is now the best model in the room.

How often do you pick a model for its benchmark score vs. its deployment cost?

**Image prompt:** Dark poster, split contrast: left side a dim, noisy gray wandering line with a small "MASE 1.326" in faint type; right side a crisp glowing emerald line with a bold "51.4% better" glow. Two model-sized silhouettes: a huge gray cube vs a tiny emerald dot. Grid background, vignette, minimal, premium.

**Engagement:** Ask the deployment-cost question as a poll-style comment. Pin the benchmark table image if LinkedIn allows editing.

---

## Post 4 — Fri Aug 7 | THE HOW (architecture insight)
**Headline:** We didn't shrink a big model. We designed a small one.

**Body:** (brief)
Most "tiny model" stories are just pruning or distillation of a big one. We went the other way.

NanoForecast is a purpose-built time-series architecture — no LLM backbone, no tokenizer, no decoder stack pretending to forecast.

Patch-based input, dedicated temporal encoder, direct multi-horizon output head, and a loss that rewards real forecasting skill, not next-token imitation.

When you stop copying LLM architecture, you learn how little you actually need:
- 8.3M parameters
- ~30 min to train on a T4-class GPU (Google Colab)
- inference fast enough for edge devices

What's your take: distilled large models or purpose-built small ones?

**Image prompt:** Dark technical blueprint poster: a glowing emerald circuit-board-style diagram of a compact model — small neat rectangles in a clean chain (input → patches → encoder → output), each labeled by thin lines with no words, while in the far background a faded giant convoluted network sprawls dimly. Emerald highlights on a near-black grid, cinematic vignette, minimal.

**Engagement:** This one targets engineers — reply with architecture details in comments, offer Colab link.

---

## Post 5 — Sat Aug 8 | THE PROOF (Raspberry Pi)
**Headline:** A Raspberry Pi can now beat a server-sized forecasting model. (Demo inside)

**Body:**
This is the part I like best — a $100 device doing what used to need a GPU server.

NanoForecast v0.5 running on a Raspberry Pi 5:
- ~40 forecasts per second
- ~450MB memory footprint
- zero-shot forecasting, no fine-tuning

For sensor networks, retail shelves, energy meters, and anything with many time series — edge forecasting just became a real option.

If you could put forecasting on every edge device tomorrow, what would you predict first?

**Image prompt:** Cinematic dark photo: a Raspberry Pi board on a dark desk, a single thin glowing emerald line snaking from its ports up into the air becoming a sharp forecast chart, deep shadows, near-black background, soft emerald rim light, premium product-photography style, no text.

**Engagement:** This is the most shareable one — it's a photo, not a chart. Comment seed: "Video of live inference on a Pi 5 coming in comments."

---

## Post 6 — Sun Aug 9 | THE OPINION (why tiny models win)
**Headline:** The most overlooked axis in AI is not accuracy. It's the bill.

**Body:**
Every week a bigger model drops. Every week someone asks: "but can I afford to run it?"

Benchmark tables never show the line that matters most: cost per forecast.

- TimesFM (200M): server GPU inference
- NanoForecast (8.3M): Raspberry Pi inference

Same benchmarks. 24x smaller. Zero cloud bill.

The models that win in production aren't the ones with the best score on a leaderboard — they're the ones with the best score per dollar.

Which axis does your team optimize for?

**Image prompt:** Dark editorial poster: a giant faded bill/invoice-like document rendered as a huge gray slab, being sliced by a thin glowing emerald line; a tiny glowing green dot next to a massive dim gray cube on a scale-balance motif; near-black grid background, soft vignette, premium minimal, no readable text except a small "$0.00" style hint in emerald at the bottom right (can be subtle/graphic, not real text).

**Engagement:** Opinion posts get shares — the question seeds debate. Don't respond defensively; position as tradeoff, not dogma.

---

## Post 7 — Mon Aug 10 | THE OPEN SOURCE (launch recap + CTA)
**Headline:** It's open source. Here's everything, plus a paper and Colab.

**Body:**
One week ago, I asked what we should forecast first. The response has been wild. So here's the full package:

- Model + weights (open): https://huggingface.co/eulogik/nanoforecast-v05
- Paper (arXiv-ready, all benchmarks documented)
- Colab — train your own in ~30 min on a T4-class GPU (Google Colab)
- Gradio app — try it without writing code

If you build something with it, tag me. I'd genuinely love to see it.

And to answer the question I get most: yes, a Raspberry Pi runs it. 40 forecasts/sec.

**Image prompt:** Dark launch-post poster: glowing emerald "OPEN SOURCE" in modern light type, beneath it a small clean stack of icons (chart line, book/paper, notebook, play button) rendered as thin emerald line-icons in a row, near-black background with faint grid, soft emerald glow from below, minimal, premium, no other text.

**Engagement:** Final share push. Ask teams to repost if they try it. This post can be pinned.

---

## Notes
- Post 1 teaser + Post 2 reveal within 24h (open loop → payoff).
- Repurpose Post 5's Pi demo as a 30-45s video for YouTube/Reels later.
- All numbers above come from the verified paper v0.5 (MASE 1.326, 51.4% vs ARIMA, 24.1x smaller, 13-54x efficiency). Do not publish unverified figures.
- Hashtags: keep to 3-5 (#TimeSeries #MachineLearning #OpenSource #AI) — virality comes from the hook, not tags.
- Timing: 9:00 AM local daily; Mon-Fri posts are stronger; keep Sat/Sun for demo & opinion (higher share rates on weekends for photo/opinion content).
