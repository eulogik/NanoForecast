# NanoForecast v0.5 — YouTube Video Plan

## Video Title Options

**Recommended**: "6.5M Parameters vs 200M: The Tiny Model That Beats Google on All 3 ETT Benchmarks"

**Alternatives**:
- "I Trained an AI Model on a Free Colab GPU That Beats Google's TimesFM on 3 Benchmarks"
- "How a 31× Smaller AI Model Beats Google's TimesFM"
- "The Free-to-Train AI Model That Competes with Google's 200M Parameter Model"

## Thumbnail Options

**Option A (Recommended)**:
- Left: "6.5M params" with small chip icon
- Right: "200M params" with Google logo
- Center: "Beats it on 3/6 benchmarks"
- Your face looking thoughtful/impressed
- Clean background, orange accent

**Option B**:
- Split screen: Laptop screen (free Colab) vs Google data center ($$$)
- Text: "31× smaller, still wins"
- Or: "David vs Goliath: AI Edition"

## Metadata

### Description
```
I trained a time series forecasting model (6.5M parameters) on a free Google Colab T4 GPU (~12 hours). It beats Google's TimesFM (200M parameters) on all three ETT benchmarks — ETTh1, ETTh2, ETTm1 — under an identical standard evaluation protocol.

This video covers:
• How a 31× smaller model can beat a 200M param model on 3 benchmarks
• The 3 silent training-pipeline fixes that improved accuracy 46.6% (zero architecture changes)
• How to train and deploy the model yourself
• What this means for AI accessibility

Links:
• Live demo: https://huggingface.co/spaces/eulogik/nanoforecast
• GitHub: https://github.com/eulogik/NanoForecast
• Paper: https://arxiv.org/abs/XXXX.XXXXX

Timestamps:
0:00 - The result that surprised me
1:30 - Full benchmark comparison
3:00 - How the architecture works (no PhD required)
5:00 - The 3 silent pipeline fixes
8:00 - The ablation study
9:30 - How to deploy it yourself
11:30 - What this means for AI
12:30 - Try it yourself

#TimeSeries #AI #Forecasting #OpenSource #MachineLearning
```

### Tags
```
time series forecasting, AI, machine learning, forecasting, Google TimesFM, open source, edge AI, Raspberry Pi, small model, efficient AI, training pipeline, model optimization, laptop AI, ONNX, streaming inference, PatchTST, ETT benchmark, NanoForecast, Eulogik
```

---

## Video Script (13 minutes)

### Scene 1: Hook (0:00 - 1:30)

**[Screen: Benchmark table, highlighting the three ETT rows]**

"Here's something I didn't expect to see.

We trained a model with 6.5 million parameters. Google's TimesFM has 200 million. Ours is 31 times smaller.

But on ETTh1 forecasting — we got a MASE of 0.685. Google got 0.705. We outperformed them.

On ETTh2 — we got 1.109. Google got 1.360. We outperformed them.

On ETTm1 — we got 0.289. Google got 0.545. We outperformed them.

**[Cut to face cam]**

I'm Gautam Kishore. I built this model as part of my work at Eulogik. And I want to be clear about what this is and isn't.

It's not that our model is better than Google's overall. On exchange_rate, electricity, and traffic, TimesFM still wins. But on all three ETT benchmarks — with 31× fewer parameters — the smaller model performed better.

The interesting question isn't "who wins." It's: how can a model trained on a free Colab GPU even be in the same conversation as a model trained on Google's infrastructure?

That's what this video is about."

---

### Scene 2: The Full Benchmark Picture (1:30 - 3:00)

**[Screen: Full benchmark table]**

"Let me show you the complete picture. Here are all 6 standard forecasting benchmarks, measured under one identical protocol — context 512, horizon 48, non-overlapping test windows, all channels, MASE scaled by seasonal-naive error. Every number you see was produced by us for every model.

**[Highlight each]**

ETTh1: NanoForecast 0.685, TimesFM 0.705 — we win.
ETTh2: NanoForecast 1.109, TimesFM 1.360 — we win.
ETTm1: NanoForecast 0.289, TimesFM 0.545 — we win.
Exchange rate: NanoForecast 4.418, TimesFM 4.383 — TimesFM wins.
Electricity: NanoForecast 2.093, TimesFM 0.923 — TimesFM wins.
Traffic: NanoForecast 1.915, TimesFM 0.765 — TimesFM wins.

**[Cut to face cam]**

So: 3-3. That's honest.

But here's the thing that makes this interesting, and it's not about winning or losing. It's about what the comparison tells us about efficiency."

**[Show size comparison]**

"Size: 6.5M params vs 200M params. That's 31× smaller.
Training cost: a free Colab T4 session vs Google's undisclosed investment.
Inference hardware: a $35 Raspberry Pi vs a GPU cluster.

The ratio matters. How much performance per parameter? Per dollar? Per watt?"

---

### Scene 3: How the Architecture Achieves This (3:00 - 5:00)

**[Screen: Architecture diagram]**

"So how does a 6.5M parameter model compete with 200M? Two reasons:

**[Highlight first]**

First, the architecture is designed for efficiency. It uses three components:

1. LongConv — a wide convolution that captures global patterns
2. DeltaNet RNN — a recurrent layer that maintains state across calls
3. A learned gated router that decides which to use per token

The DeltaNet is key — it remembers what it's seen. Feed it one new value, get an updated forecast immediately. No other time series model does this.

**[Highlight second]**

But the bigger reason — the one nobody talks about — is the training pipeline.

**[Cut to face cam]**

Here's the truth: between v0.3 and v0.5 we changed zero architecture — and improved MASE from 3.282 to 1.752. That's a 46.6% improvement, entirely from fixing the training pipeline."

---

### Scene 4: The Three Silent Pipeline Fixes (5:00 - 8:00)

**[Screen: Code]**

"Fix number 1: loss-scope handling.

**[Show incorrectly scoped loss]**

The multi-task loss was being computed and weighted over the wrong scope — including a stray 'horizon' key that always activated the multi-horizon loss path even when it was disabled.

**[Screen: Next code section]**

Fix number 2: tensor shape alignment.

When computing quantile loss — the confidence intervals — we were comparing tensors of mismatched shapes. The gradients were flowing through wrong dimensions. The model was learning from incorrect signals.

**[Screen: Augmentation visualization]**

Fix number 3: augmentation coverage.

We expanded the augmentation scheme — jitter, scaling, shifts, masking, and time reversal — applied uniformly to real and synthetic records. The model learned richer behavior instead of overfitting to easy patterns.

**[Cut to face cam]**

Three fixes. All silent — the model still trained, still converged, still looked fine. It was just 46.6% worse than it should have been: overall MASE 3.282 → 1.752.

The biggest gain was exchange_rate: 12.847 → 4.418, a 65.6% improvement.

How many models out there have the same problem?"

---

### Scene 5: Deployment and Practical Use (8:00 - 10:30)

**[Screen: Terminal + Raspberry Pi demo]**

"Here's where this becomes practical.

**[Type commands]**

pip install nanoforecast
python train_from_csv.py --csv your_data.csv --target revenue --horizon 48

That's it. Your model is trained.

Export to ONNX:
python -m nanoforecast.export.onnx --checkpoint checkpoints/nanoforecast --output model.onnx

About 13 megabytes standard, ~6.5 megabytes quantized.

**[Show Raspberry Pi]**

This is a $35 Raspberry Pi 4. The model is designed for CPU and ARM inference — no GPU, no cloud, no API bill. And the streaming mode updates forecasts per observation via the DeltaNet's recurrent state, without reprocessing history.

**[Show Gradio demo]**

Or just try the live demo at huggingface.co/spaces/eulogik/nanoforecast. Upload any CSV, get a forecast in seconds."

---

### Scene 6: What This Means (10:30 - 12:00)

**[Cut to face cam]**

"So here's what I think this means for AI.

There's a dominant narrative in the industry right now: bigger models, more compute, more parameters. And that's one path to progress.

But this project shows there's another path: better training, smarter engineering, more efficient deployment.

**[Show the 3 fixes again]**

These three pipeline issues were in our training for a long time. We didn't notice because everything worked. Loss went down. Benchmarks improved. We thought the architecture needed more work.

But the problem wasn't the architecture. It was the training.

**[Cut to face cam]**

I think there are a lot of models out there — in production, in research papers — that are performing 30, 40, 50% worse than they could be, because of silent training pipeline issues.

The lesson for anyone building AI: before you add more parameters, check your training code. Really check it. The model you have might already be better than you think."

---

### Scene 7: CTA (12:00 - 12:45)

**[Screen: Links]**

"Everything is open source under Apache 2.0.

GitHub: github.com/eulogik/NanoForecast
Live demo: huggingface.co/spaces/eulogik/nanoforecast
Paper: arxiv.org/abs/XXXX.XXXXX

If this was useful, subscribe. I'm exploring how far small, efficient models can go.

And if you've found silent bugs in your own training pipelines — drop a comment. I think this is more common than anyone admits.

Thanks for watching."

---

## Video Production Checklist

- [ ] Record screen captures (benchmarks, code, terminal, Gradio demo)
- [ ] Record face cam (hook, lesson, CTA segments)
- [ ] Record Raspberry Pi running inference (if available)
- [ ] Edit in DaVinci Resolve (free)
- [ ] Add background music
- [ ] Add captions
- [ ] Create thumbnail
- [ ] Upload to YouTube
- [ ] Add end screens and cards
- [ ] Pin comment with all links

## Post-Upload

1. Share on Twitter with timestamp link to hook
2. Post on Reddit r/MachineLearning with video link
3. Add to LinkedIn as native video
4. Respond to every comment within 24 hours