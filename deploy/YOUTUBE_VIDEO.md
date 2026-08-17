# NanoForecast v0.5 — YouTube Video Plan

## Video Title Options

**Recommended**: "8.3M Parameters vs 200M: The Tiny Model That Outperforms Google on 2 Benchmarks"

**Alternatives**:
- "I Trained an AI Model for $0.12 That Outperforms Google's on 2 Benchmarks"
- "How a 25× Smaller AI Model Outperforms Google's TimesFM"
- "The $0.12 AI Model That Competes with Google's 200M Parameter Model"

## Thumbnail Options

**Option A (Recommended)**:
- Left: "8.3M params" with small chip icon
- Right: "200M params" with Google logo
- Center: "Outperforms on 2/6 benchmarks"
- Your face looking thoughtful/impressed
- Clean background, orange accent

**Option B**:
- Split screen: Laptop screen ($0.12) vs Google data center ($$$)
- Text: "25× smaller, still wins"
- Or: "David vs Goliath: AI Edition"

## Metadata

### Description
```
I trained a time series forecasting model (8.3M parameters) on a laptop GPU for $0.12 in compute. It outperforms Google's TimesFM (200M parameters) on 2 of 6 standard benchmarks — electricity and traffic forecasting.

This video covers:
• How a 25× smaller model can outperform a 200M param model
• The 3 silent training bugs that were destroying our accuracy
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
5:00 - The 3 silent bugs
8:00 - The ablation study
9:30 - How to deploy it yourself
11:30 - What this means for AI
12:30 - Try it yourself

#TimeSeries #AI #Forecasting #OpenSource #MachineLearning
```

### Tags
```
time series forecasting, AI, machine learning, forecasting, Google TimesFM, open source, edge AI, Raspberry Pi, small model, efficient AI, training pipeline, model optimization, $0.12 training, laptop AI, ONNX, streaming inference, PatchTST, electricity forecasting, traffic forecasting, NanoForecast, Eulogik
```

---

## Video Script (13 minutes)

### Scene 1: Hook (0:00 - 1:30)

**[Screen: Benchmark table, highlighting electricity and traffic rows]**

"Here's something I didn't expect to see.

We trained a model with 8.3 million parameters. Google's TimesFM has 200 million. Ours is 25 times smaller.

But on electricity forecasting — we got a MASE of 0.709. Google got 0.89. We outperformed them.

On traffic forecasting — we got 0.535. Google got 0.62. We outperformed them.

**[Cut to face cam]**

I'm Gautam Kishore. I built this model as part of my work at Eulogik. And I want to be clear about what this is and isn't.

It's not that our model is better than Google's overall. On 4 out of 6 benchmarks, TimesFM still wins. But on 2 benchmarks — electricity and traffic — the smaller model performed better.

The interesting question isn't "who wins." It's: how can a model trained on a laptop for $0.12 even be in the same conversation as a model trained on Google's infrastructure?

That's what this video is about."

---

### Scene 2: The Full Benchmark Picture (1:30 - 3:00)

**[Screen: Full benchmark table]**

"Let me show you the complete picture. Here are all 6 standard forecasting benchmarks.

**[Highlight each]**

ETTh1: NanoForecast 0.913, TimesFM 0.52 — TimesFM wins.
ETTh2: NanoForecast 0.914, TimesFM 0.71 — TimesFM wins.
ETTm1: NanoForecast 1.305, TimesFM 0.48 — TimesFM wins.
Exchange rate: NanoForecast 3.578, TimesFM 1.12 — TimesFM wins.

**[Highlight green]**

Electricity: NanoForecast 0.709, TimesFM 0.89 — we win.
Traffic: NanoForecast 0.535, TimesFM 0.62 — we win.

**[Cut to face cam]**

So: 4-2 in Google's favor. That's honest.

But here's the thing that makes this interesting, and it's not about winning or losing. It's about what the comparison tells us about efficiency."

**[Show size comparison]**

"Size: 8.3M params vs 200M params. That's 25x smaller.
Training cost: $0.12 vs Google's undisclosed investment.
Inference hardware: a $35 Raspberry Pi vs a GPU cluster.

The ratio matters. How much performance per parameter? Per dollar? Per watt?"

---

### Scene 3: How the Architecture Achieves This (3:00 - 5:00)

**[Screen: Architecture diagram]**

"So how does an 8.3M parameter model compete with 200M? Two reasons:

**[Highlight first]**

First, the architecture is designed for efficiency. It uses three components:

1. LongConv — a wide convolution that captures global patterns
2. DeltaNet RNN — a recurrent layer that maintains state across calls
3. A learned gated router that decides which to use per token

The DeltaNet is key — it remembers what it's seen. Feed it one new value, get an updated forecast in less than a millisecond. No other time series model does this.

**[Highlight second]**

But the bigger reason — the one nobody talks about — is the training pipeline.

**[Cut to face cam]**

Here's the truth: our model was performing 51% worse than it should have been. Not because of the architecture. Because of 3 silent bugs in our training code.

When I fixed those bugs, accuracy improved by 51%. Same model. Same parameters. Just better training."

---

### Scene 4: The Three Silent Bugs (5:00 - 8:00)

**[Screen: Code]**

"Bug number 1: loss scope mismatch.

**[Show incorrectly computed loss]**

We were computing loss on the full time window — past and future. But the model should only be evaluated on its predictions for the future. The loss function was penalizing it for not predicting history correctly. One line fix.

Improvement: 21%.

**[Screen: Next code section]**

Bug number 2: tensor shape misalignment.

When computing quantile loss — the confidence intervals — we were comparing tensors of different sizes. The gradients were flowing through wrong dimensions. The model was learning from incorrect signals.

Another fix. Improvement: 35% cumulative.

**[Screen: Data mix visualization]**

Bug number 3: bad data mixing.

We were using a 50/50 mix of real and synthetic data. Synthetic sine waves and random walks are too simple. The model was learning to forecast easy patterns instead of complex real-world behavior.

Fix: tripled real data weight, added time-reversed copies.

Improvement: 51% cumulative.

**[Cut to face cam]**

Three bugs. All silent — the model still trained, still converged, still looked fine. It was just 51% worse than it should have been.

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

16 megabytes standard. 1.4 megabytes quantized. Smaller than most photos.

**[Show Raspberry Pi]**

This is a $35 Raspberry Pi 4. It runs the model at 45 milliseconds per forecast. With ONNX optimization: 12 milliseconds.

If you need real-time updates — IoT sensors, live dashboards, financial feeds — the streaming mode updates in under a millisecond per new data point.

**[Show Gradio demo]**

Or just try the live demo at huggingface.co/spaces/eulogik/nanoforecast. Upload any CSV, get a forecast in seconds."

---

### Scene 6: What This Means (10:30 - 12:00)

**[Cut to face cam]**

"So here's what I think this means for AI.

There's a dominant narrative in the industry right now: bigger models, more compute, more parameters. And that's one path to progress.

But this project shows there's another path: better training, smarter engineering, more efficient deployment.

**[Show the 3 bugs again]**

These three bugs were in our code for months. We didn't notice because everything worked. Loss went down. Benchmarks improved. We thought the architecture needed more work.

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
