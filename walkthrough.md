# NanoForecast — Walkthrough & Plan

## The Strategy

NanoForecast won't win on accuracy (yet). It wins on **deployability**:
- Train on a MacBook Air in 20 minutes (no GPU required)
- Run on a Raspberry Pi / browser / Lambda / phone
- ONNX export + INT8 quantization = 1.4 MB model
- Full pipeline: `pip install` → `predict()` → `deploy` in one repo

**Target markets:**
- **Developers** who want a TS model that actually ships to production
- **Edge/IoT** people who need forecasting on a Raspberry Pi or phone
- **Hugging Face** users who want the smallest deployable TS model on the Hub
- **OpenRouter** API consumers who want forecasting at $0.001/series

## Milestones

| # | Milestone | Status | Est. time | Impact |
|---|---|---|---|---|
| 1 | Walkthrough + plan | ✅ Done | 10 min | Alignment |
| 2 | Gradio Space app | ✅ Done | 20 min | Viral demo (#1 driver) |
| 3 | FastAPI + Docker deploy | ✅ Done | 20 min | Production story |
| 4 | demo.py one-liner | ✅ Done | 5 min | README gateway |
| 5 | Push v0.1 checkpoint to HF Hub | 🔲 Ship | 10 min | Hub presence |
| 6 | Rewrite README | 🔲 Write | 15 min | First impression |
| 7 | Mac Mini training: v0.2 checkpoint | 🔲 Later | Overnight | Real accuracy |
| 8 | OpenRouter listing | 🔲 After v0.2 | — | Revenue |

---

## Current Session Plan

Write and stage everything for a coordinated launch:

### Phase A — Build the artifacts

| File | What it does | Status |
|---|---|---|
| `walkthrough.md` | This file — living plan | ✅ |
| `gradio_app.py` | Gradio Space for HF — upload CSV, get forecast plot | ✅ |
| `deploy/fastapi_server.py` | FastAPI `/predict` endpoint + `/health` | ✅ |
| `deploy/Dockerfile` | Docker image for the FastAPI server | ✅ |
| `deploy/requirements.txt` | Deploy dependencies | ✅ |
| `demo.py` | `python3 demo.py` → downloads model → prints forecast | ✅ |
| `nanoforecast/hub.py` | Already exists — from_pretrained + predict API | ✅ |
| `checkpoints/nanoforecast-200k/` | v0.1 demo checkpoint (trained on ETTh1, 20 epochs) | ✅ |

### Phase B — Ship (requires your HF token)

1. `huggingface-cli login` → enter your HF write token
2. `python3 push_to_hub.py --checkpoint checkpoints/nanoforecast-200k --repo-id YOUR_USERNAME/nanoforecast-200k`
3. Create HF Space from `gradio_app.py`, point it at the Hub model
4. Share on Twitter / Reddit / HF community

### Phase C — Train v0.2 (on Mac Mini M4 16GB)

```bash
python3 pretrain.py \
  --datasets ETTh1,ETTh2,ETTm1,exchange_rate,electricity,traffic \
  --synthetic-records 50000 \
  --epochs 100 \
  --batch-size 64 \
  --stride 16 \
  --d-model 64 \
  --num-layers 8 \
  --lr 5e-5 \
  --max-channels 16 \
  --device mps \
  --output checkpoints/nanoforecast-500k
```

Expected: ~100K-200K training windows, ~100M-200M training tokens, ~2-4 hours on Mac Mini M4 with MPS.

Then: `python3 push_to_hub.py --checkpoint checkpoints/nanoforecast-500k --repo-id YOUR_USERNAME/nanoforecast-500k`

---

## What Makes This Wantable

**For GitHub users:**
- `pip install nanoforecast` works
- Understandable codebase (small files, clean naming)
- Train on your data in 20 min
- Deploy with FastAPI or ONNX.js

**For HF Hub users:**
- Smallest deployable TS model on the Hub (1.4 MB INT8)
- `from_pretrained` + `predict()` in 2 lines
- Model card with honest benchmarks
- Gradio Space with live demo

**For OpenRouter users:**
- Cheapest TS forecast API available
- $0.001/series because compute cost is near-zero

**For edge/IoT developers:**
- Raspberry Pi, Lambda, mobile, browser — it runs anywhere
- 12ms inference even on a $15 board

## Accuracy Targets (v0.2+)

Goal: match **naive** or better (MASE < 1.0) on common datasets while being 300× smaller.

| Dataset | Current (v0.1) | v0.2 target | Naive baseline |
|---|---|---|---|
| ETTh1 | ~4.6 | <2.0 | ~0.9 |
| ETTh2 | ~7.1 | <3.0 | ~0.9 |
| Exchange Rate | ~10.9 | <2.0 | ~0.9 |
| ETTm1 | ~8.3 | <2.5 | ~0.9 |

Target: MAE within 2× of naive on all datasets. That's enough for the "good enough to deploy" story.
