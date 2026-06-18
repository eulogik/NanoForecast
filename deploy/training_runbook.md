# NanoForecast v0.2 Training — Mac Mini Runbook

## Goal
Train a substantially better checkpoint (v0.2) on the Mac Mini M4 16GB.
Target: MASE < 2.0 on ETTh1, sensible forecasts on all datasets.

## Prerequisites on the Mac Mini

```bash
# Clone the repo
git clone https://github.com/eulogik/NanoForecast.git
cd NanoForecast

# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .                    # nanoforecast package
pip install -r requirements.txt     # deps
pip install "nanoforecast[dev]"     # pytest + ruff

# Verify
python3 -c "from nanoforecast import NanoForecast; print('ok')"
```

## Training Command

Run this on the Mac Mini (M4, MPS available):

```bash
# Clean old checkpoint
rm -rf checkpoints

# Launch training
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

### Expected behavior
- **Data loading**: downloads + caches all 6 datasets on first run
- **Training**: ~2-4 hours on M4 16GB with MPS
- **Validation loss**: should converge below ~1.0 (down from v0.1's ~11)
- **Checkpoint saved**: every epoch to `checkpoints/nanoforecast-500k/`

### If training crashes
- **bf16 autocast issues on MPS**: add `--device cpu` (slower but stable)
- **OOM**: reduce `--batch-size` to 32 or `--d-model` to 48
- **NaN loss**: reduce `--lr` to 1e-5

## Post-Training Steps

Once training finishes on the Mac Mini:

```bash
# 1. Benchmark
python3 benchmark.py \
  --checkpoint checkpoints/nanoforecast-500k \
  --datasets ETTh1,ETTh2,ETTm1,exchange_rate,electricity,traffic \
  --max-windows 64 \
  --output results/benchmark-500k.json

# 2. Push to HF Hub
huggingface-cli login  # if not already done
python3 push_to_hub.py \
  --checkpoint checkpoints/nanoforecast-500k \
  --repo-id eulogik/nanoforecast-500k \
  --benchmark-json results/benchmark-500k.json

# 3. Update the Gradio Space to use new model
#    (edit app.py: change MODEL_REPO to "eulogik/nanoforecast-500k")

# 4. Push changes back to GitHub
git add -A
git commit -m "v0.2: trained on 6 datasets, 100 epochs, d_model=64"
git push
```

## Expected Results vs v0.1

| Metric | v0.1 (current) | v0.2 (target) |
|---|---|---|
| Training data | 1 dataset, ~1000 windows | 6 datasets, ~100K windows |
| Training time | ~1 min CPU | ~2-4 hrs MPS |
| Parameters | 676K (d_model=32, layers=4) | ~1.6M (d_model=64, layers=8) |
| ETTh1 MASE | ~4.6 | < 2.0 |
| Exchange Rate MASE | ~10.9 | < 2.0 |
| Forecast quality | Noisy, barely follows trend | Should follow trend + seasonality |

## What to Monitor

During training, watch for:
1. **val_loss decreasing steadily** — target < 1.0 by epoch 50
2. **No NaN** — if NaN appears, training is unstable (lower LR or use CPU)
3. **MPS memory** — should stay under ~12 GB; if OOM, reduce batch size

## FAQ

**Q: How long should I let it run?**
A: Full 100 epochs. If val_loss plateaus before then, you can stop early.

**Q: The Mac Mini has no internet for HF push?**
A: Copy the checkpoint dir via USB/network to this machine and push from here.

**Q: Can I interrupt and resume?**
A: No built-in resume yet. Let it run uninterrupted (or wrap in tmux).
