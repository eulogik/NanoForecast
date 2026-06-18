You are training the v0.2 checkpoint of NanoForecast — a tiny time series transformer. Follow this plan exactly.

## Setup (run once)
```bash
cd ~/Code/NanoForecast
git pull
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Train
```bash
rm -rf checkpoints
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
  --output checkpoints/nanoforecast-500k 2>&1
```

## Monitor
- Check `val_loss` < 1.0 by epoch 50 (v0.1 was ~11)
- If NaN appears → stop → rerun with `--device cpu --lr 1e-5`
- If OOM → stop → reduce `--batch-size 32` or `--d-model 48`

## When training finishes
```bash
# Benchmark
python3 benchmark.py --checkpoint checkpoints/nanoforecast-500k \
  --datasets ETTh1,ETTh2,ETTm1,exchange_rate,electricity,traffic \
  --max-windows 64 --output results/benchmark-500k.json
```

Then report back:
- Training wall time
- Final val_loss
- Best epoch
- MASE on each dataset from benchmark-500k.json
- Any errors encountered

## Copy back to main machine
The files to transfer:
- `checkpoints/nanoforecast-500k/` (entire directory)
- `results/benchmark-500k.json`

Use whatever method works (scp, USB, AirDrop, cloud upload).
