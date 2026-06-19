"""demo.py — One-liner to load NanoForecast and forecast.

Usage:
  python3 demo.py

Loads a pretrained checkpoint and demonstrates all features:
  - One-shot predict
  - Streaming / online inference
  - Decomposition

For training on your own data:
  python3 train_from_csv.py --csv my_data.csv --target sales --horizon 48
"""
from __future__ import annotations

import numpy as np

from nanoforecast import NanoForecast

MODEL_SOURCE = "eulogik/nanoforecast-500k"

print(f"Loading model from {MODEL_SOURCE}...")
model = NanoForecast.from_pretrained(MODEL_SOURCE)
cfg = model.config

t = np.linspace(0, 8 * np.pi, cfg.context_length)
context = np.sin(t) + 0.1 * np.random.randn(cfg.context_length)

# --- One-shot forecast ---
print("One-shot forecast (48 steps)...")
result = model.predict(context, horizon=48, freq=1, return_components=True)
print(f"  Forecast (first 10): {result['forecast'][0, :10].round(4).tolist()}")

# --- Streaming inference ---
print("Streaming inference (8 steps, one at a time)...")
result = model.predict(context, horizon=48, freq=1, return_state=True)
state = result["state"]
for i in range(8):
    new_val = float(np.sin(8 * np.pi + i * 0.2) + 0.1 * np.random.randn())
    step = model.predict_step(new_val, state, horizon=48, freq=1)
print(f"  Final forecast (first 10): {step['forecast'][0, :10].round(4).tolist()}")

print()
print("Next steps:")
print("  Train on your data:")
print("    python3 train_from_csv.py --csv sales.csv --target revenue --horizon 48")
print("  Deploy:")
print("    python3 deploy/fastapi_server.py")
print("  Export to ONNX:")
print("    python3 -m nanoforecast.export.onnx_export")
