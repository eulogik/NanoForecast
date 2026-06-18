"""demo.py — One-liner to load NanoForecast and forecast.

Usage:
  python3 demo.py

Loads a pretrained checkpoint and generates a forecast.
After pushing to HF Hub, change MODEL_SOURCE to your repo ID.
"""
from __future__ import annotations

import numpy as np

from nanoforecast import NanoForecast

MODEL_SOURCE = "checkpoints/nanoforecast-200k"  # or "your-user/nanoforecast-200k"

print(f"🔮 NanoForecast — Loading model from {MODEL_SOURCE}...")
model = NanoForecast.from_pretrained(MODEL_SOURCE)

t = np.linspace(0, 8 * np.pi, model.config.context_length)
context = np.sin(t) + 0.1 * np.random.randn(model.config.context_length)

print("📊 Forecasting 48 steps ahead...")
result = model.predict(context, horizon=48, freq=1, return_components=True)

print(f"\n✅ Forecast (first 10 steps): {result['forecast'][0, :10].round(4).tolist()}")
print(f"   Mean: {result['forecast'].mean():.4f}")
print(f"   Std:  {result['forecast'].std():.4f}")

if result.get("trend") is not None:
    print(f"   Trend component — last 10: {result['trend'][0, -10:].round(4).tolist()}")
    print(f"   Seasonal component — last 10: {result['seasonal'][0, -10:].round(4).tolist()}")

print("\n🎯 Model loaded and predicted successfully!")
print("   Next steps:")
print("   - Try your own data:   model.predict(my_series, horizon=H)")
print("   - Deploy via FastAPI:  pip install fastapi uvicorn && python3 deploy/fastapi_server.py")
print("   - Export to ONNX:      python3 -m nanoforecast.export.onnx_export")
print("   - Train on your data:  python3 pretrain.py --data your_data.csv")
