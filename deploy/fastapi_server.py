"""FastAPI server for NanoForecast — deploy in 2 commands.

Usage:
  pip install nanoforecast fastapi uvicorn
  python deploy/fastapi_server.py

Or via Docker:
  docker build -t nanoforecast -f deploy/Dockerfile .
  docker run -p 8000:8000 nanoforecast
"""
from __future__ import annotations

import io
import json
import logging
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from nanoforecast import NanoForecast

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("nanoforecast-server")

MODEL_REPO = "eulogik/nanoforecast-500k"  # or local path like "checkpoints/nanoforecast-500k"
DEFAULT_CONTEXT = 256
DEFAULT_HORIZON = 48
DEFAULT_FREQ = 1

app = FastAPI(
    title="NanoForecast API",
    description="World's most deployable time series forecasting model. "
                "700K params, 1.4 MB ONNX, runs on Raspberry Pi.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Optional[NanoForecast] = None


@app.on_event("startup")
def load_model():
    global _model
    logger.info(f"Loading model from {MODEL_REPO}...")
    t0 = time.time()
    _model = NanoForecast.from_pretrained(MODEL_REPO)
    dt = time.time() - t0
    logger.info(f"Model loaded in {dt:.2f}s. Params: ~700K.")


def get_model() -> NanoForecast:
    if _model is None:
        raise HTTPException(503, "Model not loaded yet — server still starting.")
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_REPO}


@app.post("/predict")
def predict(
    context: str = Form(
        ...,
        description="JSON array of floats — the historical time series values. "
                    "Must be >= model context length (256).",
    ),
    horizon: int = Form(
        DEFAULT_HORIZON,
        description="Number of steps to forecast ahead.",
    ),
    freq: int = Form(
        DEFAULT_FREQ,
        description="Frequency ID: 1=hourly, 2=daily, 3=weekly, 4=monthly.",
    ),
    return_components: bool = Form(
        True,
        description="If true, return trend/seasonal/residual decomposition.",
    ),
    return_state: bool = Form(
        False,
        description="If true, return a state token for streaming predict_step calls.",
    ),
):
    model = get_model()
    try:
        series = np.array(json.loads(context), dtype=np.float32)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(400, "context must be a JSON array of floats.")

    if series.ndim != 1 or len(series) < model.config.context_length:
        raise HTTPException(
            400,
            f"Need a 1D array with >= {model.config.context_length} elements, "
            f"got shape {series.shape}.",
        )

    t0 = time.time()
    ctx = series[-model.config.context_length:]
    result = model.predict(ctx, horizon=horizon, freq=freq,
                           return_components=return_components,
                           return_state=return_state)
    dt = time.time() - t0

    response: dict = {
        "forecast": result["forecast"][0].tolist(),
        "quantiles": {
            "p10": result["quantiles"][0, :, 0].tolist(),
            "p25": result["quantiles"][0, :, 1].tolist(),
            "p50": result["quantiles"][0, :, 2].tolist(),
            "p75": result["quantiles"][0, :, 3].tolist(),
            "p90": result["quantiles"][0, :, 4].tolist(),
        },
        "context_used": model.config.context_length,
        "horizon": horizon,
        "inference_time_s": round(dt, 4),
    }

    if return_components:
        response["decomposition"] = {
            "trend": result["trend"][0].tolist(),
            "seasonal": result["seasonal"][0].tolist(),
            "residual": result["residual"][0].tolist(),
        }

    if return_state and "state" in result:
        state = result["state"]
        response["streaming_state"] = {
            "buffer": state.buffer.squeeze().tolist(),
            "num_patches_seen": state.num_patches_seen,
        }
    return response


@app.post("/predict_stream")
def predict_stream(
    value: str = Form(..., description="Single new observation (float)."),
    buffer: str = Form(..., description="Current rolling buffer as JSON array of floats."),
    delta_states: str = Form(..., description="Placeholder — pass empty list [] for now."),
    horizon: int = Form(DEFAULT_HORIZON),
    freq: int = Form(DEFAULT_FREQ),
):
    """Stream one value and return an updated forecast.

    Client sends the current rolling buffer plus the new value.
    The server returns the forecast, and the client is responsible
    for maintaining the buffer for the next call.

    For a minimal demo, this avoids server-side session management.
    The DeltaNet state is rebuilt from the buffer on each call
    (future work: serialize DeltaNet states to enable true stateless streaming).
    """
    model = get_model()
    try:
        val = float(json.loads(value) if isinstance(value, str) and value.startswith("[") else value)
        buf = np.array(json.loads(buffer), dtype=np.float32)
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(400, "value and buffer must be valid")

    # Append new value and trim to context length
    buf = np.append(buf, val)
    if len(buf) > model.config.context_length:
        buf = buf[-model.config.context_length:]

    t0 = time.time()
    result = model.predict(buf, horizon=horizon, freq=freq)
    dt = time.time() - t0

    return {
        "forecast": result["forecast"][0].tolist(),
        "quantiles": {
            "p10": result["quantiles"][0, :, 0].tolist(),
            "p50": result["quantiles"][0, :, 2].tolist(),
            "p90": result["quantiles"][0, :, 4].tolist(),
        },
        "inference_time_s": round(dt, 4),
        "buffer_remaining": len(buf),
    }


@app.post("/predict_csv")
async def predict_csv(
    file: UploadFile = File(..., description="CSV file with time series data."),
    target_col: str = Form(
        ...,
        description="Name of the column to forecast.",
    ),
    horizon: int = Form(DEFAULT_HORIZON),
    freq: int = Form(DEFAULT_FREQ),
    output_format: str = Form("json", description="json or csv"),
):
    model = get_model()
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Invalid CSV file.")

    if target_col not in df.columns:
        raise HTTPException(
            400,
            f"Column '{target_col}' not found. Columns: {list(df.columns)}",
        )

    series = df[target_col].dropna().values
    if len(series) < model.config.context_length:
        raise HTTPException(
            400,
            f"Need >= {model.config.context_length} rows, got {len(series)}.",
        )

    ctx = series[-model.config.context_length:].astype(np.float32)
    result = model.predict(ctx, horizon=horizon, freq=freq)

    if output_format == "csv":
        buf = io.StringIO()
        buf.write("step,forecast,p10,p50,p90\n")
        for i in range(horizon):
            buf.write(f"{i+1},{result['forecast'][0][i]:.6f},"
                      f"{result['quantiles'][0,i,0]:.6f},"
                      f"{result['quantiles'][0,i,2]:.6f},"
                      f"{result['quantiles'][0,i,4]:.6f}\n")
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=forecast.csv"},
        )

    return {
        "forecast": result["forecast"][0].tolist(),
        "horizon": horizon,
        "target_column": target_col,
        "total_timesteps": len(series),
        "context_used": model.config.context_length,
        "quantiles": {
            "p10": result["quantiles"][0, :, 0].tolist(),
            "p50": result["quantiles"][0, :, 2].tolist(),
            "p90": result["quantiles"][0, :, 4].tolist(),
        },
    }


@app.post("/predict_onnx")
async def predict_onnx(
    file: UploadFile = File(..., description="ONNX model file."),
    context: str = Form(..., description="JSON array of floats."),
    horizon: int = Form(DEFAULT_HORIZON),
    freq: int = Form(DEFAULT_FREQ),
):
    try:
        import onnxruntime as ort
    except ImportError:
        raise HTTPException(400, "onnxruntime not installed on server.")

    try:
        model_bytes = await file.read()
        series = np.array(json.loads(context), dtype=np.float32)
    except Exception:
        raise HTTPException(400, "Invalid model or context.")

    ctx_len = 256
    if len(series) < ctx_len:
        series = np.pad(series, (ctx_len - len(series), 0))
    ctx = series[-ctx_len:].reshape(1, ctx_len, 1)

    session = ort.InferenceSession(model_bytes)
    t0 = time.time()
    forecast = session.run(None, {"input": ctx})[0]
    dt = time.time() - t0

    return {
        "forecast": forecast[0, :horizon, 0].tolist(),
        "inference_time_s": round(dt, 4),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
