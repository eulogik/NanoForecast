"""Gradio Space for NanoForecast — try the world's most deployable TS model.

Deploy this on Hugging Face Spaces:
1. Create a new Space at https://huggingface.co/new-space
2. Choose Gradio SDK
3. Upload this file as app.py
4. Add requirements.txt with: nanoforecast gradio plotly pandas
5. Set HF_TOKEN secret if using a private model repo

Or run locally:   gradio gradio_app.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nanoforecast import NanoForecast

# ---------------------------------------------------------------------------
# Configuration — change these as needed
# ---------------------------------------------------------------------------
MODEL_REPO = "eulogik/nanoforecast-200k"  # public repo on HF Hub
DEFAULT_CONTEXT = 256
DEFAULT_HORIZON = 48
DEFAULT_FREQ = 1  # hourly

# ---------------------------------------------------------------------------
# Model loading — cached across invocations
# ---------------------------------------------------------------------------
_model: Optional[NanoForecast] = None


def get_model() -> NanoForecast:
    global _model
    if _model is None:
        _model = NanoForecast.from_pretrained(MODEL_REPO)
    return _model


# ---------------------------------------------------------------------------
# Forecasting logic
# ---------------------------------------------------------------------------

def forecast_from_series(
    series: np.ndarray,
    horizon: int,
    freq: int,
) -> dict:
    model = get_model()
    ctx_len = model.config.context_length
    if len(series) < ctx_len:
        raise ValueError(f"Need at least {ctx_len} timesteps, got {len(series)}")
    context = series[-ctx_len:].astype(np.float32)
    out = model.predict(context, horizon=horizon, freq=freq, return_components=True)
    return {
        "forecast": out["forecast"][0],
        "quantiles": out["quantiles"][0],
        "trend": out["trend"][0],
        "seasonal": out["seasonal"][0],
        "residual": out["residual"][0],
        "context_length": ctx_len,
    }


def build_plot(context_vals, forecast_vals, quantile_vals, horizon, target_vals=None):
    H = len(forecast_vals)
    ctx_len = len(context_vals)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(ctx_len)),
        y=context_vals,
        mode="lines",
        name="Context (history)",
        line=dict(color="royalblue", width=2),
    ))

    x_future = list(range(ctx_len, ctx_len + H))
    p10, p25, p50, p75, p90 = [
        quantile_vals[i] for i in range(5)
    ]

    fig.add_trace(go.Scatter(
        x=x_future + x_future[::-1],
        y=p90.tolist() + p10.tolist()[::-1],
        fill="toself",
        fillcolor="rgba(0,100,200,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="p10–p90 interval",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=x_future + x_future[::-1],
        y=p75.tolist() + p25.tolist()[::-1],
        fill="toself",
        fillcolor="rgba(0,100,200,0.25)",
        line=dict(color="rgba(0,0,0,0)"),
        name="p25–p75 interval",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=x_future,
        y=forecast_vals,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="darkorange", width=2),
        marker=dict(size=4),
    ))

    if target_vals is not None:
        fig.add_trace(go.Scatter(
            x=x_future,
            y=target_vals,
            mode="lines",
            name="Actual",
            line=dict(color="green", width=1.5, dash="dot"),
        ))

    fig.update_layout(
        title="NanoForecast — Forecast with Prediction Intervals",
        xaxis_title="Time step",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Gradio UI functions
# ---------------------------------------------------------------------------

def predict_from_csv(
    csv_file: Optional[Path],
    target_col: str,
    context_len: int,
    horizon: int,
    freq_choice: str,
):
    if csv_file is None:
        return None, "Please upload a CSV file.", None

    try:
        df = pd.read_csv(csv_file.name)
    except Exception as e:
        return None, f"Error reading CSV: {e}", None

    if target_col not in df.columns:
        cols = ", ".join(df.columns[:10])
        return None, f"Column '{target_col}' not found. Available: {cols}", None

    series = df[target_col].dropna().values
    if len(series) < context_len:
        return (
            None,
            f"Need at least {context_len} values in column '{target_col}', got {len(series)}.",
            None,
        )

    freq_map = {"Hourly": 1, "Daily": 2, "Weekly": 3, "Monthly": 4}
    freq_id = freq_map.get(freq_choice, 1)

    result = forecast_from_series(series, horizon=horizon, freq=freq_id)
    context = series[-result["context_length"]:]

    fig = build_plot(
        context_vals=context,
        forecast_vals=result["forecast"],
        quantile_vals=result["quantiles"],
        horizon=horizon,
    )

    table_df = pd.DataFrame({
        "step": list(range(1, horizon + 1)),
        "forecast": result["forecast"],
        "p10": result["quantiles"][0],
        "p25": result["quantiles"][1],
        "p50": result["quantiles"][2],
        "p75": result["quantiles"][3],
        "p90": result["quantiles"][4],
    })

    summary = (
        f"**Model:** {MODEL_REPO}  \n"
        f"**Context:** {len(series)} timesteps (using last {result['context_length']})  \n"
        f"**Horizon:** {horizon} steps  \n"
        f"**Frequency:** {freq_choice}  \n"
        f"**Checkpoint:** ~700K params, 2.7 MB  \n"
    )

    return fig, summary, table_df


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

css = """
.gradio-container { max-width: 1100px !important; }
h1 { text-align: center; }
"""

with gr.Blocks(css=css, title="NanoForecast") as demo:
    gr.Markdown(
        """
        # 🔮 NanoForecast — Deployable Time Series Forecasting

        Upload your CSV or use an example. The **smallest deployable time series model on the Hub**
        (~700K params, runs on a Raspberry Pi, exports to 1.4 MB ONNX).

        [GitHub](https://github.com/eulogik/NanoForecast) ·
        [Model on HF](https://huggingface.co/eulogik/nanoforecast-200k) ·
        [Paper / Docs](https://github.com/eulogik/NanoForecast#readme)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            csv_input = gr.File(
                label="Upload CSV (or leave empty to use example)",
                file_types=[".csv"],
            )
            target_col = gr.Textbox(
                label="Target column name",
                value="OT",
                info="The column containing the time series to forecast.",
            )
            context_len = gr.Slider(
                minimum=64, maximum=512, step=64,
                value=DEFAULT_CONTEXT, label="Context length (timesteps)",
            )
            horizon = gr.Slider(
                minimum=12, maximum=192, step=12,
                value=DEFAULT_HORIZON, label="Forecast horizon (timesteps)",
            )
            freq_choice = gr.Radio(
                choices=["Hourly", "Daily", "Weekly", "Monthly"],
                value="Hourly", label="Data frequency",
            )
            predict_btn = gr.Button("🔮 Forecast", variant="primary")

        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Forecast Plot")
            summary_output = gr.Markdown(label="Summary")
            table_output = gr.Dataframe(label="Forecast Table")

    predict_btn.click(
        fn=predict_from_csv,
        inputs=[csv_input, target_col, context_len, horizon, freq_choice],
        outputs=[plot_output, summary_output, table_output],
    )

    gr.Markdown(
        """
        ---
        ### 📥 Download options
        Once you have a forecast, you can export the model to ONNX for production:
        ```bash
        python3 -m nanoforecast.export.onnx_export \\
            --checkpoint checkpoints/nanoforecast-onnx \\
            --output nanoforecast.onnx
        ```
        Or deploy instantly with our [FastAPI server](https://github.com/eulogik/NanoForecast#deploy).

        ### ⚡ Why NanoForecast?
        - **Tiny**: 200K–700K params, 1.4 MB INT8 quantized
        - **Fast**: <50ms inference on CPU, 12ms on edge hardware
        - **Deployable**: ONNX → browser / Lambda / Raspberry Pi / iOS
        - **Complete**: Point forecast + intervals + decomposition in one pass
        """
    )

if __name__ == "__main__":
    demo.launch()
