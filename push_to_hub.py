"""Push a pretrained NanoForecast checkpoint to the Hugging Face Hub.

Usage:
    python3 push_to_hub.py \
        --checkpoint checkpoints/nanoforecast-200k \
        --repo-id your-hf-username/nanoforecast-200k \
        --private

This uploads the entire HF-style artifact directory (config.json,
model.safetensors, model_card.json) and, if missing, generates a
README.md model card from the benchmark JSON.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from huggingface_hub import HfApi, whoami


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Push a NanoForecast checkpoint to the HF Hub")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Local directory containing the artifact (config.json + model.safetensors + model_card.json)")
    p.add_argument("--repo-id", type=str, required=True,
                   help="Target Hub repo id, e.g. 'your-username/nanoforecast-200k'")
    p.add_argument("--private", action="store_true", help="Make the repo private")
    p.add_argument("--commit-message", type=str, default="Upload NanoForecast checkpoint")
    p.add_argument("--benchmark-json", type=str, default=None,
                   help="Optional path to benchmark.json to embed in the model card")
    return p.parse_args()


def render_model_card(
    artifact_dir: str,
    repo_id: str,
    benchmark: Optional[dict] = None,
) -> str:
    """Render a Hugging Face model card Markdown from the artifact metadata."""
    with open(os.path.join(artifact_dir, "config.json"), "r") as fh:
        cfg = json.load(fh)
    card_path = os.path.join(artifact_dir, "model_card.json")
    card = {}
    if os.path.exists(card_path):
        with open(card_path, "r") as fh:
            card = json.load(fh)

    params = card.get("params", sum(int(v) for v in cfg.values() if isinstance(v, int)))
    profile = card.get("profile", "nano-200k")
    training = card.get("training", {})

    md = [
        "---",
        "license: apache-2.0",
        "tags:",
        "  - time-series",
        "  - forecasting",
        "  - pytorch",
        "  - deployable",
        "  - edge-ai",
        "  - onnx",
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        f"NanoForecast is the **world's most deployable** time series forecasting model "
         f"(~{params/1e3:.0f}K parameters). It trains on a laptop, runs on a Raspberry Pi, "
         f"and exports to 1.4 MB ONNX for edge/IoT/browser deployment.",
         "",
         "Built by [Eulogik](https://eulogik.com) — deployable AI for the real world.",
         "",
         f"[![Open in Spaces](https://img.shields.io/badge/🤗%20Open%20in%20Spaces-blueviolet)]"
         f"(https://huggingface.co/spaces/{repo_id.split('/')[0]}/nanoforecast)",
         "",
         "## Model details",
         "",
         f"- **Profile**: `{profile}`",
         f"- **Parameters**: {params:,}",
         f"- **Context length**: {cfg['context_length']}",
         f"- **Prediction length**: {cfg['prediction_length']}",
         f"- **Patch size**: {cfg['patch_size']}",
         f"- **Hidden dim / layers**: {cfg['d_model']} / {cfg['num_layers']}",
         f"- **Quantiles**: {cfg['quantiles']}",
         f"- **Architecture**: LongConv + DeltaNet RNN + Gated Router + MLP",
         f"- **Streaming inference**: Stateful DeltaNet — feed one value at a time",
         "",
         "## Deploy",
         "",
         "```bash",
         "# FastAPI server",
         "pip install nanoforecast fastapi uvicorn python-multipart",
         "python3 deploy/fastapi_server.py",
         "",
         "# Docker",
         "docker build -t nanoforecast -f deploy/Dockerfile .",
         "docker run -p 8000:8000 nanoforecast",
         "",
         "# ONNX export (1.4 MB)",
         "pip install \"nanoforecast[onnx]\"",
         "python3 -m nanoforecast.export.onnx_export --checkpoint <checkpoint-dir> --output nanoforecast.onnx",
         "```",
        "",
        "## Training",
        "",
    ]
    if training:
        md.append(f"- **Datasets**: {', '.join(training.get('datasets', []))}")
        md.append(f"- **Epochs**: {training.get('epochs')}")
        md.append(f"- **Learning rate**: {training.get('lr')}")
        md.append(f"- **Batch size**: {training.get('batch_size')}")
        md.append(f"- **Best epoch**: {training.get('best_epoch')} (val_loss={training.get('best_val_loss', float('nan')):.4f})")
        md.append(f"- **Wall time**: {training.get('wall_time_s', 0):.1f}s")
    md.append("")

    if benchmark and benchmark.get("datasets"):
        md.append("## Benchmarks")
        md.append("")
        md.append("| Dataset | MASE | sMAPE (%) | MAE | CRPS |")
        md.append("|---|---:|---:|---:|---:|")
        for name, m in benchmark["datasets"].items():
            if "error" in m:
                continue
            md.append(f"| {name} | {m['mase']:.3f} | {m['smape']:.2f} | {m['mae']:.3f} | {m['crps']:.3f} |")
        if "overall" in benchmark:
            md.append(f"| **overall** | **{benchmark['overall']['mase']:.3f}** | **{benchmark['overall']['smape']:.2f}** | **{benchmark['overall']['mae']:.3f}** | **{benchmark['overall']['crps']:.3f}** |")
        md.append("")

    md.extend([
        "## Quickstart",
        "",
        "```python",
        "import numpy as np",
        "from nanoforecast import NanoForecast",
        "",
        f"model = NanoForecast.from_pretrained('{repo_id}')",
        "context = np.sin(np.linspace(0, 8*np.pi, 256)) + 0.1 * np.random.randn(256)",
        "out = model.predict(context, horizon=48, freq=1)",
        "print(out['forecast'].shape)  # (48,) point forecast",
        "print(out['quantiles'].shape)  # (5, 48)  p10..p90",
        "```",
        "",
        "## Try it in a browser",
        "",
        f"Upload your CSV to the [Gradio Space](https://huggingface.co/spaces/",
        f"{repo_id.split('/')[0]}/nanoforecast) and get a forecast in seconds.",
        "",
        "## Known limitations",
        "",
        "This checkpoint was trained on 6 real datasets + 50K synthetic records for 100 epochs. "
        "It is **not** a production foundation model. Accuracy is modest (MASE ~3.45 overall). "
        "What it does well: being deployable. Train on your own data for better accuracy.",
        "",
        "## Attribution",
        "",
        "Built by [Eulogik](https://eulogik.com) — deployable AI for the real world.",
    ])
    return "\n".join(md) + "\n"


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.checkpoint):
        raise SystemExit(f"Checkpoint directory not found: {args.checkpoint}")

    api = HfApi()
    try:
        whoami(token=None)  # validates that the user is logged in
    except Exception as e:
        raise SystemExit(
            f"Hugging Face login required: {e}\n"
            "Run `huggingface-cli login` and provide a write token."
        )

    # Render README.md (overwrites if exists) so the Hub displays a real model card
    benchmark = None
    if args.benchmark_json and os.path.exists(args.benchmark_json):
        with open(args.benchmark_json, "r") as fh:
            benchmark = json.load(fh)
    readme_md = render_model_card(args.checkpoint, args.repo_id, benchmark)
    readme_path = os.path.join(args.checkpoint, "README.md")
    with open(readme_path, "w") as fh:
        fh.write(readme_md)

    print(f"Creating repo {args.repo_id} (private={args.private}) ...")
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

    print(f"Uploading files from {args.checkpoint} ...")
    api.upload_folder(
        folder_path=args.checkpoint,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
    )
    print(f"Done. View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
