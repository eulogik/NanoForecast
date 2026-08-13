"""Convert the standalone google/timesfm-1.0-200m-pytorch checkpoint into a
transformers-native checkpoint (TimesFmModelForPrediction).

Usage:
    python3 benchmarks/timesfm_loader.py
    -> benchmarks/checkpoints/timesfm-1.0-200m/{config.json, model.safetensors}
"""
from __future__ import annotations

import json
import os
import sys

import torch

HUB = os.path.expanduser("~/.cache/huggingface/hub/models--google--timesfm-1.0-200m-pytorch")
OUT = os.path.join(os.path.dirname(__file__), "checkpoints", "timesfm-1.0-200m")


def find_ckpt() -> str:
    snap = os.path.join(HUB, "snapshots")
    if not os.path.isdir(snap):
        raise FileNotFoundError(f"no snapshot dir under {HUB}")
    for rev in os.listdir(snap):
        p = os.path.join(snap, rev, "torch_model.ckpt")
        if os.path.exists(p):
            return p
    raise FileNotFoundError("torch_model.ckpt not found")


def remap(sd: dict) -> dict:
    """standalone key -> transformers TimesFmModelForPrediction key."""
    out = {}
    for k, v in sd.items():
        if k.startswith("stacked_transformer.layers."):
            rest = k[len("stacked_transformer.layers."):]
            n, _, sub = rest.partition(".")
            if sub.startswith("self_attention."):
                sub = "self_attn." + sub[len("self_attention."):]
            if sub.startswith("self_attn.qkv_proj."):
                tail = sub[len("self_attn.qkv_proj."):]
                if tail == "weight":
                    out[f"decoder.layers.{n}.self_attn.q_proj.weight"] = v[:1280]
                    out[f"decoder.layers.{n}.self_attn.k_proj.weight"] = v[1280:2560]
                    out[f"decoder.layers.{n}.self_attn.v_proj.weight"] = v[2560:3840]
                else:
                    out[f"decoder.layers.{n}.self_attn.q_proj.bias"] = v[:1280]
                    out[f"decoder.layers.{n}.self_attn.k_proj.bias"] = v[1280:2560]
                    out[f"decoder.layers.{n}.self_attn.v_proj.bias"] = v[2560:3840]
                continue
            out[f"decoder.layers.{n}.{sub}"] = v
        elif k.startswith("input_ff_layer."):
            if k.startswith("input_ff_layer.hidden_layer.0."):
                k = k.replace("input_ff_layer.hidden_layer.0.", "input_ff_layer.input_layer.", 1)
            out["decoder." + k] = v
        elif k.startswith("horizon_ff_layer."):
            if k.startswith("horizon_ff_layer.hidden_layer.0."):
                k = k.replace("horizon_ff_layer.hidden_layer.0.", "horizon_ff_layer.input_layer.", 1)
            out[k] = v
        elif k.startswith("freq_emb."):
            out["decoder." + k] = v
        else:
            out[k] = v
    return out


def main():
    from transformers import TimesFmConfig, TimesFmModelForPrediction

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    ckpt = find_ckpt()
    print(f"loading {ckpt}")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif "model" in sd:
        sd = sd["model"]
    mapped = remap(sd)

    cfg = TimesFmConfig(
        patch_length=32, context_length=512, horizon_length=128, freq_size=3,
        num_hidden_layers=20, hidden_size=1280, intermediate_size=1280,
        head_dim=80, num_attention_heads=16, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        pad_val=1123581321.0, attention_dropout=0.0, use_positional_embedding=True,
        tolerance=1e-6,
        rms_norm_eps=1e-6, initializer_range=0.02, min_timescale=1.0, max_timescale=10000.0,
    )
    model = TimesFmModelForPrediction(cfg)
    own = model.state_dict()

    missing, unexpected, mismatch = [], [], []
    for k, v in mapped.items():
        if k not in own:
            unexpected.append(k)
        elif tuple(v.shape) != tuple(own[k].shape):
            mismatch.append((k, tuple(v.shape), tuple(own[k].shape)))
    for k in own:
        if k not in mapped and own[k].numel() > 1:
            missing.append(k)
    print(f"mapped keys: {len(mapped)}  missing: {len(missing)}  "
          f"unexpected: {len(unexpected)}  shape-mismatch: {len(mismatch)}")
    for k in missing[:10]:
        print("  MISSING", k, tuple(own[k].shape))
    for k in unexpected[:10]:
        print("  UNEXPECTED", k)
    for k, a, b in mismatch[:10]:
        print("  MISMATCH", k, a, b)

    model.load_state_dict(mapped, strict=False)

    os.makedirs(OUT, exist_ok=True)
    cfg.to_json_file(os.path.join(OUT, "config.json"))
    model.save_pretrained(OUT)
    print(f"saved -> {OUT}  (params {sum(p.numel() for p in model.parameters())/1e6:.1f}M)")


if __name__ == "__main__":
    main()
