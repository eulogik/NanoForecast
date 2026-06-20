"""Generate the v0.3 Colab training notebook as a .ipynb file."""
from __future__ import annotations
import json, os, textwrap

def cell(source: str | list[str], cell_type: str = "code") -> dict:
    if isinstance(source, str):
        source = source.splitlines(keepends=True)
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
        "outputs": [] if cell_type == "code" else None,
        "execution_count": None if cell_type == "code" else None,
    }

# ── Markdown helpers ──
M_SETUP = cell(textwrap.dedent("""\
    # NanoForecast v0.3 — Colab T4 Training

    **Target:** Train a d_model=96 (~3.6M param) model on 6 real datasets + 100K synthetic records.
    **Resumable:** If the session drops, re-run "Run all" — training picks up where it left off.
    **Drive usage:** ~50 MB for checkpoint + final artifact. Old checkpoints auto-cleaned.

    ---
    """).strip(), "markdown")

M_CONFIG = cell("## 1. Configuration & Hyperparameters", "markdown")

M_DATA = cell("## 2. Data Loading", "markdown")

M_MODEL = cell("## 3. Model Creation", "markdown")

M_TRAIN = cell("## 4. Training", "markdown")

M_COMPLETE = cell("## 5. Post-Training Cleanup", "markdown")

M_BENCH = cell("## 6. Benchmark", "markdown")

M_RESULTS = cell("## 7. Results", "markdown")

M_PUSH = cell("## 8. Push to Hugging Face Hub (Optional)", "markdown")

# ── Code cells ──
C_SETUP = cell("""\
# ── GPU check ──
import torch, sys, os, json, time, math, random, shutil, warnings, subprocess
warnings.filterwarnings("ignore")
print(f"PyTorch {torch.__version__} | Python {sys.version}")
cuda_ok = torch.cuda.is_available()
if not cuda_ok:
    raise SystemExit("ERROR: No GPU detected. Go to Runtime → Change runtime type → T4 GPU.")
props = torch.cuda.get_device_properties(0)
vram_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f} GB")

# ── Install nanoforecast + deps from GitHub ──
# Install nanoforecast + deps
# Use git with GIT_TERMINAL_PROMPT=0 to avoid auth hang in Colab
env = os.environ.copy()
env["GIT_TERMINAL_PROMPT"] = "0"
r = subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "git+https://github.com/eulogik/NanoForecast.git",
     "safetensors", "matplotlib"],
    capture_output=True, text=True, env=env,
)
if r.returncode != 0:
    print("pip install failed. Trying alternate method (archive download) ...")
    # Fallback: download the repo as a tarball and install from local copy
    subprocess.run(
        ["curl", "-sL", "https://github.com/eulogik/NanoForecast/archive/main.tar.gz",
         "-o", "/tmp/nanoforecast.tar.gz"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["tar", "xzf", "/tmp/nanoforecast.tar.gz", "-C", "/tmp"],
        check=True, capture_output=True,
    )
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "/tmp/NanoForecast-main", "safetensors", "matplotlib"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("All install methods failed:")
        print(r.stderr[-2000:] if r.stderr else "(no stderr)")
        print(r.stdout[-2000:] if r.stdout else "(no stdout)")
        raise SystemExit(1)
import nanoforecast
print(f"nanoforecast installed | safetensors ok")

# ── Mount Google Drive ──
from google.colab import drive
drive.mount("/content/drive")
DRIVE_ROOT = "/content/drive/MyDrive/nanoforecast-v03"
os.makedirs(f"{DRIVE_ROOT}/final", exist_ok=True)
os.makedirs(f"{DRIVE_ROOT}/tmp", exist_ok=True)
print(f"Drive root: {DRIVE_ROOT}")

# ── Imports ──
import numpy as np
from nanoforecast.config import NanoForecastConfig
from nanoforecast.model.core import NanoForecast
from nanoforecast.data.generator import SyntheticTimeSeriesGenerator
from nanoforecast.data.real_datasets import WindowSpec, build_mixed_pretraining_corpus, time_based_split
from nanoforecast.data.pipeline import create_dataloader
from nanoforecast.train.loss import MultiTaskLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch import nn
""")

C_CONFIG = cell("""\
# ── Model config ──
CONFIG = NanoForecastConfig(
    context_length=512,        # up from 256
    prediction_length=48,
    d_model=96,                # up from 64
    num_layers=8,              # same as v0.2
    patch_size=8,
    covariate_dim=4,
)

# ── Training hyperparams ──
TRAIN = dict(
    epochs=300,
    batch_size=32,
    lr=3e-5,
    weight_decay=0.01,
    clip_grad=1.0,
    val_fraction=0.2,
    stride=16,
    max_channels=4,
    synthetic_records=100_000,
    datasets=["ETTh1", "ETTh2", "ETTm1", "exchange_rate", "electricity", "traffic"],
    seed=42,
    checkpoint_interval=5,     # save training_state.pt every N epochs
)

# ── Loss weights ──
LOSS_KW = dict(
    quantiles=CONFIG.quantiles,
    w_point=0.5,
    w_quantile=1.0,
    w_anomaly=0.1,
    w_smooth=0.05,
)

# ── Paths ──
STATE_PATH = f"{DRIVE_ROOT}/training_state.pt"
FINAL_DIR = f"{DRIVE_ROOT}/final"
TMP_DIR = f"{DRIVE_ROOT}/tmp"

print(f"Model params (approx): d_model={CONFIG.d_model}, layers={CONFIG.num_layers}, ctx={CONFIG.context_length}")
print(f"Checkpoint: {STATE_PATH}")
print(f"Final artifact: {FINAL_DIR}")
""")

C_RESUME = cell("""\
# ── Resume detection ──
def _final_exists():
    return os.path.isfile(f"{FINAL_DIR}/config.json") and (
        os.path.isfile(f"{FINAL_DIR}/model.safetensors") or os.path.isfile(f"{FINAL_DIR}/model.pt")
    )

def detect_resume():
    if _final_exists():
        print("✓ Final model found → training already complete. Skipping to benchmark.")
        return {"action": "skip", "epoch": 0, "best_val": float("inf")}

    if os.path.isfile(STATE_PATH):
        state = torch.load(STATE_PATH, map_location="cpu", weights_only=False)
        ep = state.get("epoch", 0)
        best = state.get("best_val_loss", float("inf"))
        print(f"↻ Found training state @ epoch {ep} (best_val_loss={best:.4f}) → resuming.")
        return {"action": "resume", "epoch": ep, "best_val": best, "state": state}

    print("✗ No existing state found → starting fresh.")
    return {"action": "fresh", "epoch": 0, "best_val": float("inf")}

RESUME = detect_resume()
""")

C_DATA = cell("""\
# ── Build corpus (fast, always fresh) ──
def build_corpus():
    seed = TRAIN["seed"]
    random.seed(seed)
    np.random.seed(seed)

    spec = WindowSpec(context_len=CONFIG.context_length, prediction_len=CONFIG.prediction_length, stride=TRAIN["stride"])

    print("Loading real datasets ...")
    real = build_mixed_pretraining_corpus(spec, datasets=TRAIN["datasets"], max_channels_per_dataset=TRAIN["max_channels"])
    print(f"  real records: {len(real)}")

    print("Generating synthetic data ...")
    gen = SyntheticTimeSeriesGenerator(seed=seed)
    syn = gen.generate_dataset(num_series=TRAIN["synthetic_records"], context_len=CONFIG.context_length, prediction_len=CONFIG.prediction_length)
    print(f"  synthetic records: {len(syn)}")

    all_records = real + syn
    train_records, val_records = time_based_split(all_records, val_fraction=TRAIN["val_fraction"])
    print(f"  train: {len(train_records)} | val: {len(val_records)}")

    train_loader = create_dataloader(train_records, batch_size=TRAIN["batch_size"], augment=True, shuffle=True, drop_last=False)
    val_loader = create_dataloader(val_records, batch_size=TRAIN["batch_size"], augment=False, shuffle=False, drop_last=False)
    print(f"  train batches: {len(train_loader)} | val batches: {len(val_loader)}")
    return train_loader, val_loader

train_loader, val_loader = build_corpus()
""")

C_MODEL = cell("""\
# ── Model, loss, optimizer, scheduler ──
model = NanoForecast(CONFIG).cuda()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {n_params:,} ({n_params/1e6:.2f}M)")

loss_fn = MultiTaskLoss(**LOSS_KW)
optimizer = AdamW(model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])

# Scheduler (re-created on resume too; fast-forwarded below if needed)
steps_per_epoch = len(train_loader)
scheduler = OneCycleLR(
    optimizer,
    max_lr=TRAIN["lr"] * 10,
    epochs=TRAIN["epochs"],
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1,
    anneal_strategy="cos",
)

# ── Resume weights + scheduler state ──
START_EPOCH = 0
best_val_loss = float("inf")
best_epoch = 0
history = []

if RESUME["action"] == "resume":
    s = RESUME["state"]
    model.load_state_dict(s["model_state_dict"])
    optimizer.load_state_dict(s["optimizer_state_dict"])
    START_EPOCH = s["epoch"]
    best_val_loss = s["best_val_loss"]
    best_epoch = s.get("best_epoch", 0)
    history = s.get("history", [])
    # Fast-forward scheduler
    n_done = START_EPOCH * steps_per_epoch
    for _ in range(n_done):
        scheduler.step()
    print(f"Resumed at epoch {START_EPOCH} (fast-forwarded {n_done} scheduler steps)")
elif RESUME["action"] == "skip":
    START_EPOCH = TRAIN["epochs"]

print(f"Training: epochs {START_EPOCH + 1} → {TRAIN['epochs']}")
""")

C_TRAIN = cell("""\
# ── Training loop (resumable, self-checkpointing) ──
if START_EPOCH >= TRAIN["epochs"]:
    print("Training already complete. Skipping.")
else:
    device_type = "cuda"
    autocast_dtype = torch.bfloat16

    for epoch in range(START_EPOCH + 1, TRAIN["epochs"] + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_losses = {}
        for batch in train_loader:
            x = batch["x"].cuda()
            y = batch["y"].cuda()
            fid = batch["freq_id"].cuda()
            cov = batch["covariates"].cuda() if "covariates" in batch else None

            optimizer.zero_grad(set_to_none=True)
            if autocast_dtype in (torch.bfloat16, torch.float16):
                with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype):
                    outputs = model(x, fid, cov)
                    loss, ld = loss_fn(outputs, y, x)
            else:
                outputs = model(x, fid, cov)
                loss, ld = loss_fn(outputs, y, x)

            loss.backward()
            if TRAIN["clip_grad"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), TRAIN["clip_grad"])
            optimizer.step()
            scheduler.step()

            for k, v in ld.items():
                train_losses[k] = train_losses.get(k, 0.0) + v

        for k in train_losses:
            train_losses[k] /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_losses = {}
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].cuda()
                y = batch["y"].cuda()
                fid = batch["freq_id"].cuda()
                cov = batch["covariates"].cuda() if "covariates" in batch else None
                if autocast_dtype in (torch.bfloat16, torch.float16):
                    with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype):
                        outputs = model(x, fid, cov)
                        _, ld = loss_fn(outputs, y, x)
                else:
                    outputs = model(x, fid, cov)
                    _, ld = loss_fn(outputs, y, x)
                for k, v in ld.items():
                    val_losses[f"val_{k}"] = val_losses.get(f"val_{k}", 0.0) + v

        for k in val_losses:
            val_losses[k] /= len(val_loader)

        elapsed = time.time() - t0
        val_total = val_losses.get("val_loss_total", 0.0)
        train_total = train_losses.get("loss_total", 0.0)
        print(f"E {epoch:03d}/{TRAIN['epochs']:03d} | "
              f"loss={train_total:.4f} | val_loss={val_total:.4f} | "
              f"{elapsed:.0f}s | lr={scheduler.get_last_lr()[0]:.2e}")

        metrics = {"epoch": epoch, **train_losses, **val_losses, "time_s": elapsed}
        history.append(metrics)

        # ── Best model tracking ──
        is_best = val_total < best_val_loss
        if is_best:
            best_val_loss = val_total
            best_epoch = epoch

        # ── Save checkpoint (every checkpoint_interval epochs OR at best) ──
        if epoch % TRAIN["checkpoint_interval"] == 0 or is_best or epoch == TRAIN["epochs"]:
            ckpt = {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": CONFIG,
                "history": history,
                "train_kw": TRAIN,
            }
            torch.save(ckpt, STATE_PATH)
            if is_best:
                print(f"  ★ New best val_loss={best_val_loss:.4f} (saved)")

        # ── GPU memory check ──
        if epoch % 20 == 0:
            alloc = torch.cuda.memory_allocated() / 1e9
            res = torch.cuda.memory_reserved() / 1e9
            print(f"  VRAM: {alloc:.1f}GB allocated / {res:.1f}GB reserved")

    # ── Save final artifact ──
    model.eval()
    model.save_pretrained(FINAL_DIR)
    meta = {
        "model_name": "NanoForecast",
        "profile": f"d{CONFIG.d_model}-L{CONFIG.num_layers}",
        "params": n_params,
        "config": CONFIG,
        "training": {
            "datasets": TRAIN["datasets"],
            "synthetic_records": TRAIN["synthetic_records"],
            "epochs": TRAIN["epochs"],
            "lr": TRAIN["lr"],
            "batch_size": TRAIN["batch_size"],
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "wall_time_s": sum(m.get("time_s", 0) for m in history),
        },
    }
    with open(f"{FINAL_DIR}/model_card.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"\\n✅ Training complete! Best epoch={best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"   Final artifact saved to {FINAL_DIR}")
""")

C_CLEANUP = cell("""\
# ── Remove training state → prevents re-training on next "Run all" ──
if os.path.isfile(STATE_PATH):
    os.remove(STATE_PATH)
    print("Removed training_state.pt — re-running this notebook will skip training.")

# ── Clean temp dir ──
if os.path.isdir(TMP_DIR):
    shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)
    print("Cleaned tmp/")

# ── Verify final artifact ──
artifacts = ["config.json", "model_card.json"]
if os.path.isfile(f"{FINAL_DIR}/model.safetensors"):
    artifacts.append("model.safetensors")
elif os.path.isfile(f"{FINAL_DIR}/model.pt"):
    artifacts.append("model.pt")
for fn in artifacts:
    path = f"{FINAL_DIR}/{fn}"
    size = os.path.getsize(path) / 1e3
    print(f"  {fn}: {size:.0f} KB")
""")

C_BENCH = cell("""\
# ── Benchmark on all 6 datasets via benchmark.py CLI ──
# Takes ~5-10 minutes. Set SKIP_BENCH to True to skip.

SKIP_BENCH = False  # ← set to True to skip

if not SKIP_BENCH and os.path.isfile(f"{FINAL_DIR}/config.json"):
    print("Running benchmark (6 datasets, max 128 windows each) ...")
    out_path = f"{DRIVE_ROOT}/benchmark-v03.json"
    datasets = ",".join(TRAIN["datasets"])
    # Clone the repo to get benchmark.py (the entry point isn't installed as a module)
    repo_tmp = "/tmp/nanoforecast-repo"
    if not os.path.isdir(repo_tmp):
        subprocess.run(["git", "clone", "--depth=1", "https://github.com/eulogik/NanoForecast.git", repo_tmp], check=True, capture_output=True)
    cmd = [
        sys.executable, f"{repo_tmp}/benchmark.py",
        "--checkpoint", FINAL_DIR,
        "--datasets", datasets,
        "--max-windows", "128",
        "--output", out_path,
        "--device", "cuda",
    ]
    subprocess.run(cmd, check=True)

    # Print summary from results
    if os.path.isfile(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"\\n{'='*60}")
        print(f"{'Dataset':<20} {'MASE':<10} {'sMAPE':<10} {'MAE':<10} {'CRPS':<10}")
        print(f"{'-'*60}")
        for name, m in results.get("datasets", {}).items():
            if "error" in m:
                print(f"{name:<20} ERROR: {m['error']}")
            else:
                print(f"{name:<20} {m['mase']:<10.3f} {m['smape']:<10.2f} {m['mae']:<10.3f} {m['crps']:<10.3f}")
        if "overall" in results:
            o = results["overall"]
            print(f"{'-'*60}")
            print(f"{'OVERALL':<20} {o['mase']:<10.3f} {o['smape']:<10.2f} {o['mae']:<10.3f} {o['crps']:<10.3f}")
else:
    print("Benchmark skipped.")
""")

C_RESULTS = cell("""\
# ── Plot training curves ──
if history:
    import matplotlib.pyplot as plt
    epochs_ = [m["epoch"] for m in history]
    train_losses_ = [m.get("loss_total", 0) for m in history]
    val_losses_ = [m.get("val_loss_total", 0) for m in history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_, train_losses_, label="Train loss")
    plt.plot(epochs_, val_losses_, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("NanoForecast v0.3 Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{DRIVE_ROOT}/training_curve.png", dpi=150)
    plt.show()
    print(f"Training curve saved to {DRIVE_ROOT}/training_curve.png")

# ── Benchmark comparison vs v0.2 (if available) ──
v02_path = f"{DRIVE_ROOT}/benchmark-v02.json"
if os.path.isfile(v02_path):
    with open(v02_path) as f:
        v02 = json.load(f)
    v03_path = f"{DRIVE_ROOT}/benchmark-v03.json"
    if os.path.isfile(v03_path):
        with open(v03_path) as f:
            v03 = json.load(f)
        print(f"\\n{'Dataset':<20} {'v0.2 MASE':<12} {'v0.3 MASE':<12} {'Δ':<12}")
        print(f"{'-'*56}")
        for dset in v02.get("datasets", {}):
            v02m = v02["datasets"][dset].get("mase", float("nan"))
            v03e = v03["datasets"].get(dset, {})
            v03m = v03e.get("mase", float("nan")) if "error" not in v03e else float("nan")
            if math.isnan(v02m) or math.isnan(v03m):
                print(f"{dset:<20} {v02m:<12.3f} {v03m:<12.3f} {'N/A':<12}")
            else:
                delta = v03m - v02m
                sign = "↓" if delta < 0 else "↑" if delta > 0 else "="
                print(f"{dset:<20} {v02m:<12.3f} {v03m:<12.3f} {sign} {abs(delta):.3f}")
        print(f"{'-'*56}")
""")

C_PUSH = cell("""\
# ── Push to Hugging Face Hub ──
# Set HF_TOKEN in Colab secrets (🔑 key icon in the left sidebar) before running.
# Or uncomment and paste your token below.

from huggingface_hub import HfApi, login
from google.colab import userdata

try:
    token = userdata.get("HF_TOKEN")
except Exception:
    token = None

if not token:
    print("No HF_TOKEN found in Colab secrets.")
    print("To push: set HF_TOKEN in the 🔑 Secrets panel, then re-run this cell.")
    token = None  # ← paste your token here as a fallback

if token and os.path.isfile(f"{FINAL_DIR}/config.json"):
    api = HfApi(token=token)
    REPO_ID = f"eulogik/nanoforecast-v03"  # change if needed
    api.create_repo(repo_id=REPO_ID, private=False, exist_ok=True)
    api.upload_folder(
        folder_path=FINAL_DIR,
        repo_id=REPO_ID,
        commit_message="Upload NanoForecast v0.3 checkpoint",
    )
    print(f"✅ Model pushed to https://huggingface.co/{REPO_ID}")

    # Also push benchmark results
    bm_path = f"{DRIVE_ROOT}/benchmark-v03.json"
    if os.path.isfile(bm_path):
        api.upload_file(
            path_or_fileobj=bm_path,
            path_in_repo="benchmark-v03.json",
            repo_id=REPO_ID,
        )
        print("  + benchmark results uploaded")
else:
    print("Skipping push (no token or no final model).")
""")

# ── Build notebook ──
NOTEBOOK = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
        "colab": {
            "name": "NanoForecast v0.3 — Colab T4 Training",
            "provenance": [],
            "gpuType": "T4",
            "toc_visible": True,
        },
        "accelerator": "GPU",
    },
    "cells": [
        M_SETUP,
        C_SETUP,
        M_CONFIG,
        C_CONFIG,
        C_RESUME,
        M_DATA,
        C_DATA,
        M_MODEL,
        C_MODEL,
        M_TRAIN,
        C_TRAIN,
        M_COMPLETE,
        C_CLEANUP,
        M_BENCH,
        C_BENCH,
        M_RESULTS,
        C_RESULTS,
        M_PUSH,
        C_PUSH,
    ],
}

out_path = os.path.join(os.path.dirname(__file__), "colab_training_v03.ipynb")
with open(out_path, "w") as fh:
    json.dump(NOTEBOOK, fh, indent=1, ensure_ascii=False)
print(f"Generated notebook: {out_path}")
print(f"Cells: {len(NOTEBOOK['cells'])}")
