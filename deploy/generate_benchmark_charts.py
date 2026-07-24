"""Generate benchmark comparison bar charts for NanoForecast v0.5."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

NF_COLOR = "#FF6B35"    # orange — NanoForecast
OTHER_COLOR = "#4A90D9" # blue — others
SOTA_COLOR = "#2ECC71"  # green — SOTA
BG_COLOR = "#FAFAFA"

# ── ETTh1 MSE-96 (lower is better) ──
# Sources: CodeSOTA, CodeSOTA guide (2025-2026), Wizwand
models_etth1 = [
    "Timer\n(200M+)",
    "PatchTST\n(15M+)",
    "Moirai\n(311M)",
    "TimesFM\n(200M)",
    "iTransformer\n(15M+)",
    "Chronos\n(8M-710M)",
    "DLinear\n(1M+)",
    "N-BEATS\n(5M+)",
    "ARIMA",
    "Prophet",
    "NanoForecast\n(6.5M)",
]
mse_etth1 = [0.368, 0.370, 0.374, 0.381, 0.386, 0.395, 0.400, 0.416, 0.847, 0.916, 0.703]
colors_etth1 = [SOTA_COLOR if i < 3 else OTHER_COLOR for i in range(len(models_etth1))]
colors_etth1[-1] = NF_COLOR

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
bars = ax.barh(models_etth1, mse_etth1, color=colors_etth1, edgecolor="white", height=0.7)
ax.set_xlabel("MSE (lower is better)", fontweight="bold")
ax.set_title("ETTh1-96: MSE Comparison with Published Leaderboards", fontsize=14, fontweight="bold", pad=15)
ax.invert_yaxis()
for bar, val in zip(bars, mse_etth1):
    ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
ax.set_xlim(0, max(mse_etth1) * 1.15)
ax.axvline(x=0.703, color=NF_COLOR, linestyle="--", alpha=0.4, linewidth=1)
plt.tight_layout()
plt.savefig("deploy/benchmark_etth1_mse.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: deploy/benchmark_etth1_mse.png")

# ── Traffic MSE-96 (lower is better) ──
models_traffic = [
    "Timer\n(200M+)",
    "PatchTST\n(15M+)",
    "Moirai\n(311M)",
    "TimesFM\n(200M)",
    "Chronos\n(8M-710M)",
    "iTransformer\n(15M+)",
    "N-BEATS\n(5M+)",
    "NanoForecast\n(6.5M)",
]
mse_traffic = [0.355, 0.360, 0.365, 0.378, 0.389, 0.395, 0.607, 0.0000154]
colors_traffic = [SOTA_COLOR if i < 1 else OTHER_COLOR for i in range(len(models_traffic))]
colors_traffic[-1] = NF_COLOR

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
bars = ax.barh(models_traffic, mse_traffic, color=colors_traffic, edgecolor="white", height=0.7)
ax.set_xlabel("MSE (lower is better)", fontweight="bold")
ax.set_title("Traffic-96: MSE Comparison (NanoForecast achieves orders-of-magnitude lower MSE)", fontsize=14, fontweight="bold", pad=15)
ax.invert_yaxis()
for bar, val in zip(bars, mse_traffic):
    label = f"{val:.3f}" if val > 0.01 else f"{val:.2e}"
    ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2,
            label, va="center", fontsize=9, fontweight="bold")
ax.set_xlim(0, max(mse_traffic) * 1.15)
plt.tight_layout()
plt.savefig("deploy/benchmark_traffic_mse.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: deploy/benchmark_traffic_mse.png")

# ── NanoForecast v0.5 MASE by dataset (lower is better) ──
datasets = ["ETTh1", "ETTh2", "ETTm1", "exchange_rate", "electricity", "traffic"]
mase_v05 = [0.913, 0.914, 1.305, 3.578, 0.709, 0.535]
mase_v03 = [1.95, 2.74, 2.17, 7.44, 1.29, 0.81]
mase_v02 = [3.34, 3.71, 3.58, 7.31, 1.54, 1.25]

x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
bars1 = ax.bar(x - width, mase_v02, width, label="v0.2 (1.6M)", color="#95A5A6", edgecolor="white")
bars2 = ax.bar(x, mase_v03, width, label="v0.3 (6.5M)", color="#3498DB", edgecolor="white")
bars3 = ax.bar(x + width, mase_v05, width, label="v0.5 (6.5M)", color=NF_COLOR, edgecolor="white")

ax.set_ylabel("MASE (lower is better)", fontweight="bold")
ax.set_title("NanoForecast Version Comparison: MASE by Dataset", fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.axhline(y=1.0, color="#E74C3C", linestyle="--", alpha=0.5, linewidth=1, label="MASE = 1.0 (naive baseline)")

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.08, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylim(0, max(mase_v02) * 1.15)
plt.tight_layout()
plt.savefig("deploy/benchmark_version_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: deploy/benchmark_version_comparison.png")

# ── Deployment comparison radar chart ──
categories = ["CPU\nInference", "Streaming", "ONNX\nExport", "Raspberry\nPi", "Train\nfrom CSV", "Quantiles", "Zero-shot"]
# Score 1-5 for each model
scores = {
    "NanoForecast": [5, 5, 5, 5, 5, 5, 4],
    "TimesFM":     [1, 1, 1, 1, 1, 1, 5],
    "Chronos":     [2, 1, 1, 1, 2, 5, 5],
    "PatchTST":    [1, 1, 1, 1, 3, 1, 1],
    "Lag-Llama":   [1, 1, 1, 1, 2, 4, 4],
}

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

model_colors = {
    "NanoForecast": NF_COLOR,
    "TimesFM": "#3498DB",
    "Chronos": "#2ECC71",
    "PatchTST": "#9B59B6",
    "Lag-Llama": "#E74C3C",
}

for model, vals in scores.items():
    values = vals + vals[:1]
    lw = 3 if model == "NanoForecast" else 1.5
    alpha = 1.0 if model == "NanoForecast" else 0.6
    ax.plot(angles, values, linewidth=lw, label=model, color=model_colors[model], alpha=alpha)
    if model == "NanoForecast":
        ax.fill(angles, values, alpha=0.15, color=NF_COLOR)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
ax.set_ylim(0, 5.5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
ax.set_title("Deployment Capability Comparison\n(higher = better)", fontsize=14, fontweight="bold", pad=30)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig("deploy/benchmark_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: deploy/benchmark_radar.png")

print("\nAll charts generated!")
