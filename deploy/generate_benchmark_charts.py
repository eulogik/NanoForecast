"""Generate benchmark charts for NanoForecast v0.5.

Every number in these charts is a standard-protocol measurement
(H=48, C=512, non-overlapping test windows, seasonal-naive MASE scale,
all series; see benchmark_standard.py) produced by us under the identical
protocol for every model. No published-leaderboard numbers are mixed in.

Usage:
    python3 deploy/generate_benchmark_charts.py
"""
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

NF_COLOR = "#FF6B35"    # orange — NanoForecast
V03_COLOR = "#95A5A6"   # gray — v0.3
TFM_COLOR = "#3498DB"   # blue — TimesFM
PTST_COLOR = "#9B59B6"  # purple — PatchTST
BG_COLOR = "#FAFAFA"

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "Exchange", "Electricity", "Traffic"]

# Standard-protocol MASE (benchmark_standard.py, fixed harness).
MASE_V05 = [0.685, 1.109, 0.289, 4.418, 2.093, 1.915]
MASE_V03 = [0.676, 1.357, 0.291, 12.847, 2.418, 2.102]
MASE_TFM = [0.705, 1.360, 0.545, 4.383, 0.923, 0.765]
MASE_PTST = [0.781, 1.467, 0.488, 3.861, 1.347, 1.379]

PARAMS = {"NanoForecast v0.5": 6.5, "PatchTST": 15, "TimesFM": 200}
EFFICIENCY = {"NanoForecast v0.5": 0.088, "PatchTST": 0.043, "TimesFM": 0.0035}


def _style_ax(ax, title, ylabel):
    fig = ax.figure
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def chart_main_mase():
    """MASE by dataset, the three compared systems (paper Table 1)."""
    x = np.arange(len(DATASETS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(13, 6))
    _style_ax(ax, "Standard-Protocol MASE by Dataset (lower is better)", "MASE")
    ax.bar(x - width, MASE_TFM, width, label="TimesFM (200M)", color=TFM_COLOR, edgecolor="white")
    ax.bar(x, MASE_PTST, width, label="PatchTST (15M+)", color=PTST_COLOR, edgecolor="white")
    ax.bar(x + width, MASE_V05, width, label="NanoForecast v0.5 (6.5M)", color=NF_COLOR, edgecolor="white")
    for xi, (a, b, c) in enumerate(zip(MASE_TFM, MASE_PTST, MASE_V05)):
        for xv, v in ((xi - width, a), (xi, b), (xi + width, c)):
            ax.text(xv, v + 0.12, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontweight="bold")
    ax.set_ylim(0, 5.0)
    ax.legend(fontsize=10, loc="upper left")
    ax.annotate("NanoForecast wins all three ETT sets",
                xy=(1, 1.35), xytext=(0.4, 2.0),
                arrowprops=dict(arrowstyle="->", color=NF_COLOR, lw=1.5),
                fontsize=9, color=NF_COLOR, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "benchmark_mase_standard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


def chart_v03_vs_v05():
    """Pipeline-refinement ablation: released v0.3 vs v0.5 checkpoints."""
    x = np.arange(len(DATASETS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(13, 6))
    _style_ax(ax, "Training-Pipeline Refinement: v0.3 vs v0.5 (same architecture)",
              "MASE (lower is better)")
    ax.bar(x - width / 2, MASE_V03, width, label="v0.3 (original pipeline)",
           color=V03_COLOR, edgecolor="white")
    ax.bar(x + width / 2, MASE_V05, width, label="v0.5 (three pipeline fixes)",
           color=NF_COLOR, edgecolor="white")
    for xi, (a, b) in enumerate(zip(MASE_V03, MASE_V05)):
        ax.text(xi - width / 2, a + 0.12, f"{a:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(xi + width / 2, b + 0.12, f"{b:.2f}", ha="center", fontsize=8, fontweight="bold")
        if b < a:
            ax.annotate("", xy=(xi + width / 2, b), xytext=(xi - width / 2, a),
                        arrowprops=dict(arrowstyle="-|>", color="#2ECC71", lw=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontweight="bold")
    ax.set_ylim(0, 14.0)
    ax.legend(fontsize=10, loc="upper left")
    ax.text(0.02, 0.92, "Overall MASE 3.282 → 1.752  (−46.6%)",
            transform=ax.transAxes, fontsize=11, fontweight="bold", color="#2ECC71")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "benchmark_v03_vs_v05.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


def chart_params():
    """Parameter count (log scale)."""
    names = list(PARAMS.keys())
    vals = list(PARAMS.values())
    colors = [NF_COLOR, PTST_COLOR, TFM_COLOR]
    fig, ax = plt.subplots(figsize=(10, 5))
    _style_ax(ax, "Parameter Count (log scale)", "Parameters (millions)")
    bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.55, log=True)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.15, f"{v}M",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(1, 1000)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "benchmark_params.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


def chart_efficiency():
    """Efficiency ratio E = MASE^-1 / params (higher is better)."""
    names = list(EFFICIENCY.keys())
    vals = list(EFFICIENCY.values())
    colors = [NF_COLOR, PTST_COLOR, TFM_COLOR]
    fig, ax = plt.subplots(figsize=(10, 5))
    _style_ax(ax, "Parameter Efficiency  E = MASE$^{-1}$ / params (higher is better)",
              "Efficiency ratio")
    bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.10)
    ax.text(0.02, 0.85, "25× more parameter-efficient than TimesFM, 2× vs PatchTST",
            transform=ax.transAxes, fontsize=10, fontweight="bold", color="#2ECC71")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "benchmark_efficiency.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    chart_main_mase()
    chart_v03_vs_v05()
    chart_params()
    chart_efficiency()
    print("\nAll charts generated.")