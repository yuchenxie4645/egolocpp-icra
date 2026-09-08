#!/usr/bin/env python3
"""Plot loss / mean-token-accuracy / grad-norm curves for the v3 training runs."""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "v3-plots")
os.makedirs(OUT, exist_ok=True)

NAMES = {
    "contact_sft_grpo_run": ("SFT+GRPO  contact", "#2ca02c", "-"),
    "separation_sft_grpo_run": ("SFT+GRPO  separation", "#2ca02c", "--"),
}
ORDER = list(NAMES)

runs = {}
for path in glob.glob(os.path.join(HERE, "v3-*", "*_run", "checkpoint-*", "trainer_state.json")):
    run = path.split(os.sep)[-3]
    hist = json.load(open(path)).get("log_history", [])
    pts = [h for h in hist if "loss" in h]
    if pts:
        runs[run] = pts


def smooth(values, window=15):
    out, acc = [], []
    for v in values:
        acc.append(v)
        if len(acc) > window:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def series(pts, key):
    xs, ys = [], []
    for h in pts:
        v = h.get(key)
        if v is not None:
            xs.append(h["epoch"])
            ys.append(v)
    return xs, ys


def make_plot(key, title, ylabel, filename, subtitle, logy=False, only=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for run in ORDER:
        if run not in runs:
            continue
        if only and run not in only:
            continue
        label, color, style = NAMES[run]
        xs, ys = series(runs[run], key)
        if not xs:
            continue
        ax.plot(xs, ys, color=color, ls=style, lw=0.6, alpha=0.22)
        ax.plot(xs, smooth(ys), color=color, ls=style, lw=1.9, label=label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return None
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color="#555")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9, framealpha=0.9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = os.path.join(OUT, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)
    return out


make_plot(
    "loss", "SFT+GRPO - training loss", "DAPO policy loss", "1_loss.png",
    "v3 GRPO run initialised from the SFT adapter. Bold = 15-step moving average, faint = raw.",
)
# GRPO has no teacher forcing, so mean_token_accuracy is never logged. The
# equivalent accuracy signal is the fraction of sampled completions that pick
# the exactly-correct grid cell.
make_plot(
    "rewards/exact_cell_metric/mean", "SFT+GRPO - exact cell accuracy",
    "Exact cell accuracy (fraction of rollouts)", "2_exact_cell_accuracy.png",
    "GRPO logs no mean_token_accuracy (no teacher forcing); this is the equivalent "
    "accuracy signal. Bold = 15-step moving average, faint = raw.",
)
make_plot(
    "grad_norm", "SFT+GRPO - gradient norm", "Gradient norm (log scale)", "3_grad_norm.png",
    "v3 GRPO run initialised from the SFT adapter. Bold = 15-step moving average, faint = raw. "
    "Clipped at max_grad_norm=0.3.",
    logy=True,
)
