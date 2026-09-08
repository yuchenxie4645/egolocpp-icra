#!/usr/bin/env python3
"""Plot loss curves for the v3 SFT runs (contact + separation)."""

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
    "contact_run":    ("SFT contact",    "#1f77b4", "-"),
    "separation_run": ("SFT separation", "#ff7f0e", "--"),
}
ORDER = list(NAMES)

runs = {}
for path in glob.glob(os.path.join(HERE, "v3-sft", "*_run", "checkpoint-*", "trainer_state.json")):
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


fig, ax = plt.subplots(figsize=(9, 5))
plotted = 0
for run in ORDER:
    if run not in runs:
        continue
    label, color, style = NAMES[run]
    xs, ys = series(runs[run], "loss")
    if not xs:
        continue
    ax.plot(xs, ys, color=color, ls=style, lw=0.6, alpha=0.22)
    ax.plot(xs, smooth(ys), color=color, ls=style, lw=1.9, label=label)
    plotted += 1

if plotted == 0:
    plt.close(fig)
    raise SystemExit("no SFT runs found")

ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("v3 SFT - training loss", fontsize=13, fontweight="bold", pad=14)
ax.text(0.0, 1.015,
        "Bold = 15-step moving average, faint = raw. Cross-entropy on the SFT mix.",
        transform=ax.transAxes, fontsize=8.5, color="#555")
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=9, framealpha=0.9)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
out = os.path.join(OUT, "sft_loss.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("wrote", out)
