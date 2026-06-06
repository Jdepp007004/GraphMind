"""
Generate GraphMindRL vs Markov-2 scatter plot (one point per user).
X axis = Markov-2 hit rate, Y axis = GraphMindRL hit rate.
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_CSV    = os.path.join(PROJECT_ROOT, "results", "benchmark_results_fast.csv")
OUT_DIR      = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

rows = list(csv.DictReader(open(BENCH_CSV, encoding="utf-8")))

# Build per-user dicts
m2_hr  = {}
rl_hr  = {}
m2_f1  = {}
rl_f1  = {}

for r in rows:
    uid = r["user_id"]
    pol = r["policy"]
    if pol == "Markov-2":
        m2_hr[uid] = float(r["hit_rate"])
        m2_f1[uid] = float(r["f1"])
    elif pol == "GraphMindRL":
        rl_hr[uid] = float(r["hit_rate"])
        rl_f1[uid] = float(r["f1"])

common = sorted(set(m2_hr) & set(rl_hr))
xs_hr = [m2_hr[u] for u in common]
ys_hr = [rl_hr[u] for u in common]
xs_f1 = [m2_f1[u] for u in common]
ys_f1 = [rl_f1[u] for u in common]

# ── Scatter: Hit Rate ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0f1117")

for ax in axes:
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="#c0c0d0", labelsize=10)
    ax.spines["bottom"].set_color("#3a3d55")
    ax.spines["left"].set_color("#3a3d55")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Hit Rate scatter
lims_hr = [min(min(xs_hr), min(ys_hr)) - 0.005, max(max(xs_hr), max(ys_hr)) + 0.005]
axes[0].plot(lims_hr, lims_hr, "--", color="#555577", alpha=0.6, linewidth=1, label="y = x (parity)")
sc = axes[0].scatter(
    xs_hr, ys_hr,
    c=np.array(ys_hr) - np.array(xs_hr),
    cmap="RdYlGn", s=80, edgecolors="#ffffff", linewidths=0.5, alpha=0.92,
    vmin=-0.04, vmax=0.04,
)
# Annotate outliers
for uid, xv, yv in zip(common, xs_hr, ys_hr):
    if abs(yv - xv) > 0.02:
        axes[0].annotate(uid, (xv, yv), fontsize=7, color="#c0c0d0",
                         xytext=(4, 4), textcoords="offset points")
cb = plt.colorbar(sc, ax=axes[0])
cb.ax.yaxis.set_tick_params(color="#c0c0d0")
cb.set_label("RL − Markov-2", color="#c0c0d0", fontsize=10)
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#c0c0d0")
axes[0].set_xlabel("Markov-2  Hit Rate", color="#c0c0d0", fontsize=11)
axes[0].set_ylabel("GraphMindRL  Hit Rate", color="#c0c0d0", fontsize=11)
axes[0].set_title("Hit Rate: GraphMindRL vs Markov-2\n(one point per user)",
                  color="#e0e0f0", fontsize=12, pad=12)
axes[0].set_xlim(lims_hr); axes[0].set_ylim(lims_hr)
above = sum(1 for x, y in zip(xs_hr, ys_hr) if y >= x)
axes[0].legend([plt.Line2D([0],[0], linestyle="--", color="#555577")],
               [f"Parity  ({above}/{len(common)} RL ≥ Markov-2)"],
               facecolor="#1a1d2e", labelcolor="#c0c0d0", fontsize=9)

# F1 scatter
lims_f1 = [min(min(xs_f1), min(ys_f1)) - 0.01, max(max(xs_f1), max(ys_f1)) + 0.01]
axes[1].plot(lims_f1, lims_f1, "--", color="#555577", alpha=0.6, linewidth=1)
sc2 = axes[1].scatter(
    xs_f1, ys_f1,
    c=np.array(ys_f1) - np.array(xs_f1),
    cmap="RdYlGn", s=80, edgecolors="#ffffff", linewidths=0.5, alpha=0.92,
    vmin=-0.06, vmax=0.06,
)
for uid, xv, yv in zip(common, xs_f1, ys_f1):
    if abs(yv - xv) > 0.03:
        axes[1].annotate(uid, (xv, yv), fontsize=7, color="#c0c0d0",
                         xytext=(4, 4), textcoords="offset points")
cb2 = plt.colorbar(sc2, ax=axes[1])
cb2.ax.yaxis.set_tick_params(color="#c0c0d0")
cb2.set_label("RL − Markov-2", color="#c0c0d0", fontsize=10)
plt.setp(plt.getp(cb2.ax.axes, "yticklabels"), color="#c0c0d0")
axes[1].set_xlabel("Markov-2  F1", color="#c0c0d0", fontsize=11)
axes[1].set_ylabel("GraphMindRL  F1", color="#c0c0d0", fontsize=11)
axes[1].set_title("F1 Score: GraphMindRL vs Markov-2\n(one point per user)",
                  color="#e0e0f0", fontsize=12, pad=12)
axes[1].set_xlim(lims_f1); axes[1].set_ylim(lims_f1)
above_f1 = sum(1 for x, y in zip(xs_f1, ys_f1) if y >= x)
axes[1].legend([plt.Line2D([0],[0], linestyle="--", color="#555577")],
               [f"Parity  ({above_f1}/{len(common)} RL ≥ Markov-2)"],
               facecolor="#1a1d2e", labelcolor="#c0c0d0", fontsize=9)

plt.tight_layout(pad=2.5)
out_path = os.path.join(OUT_DIR, "graphmind_vs_markov2.png")
plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out_path}")
print(f"Users: {len(common)}, RL >= Markov-2 hit rate: {above}/{len(common)}, F1: {above_f1}/{len(common)}")
