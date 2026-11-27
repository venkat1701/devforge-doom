"""Figures for the ICLR-style report. Vector PDFs sized to the 5.5in ICLR text block.

usage (from docs/report): python make_figs.py data figs
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

S = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.6,
    "axes.edgecolor": "#8a8a8a", "axes.linewidth": 0.6, "xtick.color": "#555555", "ytick.color": "#555555",
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "axes.titlelocation": "left", "axes.titleweight": "bold",
})
TEXT2 = "#555555"
# Validated categorical palette (light): fixed order, colour follows the entity in every figure.
C = {"official": "#257de0", "shaped": "#f47230", "gen": "#00b894", "v4": "#f2a600", "v7": "#6a4ce2", "impala": "#008a17"}
M = {"official": "s", "shaped": "D", "gen": "o", "v4": "^", "v7": "*", "impala": "P"}
LABEL = {"official": "Official reward", "shaped": "Category shaping (Freedoom)", "gen": "Generated maps (v1-v3)",
         "v4": "+ Route potential (v4)", "v7": "+ Trigger discipline (v7)", "impala": "IMPALA encoder + v7"}

ev = {}
for f in ("agg_old.json", "agg_new.json"):
    for e in json.load(open(S / f))["evals"]:
        ev[e["file"]] = e
runs = {**json.load(open(S / "agg_old.json"))["runs"], **json.load(open(S / "agg_new.json"))["runs"]}

def stage(name):
    if name.startswith("ppo_learned"): return "official"
    if name.startswith("ppo_progress"): return "shaped"
    if name.startswith(("gen_", "genv2_", "genv3T0_", "v3x256T0_")): return "gen"
    if name.startswith("v4routeT0_"): return "v4"
    if name.startswith("v7T0mask_"): return "v7"
    return None

pts = defaultdict(list)
for k, e in ev.items():
    if not k.startswith("evals/") or e["suite"] != "gen_dev" or e["step"] is None:
        continue
    s = stage(e["name"])
    if s:
        pts[s].append((e["step"] / 1e6, e["mean"]))
for s in pts:
    pts[s].sort()

# ---------------- Figure 3: held-out progress across the lineage ----------------
fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.5), constrained_layout=True)
panels = [("exit", "Exit rate", "(a) Maps finished", (0, 1.0)),
          ("locked_opened", "Per episode", "(b) Keycard doors opened", (0, 1.0)),
          ("blind_fire", "Share of presses", "(c) Firing at nothing", (0, 0.8))]
order = ["official", "shaped", "gen", "v4", "v7"]
for ax, (key, ylab, title, ylim) in zip(axes, panels):
    for s in order:
        xy = [(x, m.get(key)) for x, m in pts.get(s, []) if m.get(key) is not None]
        if not xy:
            continue
        xs, ys = zip(*xy)
        ax.plot(xs, ys, color=C[s], lw=1.1, marker=M[s], ms=4.2 if s != "v7" else 6, mec="white", mew=0.5,
                label=LABEL[s], zorder=3)
    ax.set_title(title, fontsize=7.5)
    ax.set_ylabel(ylab); ax.set_xlabel("Environment steps (M)")
    ax.set_ylim(*ylim); ax.set_xlim(0, 80)
# submitted checkpoint annotation (800-episode confirmation)
fin = ev["final16/v7_67174400_T025.json"]["mean"]
for ax, key in zip(axes, ("exit", "locked_opened", "blind_fire")):
    ax.scatter([67.2], [fin[key]], s=46, facecolor="none", edgecolor="#191919", lw=0.9, zorder=5)
axes[0].annotate("submitted\n(800 ep.)", xy=(67.2, fin["exit"]), xytext=(44, 0.32), fontsize=6.3, color="#191919",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#191919"))
axes[2].annotate("v7 trigger\npenalties", xy=(55.7, 0.37), xytext=(33, 0.12), fontsize=6.2, color=TEXT2,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=TEXT2))
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, handlelength=1.6, columnspacing=1.2)
fig.savefig(OUT / "fig_lineage.pdf"); fig.savefig(OUT / "fig_lineage.png", dpi=260)
plt.close(fig)

# ---------------- Figure 4: tiers, firing during training, throughput ----------------
fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.42), constrained_layout=True, width_ratios=[1.25, 1.1, 0.85])
# (a) per-tier exit
tiers = ["1", "2", "3", "4"]; names = ["Combat", "Doors", "Keycard", "Gauntlet"]
test_tiers = {"1": 0.944, "2": 0.917, "3": 0.907, "4": 0.556}  # reserved generated test maps, computed from episodes_raw
series = [("v4", ev["final/v4_51609600_T0.json"]["exit_by_tier"], "v4", None),
          ("v7", ev["final16/v7_67174400_T025.json"]["exit_by_tier"], "v7 (submitted)", None),
          ("v7", test_tiers, "v7, reserved test", "////")]
w = 0.26
ax = axes[0]
for i, (s, d, lab, hatch) in enumerate(series):
    xs = [j + (i - 1) * (w + 0.02) for j in range(4)]
    ys = [d[t] for t in tiers]
    ax.bar(xs, ys, width=w, color=C[s] if not hatch else "white", edgecolor=C[s], lw=0.8 if hatch else 0,
           hatch=hatch, label=lab, zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels(names)
ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("Exit rate"); ax.set_title("(a) Exit rate by map tier")
ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=5.8, handlelength=1.0, columnspacing=0.7, handletextpad=0.3, borderaxespad=0.1)
ax.grid(axis="x", visible=False)
# (b) training-time rounds wasted
ax = axes[1]
for key, s, lab in (("v4", "v4", "v4"), ("v7", "v7", "v7"), ("impala_v7", "impala", "IMPALA + v7")):
    se = [p for p in runs[key]["series"] if p["kind"] == "gen" and p.get("blind_frac") is not None]
    xs = [p["step"] / 1e6 for p in se]; ys = [p["blind_frac"] for p in se]
    ax.plot(xs, ys, color=C[s], lw=1.2, label=lab, zorder=3)
    if s == "impala":
        ax.text(37.5, 0.24, lab, color="#191919", fontsize=6.2, va="center")
    else:
        ax.text(xs[-1] + 0.8, ys[-1], lab, color="#191919", fontsize=6.2, va="center")
ax.set_ylim(0, 0.8); ax.set_xlim(36, 86)
ax.set_xlabel("Environment steps (M)"); ax.set_ylabel("Share of rounds")
ax.set_title("(b) Wasted rounds in training")
# (c) throughput
ax = axes[2]
tp = [("32\nofficial", runs["native"]["sps"]["median"]), ("32\nshaped", runs["freedoom_shaped"]["sps"]["median"]),
      ("64", runs["gen"]["sps"]["median"]), ("256", runs["v3_256"]["sps"]["median"])]
ax.bar(range(4), [v / 1000 for _, v in tp], width=0.62, color="#6b6b6b", zorder=3)
for i, (_, v) in enumerate(tp):
    ax.text(i, v / 1000 + 0.12, f"{v/1000:.1f}k", ha="center", fontsize=6.2, color="#191919")
ax.set_xticks(range(4)); ax.set_xticklabels([n for n, _ in tp], fontsize=6.2)
ax.set_xlabel("Parallel simulators"); ax.set_ylabel("Thousand / s"); ax.set_ylim(0, 7.2)
ax.set_title("(c) Training throughput"); ax.grid(axis="x", visible=False)
fig.savefig(OUT / "fig_results.pdf"); fig.savefig(OUT / "fig_results.png", dpi=260)
plt.close(fig)
print("ok", sorted(p.name for p in OUT.iterdir()))
