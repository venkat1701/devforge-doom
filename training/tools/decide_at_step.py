"""Wait until every listed run has a checkpoint at or past --step, evaluate each one on the dev suite
through the real DoomAgent (doomrl.evaluate), and write a comparison with 95% bootstrap CIs.

    PYTHONPATH=training:submission uv run --group train python training/tools/decide_at_step.py \
        --runs training/runs/ppo_map_s0 training/runs/ppo_map_s1 training/runs/drqn_map_s0 training/runs/drqn_map_s1 \
        --step 2000000 --out training/runs/decision_2M

Arms are grouped by run-name prefix (text before _s<seed>). The decision rule is printed, not applied:
an arm "wins" a metric only if its CI does not overlap the other arm's CI.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]

from doomrl.evaluate import METRICS, bootstrap_ci, evaluate  # noqa: E402


def checkpoint_at(run: Path, step: int):
    cks = sorted((int(m.group(1)), p) for p in run.glob("model_*.pt") if (m := re.match(r"model_(\d+)\.pt", p.name)))
    for s, p in cks:
        if s >= step:
            return s, p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--poll", type=int, default=120)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = [Path(r) for r in a.runs]
    found = {}
    while len(found) < len(runs):
        for r in runs:
            if r not in found and (ck := checkpoint_at(r, a.step)):
                found[r] = ck
                print(f"{time.strftime('%H:%M:%S')} {r.name}: checkpoint at {ck[0]:,}", flush=True)
        if len(found) < len(runs):
            time.sleep(a.poll)
    results = {}
    for r, (s, p) in found.items():
        d = out / f"{r.name}_{s}"
        d.mkdir(exist_ok=True)
        shutil.copy2(p, d / "model.pt"); shutil.copy2(r / "model_config.json", d / "model_config.json")
        res = evaluate(d, "dev", a.seeds, [2100, 4200], a.workers)
        (d / "eval_dev.json").write_text(json.dumps(res, indent=1))
        results[r.name] = res
        print(f"{time.strftime('%H:%M:%S')} evaluated {r.name} @ {s:,}: cells {res['summary']['cells']['mean']:.1f} "
              f"exit {res['summary']['exit']['mean']:.3f}", flush=True)
    arms = {}
    for name, res in results.items():
        arms.setdefault(re.sub(r"_s\d+$", "", name), []).extend(res["episodes_raw"])
    lines = [f"Decision at >= {a.step:,} steps, dev suite (MAP17-24), {a.seeds} seeds x 2 tick caps per run"]
    table = {}
    for arm, eps in arms.items():
        table[arm] = {k: (float(np.mean([e[k] for e in eps])), bootstrap_ci([e[k] for e in eps])) for k in METRICS}
        lines.append(f"{arm:12s} n={len(eps):3d} " + " | ".join(
            f"{k} {v[0]:.2f} [{v[1][0]:.2f},{v[1][1]:.2f}]" for k, v in table[arm].items() if k in
            ("exit", "keys", "doors", "unlocks", "cells", "kills", "dead")))
    if len(table) == 2:
        (a1, t1), (a2, t2) = table.items()
        for k in ("exit", "keys", "doors", "unlocks", "cells", "kills", "dead"):
            lo1, hi1 = t1[k][1]; lo2, hi2 = t2[k][1]
            better_high = k != "dead"
            if hi1 < lo2 or hi2 < lo1:
                winner = (a1 if t1[k][0] > t2[k][0] else a2) if better_high else (a1 if t1[k][0] < t2[k][0] else a2)
                lines.append(f"  {k:8s}: {winner} better (CIs separate)")
            else:
                lines.append(f"  {k:8s}: no separation (CIs overlap)")
    text = "\n".join(lines)
    (out / "DECISION.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
