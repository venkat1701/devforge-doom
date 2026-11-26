"""Evaluate a checkpoint through the exact submission code path: submission/agent.py DoomAgent, the
OFFICIAL doom_env wrapper (never the fast training env), unmodified engine settings.

Review fixes built in:
  * suites are disjoint (maps.py); the 'test' suite (MAP25-32) needs --final, runs once, and a second
    look needs --allow-test-reuse; every use is logged with the checkpoint hash in
    training/docs/TEST_SUITE_USAGE.log (committed, so reuse is visible to reviewers)
  * same action-selection rule as the submission for every algorithm (whatever DoomAgent does)
  * the agent samples with its own torch.Generator, reseeded at every reset(), so scores do not depend
    on how episodes are split across worker processes
  * both tick caps (2,100 and 4,200) by default: hidden maps go up to 4,200 ticks (1,050 decisions)
  * many episodes per map (--seeds), with 95% bootstrap confidence intervals on suite means
  * agent compute time per step is measured against the 40 ms budget (docs/EVALUATION.md)

Metrics mirror the official categories (docs/EVALUATION.md "Scoring"): exit, progression (keys, doors,
unlocks), exploration (unique 64-unit cells), combat & survival (kills, death).

    uv run python -m training.doomrl.evaluate --weights <dir> --suite dev --seeds 5
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]

from doomrl.maps import SUITES  # noqa: E402

CELL, PLAYER_HEIGHT = 64.0, 56.0
METRICS = ("exit", "keys", "doors", "unlocks", "cells", "kills", "dead", "steps", "ms_mean", "ms_max")


def run_episode(agent, wad, map_name, seed, ticks):
    from doom_env import make_env
    env = make_env(wad, map_name, episode_ticks=ticks)   # official default: rgb/depth rendered (pixel models)
    obs, _ = env.reset(seed=seed)
    agent.reset()
    p = env.privileged_state()
    lock = p["door_lock"]
    opened = set(np.flatnonzero((lock >= 0) & (p["sector_ceiling"] - p["sector_floor"] >= PLAYER_HEIGHT)).tolist())
    cells = {(int(p["x"] // CELL), int(p["y"] // CELL))}
    k0 = p["kills"]
    doors = unlocks = 0
    times = []
    done = False
    steps = 0
    info = {"exit_reached": False}
    while not done:
        t0 = time.perf_counter()
        a = agent.act(obs)
        times.append(time.perf_counter() - t0)
        obs, _, term, trunc, info = env.step(a)
        steps += 1
        p = env.privileged_state()
        cells.add((int(p["x"] // CELL), int(p["y"] // CELL)))
        gap = p["sector_ceiling"] - p["sector_floor"]
        for sid in np.flatnonzero((lock >= 0) & (gap >= PLAYER_HEIGHT)):
            if sid not in opened:
                opened.add(sid)
                doors += int(lock[sid] == 0); unlocks += int(lock[sid] > 0)
        done = term or trunc
    env.close()
    ms = np.array(times) * 1000
    return {"exit": float(info["exit_reached"]), "keys": float(np.sum(p["keys"])), "doors": doors,
            "unlocks": unlocks, "cells": len(cells), "kills": p["kills"] - k0, "dead": float(p["dead"]),
            "steps": steps, "ms_mean": float(ms.mean()), "ms_max": float(ms.max())}


def _worker(args):
    weights, jobs = args
    import agent as agent_mod                   # the submission module, unmodified code path
    from doom_env import observation_space, action_space
    agent_mod.WEIGHTS_DIR = Path(weights)       # constructor loads (and fails closed on) this checkpoint
    agent = agent_mod.DoomAgent(observation_space(), action_space(), "cpu")
    return [dict(run_episode(agent, w, m, s, tk), map=f"{Path(w).stem}:{m}", seed=s, ticks=tk) for w, m, s, tk in jobs]


def bootstrap_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = x[rng.integers(len(x), size=(n, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


TEST_LOG = ROOT / "training" / "docs" / "TEST_SUITE_USAGE.log"     # tracked in git, so reuse is visible


def guard_test_suite(weights, final: bool, allow_reuse: bool) -> None:
    """The test suite (MAP25-32) is for ONE final measurement. Review fix: --final alone still allowed
    repeated looks. Now a second use refuses unless --allow-test-reuse, and every use is logged
    (timestamp, checkpoint sha256) in a file that is committed with the repo."""
    import hashlib
    if not final:
        raise SystemExit("refusing to evaluate on the held-out test suite without --final")
    prior = TEST_LOG.read_text().strip().splitlines() if TEST_LOG.is_file() else []
    if prior and not allow_reuse:
        raise SystemExit(f"test suite already used {len(prior)} time(s) (see {TEST_LOG}); "
                         "pass --allow-test-reuse to look again (it will be logged)")
    sha = hashlib.sha256((Path(weights) / "model.pt").read_bytes()).hexdigest()[:16]
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_LOG, "a") as fh:
        fh.write(f"{datetime.datetime.now().isoformat()} model_sha256={sha} weights={weights}"
                 f"{' REUSE' if prior else ''}\n")


def evaluate(weights, suite, seeds, ticks, workers, final=False, allow_reuse=False):
    if suite == "test":
        guard_test_suite(weights, final, allow_reuse)
    tick_list = ticks if isinstance(ticks, (list, tuple)) else [ticks]
    jobs = [(w, m, 10_000 + s, tk) for w, m in SUITES[suite] for s in range(seeds) for tk in tick_list]
    chunks = [jobs[i::workers] for i in range(workers)]
    with mp.get_context("forkserver").Pool(workers) as pool:
        eps = [e for part in pool.map(_worker, [(str(weights), c) for c in chunks if c]) for e in part]
    summary = {}
    for k in METRICS:
        vals = [e[k] for e in eps]
        lo, hi = bootstrap_ci(vals)
        summary[k] = {"mean": float(np.mean(vals)), "ci95": [lo, hi]}
    per_map = {}
    for e in eps:
        per_map.setdefault(e["map"], []).append(e)
    per_map = {m: {k: float(np.mean([e[k] for e in v])) for k in METRICS} for m, v in per_map.items()}
    return {"weights": str(weights), "suite": suite, "episodes": len(eps), "seeds_per_map": seeds,
            "summary": summary, "per_map": per_map, "episodes_raw": eps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--suite", choices=("dev", "test", "train"), default="dev")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ticks", type=int, nargs="+", default=[2100, 4200],
                    help="tick caps to evaluate (hidden maps use 2,100-4,200, docs/CONTRACT.md §5)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--allow-test-reuse", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = evaluate(a.weights, a.suite, a.seeds, a.ticks, a.workers, a.final, a.allow_test_reuse)
    out = Path(a.out or Path(a.weights) / f"eval_{a.suite}.json")
    out.write_text(json.dumps(res, indent=1))
    s = res["summary"]
    print(f"{a.suite}: {res['episodes']} episodes | " + " | ".join(
        f"{k} {s[k]['mean']:.2f} [{s[k]['ci95'][0]:.2f},{s[k]['ci95'][1]:.2f}]" for k in METRICS))


if __name__ == "__main__":
    main()
