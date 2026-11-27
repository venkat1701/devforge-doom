"""Official-wrapper evaluation of immutable exports; no fabricated aggregate competition score."""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]
from doomrl.maps import SUITES
from doomrl.progress import FOV_HALF_DEG, EpisodeProgress, enemy_in_view, total_ammo
from doom_model.artifacts import verify_artifact, file_hash

METRICS = ("exit", "keys", "kills", "dead", "timeout", "health", "native_return", "steps", "ms_mean", "ms_max",
           "cells", "doors_opened", "locked_opened", "damage", "distance", "revisit_ratio", "proxy_score",
           "fire_rate", "blind_fire", "ammo_spent", "ended_dry", "on_target_fire")


def proxy_score(m):
    """Rough stand-in for the private per-map score (weights are hidden): exit largest, then progression,
    exploration and combat. For ranking our own checkpoints only; it is NOT the competition score."""
    exit_part = 10. * m["exit"] * (1. + .1 * max(0., m["health"]) / 100.)
    progression = 2. * m["keys"] + 1. * m["doors_opened"] + 2. * m["locked_opened"]
    exploration = .1 * m["cells"]
    combat = .5 * m["kills"] - .01 * m["damage"] - 1. * m["dead"]
    return exit_part + progression + exploration + combat
TEST_LOG = ROOT / "training" / "test_suite_usage.jsonl"


def run_episode(agent, wad, map_name, seed, ticks):
    from doom_env import make_env
    env = make_env(wad, map_name, episode_ticks=ticks)
    try:
        obs, _ = env.reset(seed=seed)
        agent.reset()
        kills0 = env.privileged_state()["kills"]
        progress = EpisodeProgress(env.privileged_state())
        timings, reward_sum, steps = [], 0., 0
        fires = blind = on_target = 0
        spent = 0.
        terminated = truncated = False
        while not (terminated or truncated):
            t0 = time.perf_counter()
            action = agent.act(obs)
            timings.append(time.perf_counter() - t0)
            before, seen = total_ammo(obs), enemy_in_view(obs, FOV_HALF_DEG)
            on_target += int(np.asarray(action)[6] > 0 and enemy_in_view(obs, 8.0))
            fires += int(np.asarray(action)[6] > 0)
            blind += int(np.asarray(action)[6] > 0 and not seen)
            obs, reward, terminated, truncated, _ = env.step(action)
            spent += max(0., before - total_ammo(obs))
            reward_sum += reward
            progress.update(env.privileged_state())
            steps += 1
        p = env.privileged_state()
        ms = np.asarray(timings) * 1000.
        s = progress.summary()
        m = {"exit": float(p["exit_reached"]), "keys": float(np.sum(p["keys"])),
             "kills": p["kills"] - kills0, "dead": float(p["dead"]), "timeout": float(p["timeout"]),
             "health": p["health"], "native_return": reward_sum, "steps": steps,
             "ms_mean": float(ms.mean()), "ms_max": float(ms.max()),
             "cells": float(s["cells"]), "doors_opened": float(s["doors_opened"]),
             "locked_opened": float(s["locked_opened"]), "damage": float(s["damage"]),
             "distance": float(s["distance"]), "revisit_ratio": float(s["revisit_ratio"]),
             "fire_rate": fires / max(1, steps), "blind_fire": blind / max(1, fires), "ammo_spent": spent,
             "on_target_fire": on_target / max(1, fires),
             "ended_dry": float(total_ammo(obs) <= 0)}
        m["proxy_score"] = proxy_score(m)
        return m
    finally:
        env.close()


def _worker(args):
    weights, jobs, expected_hash = args
    import agent as agent_mod
    from doom_env import observation_space, action_space
    if file_hash(Path(weights) / "manifest.json") != expected_hash:
        raise ValueError("evaluation artifact changed")
    agent_mod.WEIGHTS_DIR = Path(weights)
    agent = agent_mod.DoomAgent(observation_space(), action_space(), "cpu")
    return [dict(run_episode(agent, w, m, s, tk), map=f"{Path(w).name}:{m}", seed=s, ticks=tk)
            for w, m, s, tk in jobs]


def bootstrap_ci(values, n=2000, seed=0):
    """Input units must already be independent clusters (here: map means)."""
    x = np.asarray(values, float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("bootstrap requires nonempty finite values")
    if len(x) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(n, len(x)))].mean(1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def _summary(eps):
    maps = sorted({e["map"] for e in eps})
    result = {}
    for key in METRICS:
        means = [float(np.mean([e[key] for e in eps if e["map"] == m])) for m in maps]
        ci = bootstrap_ci(means)
        item = {"mean": float(np.mean(means)), "ci95": ci, "independent_maps": len(maps)}
        if key == "exit":
            successes = sum(int(e["exit"]) for e in eps)
            item["observed_successes"] = successes
            if successes in (0, len(eps)):
                item["ci95"] = None
                item["uncertainty_note"] = "boundary sample: cluster bootstrap cannot estimate unseen success variability"
        result[key] = item
    return result


def summarize_episodes(eps):
    if not eps:
        raise ValueError("no evaluation episodes")
    for e in eps:
        if not all(np.isfinite(e[k]) for k in METRICS):
            raise ValueError("nonfinite evaluation metric")
    return {"summary": _summary(eps),
            "by_ticks": {str(t): _summary([e for e in eps if e["ticks"] == t])
                         for t in sorted({e["ticks"] for e in eps})},
            "per_map": {m: _summary([e for e in eps if e["map"] == m])
                        for m in sorted({e["map"] for e in eps})}}


def compare_paired(baseline, candidate):
    def index(eps):
        result = {(e["map"], e["seed"], e["ticks"]): e for e in eps}
        if len(result) != len(eps):
            raise ValueError("duplicate evaluation job")
        return result
    a, b = index(baseline), index(candidate)
    if not a or a.keys() != b.keys():
        raise ValueError("paired comparison requires identical map/seed/tick jobs")
    result = {}
    for ticks in sorted({k[2] for k in a}):
        maps = sorted({k[0] for k in a if k[2] == ticks})
        result[str(ticks)] = {}
        for metric in METRICS:
            diffs = [np.mean([b[k][metric] - a[k][metric] for k in a if k[0] == m and k[2] == ticks])
                     for m in maps]
            result[str(ticks)][metric] = {"mean_difference": float(np.mean(diffs)),
                                         "ci95": bootstrap_ci(diffs), "independent_maps": len(maps)}
            if metric == "exit" and all(a[k]["exit"] == b[k]["exit"] == 0 for k in a if k[2] == ticks):
                result[str(ticks)][metric]["ci95"] = None
    return result


def guard_test_suite(weights, final, allow_reuse=False):
    """Lock + append makes concurrent test-set claims visible; failed attempts remain recorded."""
    import fcntl
    if not final:
        raise ValueError("held-out evaluation requires --final")
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TEST_LOG.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        if stream.read().strip() and not allow_reuse:
            raise ValueError("held-out test suite already accessed")
        stream.write(json.dumps({"time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 "manifest_sha256": file_hash(Path(weights) / "manifest.json"),
                                 "status": "started", "reuse": allow_reuse}) + "\n")
        stream.flush()


def evaluate(weights, suite, seeds, ticks, workers, final=False, allow_reuse=False, timeout=1800.):
    ticks = ticks if isinstance(ticks, (list, tuple)) else [ticks]
    if seeds <= 0 or workers <= 0 or not ticks or any(t <= 0 for t in ticks) or timeout <= 0:
        raise ValueError("positive seeds, workers, tick caps and timeout required")
    if len(set(ticks)) != len(ticks):
        raise ValueError("duplicate tick caps")
    weights = Path(weights).resolve()
    manifest = verify_artifact(weights)
    identity = file_hash(weights / "manifest.json")
    if suite == "test":
        guard_test_suite(weights, final, allow_reuse)
    jobs = [(w, m, 10_000 + s, tk) for w, m in SUITES[suite] for s in range(seeds) for tk in ticks]
    chunks = [jobs[i::workers] for i in range(workers)]
    started = time.monotonic()
    with mp.get_context("forkserver").Pool(min(workers, len(jobs))) as pool:
        pending = pool.map_async(_worker, [(str(weights), c, identity) for c in chunks if c])
        eps = [e for part in pending.get(timeout=timeout) for e in part]
    verify_artifact(weights)
    if file_hash(weights / "manifest.json") != identity:
        raise ValueError("evaluation artifact changed during run")
    return dict(weights=str(weights), manifest_sha256=identity, provenance=manifest["provenance"],
                suite=suite, episodes=len(eps), seeds_per_map=seeds, elapsed_seconds=time.monotonic() - started,
                **summarize_episodes(eps), episodes_raw=eps,
                limitations=["direct-call latency is not sandbox validation",
                             "terminal key observation may be stale in official wrapper",
                             "metrics are not the private aggregate competition score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--suite", choices=tuple(SUITES), default="dev")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ticks", type=int, nargs="+", default=[2100, 4200])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=1800.)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--allow-test-reuse", action="store_true")
    ap.add_argument("--out", required=True, help="result outside the immutable export")
    a = ap.parse_args()
    result = evaluate(a.weights, a.suite, a.seeds, a.ticks, a.workers, a.final, a.allow_test_reuse, a.timeout)
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result["by_ticks"], indent=2))


if __name__ == "__main__":
    main()
