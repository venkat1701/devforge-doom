"""Throughput: official DoomEnv vs FastDoomEnv, render_buffers=False, single process.

    uv run --group train python training/fastenv/bench.py [--steps 1500] [--maps MAP01 MAP20]

Both envs replay the same macro-action sequence, alternating in chunks of 100 steps. Reports
env.step() throughput and the time spent inside _observe (the part FastDoomEnv replaces). Note that
_observe also pays for freeing the previous step's vizdoom sector/line objects; the rest of a step is
the engine (make_action + get_state).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from doom_env import make_env  # noqa: E402
from fastenv import HAVE_NUMBA, make_fast_env  # noqa: E402
from test_equivalence import MAPS, Explorer  # noqa: E402


class _Timed:
    """Wraps env._observe to accumulate the time spent inside it."""

    def __init__(self, env):
        self.inner, self.spent = env._observe, 0.0
        env._observe = self

    def __call__(self, *a, **k):
        t = time.perf_counter()
        out = self.inner(*a, **k)
        self.spent += time.perf_counter() - t
        return out


def bench(wad: str, map_name: str, steps: int, seed: int, chunk: int = 100) -> dict:
    """Run the official and the fast env on the same seed and macro-action sequence, alternating
    in chunks of `chunk` steps so both see the same machine load (a shared or busy machine
    otherwise skews a sequential A-then-B comparison)."""
    envs = {"official": make_env(wad, map_name, render_buffers=False),
            "fast": make_fast_env(wad, map_name, render_buffers=False)}
    timers = {k: _Timed(e) for k, e in envs.items()}
    explorers = {k: Explorer(seed) for k in envs}
    step_time = dict.fromkeys(envs, 0.0)
    done_steps = dict.fromkeys(envs, 0)
    for k, e in envs.items():
        e.reset(seed=seed)
        for _ in range(20):  # warm-up (numba dispatch, caches)
            e.step(explorers[k]())
        explorers[k] = Explorer(seed)
        e.reset(seed=seed)
        timers[k].spent = 0.0
    while min(done_steps.values()) < steps:
        for k, e in envs.items():
            n = min(chunk, steps - done_steps[k])
            for _ in range(n):
                a = explorers[k]()
                t = time.perf_counter()
                _, _, term, trunc, _ = e.step(a)
                step_time[k] += time.perf_counter() - t
                if term or trunc:  # reset time is excluded from step time
                    e.reset(seed=seed)
            done_steps[k] += n
    for e in envs.values():
        e.close()
    return {k: (steps / step_time[k], 1000.0 * timers[k].spent / steps) for k in envs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=list(MAPS), choices=list(MAPS))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    print(f"numba kernels: {HAVE_NUMBA}; {args.steps} steps/map, render_buffers=False, 1 process")
    print(f"{'map':10s} {'official st/s':>13s} {'fast st/s':>10s} {'speedup':>8s} {'obs ms (off)':>12s} {'obs ms (fast)':>13s} {'obs speedup':>11s}")
    for m in args.maps:
        wad, map_name, _ = MAPS[m]
        r = bench(wad, map_name, args.steps, args.seed)
        (so, oo), (sf, of) = r["official"], r["fast"]
        print(f"{m:10s} {so:13,.0f} {sf:10,.0f} {sf / so:7.2f}x {oo:12.3f} {of:13.3f} {oo / of:10.1f}x")


if __name__ == "__main__":
    main()
