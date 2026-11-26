"""Probe data collection: drive a random macro-action explorer and log observations + ground truth.

Used by the hypothesis probes (throughput, odometry, scan matching). Ground truth comes from
env.privileged_state(), which is training-only.

    uv run python training/probes/collect.py --out <dir> [--maps MAP01 MAP02 ...]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import vizdoom as vd

from doom_env import make_env

IWAD = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")

# Macro actions (MultiBinary(9): F B L R TL TR ATK USE WPN), biased toward moving forward.
MACROS = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # forward
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0],  # forward + turn left
    [1, 0, 0, 0, 0, 1, 0, 0, 0],  # forward + turn right
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # turn left
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # turn right
    [0, 0, 1, 0, 0, 0, 0, 0, 0],  # strafe left
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # strafe right
    [0, 1, 0, 0, 0, 0, 0, 0, 0],  # backward
    [1, 0, 0, 0, 0, 0, 1, 0, 0],  # forward + attack
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # use
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # idle
], dtype=np.int8)


def run_episode(wad: str, map_name: str, seed: int, ticks: int, render: bool):
    env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=render)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    log = {k: [] for k in ("action", "vel", "dist", "hit", "x", "y", "angle", "health")}

    def record(o, a):
        p = env.privileged_state()
        log["action"].append(a)
        log["vel"].append(o["vitals"][14:16].copy())
        log["dist"].append(o["rays"]["distance"].copy())
        log["hit"].append(o["rays"]["hit_type"].astype(np.int8))
        log["x"].append(p["x"]); log["y"].append(p["y"]); log["angle"].append(p["angle"])
        log["health"].append(p["health"])

    step_time = 0.0
    steps = 0
    macro, hold = MACROS[0], 0
    done = False
    while not done:
        if hold <= 0:
            macro, hold = MACROS[rng.integers(len(MACROS))], int(rng.integers(1, 13))
        hold -= 1
        record(obs, macro)
        t0 = time.perf_counter()
        obs, _, term, trunc, _ = env.step(macro)
        step_time += time.perf_counter() - t0
        steps += 1
        done = term or trunc
    # No terminal record: at episode end the wrapper returns the previous observation
    # (wrapper.py:142) but refreshes game variables (wrapper.py:144), so rays/velocity and
    # privileged position would describe different states.
    env.close()
    out = {k: np.asarray(v) for k, v in log.items()}
    return out, steps, step_time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--maps", nargs="+", default=[f"MAP{i:02d}" for i in range(1, 9)])
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--ticks", type=int, default=4200)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Throughput: same map with and without pixel buffers.
    for render in (False, True):
        _, n, t = run_episode(IWAD, "MAP01", 123, 1400, render)
        print(f"throughput render_buffers={render}: {n / t:,.0f} steps/s ({1000 * t / n:.2f} ms/step)")

    for m in args.maps:
        for ep in range(args.episodes):
            data, n, t = run_episode(IWAD, m, 1000 * ep + 7, args.ticks, False)
            np.savez_compressed(out / f"{m}_ep{ep}.npz", **data)
            print(f"{m} ep{ep}: {n} steps, {n / t:,.0f} steps/s, died={data['health'][-1] <= 0}")


if __name__ == "__main__":
    main()
