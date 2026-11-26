"""Checks that the agent-side features used at evaluation match the versions validated in the probes,
and that DoomAgent honours the contract (docs/CONTRACT.md §3, §6).

  1. Odometry (submission/doom_model/features.py) step-by-step == probes/memory_probe.dead_reckon (batch)
  2. SpatialMap semantics: crop row 0 = ahead, left columns = left; out-of-range writes dropped
     (the probe's map_crops had a reversed crop and clipped onto the border; fixed here, see review)
  3. DoomAgent on a real env: valid MultiBinary(9) actions, reset() clears state, deterministic

    uv run --with torch==2.15.0 python training/tests/test_features.py <dir with probe .npz files>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training" / "probes")]

from doom_model.features import Odometry, SpatialMap  # noqa: E402
import memory_probe as MP  # noqa: E402


def check_odometry_and_map(npz: Path) -> tuple[float, int]:
    z = dict(np.load(npz))
    ref_pose = MP.dead_reckon(z)
    odo = Odometry()
    poses = [(0.0, 0.0, 0.0)]
    for t in range(len(z["x"]) - 1):
        poses.append(odo.step(z["action"][t], z["vel"][t], z["vel"][t + 1],
                              z["dist"][t].astype(np.float64), z["dist"][t + 1].astype(np.float64)))
    poses = np.array(poses)
    pose_err = float(np.abs(poses - ref_pose).max())
    return pose_err


def check_map_semantics() -> list[str]:
    from doom_model.features import CELL, CROP, GRID
    fails = []
    for th in (0.0, 1.0, -2.5):                       # several headings
        c, s = np.cos(th), np.sin(th)
        sm = SpatialMap()
        ahead = (5 * CELL * c, 5 * CELL * s)          # 5 cells straight ahead
        left = (-5 * CELL * s, 5 * CELL * c)          # 5 cells to the left (CCW +90 deg)
        for (wx, wy), ch in ((ahead, 1), (left, 2)):
            cx, cy, ok = sm._cell(np.array([wx]), np.array([wy]))
            sm.grid[ch, cy, cx] = 255
        crop = sm.crops((0.0, 0.0, th))
        r, col = np.argwhere(crop[1])[0]
        if not (r < CROP / 2 and abs(col - (CROP / 2 - 0.5)) <= 1):
            fails.append(f"theta={th}: wall ahead at row {r}, col {col}")
        r, col = np.argwhere(crop[2])[0]
        if not (col < CROP / 2 and abs(r - (CROP / 2 - 0.5)) <= 1):
            fails.append(f"theta={th}: door left at row {r}, col {col}")
    sm = SpatialMap()                                  # far outside the grid: nothing written
    far = GRID * CELL
    sm.write((far, far, 0.0), np.full(64, 0.5, np.float32), np.ones(64, np.int64))
    if sm.grid.any():
        fails.append("out-of-range write was not dropped")
    return fails


def check_agent() -> None:
    import os
    import tempfile
    import torch
    import vizdoom as vd
    from doom_env import make_env
    import agent as agent_mod
    from agent import DoomAgent
    from doom_model import AgentNet, ModelConfig
    # 1) fail closed: no weights -> initialisation error (review fix: never ship a random network)
    empty = Path(tempfile.mkdtemp())
    agent_mod.WEIGHTS_DIR = empty
    try:
        DoomAgent(None, None, "cpu")
        raise AssertionError("DoomAgent started without weights")
    except FileNotFoundError:
        pass
    # 2) production-shaped weights (default ModelConfig = GRU 512 + map) for the contract checks
    wdir = Path(tempfile.mkdtemp())
    cfg = ModelConfig(aux=False)
    cfg.save(wdir / "model_config.json")
    torch.save(AgentNet(cfg).state_dict(), wdir / "model.pt")
    agent_mod.WEIGHTS_DIR = wdir
    env = make_env(os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad"), "MAP01", episode_ticks=400,
                   render_buffers=False)
    agent = DoomAgent(env.observation_space, env.action_space, "cpu")
    runs = []
    for _ in range(2):
        agent.reset()
        obs, _ = env.reset(seed=3)
        acts = []
        done = False
        while not done:
            a = agent.act(obs)
            assert a.shape == (9,) and set(np.unique(a)) <= {0, 1}, a
            assert not (a[0] and a[1]) and not (a[2] and a[3]) and not (a[4] and a[5])
            acts.append(a)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
        runs.append(np.stack(acts))
    env.close()
    # reset() reseeds the sampler and clears all state: same env seed -> identical action sequence
    assert len(runs[0]) == len(runs[1]) and np.array_equal(runs[0], runs[1]), "reset() did not clear state"
    print(f"agent: fails closed without weights; {len(runs[0])} valid actions x 2 episodes; "
          f"reset() reproducible; contract OK")


if __name__ == "__main__":
    files = sorted(Path(sys.argv[1]).glob("*.npz"))[:6]
    worst_pose = 0.0
    for f in files:
        e = check_odometry_and_map(f)
        worst_pose = max(worst_pose, e)
        print(f"{f.name:14s} max |pose online - pose batch| = {e:.2e}")
    map_fails = check_map_semantics()
    print("map semantics:", "OK" if not map_fails else map_fails)
    check_agent()
    # tolerance: batch version accumulates float32 inputs with numpy cumsum, online version float64
    # step by step; differences are ~1e-6 map units (a cell is 48 units)
    ok = worst_pose < 1e-3 and not map_fails
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
