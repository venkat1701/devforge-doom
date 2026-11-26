"""Training environments: map sampling, agent-side features, shaped reward, privileged critic inputs,
auxiliary labels, and a spawn-safe process pool.

Everything the network sees comes from AgentSideFeatures (submission/doom_model/features.py), the same
code the evaluated agent runs. Privileged state (env.privileged_state(), training only, docs/CONTRACT.md
§7) is used only for: the shaped reward, the critic's extra inputs, and auxiliary-head labels.

Shaped reward, aligned with the official scoring categories (docs/EVALUATION.md "Scoring": exit is the
largest; progression; exploration; combat and survival). Review fixes: doors and unlocks are rewarded,
and exploration is capped so it can never outweigh reaching the exit.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "submission") not in sys.path:
    sys.path.insert(0, str(ROOT / "submission"))

from doom_model.features import AgentSideFeatures  # noqa: E402

from .maps import sample_map  # noqa: E402

CELL = 64.0
PLAYER_HEIGHT = 56.0
DEFAULT_REWARD = dict(
    exit=10.0, exit_health=2.0, exit_time=2.0,      # exit + efficiency bonus (health, time left)
    key=2.0, door=0.5, unlock=3.0,                  # progression
    cell=0.02, cell_cap=4.0,                        # exploration, capped per episode (< exit)
    kill=0.5, hp_loss=0.002, death=2.0,             # combat & survival
)
PRIV_DIM = 8
REVISIT_GAP = 30                                    # steps; aux label "visited this cell before"


def _make_env(wad, map_name, ticks, fast, render=False):
    """render: fill rgb/depth (needed only when the model uses pixels)."""
    if fast:
        try:
            sys.path.insert(0, str(ROOT / "training"))
            from fastenv import make_fast_env
            return make_fast_env(wad, map_name, episode_ticks=ticks, render_buffers=render)
        except ImportError:
            pass
    from doom_env import make_env
    return make_env(wad, map_name, episode_ticks=ticks, render_buffers=render)


class TrainEpisodeRunner:
    """One env slot: samples a map per episode (LRU cache of engine instances), tracks reward state."""

    def __init__(self, seed: int, ticks, fast: bool, use_map: bool, reward=None, fixed_map=None,
                 cache: int = 4, use_pixels: bool = False):
        """ticks: int or list of ints; a tick cap is sampled per episode (hidden maps use 2,100-4,200
        ticks, docs/CONTRACT.md §5, so training must cover decisions beyond 525)."""
        self.rng = np.random.default_rng(seed)
        self.tick_choices = list(ticks) if isinstance(ticks, (list, tuple)) else [int(ticks)]
        self.ticks, self.fast, self.fixed_map = self.tick_choices[0], fast, fixed_map
        self.rw = dict(DEFAULT_REWARD, **(reward or {}))
        self.feats = AgentSideFeatures(use_map=use_map, use_pixels=use_pixels)
        self.render = use_pixels
        self.cache: dict = {}
        self.cache_size = cache
        self.global_step = 0

    def _env(self, key):
        if key not in self.cache:
            if len(self.cache) >= self.cache_size:
                old = next(iter(self.cache))
                self.cache.pop(old).close()
            self.cache[key] = _make_env(key[0], key[1], key[2], self.fast, self.render)
        else:
            self.cache[key] = self.cache.pop(key)                  # move to most-recent
        return self.cache[key]

    def reset(self, seed=None):
        wad_map = self.fixed_map or sample_map(self.rng, self.global_step)
        self.ticks = int(self.rng.choice(self.tick_choices))
        self.map = wad_map
        self.env = self._env((wad_map[0], wad_map[1], self.ticks))
        obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 30)) if seed is None else seed)
        p = self.env.privileged_state()
        lock = p["door_lock"]
        gap = p["sector_ceiling"] - p["sector_floor"]
        self.s = {
            "x0": p["x"], "y0": p["y"], "cells": {}, "cell_r": 0.0,
            "kills": p["kills"], "kills0": p["kills"], "health": p["health"], "keys": 0.0,
            "door_ids": np.flatnonzero(lock == 0), "lock_ids": np.flatnonzero(lock > 0),
            "opened": set(np.flatnonzero((lock >= 0) & (gap >= PLAYER_HEIGHT)).tolist()),  # open at spawn
            "doors": 0, "unlocks": 0, "t": 0, "ret": 0.0, "exit": False, "dead": False,
        }
        self._visit(p)
        f = self.feats.reset(obs)
        return self._pack(f, p, 0.0, False, False)

    def _visit(self, p):
        c = (int(p["x"] // CELL), int(p["y"] // CELL))
        s = self.s
        rev = c in s["cells"] and s["t"] - s["cells"][c] >= REVISIT_GAP
        new = c not in s["cells"]
        s["cells"].setdefault(c, s["t"])
        return new, rev

    def step(self, a: int):
        obs, _, term, trunc, info = self.env.step(self._buttons(a))
        p = self.env.privileged_state()
        s, rw = self.s, self.rw
        s["t"] += 1
        r = 0.0
        new, _ = self._visit(p)
        if new and s["cell_r"] < rw["cell_cap"]:
            inc = min(rw["cell"], rw["cell_cap"] - s["cell_r"])
            r += inc; s["cell_r"] += inc
        r += rw["kill"] * (p["kills"] - s["kills"]); s["kills"] = p["kills"]
        keys = float(np.sum(p["keys"]))
        r += rw["key"] * (keys - s["keys"]); s["keys"] = keys
        gap = p["sector_ceiling"] - p["sector_floor"]
        for sid in s["door_ids"]:
            if sid not in s["opened"] and gap[sid] >= PLAYER_HEIGHT:
                s["opened"].add(sid); s["doors"] += 1; r += rw["door"]
        for sid in s["lock_ids"]:
            if sid not in s["opened"] and gap[sid] >= PLAYER_HEIGHT:
                s["opened"].add(sid); s["unlocks"] += 1; r += rw["unlock"]
        lost = max(0.0, s["health"] - p["health"]); s["health"] = p["health"]
        r -= rw["hp_loss"] * lost
        if info["exit_reached"]:
            s["exit"] = True
            r += rw["exit"] + rw["exit_health"] * max(0.0, p["health"]) / 100.0 \
                + rw["exit_time"] * (1.0 - p["tic"] / p["episode_ticks"])
        if p["dead"]:
            s["dead"] = True; r -= rw["death"]
        s["ret"] += r
        done = term or trunc
        f = None if done else self.feats.step(a, obs)
        return self._pack(f, p, r, done, bool(term))

    @staticmethod
    def _buttons(a):
        from doom_model.actions import ACTIONS
        return ACTIONS[a]

    def _pack(self, f, p, r, done, term):
        s = self.s
        x, y, th = p["x"], p["y"], np.radians(p["angle"])
        dx, dy = s["x0"] - x, s["y0"] - y
        home = np.array([dx * np.cos(th) + dy * np.sin(th), dx * np.sin(th) - dy * np.cos(th)], np.float32) / 1024
        c = (int(x // CELL), int(y // CELL))
        revisit = float(c in s["cells"] and s["t"] - s["cells"][c] >= REVISIT_GAP)
        priv = np.array([s["t"] * 4 / self.ticks, len(s["cells"]) / 200, (s["kills"] - s["kills0"]) / 10,
                         s["keys"] / 3, max(0.0, p["health"]) / 100, s["doors"] / 10, s["unlocks"] / 3,
                         float(s["exit"])], np.float32)
        out = {"f": f, "r": r, "done": done, "term": term, "priv": priv,
               "aux": np.array([revisit, *home], np.float32)}
        if done:
            out["episode"] = {"map": f"{Path(self.map[0]).stem}:{self.map[1]}", "ticks": self.ticks, "steps": s["t"],
                              "cells": len(s["cells"]), "kills": s["kills"] - s["kills0"], "keys": s["keys"],
                              "doors": s["doors"], "unlocks": s["unlocks"], "exit": s["exit"],
                              "dead": s["dead"], "return": s["ret"]}
        return out


def _worker(conn, kw):
    runner = TrainEpisodeRunner(**kw)
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            out = runner.step(data)
            if out["done"]:
                ep = out["episode"]
                nxt = runner.reset()
                nxt.update(r=out["r"], done=True, term=out["term"], episode=ep,
                           aux_last=out["aux"], priv_last=out["priv"])
                out = nxt
            conn.send(out)
        elif cmd == "reset":
            conn.send(runner.reset(seed=data))
        elif cmd == "set_step":
            runner.global_step = data
        elif cmd == "close":
            for e in runner.cache.values():
                e.close()
            conn.close()
            return


class ProcEnvs:
    """N env slots in subprocesses. 'forkserver' start method: no fork after CUDA initialisation
    (review fix; forking a process that has initialised CUDA is unsafe)."""

    def __init__(self, n, seed, ticks=2100, fast=True, use_map=True, reward=None, fixed_maps=None, use_pixels=False):
        ctx = mp.get_context("forkserver")
        self.n = n
        self.conns, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            kw = dict(seed=seed * 10_000 + i, ticks=ticks, fast=fast, use_map=use_map, reward=reward,
                      use_pixels=use_pixels,
                      fixed_map=None if fixed_maps is None else fixed_maps[i])
            p = ctx.Process(target=_worker, args=(b, kw), daemon=True)
            p.start()
            self.conns.append(a); self.procs.append(p)

    def reset(self, seeds=None):
        for i, c in enumerate(self.conns):
            c.send(("reset", None if seeds is None else seeds[i]))
        return [c.recv() for c in self.conns]

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(("step", int(a)))
        return [c.recv() for c in self.conns]

    def set_global_step(self, step):
        for c in self.conns:
            c.send(("set_step", int(step)))

    def close(self):
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=10)


def stack_feats(outs, key="f"):
    fs = [o[key] for o in outs]
    return {k: np.stack([f[k] for f in fs]) for k in fs[0]}


def ram_stats(envs: "ProcEnvs") -> dict:
    """Memory of the learner + env workers (Linux). Uses PSS (proportional set size: shared pages split
    among the processes sharing them), so the sum is not inflated by pages shared with the forkserver
    parent, unlike summing RSS. Logged every update so the job's real footprint is measured."""
    import os
    def pss(pid):
        try:
            with open(f"/proc/{pid}/smaps_rollup") as fh:
                for line in fh:
                    if line.startswith("Pss:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            return 0
        return 0
    if not os.path.exists("/proc/self/smaps_rollup"):
        return {}
    return {"pss_learner_gb": pss(os.getpid()) / 2**30,
            "pss_workers_gb": sum(pss(p.pid) for p in envs.procs) / 2**30}
