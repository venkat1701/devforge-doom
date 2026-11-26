"""Shared pieces for the RL algorithm probe: action set, observation encoder, cores, vector env.

Every arm (PPO-GRU, DRQN, DQN-stack4) uses exactly these, so differences come from the algorithm.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import torch
import torch.nn as nn
import vizdoom as vd

from doom_env import make_env

IWAD = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
TRAIN_MAPS = [f"MAP{i:02d}" for i in range(1, 17)]
TEST_MAPS = [f"MAP{i:02d}" for i in range(17, 25)]
CELL = 64.0


# ----------------------------------------------------------------------------- actions
def _build_actions() -> np.ndarray:
    """56 discrete combos: move{-,F,B} x strafe{-,L,R} x turn{-,TL,TR} x attack{0,1} + USE + NEXT_WEAPON."""
    acts = []
    for mv, st, tn, atk in itertools.product((None, 0, 1), (None, 2, 3), (None, 4, 5), (0, 1)):
        a = np.zeros(9, np.int8)
        for i in (mv, st, tn):
            if i is not None:
                a[i] = 1
        a[6] = atk
        acts.append(a)
    use = np.zeros(9, np.int8); use[7] = 1
    wpn = np.zeros(9, np.int8); wpn[8] = 1
    return np.stack(acts + [use, wpn])


ACTIONS = _build_actions()
N_ACTIONS = len(ACTIONS)

# ----------------------------------------------------------------------------- observations
VITALS_SCALE = np.array([100, 100, 7, 50, 200, 50, 50, 300, 1, 1, 1, 1, 1, 1, 8, 8, 8], np.float32)


def batch_obs(obs_list, prev_actions) -> dict[str, np.ndarray]:
    """List of doom_env observations -> stacked arrays (no pixels)."""
    return {
        "dist": np.stack([o["rays"]["distance"] for o in obs_list]).astype(np.float32),
        "hit": np.stack([o["rays"]["hit_type"] for o in obs_list]).astype(np.int64),
        "ent": np.stack([o["visible_entities"]["features"] for o in obs_list]).astype(np.float32),
        "etype": np.stack([o["visible_entities"]["entity_type"] for o in obs_list]).astype(np.int64),
        "emask": np.stack([o["visible_entities"]["mask"] for o in obs_list]).astype(np.float32),
        "vit": (np.stack([o["vitals"] for o in obs_list]) / VITALS_SCALE).astype(np.float32),
        "prev": np.asarray(prev_actions, np.int64),
    }


DEV = torch.device("cpu")


def set_device(name: str) -> None:
    global DEV
    DEV = torch.device(name)


def to_torch(b: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(DEV) for k, v in b.items()}


def tt(x) -> torch.Tensor:
    """numpy -> tensor on the active device."""
    return torch.from_numpy(np.asarray(x)).to(DEV)


class Encoder(nn.Module):
    """Per-step embedding e_t from rays (circular conv), entities (masked set pool), vitals + prev action."""

    def __init__(self, dim: int = 256):
        super().__init__()
        def c1(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2, padding_mode="circular"), nn.ReLU())
        self.rays = nn.Sequential(c1(7, 32), c1(32, 64), nn.Conv1d(64, 64, 4, stride=4), nn.ReLU(), nn.Flatten(),
                                  nn.Linear(64 * 16, 256), nn.ReLU())
        self.etype = nn.Embedding(12, 8)
        self.ent = nn.Sequential(nn.Linear(13, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.prev = nn.Embedding(N_ACTIONS, 16)
        self.vit = nn.Sequential(nn.Linear(17 + 16, 64), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(256 + 128 + 64, dim), nn.ReLU())
        self.dim = dim

    def forward(self, o: dict[str, torch.Tensor]) -> torch.Tensor:
        lead = o["dist"].shape[:-1]                       # (...,) batch dims, flattened then restored
        f = lambda x, tail: x.reshape(-1, *tail)
        hit = nn.functional.one_hot(f(o["hit"], (64,)), 6).float().transpose(1, 2)
        r = self.rays(torch.cat([f(o["dist"], (64,))[:, None], hit], 1))
        e = self.ent(torch.cat([f(o["ent"], (32, 5)).clamp(-5, 5), self.etype(f(o["etype"], (32,)))], -1))
        m = f(o["emask"], (32,))[..., None]
        e_mean = (e * m).sum(1) / m.sum(1).clamp(min=1)
        e_max = (e * m - (1 - m) * 1e4).max(1).values * (m.sum(1) > 0)
        v = self.vit(torch.cat([f(o["vit"], (17,)).clamp(-10, 10), self.prev(o["prev"].reshape(-1))], -1))
        return self.out(torch.cat([r, e_mean, e_max, v], -1)).reshape(*lead, self.dim)


class GRUCore(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=False)
        self.dim = dim

    def init_state(self, n):
        return torch.zeros(1, n, self.dim)

    def forward(self, x, h, done=None):
        """x: (T, N, D); h: (1, N, D); done: (T, N) episode-start mask (reset state before step)."""
        if done is None:
            return self.gru(x, h)
        outs = []
        for t in range(x.shape[0]):
            h = h * (1 - done[t])[None, :, None]
            o, h = self.gru(x[t:t + 1], h)
            outs.append(o)
        return torch.cat(outs), h


class Stack4Core(nn.Module):
    """Feed-forward 'frame stack': last 4 embeddings concatenated (state = the 3 previous embeddings)."""

    def __init__(self, dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4 * dim, dim), nn.ReLU())
        self.dim = dim

    def init_state(self, n):
        return torch.zeros(1, n, 3 * self.dim)

    def forward(self, x, h, done=None):
        outs = []
        hist = h[0].reshape(x.shape[1], 3, self.dim)
        for t in range(x.shape[0]):
            if done is not None:
                hist = hist * (1 - done[t])[:, None, None]
            outs.append(self.mlp(torch.cat([hist.reshape(x.shape[1], -1), x[t]], -1)))
            hist = torch.cat([hist[:, 1:], x[t][:, None]], 1)
        return torch.stack(outs), hist.reshape(1, x.shape[1], -1)


# ----------------------------------------------------------------------------- env + shaped reward
class ShapedVecEnv:
    """One env per map, auto-reset. Reward uses privileged_state (training scaffolding only):
    +0.1 per new 64-unit cell, +1 per kill, +0.5 per key, +5 exit, -1 death, -0.005 per hp lost."""

    def __init__(self, maps, ticks=2100, seed=0):
        self.envs = [make_env(IWAD, m, episode_ticks=ticks, render_buffers=False) for m in maps]
        self.maps = list(maps)
        self.rng = np.random.default_rng(seed)
        self.n = len(self.envs)
        self.obs = [None] * self.n
        self.prev = np.zeros(self.n, np.int64)
        self.stats = [None] * self.n
        self.finished: list[dict] = []
        for i in range(self.n):
            self._reset(i)

    def _reset(self, i):
        self.obs[i], _ = self.envs[i].reset(seed=int(self.rng.integers(1 << 30)))
        p = self.envs[i].privileged_state()
        self.prev[i] = 0
        self.stats[i] = {"map": self.maps[i], "cells": {(int(p["x"] // CELL), int(p["y"] // CELL))},
                         "kills": p["kills"], "health": p["health"], "keys": 0.0, "ret": 0.0, "steps": 0,
                         "exit": False, "dead": False, "kills0": p["kills"]}

    def observe(self):
        return batch_obs(self.obs, self.prev)

    def step(self, action_idx):
        rew = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, np.float32)
        for i, a in enumerate(action_idx):
            o, _, term, trunc, info = self.envs[i].step(ACTIONS[a])
            p = self.envs[i].privileged_state()
            s = self.stats[i]
            cell = (int(p["x"] // CELL), int(p["y"] // CELL))
            r = 0.0
            if cell not in s["cells"]:
                s["cells"].add(cell); r += 0.1
            r += 1.0 * (p["kills"] - s["kills"]); s["kills"] = p["kills"]
            keys = float(o["vitals"][8:14].sum())
            r += 0.5 * (keys - s["keys"]); s["keys"] = keys
            r -= 0.005 * max(0.0, s["health"] - p["health"]); s["health"] = p["health"]
            if info["exit_reached"]:
                r += 5.0; s["exit"] = True
            if p["dead"]:
                r -= 1.0; s["dead"] = True
            s["ret"] += r; s["steps"] += 1
            rew[i] = r
            self.obs[i] = o
            self.prev[i] = a
            if term or trunc:
                done[i] = 1.0
                self.finished.append({k: (len(v) if k == "cells" else v) for k, v in s.items()})
                self._reset(i)
        return rew, done

    def pop_finished(self):
        out, self.finished = self.finished, []
        return out

    def close(self):
        for e in self.envs:
            e.close()


def summarize(eps: list[dict]) -> dict:
    if not eps:
        return {}
    return {
        "cells": float(np.mean([e["cells"] for e in eps])),
        "kills": float(np.mean([e["kills"] - e["kills0"] for e in eps])),
        "keys": float(np.mean([e["keys"] for e in eps])),
        "exit": float(np.mean([e["exit"] for e in eps])),
        "dead": float(np.mean([e["dead"] for e in eps])),
        "return": float(np.mean([e["ret"] for e in eps])),
        "n": len(eps),
    }


# ----------------------------------------------------------------------------- multi-process version
def _worker(conn, map_name, ticks, seed):
    venv = ShapedVecEnv([map_name], ticks=ticks, seed=seed)
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            r, d = venv.step(np.array([data]))
            conn.send((venv.observe(), float(r[0]), float(d[0]), venv.pop_finished()))
        elif cmd == "reseed":
            venv.rng = np.random.default_rng(data)
            venv._reset(0)
            venv.pop_finished()
            conn.send(venv.observe())
        elif cmd == "obs":
            conn.send(venv.observe())
        elif cmd == "close":
            venv.close()
            conn.close()
            return


class ParallelShapedVecEnv:
    """Same interface as ShapedVecEnv, one subprocess per env: observation computation in the
    official wrapper (rays/entities in numpy) is the bottleneck, ~1-3.5 ms per step per core."""

    def __init__(self, maps, ticks=2100, seed=0):
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        self.maps = list(maps)
        self.n = len(self.maps)
        self.conns, self.procs = [], []
        for i, m in enumerate(self.maps):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, m, ticks, seed * 1000 + i), daemon=True)
            p.start()
            self.conns.append(a); self.procs.append(p)
        for c in self.conns:
            c.send(("obs", None))
        self._obs = [c.recv() for c in self.conns]
        self.finished: list[dict] = []

    def observe(self):
        return {k: np.concatenate([o[k] for o in self._obs]) for k in self._obs[0]}

    def step(self, action_idx):
        for c, a in zip(self.conns, action_idx):
            c.send(("step", int(a)))
        rew = np.zeros(self.n, np.float32); done = np.zeros(self.n, np.float32)
        for i, c in enumerate(self.conns):
            self._obs[i], rew[i], done[i], fin = c.recv()
            self.finished.extend(fin)
        return rew, done

    def reseed(self, seed):
        for i, c in enumerate(self.conns):
            c.send(("reseed", seed * 1000 + i))
        self._obs = [c.recv() for c in self.conns]
        self.finished = []

    def pop_finished(self):
        out, self.finished = self.finished, []
        return out

    def close(self):
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=10)
