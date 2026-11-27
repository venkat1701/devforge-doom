"""Uniform train-map environments: official wrapper reward, optionally plus training-only progress shaping."""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submission"))
from doom_model.features import AgentSideFeatures
from doom_model.actions import ACTIONS
from .maps import resolve_training_maps, sample_map
from .progress import EpisodeProgress, RouteGuide, enemy_in_view, resolve_shaping, total_ammo
from .maps import gen_layout

PRIV_DIM = 7  # direct state only: time fraction, health, kills, items, damage, keys, terminal exit


def _make_env(wad, map_name, ticks, fast, render=False):
    if fast:
        from fastenv import make_fast_env
        return make_fast_env(wad, map_name, episode_ticks=ticks, render_buffers=render)
    from doom_env import make_env
    return make_env(wad, map_name, episode_ticks=ticks, render_buffers=render)


def privileged_vector(p):
    return np.asarray([p["tic"] / p["episode_ticks"], p["health"] / 100.,
                       p["kills"], p["items"], p["damage_taken"] / 100.,
                       np.sum(p["keys"]), float(p["exit_reached"])], dtype=np.float32)


class TrainEpisodeRunner:
    def __init__(self, seed, ticks, fast, use_map=False, reward=None, fixed_map=None,
                 cache=4, use_pixels=True, training_maps=None, prefetch=False):
        self.shaping = resolve_shaping(reward)  # None = official wrapper reward only
        self.maps = resolve_training_maps(training_maps)
        if fixed_map is not None and tuple(fixed_map) not in self.maps:
            raise ValueError("fixed map must belong to the training manifest")
        self.rng = np.random.default_rng(seed)
        self.tick_choices = list(ticks) if isinstance(ticks, (list, tuple)) else [ticks]
        if not self.tick_choices or any(type(t) is not int or t <= 0 for t in self.tick_choices):
            raise ValueError("episode tick caps must be positive integers")
        if cache <= 0:
            raise ValueError("cache must be positive")
        self.fast, self.fixed_map = fast, fixed_map
        self.feats = AgentSideFeatures(use_map, use_pixels)
        self.render, self.cache_size = use_pixels, cache
        self.cache = {}
        self.global_step = 0
        # prefetch: draw the NEXT episode's (map, cap) at reset and build its engine in a background
        # thread, so a cold engine start (~150 ms) no longer stalls the synchronous batch. Same
        # uniform distribution; only the order of RNG draws differs from prefetch=False.
        self.prefetch, self._next = prefetch, None

    def _env(self, key):
        if key not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache))).close()
            self.cache[key] = _make_env(*key, self.fast, self.render)
        else:
            self.cache[key] = self.cache.pop(key)
        return self.cache[key]

    def _draw_key(self):
        m = self.fixed_map or sample_map(self.rng, maps=self.maps)
        return (*m, int(self.rng.choice(self.tick_choices)))

    def _schedule_next(self):
        key, box = self._draw_key(), {}
        thread = None
        if key not in self.cache:
            def build():
                try:
                    box["env"] = _make_env(*key, self.fast, self.render)
                except BaseException as error:  # surfaced on the next reset
                    box["error"] = error
            thread = threading.Thread(target=build, name="doom-prefetch", daemon=True)
            thread.start()
        self._next = (key, thread, box)

    def _take_next(self):
        key, thread, box = self._next
        self._next = None
        if thread is not None:
            thread.join()
        if "error" in box:
            raise RuntimeError(f"prefetched engine failed: {box['error']}") from box["error"]
        if "env" in box:
            if key in self.cache:
                box["env"].close()
            else:
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache))).close()
                self.cache[key] = box["env"]
        return key

    def reset(self, seed=None):
        key = self._take_next() if self._next is not None else self._draw_key()
        self.map, self.ticks = tuple(key[:-1]), key[-1]
        self.env = self._env(key)
        obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 30)) if seed is None else seed)
        if self.prefetch:
            self._schedule_next()
        p = self.env.privileged_state()
        self.steps, self.ret = 0, 0.
        self.kills0 = p["kills"]
        self.native_ret = 0.
        self.progress = EpisodeProgress(p) if self.shaping else None
        layout = gen_layout(self.map[0], self.map[1]) if self.shaping and self.shaping.get("route") else None
        self.route = RouteGuide(layout) if layout else None
        self.phi = self.route.potential(p) if self.route else 0.0
        self._obs = obs
        return self._pack(self.feats.reset(obs), p, 0., False, False)

    def step(self, a):
        if not 0 <= a < len(ACTIONS):
            raise ValueError("action outside vocabulary")
        acted_on = self._obs
        obs, reward, term, trunc, _ = self.env.step(ACTIONS[a])
        self._obs = obs
        p = self.env.privileged_state()
        self.steps += 1
        self.native_ret += float(reward)
        if self.progress is not None:
            reward = float(reward) + self.progress.bonus(self.progress.update(p), self.shaping)
            if self.route is not None:
                phi = self.route.potential(p)
                shaped = self.shaping["route"] * (phi - self.phi)
                self.phi = phi
                self.progress.components["route"] += shaped
                reward += shaped
            if not (term or trunc):  # the wrapper repeats the last observation at episode end
                seen = enemy_in_view(acted_on, self.shaping.get("aim_cone_deg"))
                reward += self.progress.fired(total_ammo(acted_on) - total_ammo(obs), seen, self.shaping)
                reward += self.progress.pressed(bool(ACTIONS[a][6]), seen, self.shaping)
                reward += self.progress.tick(acted_on, self.shaping)
        self.ret += float(reward)
        f = None if term or trunc else self.feats.step(a, obs)
        return self._pack(f, p, float(reward), bool(term), bool(trunc))

    def _pack(self, f, p, reward, term, trunc):
        out = {"f": f, "priv": privileged_vector(p), "r": reward,
               "done": term or trunc, "term": term, "trunc": trunc,
               "map": f"{Path(self.map[0]).name}:{self.map[1]}"}
        if out["done"]:
            out["episode"] = {
                "map": out["map"], "ticks": self.ticks, "steps": self.steps,
                "return": self.ret, "kills": p["kills"] - self.kills0,
                "keys": float(np.sum(p["keys"])), "items": p["items"], "health": p["health"],
                "damage_taken": p["damage_taken"], "exit": bool(p["exit_reached"]),
                "dead": bool(p["dead"]), "timeout": bool(p["timeout"]),
                "native_return": self.native_ret,
            }
            if self.progress is not None:
                s = self.progress.summary()
                out["episode"].update({k: s[k] for k in ("cells", "doors_opened", "locked_opened", "keys_picked",
                                                         "distance", "revisit_ratio", "shaping", "damage",
                                                         "ammo_spent", "blind_ammo")})
        return out

    def close(self):
        if self._next is not None:
            _, thread, box = self._next
            self._next = None
            if thread is not None:
                thread.join()
            if "env" in box:
                box["env"].close()
        for env in self.cache.values():
            env.close()
        self.cache.clear()


def worker_kwargs(n, seed, ticks, fast, use_map, reward, use_pixels, training_maps, fixed_maps):
    """Per-worker runner arguments; next-episode engine prefetch is on for every pool worker."""
    return [dict(seed=seed * 10_000 + i, ticks=ticks, fast=fast, use_map=use_map,
                 reward=reward, use_pixels=use_pixels, training_maps=training_maps,
                 fixed_map=None if fixed_maps is None else fixed_maps[i],
                 prefetch=True) for i in range(n)]


def _worker(conn, kw):
    runner = None
    try:
        runner = TrainEpisodeRunner(**kw)
        while True:
            cmd, data = conn.recv()
            if cmd == "close":
                break
            if cmd == "step":
                out = runner.step(data)
                if out["done"]:
                    terminal = out
                    out = runner.reset()
                    out.update({k: terminal[k] for k in ("r", "done", "term", "trunc", "episode")})
                    out["transition_map"] = terminal["map"]
                conn.send(out)
            elif cmd == "reset":
                conn.send(runner.reset(seed=data))
            elif cmd == "set_step":
                runner.global_step = data
                conn.send(None)
            else:
                raise ValueError(f"unknown environment command {cmd}")
    except EOFError:
        pass
    except Exception as error:
        try:
            conn.send({"worker_error": f"{type(error).__name__}: {error}"})
        except (OSError, EOFError):
            pass
    finally:
        if runner is not None:
            runner.close()
        conn.close()


class ProcEnvs:
    def __init__(self, n, seed, ticks=2100, fast=False, use_map=False, reward=None,
                 fixed_maps=None, use_pixels=True, training_maps=None, timeout=120.):
        if type(n) is not int or n <= 0 or timeout <= 0:
            raise ValueError("worker count and timeout must be positive")
        if fixed_maps is not None and len(fixed_maps) != n:
            raise ValueError("one fixed map is required per worker")
        maps = resolve_training_maps(training_maps)
        self.n, self.timeout, self.closed = n, timeout, False
        self.conns, self.procs = [], []
        ctx = mp.get_context("forkserver")
        try:
            for kw in worker_kwargs(n, seed, ticks, fast, use_map, reward, use_pixels, maps, fixed_maps):
                a, b = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(b, kw), daemon=True)
                p.start()
                b.close()
                self.conns.append(a)
                self.procs.append(p)
        except BaseException:
            self.close()
            raise

    def _request(self, cmd, values):
        if self.closed:
            raise RuntimeError("environment pool closed")
        if len(values) != self.n:
            raise ValueError("one value per worker required")
        for c, value in zip(self.conns, values):
            c.send((cmd, value))
        result = []
        for c in self.conns:
            if not c.poll(self.timeout):
                raise TimeoutError(f"environment worker timed out during {cmd}")
            value = c.recv()
            if isinstance(value, dict) and "worker_error" in value:
                raise RuntimeError(value["worker_error"])
            result.append(value)
        return result

    def reset(self, seeds=None):
        return self._request("reset", [None] * self.n if seeds is None else seeds)

    def step(self, actions):
        return self._request("step", [int(a) for a in actions])

    def set_global_step(self, step):
        self._request("set_step", [int(step)] * self.n)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for c in self.conns:
            try:
                c.send(("close", None))
            except (OSError, EOFError):
                pass
            c.close()
        for p in self.procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)


def stack_feats(outs, key="f"):
    fs = [o[key] for o in outs]
    return {k: np.stack([f[k] for f in fs]) for k in fs[0]}


def ram_stats(envs):
    def pss(pid):
        try:
            with open(f"/proc/{pid}/smaps_rollup") as stream:
                for line in stream:
                    if line.startswith("Pss:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        return 0
    if not os.path.exists("/proc/self/smaps_rollup"):
        return {}
    return {"pss_learner_gb": pss(os.getpid()) / 2**30,
            "pss_workers_gb": sum(pss(p.pid) for p in envs.procs) / 2**30}
