"""Reset prefetch: the next episode's engine is built off-thread so cold starts do not stall the batch."""
import json
import struct
import sys
import threading
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(Path(__file__).parent)]
from test_native_training import FakeEnv
from doomrl import envs


class CountingFactory:
    def __init__(self, delay=0.0):
        self.keys, self.threads, self.delay = [], [], delay

    def __call__(self, wad, map_name, ticks, fast, render=False):
        self.keys.append((wad, map_name, ticks))
        self.threads.append(threading.current_thread() is not threading.main_thread())
        time.sleep(self.delay)
        return FakeEnv()


def episodes(runner, n):
    maps = []
    for _ in range(n):
        runner.reset()
        maps.append((*runner.map, runner.ticks))
        out = runner.step(0)
        while not out["done"]:
            out = runner.step(0)
    return maps


def test_prefetch_builds_next_episode_off_thread_and_uses_it(monkeypatch):
    factory = CountingFactory(delay=0.01)
    monkeypatch.setattr(envs, "_make_env", factory)
    runner = envs.TrainEpisodeRunner(0, [8, 16], False, cache=2, prefetch=True)
    played = episodes(runner, 12)
    runner.close()
    # every episode after the first plays the key scheduled at the previous reset
    assert len(played) == 12
    assert all(k in runner.maps or (k[0], k[1]) in runner.maps for k in played)
    assert any(factory.threads[1:]), "next-episode engines must be built in the background"
    assert not factory.threads[0], "the very first engine is built synchronously"


def test_prefetch_same_map_distribution_and_valid_keys(monkeypatch):
    monkeypatch.setattr(envs, "_make_env", CountingFactory())
    runner = envs.TrainEpisodeRunner(3, [8, 16], False, cache=4, prefetch=True)
    played = episodes(runner, 40)
    runner.close()
    assert {t for *_, t in played} <= {8, 16}
    assert all((w, m) in runner.maps for w, m, _ in played)
    assert len(set(played)) > 5


def test_prefetch_error_surfaces_on_next_reset(monkeypatch):
    calls = {"n": 0}

    def factory(*a, **kw):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("engine failed to start")
        return FakeEnv()
    monkeypatch.setattr(envs, "_make_env", factory)
    runner = envs.TrainEpisodeRunner(0, 8, False, cache=1, prefetch=True)
    runner.reset()
    runner.step(0); runner.step(0)
    try:
        for _ in range(3):  # may hit the cached map; a failed prefetch must raise, never hang or hide
            runner.reset(); runner.step(0); runner.step(0)
        raised = False
    except RuntimeError as e:
        raised = "engine failed to start" in str(e)
    runner.close()
    assert raised


def test_prefetch_off_keeps_original_draw_order(monkeypatch):
    monkeypatch.setattr(envs, "_make_env", CountingFactory())
    a = envs.TrainEpisodeRunner(5, [8, 16], False, prefetch=False)
    rng = np.random.default_rng(5)
    a.reset()
    expected_map = a.maps[int(rng.integers(len(a.maps)))]
    assert tuple(a.map) == tuple(expected_map)
    a.close()


def test_worker_kwargs_enable_prefetch(monkeypatch):
    kws = envs.worker_kwargs(n=3, seed=2, ticks=8, fast=False, use_map=False, reward=None,
                             use_pixels=True, training_maps=None, fixed_maps=None)
    assert [kw["prefetch"] for kw in kws] == [True, True, True]
    assert [kw["seed"] for kw in kws] == [20000, 20001, 20002]
