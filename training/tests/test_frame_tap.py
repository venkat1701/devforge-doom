"""Learner-side live frame tap: monitoring only, never affects training or the environment spec."""
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]
from doomrl import frame_tap
from doomrl.frame_tap import FrameTap


def out(r=0.1, done=False, seed=0):
    pix = np.random.default_rng(seed).integers(256, size=(4, 60, 80), dtype=np.uint8)
    return {"f": {"pix": pix}, "r": r, "done": done, "map": "freedoom2.wad:MAP01"}


def decode_rgb(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", data[16:24])
    i = data.index(b"IDAT")
    n = struct.unpack(">I", data[i - 4:i])[0]
    raw = np.frombuffer(zlib.decompress(data[i + 4:i + 4 + n]), np.uint8).reshape(h, w * 3 + 1)
    assert (raw[:, 0] == 0).all()
    return raw[:, 1:].reshape(h, w, 3)


def test_writes_exact_network_rgb_and_meta(tmp_path):
    t = [100.0]
    tap = FrameTap(tmp_path, hz=10, clock=lambda: t[0])
    o = out()
    tap.observe(o, global_step=4096, update=1)
    rgb = decode_rgb((tmp_path / "train.png").read_bytes())
    assert rgb.shape == (60, 80, 3)
    assert np.array_equal(rgb, o["f"]["pix"][:3].transpose(1, 2, 0))
    meta = json.loads((tmp_path / "train.json").read_text())
    assert meta["global_step"] == 4096 and meta["update"] == 1 and meta["map"] == "freedoom2.wad:MAP01"
    assert meta["episode_step"] == 1 and abs(meta["episode_return"] - 0.1) < 1e-9 and "LIVE TRAINING" in meta["label"]


def test_rate_limited_and_episode_counters_reset(tmp_path):
    t = [100.0]
    tap = FrameTap(tmp_path, hz=10, clock=lambda: t[0])
    tap.observe(out(), 1, 0)
    tap.observe(out(r=0.5), 2, 0)          # within 0.1 s: counted, not written
    assert json.loads((tmp_path / "train.json").read_text())["global_step"] == 1
    t[0] += 0.2
    tap.observe(out(r=-1.0, done=True), 3, 0)  # episode boundary resets the counters
    meta = json.loads((tmp_path / "train.json").read_text())
    assert meta["global_step"] == 3 and meta["episode_step"] == 0 and meta["episode_return"] == 0.0


def test_failures_are_swallowed(tmp_path):
    bad = tmp_path / "not-a-dir"
    bad.write_text("x")
    tap = FrameTap(bad, hz=1e9)
    tap.observe(out(), 1, 0)
    tap.observe({"f": None, "r": 0.0, "done": False}, 2, 0)
    tap.observe({"r": "nan?"}, 3, 0)


def test_disabled_without_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv("DOOM_FRAME_TAP_DIR", raising=False)
    assert FrameTap.from_env() is None
    monkeypatch.setenv("DOOM_FRAME_TAP_DIR", str(tmp_path))
    assert FrameTap.from_env().dir == tmp_path


def test_trainer_hook_present_and_env_spec_sources_untouched():
    src = (ROOT / "training/doomrl/ppo.py").read_text()
    assert "FrameTap.from_env()" in src and "tap.observe(outs[0]" in src
    spec_src = (ROOT / "training/doomrl/remote_envs.py").read_text()
    assert "frame_tap" not in spec_src  # the tap is not part of the hashed environment code
