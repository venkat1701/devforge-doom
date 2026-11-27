"""Checkpoint spectator: plays the newest verified candidate_* on TRAIN maps and writes PNG + meta.

Separate process; never touches the training process (read-only on the run dir, writes only --frames-dir).
Run: python -m monitor.spectator --run-dir RUN --frames-dir FRAMES
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]

EPISODE_CAP = 2100
NATIVE_TICS_PER_S = 35


def encode_png(rgb) -> bytes:
    """Encode an HxWx3 uint8 array (anything with .tobytes and .shape) as PNG using zlib/struct only."""
    h, w, c = rgb.shape
    if c != 3:
        raise ValueError("expected HxWx3")
    raw = rgb.tobytes()
    stride = w * 3
    body = b"".join(b"\x00" + raw[y * stride:(y + 1) * stride] for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(body, 1)) + chunk(b"IEND", b""))


def atomic_write(path: Path, data: bytes):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def candidate_steps(run_dir):
    found = []
    for d in Path(run_dir).glob("candidate_*"):
        m = re.fullmatch(r"candidate_(\d+)", d.name)
        if m and d.is_dir():
            found.append((int(m[1]), d))
    return sorted(found, reverse=True)


def select_candidate(run_dir, previous=None):
    """Newest candidate whose manifest verifies. Returns dict or None.

    If a newer candidate fails verification, fall back to the newest one that verifies (preferring to keep
    `previous` if it is still the newest verifying one) and flag it old=True."""
    from doom_model.artifacts import verify_artifact, file_hash
    newest = None
    for steps, d in candidate_steps(run_dir):
        newest = newest or steps
        try:
            verify_artifact(d)
            sha = file_hash(d / "manifest.json")
        except Exception:
            continue
        return {"steps": steps, "dir": d, "manifest_sha256": sha, "mtime": (d / "manifest.json").stat().st_mtime,
                "old": steps != newest}
    if previous is not None:
        return dict(previous, old=True)
    return None


def build_agent(candidate_dir):
    import agent as agent_mod
    from doom_env import observation_space, action_space
    agent_mod.WEIGHTS_DIR = Path(candidate_dir)
    return agent_mod.DoomAgent(observation_space(), action_space(), "cpu")


def play_episode(cand, episode_no, frames_dir, rng, maps, sleep=time.sleep, clock=time.time):
    import numpy as np
    from doom_env import make_env
    from doom_env.spaces import FRAME_SKIP
    wad, map_name = maps[int(rng.integers(len(maps)))]
    agent = build_agent(cand["dir"])
    env = make_env(wad, map_name, episode_ticks=EPISODE_CAP)
    step_s = FRAME_SKIP / NATIVE_TICS_PER_S
    try:
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        agent.reset()
        steps, done = 0, False
        while not done:
            t0 = clock()
            meta = {"available": True, "checkpoint_steps": cand["steps"], "manifest_sha256": cand["manifest_sha256"],
                    "checkpoint_mtime": cand["mtime"], "old": bool(cand["old"]),
                    "map": f"{Path(wad).name}:{map_name}", "episode": episode_no, "tic": steps * FRAME_SKIP,
                    "episode_cap_tics": EPISODE_CAP, "frame_time": clock()}
            atomic_write(Path(frames_dir) / "spectator.png", encode_png(np.asarray(obs["rgb"], np.uint8)))
            atomic_write(Path(frames_dir) / "spectator.json", json.dumps(meta).encode())
            obs, _, term, trunc, _ = env.step(agent.act(obs))
            steps += 1
            done = term or trunc
            sleep(max(0.0, step_s - (clock() - t0)))
        return steps
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--max-episodes", type=int, default=None, help="testing only")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from doomrl.maps import resolve_training_maps, SUITES
    maps = resolve_training_maps(None)
    assert set(maps) <= set(SUITES["train"])
    Path(a.frames_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    cand, ep = None, 0
    while a.max_episodes is None or ep < a.max_episodes:
        try:
            cand = select_candidate(a.run_dir, cand)
            if cand is None:
                time.sleep(5)
                continue
            ep += 1
            play_episode(cand, ep, a.frames_dir, rng, maps)
        except Exception as e:  # keep spectating; never affect training
            print(f"spectator error: {type(e).__name__}: {e}", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
