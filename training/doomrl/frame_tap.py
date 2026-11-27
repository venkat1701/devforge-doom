"""Optional live view of training worker 0, written by the learner (monitoring only).

Enabled by DOOM_FRAME_TAP_DIR. It reads the observation the learner already received, so it never
touches the simulators, the environment spec or the update. Any failure is swallowed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import time
import zlib

import numpy as np


def encode_png(rgb) -> bytes:
    """HxWx3 uint8 -> PNG bytes (stdlib only)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    body = np.concatenate([np.zeros((h, 1), np.uint8), rgb.reshape(h, w * 3)], 1).tobytes()

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(body, 1)) + chunk(b"IEND", b""))


class FrameTap:
    LABEL = "LIVE TRAINING ROLLOUT - worker 0, the 80x60 RGB input the network sees (stochastic policy)"

    def __init__(self, directory, hz=10.0, clock=time.time):
        self.dir, self.period, self.clock = Path(directory), 1.0 / hz, clock
        self.last, self.ep_return, self.ep_step = 0.0, 0.0, 0

    @classmethod
    def from_env(cls):
        d = os.environ.get("DOOM_FRAME_TAP_DIR")
        return cls(d) if d else None

    def observe(self, out, global_step, update):
        """Call once per rollout step with worker 0's step result."""
        try:
            if out.get("done"):
                self.ep_return, self.ep_step = 0.0, 0
            else:
                self.ep_return += float(out.get("r", 0.0))
                self.ep_step += 1
            now = self.clock()
            if now - self.last < self.period:
                return
            f = out.get("f")
            if f is None or "pix" not in f:
                return
            self.last = now
            rgb = np.asarray(f["pix"])[:3].transpose(1, 2, 0)
            meta = {"label": self.LABEL, "worker": 0, "map": out.get("map"), "episode_step": self.ep_step,
                    "episode_return": self.ep_return, "global_step": int(global_step), "update": int(update),
                    "frame_time": now}
            for name, data in (("train.png", encode_png(rgb)), ("train.json", json.dumps(meta).encode())):
                tmp = self.dir / f".{name}.{os.getpid()}.tmp"
                tmp.write_bytes(data)
                os.replace(tmp, self.dir / name)
        except Exception:
            pass  # monitoring must never affect training
