"""RingSeqReplay (training/doomrl/drqn.py) correctness:
  * every sampled sequence = L consecutive steps of ONE env's stream (encoded step ids)
  * the stored recurrent state h0 is the one recorded at the sequence's first step
  * after the ring wraps, sequences with any overwritten step are never sampled
  * storage holds each step once (array size = n_envs x cap, no per-sequence copies)
  * priority update targets the sampled slots

    uv run python training/tests/test_replay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]
from doomrl.drqn import RingSeqReplay  # noqa: E402


def main() -> int:
    n_envs, L, P, H, cap_total = 3, 8, 4, 5, 3 * 40
    example = {"dist": np.zeros((n_envs, 64), np.float32), "crops": np.zeros((n_envs, 2, 3, 3), np.uint8)}
    rb = RingSeqReplay(n_envs, cap_total, L, P, H, example, alpha=0.9, beta=0.6, eta=0.9, ram_fraction=0.9)
    assert rb.obs["crops"].shape[:2] == (n_envs, rb.cap) and rb.cap == 40, rb.obs["crops"].shape
    fails = []
    rng = np.random.default_rng(0)
    for step in range(200):                          # 200 > cap: forces several wraps
        for i in range(n_envs):
            if rng.random() < 0.3 and step > 0:      # envs advance unevenly
                continue
            s = int(rb.w[i])
            d = np.zeros(64, np.float32); d[0], d[1] = i, s
            rb.add_step(i, {"dist": d, "crops": np.full((2, 3, 3), s % 256, np.uint8)},
                        a=s % 56, r=float(s), d=0.0, st=0.0, h_before=np.full(H, 1000 * i + s, np.float32))
        if len(rb) >= 4 and step % 5 == 0:
            obs, act, rew, done, st, h0, w = rb.sample(4, torch.device("cpu"))
            env_ids, step_ids = obs["dist"][..., 0], obs["dist"][..., 1]          # (L, B)
            for b in range(4):
                e = env_ids[:, b]
                sid = step_ids[:, b]
                if not torch.all(e == e[0]):
                    fails.append(f"step {step}: mixed envs in a sequence")
                if not torch.all(sid[1:] - sid[:-1] == 1):
                    fails.append(f"step {step}: non-consecutive steps {sid.tolist()}")
                if int(sid[0]) % P:
                    fails.append(f"step {step}: sequence start {int(sid[0])} not on period")
                i = int(e[0])
                if int(sid[0]) < rb.w[i] - rb.cap or int(sid[-1]) >= rb.w[i]:
                    fails.append(f"step {step}: overwritten/unwritten step sampled (w={rb.w[i]})")
                if not torch.allclose(h0[0, b], torch.full((H,), 1000.0 * i + float(sid[0]))):
                    fails.append(f"step {step}: wrong h0 {h0[0, b].tolist()} for start {int(sid[0])}")
                if not torch.all(rew[:, b] == sid):
                    fails.append(f"step {step}: rewards misaligned")
            rb.update(torch.rand(L - 3, 4))
            if not np.all(rb.prio[rb.last] > 0):
                fails.append("priority update did not land on sampled slots")
    print(f"replay: cap/env={rb.cap}, valid sequences now={len(rb)}, checks failed={len(fails)}")
    for f in fails[:5]:
        print("  ", f)
    print("PASS" if not fails else "FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
