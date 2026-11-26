"""Training entry point.

    uv run --group train python -m doomrl.train --config training/configs/ppo_map.json --out training/runs/ppo_map_s0

(run from the repo root with PYTHONPATH=training:submission, or use the provided scripts/train.sh)
Writes <out>/model.pt + model_config.json (submission-ready), step-tagged checkpoints, episodes.jsonl,
updates.jsonl and the resolved config. Evaluate with doomrl.evaluate (dev suite).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]


class Logger:
    def __init__(self, out: Path, every_eps: int = 64):
        out.mkdir(parents=True, exist_ok=True)
        self.ep_f = open(out / "episodes.jsonl", "a")
        self.up_f = open(out / "updates.jsonl", "a")
        self.recent = deque(maxlen=every_eps)
        self.every, self.n, self.t0 = every_eps, 0, time.time()

    def episode(self, ep: dict, steps: int):
        ep = dict(ep, global_step=steps)
        self.ep_f.write(json.dumps(ep) + "\n")
        self.recent.append(ep)
        self.n += 1
        if self.n % self.every == 0:
            keys = ("cells", "kills", "keys", "doors", "unlocks", "exit", "dead", "return")
            m = {k: sum(float(e[k]) for e in self.recent) / len(self.recent) for k in keys}
            print(f"[{(time.time() - self.t0) / 60:6.1f} min] step {steps:>10,} | " +
                  " ".join(f"{k} {v:.2f}" for k, v in m.items()), flush=True)
            self.ep_f.flush()

    def update(self, steps: int, stats: dict):
        self.up_f.write(json.dumps(dict(stats, global_step=steps)) + "\n")
        self.up_f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--total_steps", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--num_envs", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--remote_envs", default=None, help="ip:port of an env server (doomrl.remote_envs); "
                    "kept on the command line so infrastructure addresses never land in committed configs")
    ap.add_argument("--init_from", default=None, help="warm start from a model.pt / model_<step>.pt")
    ap.add_argument("--start_step", type=int, default=None, help="global step of the warm-start checkpoint")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    for k in ("seed", "total_steps", "num_envs", "remote_envs", "init_from", "start_step"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    from doomrl.maps import assert_disjoint
    assert_disjoint()
    log = Logger(out)
    if cfg["algo"] == "ppo":
        from doomrl.ppo import train
    elif cfg["algo"] == "drqn":
        from doomrl.drqn import train
    else:
        raise SystemExit(f"unknown algo {cfg['algo']}")
    train(cfg, out, log)


if __name__ == "__main__":
    main()
