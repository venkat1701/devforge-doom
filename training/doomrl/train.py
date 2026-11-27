"""Training entry point.

    uv run --group train python -m training.doomrl.train --config training/configs/ppo_learned.json --out training/runs/learned

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
            keys = ("kills", "keys", "exit", "dead", "timeout", "return")
            m = {k: sum(float(e[k]) for e in self.recent) / len(self.recent) for k in keys}
            print(f"[{(time.time() - self.t0) / 60:6.1f} min] step {steps:>10,} | " +
                  " ".join(f"{k} {v:.2f}" for k, v in m.items()), flush=True)
            self.ep_f.flush()

    def update(self, steps: int, stats: dict):
        self.up_f.write(json.dumps(dict(stats, global_step=steps), allow_nan=False) + "\n")
        self.up_f.flush()

    def close(self):
        self.ep_f.close()
        self.up_f.close()


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
    ap.add_argument("--init_config", default=None, help="explicit source model configuration for migration")
    ap.add_argument("--allow_map_removal", action="store_true")
    ap.add_argument("--resume", default=None, help="trusted full learner checkpoint; starts fresh simulator episodes")
    ap.add_argument("--max_seconds", type=float, default=None, help="stop between complete updates")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    for k in ("seed", "total_steps", "num_envs", "remote_envs", "init_from", "start_step",
              "init_config", "resume", "max_seconds"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    cfg["allow_map_removal"] = a.allow_map_removal
    if cfg.get("max_seconds") is not None and cfg["max_seconds"] <= 0:
        raise SystemExit("max_seconds must be positive")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    from doomrl.maps import assert_disjoint
    assert_disjoint()
    log = Logger(out)
    try:
        if cfg["algo"] != "ppo":
            raise SystemExit("this learned-agent training entry point supports PPO only")
        from doomrl.ppo import train
        train(cfg, out, log)
    finally:
        log.close()


if __name__ == "__main__":
    main()
