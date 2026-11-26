"""Live training dashboard: tails every run's episodes.jsonl / updates.jsonl and mirrors them into
TensorBoard event files, without touching the training processes.

    PYTHONPATH=training:submission uv run --group train python training/tools/live_dashboard.py \
        --runs training/runs --tb training/runs/_tensorboard &
    uv run --group train tensorboard --logdir training/runs/_tensorboard --host 127.0.0.1 --port 6006

View from your machine through an SSH tunnel (nothing is exposed to the internet):
    ssh -i <your-key> -N -L 6006:127.0.0.1:6006 <user>@<vm-ip>   then open http://localhost:6006

Per run it writes:
  episode/<metric>  rolling mean over the last 64 finished episodes (cells, kills, keys, doors, unlocks,
                    exit, dead, return, steps), x-axis = global env step
  episode_4200/...  the same, only episodes with the 4,200-tick cap (long-horizon behaviour)
  update/<metric>   learner stats as logged (losses, entropy, lr, steps/s, PSS memory)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

EP_KEYS = ("cells", "kills", "keys", "doors", "unlocks", "exit", "dead", "return", "steps")


class Tail:
    def __init__(self, path: Path):
        self.path, self.pos, self.buf = path, 0, ""

    def lines(self):
        if not self.path.is_file():
            return []
        with open(self.path) as fh:
            fh.seek(self.pos)
            chunk = fh.read()
            self.pos = fh.tell()
        self.buf += chunk
        *full, self.buf = self.buf.split("\n")
        return [json.loads(x) for x in full if x.strip()]


class RunMirror:
    def __init__(self, run: Path, tb_root: Path):
        self.ep, self.up = Tail(run / "episodes.jsonl"), Tail(run / "updates.jsonl")
        self.w = SummaryWriter(str(tb_root / run.name))
        self.win = deque(maxlen=64)
        self.win_long = deque(maxlen=64)

    def poll(self):
        for e in self.ep.lines():
            step = int(e.get("global_step", 0))
            self.win.append(e)
            for k in EP_KEYS:
                self.w.add_scalar(f"episode/{k}", sum(float(x[k]) for x in self.win) / len(self.win), step)
            if e.get("ticks") == 4200:
                self.win_long.append(e)
                for k in EP_KEYS:
                    self.w.add_scalar(f"episode_4200/{k}",
                                      sum(float(x[k]) for x in self.win_long) / len(self.win_long), step)
        for u in self.up.lines():
            step = int(u.get("global_step", 0))
            for k, v in u.items():
                if k != "global_step" and isinstance(v, (int, float)):
                    self.w.add_scalar(f"update/{k}", v, step)
        self.w.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="training/runs")
    ap.add_argument("--tb", default="training/runs/_tensorboard")
    ap.add_argument("--every", type=float, default=20.0, help="seconds between polls")
    a = ap.parse_args()
    root, tb = Path(a.runs), Path(a.tb)
    mirrors: dict[str, RunMirror] = {}
    while True:
        for run in sorted(p for p in root.iterdir() if (p / "episodes.jsonl").is_file()):
            if run.name not in mirrors:
                mirrors[run.name] = RunMirror(run, tb)
                print(f"mirroring {run.name}", flush=True)
            mirrors[run.name].poll()
        time.sleep(a.every)


if __name__ == "__main__":
    main()
