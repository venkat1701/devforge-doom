"""Run the official validate.py against a CANDIDATE checkpoint, without touching submission/weights.

Review fix: validate.py on the repo validated whatever happened to be in submission/ (earlier a random
256-wide default network), so its latency/RAM numbers said nothing about the model we intend to ship.
This copies the working tree (including untracked files such as submission/doom_model/) to a temporary
directory, installs <weights>/model.pt + model_config.json as submission/weights/, and runs
`validate.py --repo <copy>` (optionally --docker, which also measures RAM and pins 1 CPU like the scorer).

    uv run python training/tools/validate_candidate.py --weights training/runs/ppo_map_s0 [--docker]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IGNORE = shutil.ignore_patterns(".venv", "__pycache__", "runs", "*.pyc", ".DS_Store")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="dir with model.pt + model_config.json")
    ap.add_argument("--ticks", type=int, default=4200, help="longest hidden-map cap (docs/CONTRACT.md §5)")
    ap.add_argument("--docker", action="store_true")
    ap.add_argument("--image", default=None)
    a, rest = ap.parse_known_args()
    w = Path(a.weights)
    for f in ("model.pt", "model_config.json"):
        if not (w / f).is_file():
            raise SystemExit(f"missing {w / f}")
    with tempfile.TemporaryDirectory(prefix="doom_candidate_") as tmp:
        repo = Path(tmp) / "repo"
        shutil.copytree(ROOT, repo, ignore=IGNORE, symlinks=True)
        dst = repo / "submission" / "weights"
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("model.pt", "model_config.json"):
            shutil.copy2(w / f, dst / f)
        cmd = [sys.executable, str(ROOT / "validate.py"), "--repo", str(repo), "--ticks", str(a.ticks)]
        if a.docker:
            cmd.append("--docker")
        if a.image:
            cmd += ["--image", a.image]
        print("candidate:", w, "| config:", (w / "model_config.json").read_text().replace("\n", " "), flush=True)
        return subprocess.call(cmd + rest)


if __name__ == "__main__":
    sys.exit(main())
