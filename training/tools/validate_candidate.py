"""Run the official validate.py against a CANDIDATE checkpoint, without touching submission/weights.

Review fix: validate.py on the repo validated whatever happened to be in submission/ (earlier a random
256-wide default network), so its latency/RAM numbers said nothing about the model we intend to ship.
This copies the working tree (including untracked files such as submission/doom_model/) to a temporary
directory, installs the exact checksummed export as submission/weights/, and runs
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
sys.path.insert(0, str(ROOT / "submission"))
from doom_model.artifacts import file_hash, verify_artifact

IGNORE = shutil.ignore_patterns(".venv", "__pycache__", "runs", "*.pyc", ".DS_Store")


def check_identity(directory: Path, identity: str) -> dict:
    """Verify file contents and retain the exact manifest, including provenance."""
    manifest = verify_artifact(directory)
    if file_hash(directory / "manifest.json") != identity:
        raise ValueError("candidate artifact identity changed")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="immutable export directory with manifest.json")
    ap.add_argument("--ticks", type=int, default=4200, help="longest hidden-map cap (docs/CONTRACT.md §5)")
    ap.add_argument("--docker", action="store_true")
    ap.add_argument("--image", default=None)
    a, rest = ap.parse_known_args()
    w = Path(a.weights).resolve()
    identity = file_hash(w / "manifest.json")
    manifest = check_identity(w, identity)

    def ignore(directory, names):
        ignored = set(IGNORE(directory, names))
        if Path(directory) == ROOT:
            ignored.add("submission")
        elif Path(directory) == ROOT / "submission":
            ignored.add("weights")
        return ignored

    with tempfile.TemporaryDirectory(prefix="doom_candidate_") as tmp:
        repo = Path(tmp) / "repo"
        # Copy submission separately so even a symlinked source submission or
        # weights directory cannot redirect writes back into the working tree.
        shutil.copytree(ROOT, repo, ignore=ignore, symlinks=True)
        shutil.copytree(ROOT / "submission", repo / "submission", ignore=ignore, symlinks=True)
        dst = repo / "submission" / "weights"
        dst.mkdir()
        for f in (*manifest["files"], "manifest.json"):
            shutil.copy2(w / f, dst / f)
        check_identity(dst, identity)
        # Keep controlled arguments last: forwarded flags cannot select another repo.
        cmd = [sys.executable, str(ROOT / "validate.py"), *rest,
               "--repo", str(repo), "--ticks", str(a.ticks)]
        if a.docker:
            cmd.append("--docker")
        if a.image:
            cmd += ["--image", a.image]
        print("candidate:", w, "| manifest_sha256:", identity,
              "| config:", (dst / "model_config.json").read_text().replace("\n", " "), flush=True)
        result = subprocess.call(cmd)
        check_identity(dst, identity)
        return result


if __name__ == "__main__":
    sys.exit(main())
