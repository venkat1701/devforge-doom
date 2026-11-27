"""Isolated demonstration BC -> learned-memory PPO + RND experiment.

No remote access, launch-script changes, checkpoint promotion or job control.
Run only on separately allocated compute; this command does not detect or stop
other trainers. Runtime bounds apply to PPO; BC is bounded by its episode/epoch
configuration. A successful run proves execution, not improved gameplay.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]

from doom_model.artifacts import file_hash, load_policy
from doomrl.behavior_cloning import train_bc, validate_config
from doomrl.demonstrations import load_dataset
from doomrl.ppo import model_config, train, validate_budget
from doomrl.rnd import RNDConfig
from doomrl.train import Logger


def run_experiment(config, bc_parameters, dataset, initial_candidate, out, *, device, ppo_max_seconds):
    """Compose all three components, with explicit dataset/source and a fresh run directory."""
    cfg = copy.deepcopy(config)
    if (type(ppo_max_seconds) not in (int, float)
            or not math.isfinite(ppo_max_seconds) or ppo_max_seconds <= 0):
        raise ValueError("ppo_max_seconds must be finite and positive")
    if any(name in cfg for name in ("resume", "init_from", "init_config", "start_step", "remote_envs")):
        raise ValueError("combined experiment requires a fresh, local learner configuration")
    if cfg.get("reward") not in (None, "official-wrapper"):
        raise ValueError("combined learned experiment does not permit heuristic reward shaping")
    if cfg.get("algo") != "ppo" or not cfg.get("model", {}).get("memory_slots", 0):
        raise ValueError("combined experiment requires PPO and enabled learned memory")
    if not cfg["model"].get("use_pixels", True) or cfg.get("rnd") is None:
        raise ValueError("combined experiment requires RGBD and RND")
    RNDConfig(**cfg["rnd"])
    if validate_budget(cfg) <= 0:
        raise ValueError("combined experiment needs at least one complete PPO rollout")
    architecture = model_config(cfg)  # Validate PPO before spending time on BC.
    cfg["model"] = dataclasses.asdict(architecture)
    cfg["disable_frame_tap"] = True  # Also persists for subsequent standalone resumes.
    bc_cfg = {"seed": cfg["seed"], "device": device, "torch_threads": cfg["torch_threads"],
              "model": dataclasses.asdict(dataclasses.replace(architecture, privileged_dim=0)),
              "bc": copy.deepcopy(bc_parameters)}
    validate_config(bc_cfg)
    data = load_dataset(dataset)
    initial_candidate = Path(initial_candidate).resolve()
    load_policy(initial_candidate)
    identity = file_hash(initial_candidate / "manifest.json")
    out = Path(out).resolve()
    # Exclusive ownership of a NEW output tree. Never use the active v3 directory.
    out.mkdir(parents=True, exist_ok=False)
    inputs = {"config": cfg, "bc": bc_cfg, "dataset_sha256": data.sha256,
              "initial_policy_sha256": identity, "device": device,
              "ppo_max_seconds": ppo_max_seconds}
    (out / "experiment.json").write_text(json.dumps(inputs, indent=2, allow_nan=False))
    candidate = train_bc(bc_cfg, data, initial_candidate, out / "bc")
    cfg.update(device=device, max_seconds=ppo_max_seconds,
               init_from=str(candidate / "model.pt"),
               init_config=str(candidate / "model_config.json"),
               bc_manifest_sha256=file_hash(candidate / "manifest.json"))
    run_dir = out / "ppo"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, allow_nan=False))
    logger = Logger(run_dir)
    try:
        train(cfg, run_dir, logger)
    finally:
        logger.close()
    pointer = run_dir / "progress.json"
    if not pointer.exists():
        raise RuntimeError("PPO stopped without a completed checkpoint; no experiment candidate")
    progress = json.loads(pointer.read_text())
    load_policy(progress["candidate"])
    summary = {"bc_candidate": str(candidate), "ppo": progress,
               "promotion": "none; independent evaluation required"}
    (out / "result.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="PPO + memory + RND JSON")
    parser.add_argument("--bc-config", required=True, help="JSON containing BC optimizer/split parameters")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--init-candidate", required=True)
    parser.add_argument("--out", required=True, help="new experiment directory; existing paths refused")
    parser.add_argument("--device", required=True, help="explicit separately allocated device, e.g. cpu")
    parser.add_argument("--ppo-max-seconds", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(run_experiment(
        json.loads(Path(args.config).read_text()), json.loads(Path(args.bc_config).read_text()),
        args.dataset, args.init_candidate, args.out, device=args.device,
        ppo_max_seconds=args.ppo_max_seconds), indent=2))


if __name__ == "__main__":
    main()
