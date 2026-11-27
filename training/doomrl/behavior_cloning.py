"""Categorical BC with episode-consistent truncated BPTT and verified PPO-compatible exports.

Objective reference (MIT): https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/algorithms/bc.py
The recurrent adaptation here is independently implemented: no entropy bonus or critic objective.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]

from doom_model import AgentNet, ModelConfig
from doom_model.artifacts import check_finite, export_policy, file_hash, load_policy, verify_artifact
from doomrl.demonstrations import (
    Dataset, Episode, _json, _new_staging, _positive_int, _publish_directory, _seed, load_dataset,
)
from doomrl.migrate import transfer_policy


def _finite_positive(name, value, *, zero=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value < 0 if zero else value <= 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def validate_config(cfg):
    """No implicit optimizer, schedule, split, device, or architecture hyperparameters."""
    required = {"seed", "device", "torch_threads", "model", "bc"}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError(f"BC config requires exactly {sorted(required)}")
    _seed("seed", cfg["seed"])
    _positive_int("torch_threads", cfg["torch_threads"])
    if not isinstance(cfg["device"], str):
        raise ValueError("explicit torch device required")
    torch.device(cfg["device"])
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    explicit = {"hidden", "embed", "use_pixels"} | (fields & {"memory_slots", "memory_dim"})
    if not isinstance(cfg["model"], dict) or not explicit <= cfg["model"].keys():
        raise ValueError(f"explicit model fields required: {sorted(explicit)}")
    model = ModelConfig(**cfg["model"])
    if model.head != "actor_critic" or model.privileged_dim:
        raise ValueError("BC requires categorical actor_critic with privileged_dim=0")
    p = cfg["bc"]
    required_bc = {"epochs", "lr", "adam_betas", "adam_eps", "weight_decay", "chunk_steps",
                   "max_grad_norm", "validation_fraction", "validation_strategy"}
    if not isinstance(p, dict) or set(p) != required_bc:
        raise ValueError(f"bc requires exactly {sorted(required_bc)}")
    for name in ("epochs", "chunk_steps"):
        _positive_int(name, p[name])
    for name in ("lr", "adam_eps", "max_grad_norm"):
        _finite_positive(name, p[name])
    _finite_positive("weight_decay", p["weight_decay"], zero=True)
    betas = p["adam_betas"]
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("adam_betas requires two values")
    for beta in betas:
        _finite_positive("adam beta", beta, zero=True)
        if beta >= 1:
            raise ValueError("adam beta must be less than one")
    _finite_positive("validation_fraction", p["validation_fraction"])
    if p["validation_fraction"] >= 1:
        raise ValueError("validation_fraction must be less than one")
    if p["validation_strategy"] not in ("heldout_maps", "episodes"):
        raise ValueError("validation_strategy must be heldout_maps or episodes")
    return model


def split_episodes(dataset, *, fraction, strategy, seed):
    """Hold out training maps or complete map/seed episodes; retain original episode order."""
    rng = np.random.default_rng(seed)
    if strategy == "heldout_maps":
        units = list(dict.fromkeys(tuple(e.metadata["map"]) for e in dataset.episodes))
        labels = [tuple(e.metadata["map"]) for e in dataset.episodes]
    elif strategy == "episodes":
        units = list(range(len(dataset.episodes)))
        labels = units
    else:
        raise ValueError("unsupported validation strategy")
    if len(units) < 2:
        raise ValueError(f"{strategy} validation needs at least two independent units")
    count = min(len(units) - 1, max(1, math.ceil(fraction * len(units))))
    heldout = {units[int(i)] for i in rng.permutation(len(units))[:count]}
    train = tuple(i for i, label in enumerate(labels) if label not in heldout)
    validation = tuple(i for i, label in enumerate(labels) if label in heldout)
    return train, validation


def _detach_state(state):
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(_detach_state(value) for value in state)
    if isinstance(state, dict):
        return {key: _detach_state(value) for key, value in state.items()}
    raise TypeError("unsupported recurrent state type")


def episode_cross_entropy(net, episode: Episode, *, chunk_steps, device, backward=False) -> float:
    """Reset once; carry/detach state at chunk boundaries. Caller steps only after this returns.

    Parameters stay fixed throughout an episode. Sum-reduced chunk CE divided by the full
    episode length gives identical scalar loss for whole-episode and chunked execution;
    only cross-chunk gradient paths are truncated. No critic is called.
    """
    _positive_int("chunk_steps", chunk_steps)
    n = len(episode.actions)
    if not n:
        raise ValueError("empty episode")
    h = net.init_state(1, device)
    total = 0.0
    with torch.set_grad_enabled(backward):
        for start in range(0, n, chunk_steps):
            stop = min(n, start + chunk_steps)
            features = {k: torch.tensor(v[start:stop], device=device)[:, None]
                        for k, v in episode.features.items()}
            actions = torch.tensor(episode.actions[start:stop], device=device)
            starts = torch.zeros(stop - start, 1, device=device)
            starts[0, 0] = float(start == 0)
            result = net.policy(features, h, starts)
            logits = result["logits"][:, 0]
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite policy logits")
            loss = F.cross_entropy(logits, actions, reduction="sum") / n
            if not torch.isfinite(loss):
                raise ValueError("nonfinite BC loss")
            h = _detach_state(result["h"])
            if backward:
                loss.backward()
            total += float(loss.detach())
    return total


def _evaluate(net, episodes, indices, p, device):
    net.eval()
    count = sum(len(episodes[i].actions) for i in indices)
    return sum(episode_cross_entropy(net, episodes[i], chunk_steps=p["chunk_steps"], device=device)
               * len(episodes[i].actions) for i in indices) / count


def train_bc(cfg: dict, dataset, init_candidate, out) -> Path:
    """Train from a verified source export and return <out>/candidate; existing out is forbidden.

    dataset is a dataset directory or Dataset (reloaded for integrity). All BC config fields
    are explicit; use validation_strategy='heldout_maps' when multiple training maps exist.
    """
    model = validate_config(cfg)
    dataset = load_dataset(dataset.directory if isinstance(dataset, Dataset) else dataset)
    if model.use_pixels != dataset.manifest["use_pixels"]:
        raise ValueError("model/dataset pixel feature mismatch")
    init_candidate, out = Path(init_candidate), Path(out)
    source_hash = file_hash(init_candidate / "manifest.json")
    source_net, source_config, _ = load_policy(init_candidate)
    train_ids, validation_ids = split_episodes(
        dataset, fraction=cfg["bc"]["validation_fraction"],
        strategy=cfg["bc"]["validation_strategy"], seed=cfg["seed"])
    staging = _new_staging(out)
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(cfg["device"])
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.set_num_threads(cfg["torch_threads"])
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(cfg["seed"])
            random.seed(cfg["seed"])
            np.random.seed(cfg["seed"])
            net = AgentNet(model).to(device)
            migration = transfer_policy(net, source_net.state_dict(), dataclasses.asdict(source_config))
            # Policy-only objective and optimizer; no critic parameter is trained.
            for parameter in net.v.parameters():
                parameter.requires_grad_(False)
            parameters = [parameter for parameter in net.parameters() if parameter.requires_grad]
            p = cfg["bc"]
            optimizer = torch.optim.Adam(parameters, lr=p["lr"], betas=tuple(p["adam_betas"]),
                                         eps=p["adam_eps"], weight_decay=p["weight_decay"])
            history = [{"epoch": 0,
                        "train_cross_entropy": _evaluate(net, dataset.episodes, train_ids, p, device),
                        "validation_cross_entropy": _evaluate(net, dataset.episodes, validation_ids, p, device)}]
            updates = 0
            for epoch in range(1, p["epochs"] + 1):
                net.train()
                for index in train_ids:
                    optimizer.zero_grad(set_to_none=True)
                    episode_cross_entropy(net, dataset.episodes[index], chunk_steps=p["chunk_steps"],
                                          device=device, backward=True)
                    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                           for parameter in parameters):
                        raise ValueError("nonfinite BC gradient")
                    torch.nn.utils.clip_grad_norm_(parameters, p["max_grad_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    check_finite(net.state_dict())
                    updates += 1
                history.append({"epoch": epoch,
                                "train_cross_entropy": _evaluate(net, dataset.episodes, train_ids, p, device),
                                "validation_cross_entropy": _evaluate(net, dataset.episodes, validation_ids, p, device)})
            # Revalidate inputs before publishing; no output is published if a source changed.
            verify_artifact(init_candidate)
            if file_hash(init_candidate / "manifest.json") != source_hash:
                raise ValueError("initial policy changed during training")
            if load_dataset(dataset.directory).sha256 != dataset.sha256:
                raise ValueError("dataset changed during training")
            provenance = {
                "purpose": "behavior_cloning", "objective": "categorical_cross_entropy",
                "dataset_sha256": dataset.sha256, "source_manifest_sha256": source_hash,
                "demonstration_source": dataset.manifest["source"], "config": cfg,
                "train_episodes": list(train_ids), "validation_episodes": list(validation_ids),
                "updates": updates, "migration": migration,
                "sequence_semantics": "episode reset; carried detached state per chunk; one step per episode",
                "history": history,
            }
            candidate = export_policy(net, model, staging / "candidate", provenance)
            load_policy(candidate)
            _json(staging / "training.json", provenance)
            _publish_directory(staging, out)
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if staging.exists():
            shutil.rmtree(staging)
    return out / "candidate"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--init-candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(train_bc(json.loads(Path(args.config).read_text()), args.dataset, args.init_candidate, args.out))


if __name__ == "__main__":
    main()
