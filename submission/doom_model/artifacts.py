"""Immutable, checksummed policy exports shared by training and submission."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .actions import ACTIONS
from .config import ModelConfig
from .features import FEATURE_SCHEMA
from .network import AgentNet


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def action_hash():
    return hashlib.sha256(ACTIONS.tobytes()).hexdigest()


def check_finite(state):
    if not state or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("checkpoint must contain finite tensors")


def export_policy(net, cfg, destination, provenance):
    destination = Path(destination)
    state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if not k.startswith("v.")}
    check_finite(state)
    eval_cfg = dataclasses.replace(cfg, privileged_dim=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    tmp = Path(tempfile.mkdtemp(prefix=".export-", dir=destination.parent))
    try:
        torch.save(state, tmp / "model.pt")
        eval_cfg.save(tmp / "model_config.json")
        manifest = {
            "version": 2, "feature_schema": FEATURE_SCHEMA, "action_sha256": action_hash(),
            "files": {name: file_hash(tmp / name) for name in ("model.pt", "model_config.json")},
            "provenance": provenance,
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))
        # All files are closed before publication; a reader sees a complete directory or nothing.
        os.rename(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return destination


def verify_artifact(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("version") != 2 or manifest.get("feature_schema") != FEATURE_SCHEMA
            or manifest.get("action_sha256") != action_hash()):
        raise ValueError("artifact schema/action mismatch")
    if set(manifest.get("files", {})) != {"model.pt", "model_config.json"}:
        raise ValueError("artifact file manifest mismatch")
    for name, digest in manifest["files"].items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"artifact hash mismatch: {name}")
    return manifest


def copy_policy(source, destination):
    """Install an identical verified export into an absent or empty weights directory."""
    source, destination = Path(source), Path(destination)
    identity = file_hash(source / "manifest.json")
    manifest = verify_artifact(source)
    if destination.is_symlink() or (destination.exists() and (
            not destination.is_dir() or any(destination.iterdir()))):
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".package-", dir=destination.parent))
    try:
        for name in (*manifest["files"], "manifest.json"):
            shutil.copy2(source / name, tmp / name)
        verify_artifact(tmp)
        if file_hash(tmp / "manifest.json") != identity:
            raise ValueError("artifact identity changed during packaging")
        os.rename(tmp, destination)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return destination


def load_policy(directory):
    directory = Path(directory)
    manifest = verify_artifact(directory)
    cfg = ModelConfig.load(directory / "model_config.json")
    if cfg.privileged_dim:
        raise ValueError("inference configuration must not request privileged inputs")
    net = AgentNet(cfg)
    if cfg.head == "actor_critic":
        del net.v
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    check_finite(state)
    net.load_state_dict(state, strict=True)
    return net.eval(), cfg, manifest
