"""Explicit architecture transfer; never silently ignore mismatched policy tensors."""
from __future__ import annotations

import torch

from doom_model.artifacts import check_finite
from doom_model.network import Encoder


def transfer_policy(net, source, source_cfg, *, allow_map_removal=False):
    """Transfer policy weights, optionally adding target-configured latent memory.

    Legacy configs without memory fields mean disabled memory. A declared memory
    source must have complete configuration and weights with the same slot layout.
    Adding memory preserves initial logits but always requires retraining.
    """
    check_finite(source)
    if source_cfg.get("use_episodic", False):
        raise ValueError("pose-memory checkpoint needs a separately trained replacement")
    removes_map = bool(source_cfg.get("use_map", False))
    if removes_map and not allow_map_removal:
        raise ValueError("map removal requires explicit allow_map_removal")
    if source_cfg.get("head", "actor_critic") != net.cfg.head:
        raise ValueError("policy head mismatch")
    memory_fields = {"memory_slots", "memory_dim"}
    declared = memory_fields.intersection(source_cfg)
    if declared and declared != memory_fields:
        raise ValueError("incomplete source memory configuration")
    source_slots = source_cfg.get("memory_slots", 0)
    source_dim = source_cfg.get("memory_dim", 64)
    if type(source_slots) is not int or source_slots < 0:
        raise ValueError("source memory_slots must be a nonnegative integer")
    if type(source_dim) is not int or source_dim <= 0:
        raise ValueError("source memory_dim must be a positive integer")
    if source_slots:
        if not net.cfg.memory_slots:
            raise ValueError("memory removal is not supported")
        if (source_slots, source_dim) != (net.cfg.memory_slots, net.cfg.memory_dim):
            raise ValueError("memory configuration mismatch")
    elif any(name.startswith("memory.") for name in source):
        raise ValueError("memory parameters require an enabled source memory configuration")
    adds_memory = bool(net.cfg.memory_slots and not source_slots)
    # Swapping the pixel encoder (e.g. nature -> impala): every non-pixel weight transfers; the new encoder starts
    # fresh with its zero-initialised projection, so the start policy is the source policy without pixel input.
    swaps_pixels = source_cfg.get("pixel_encoder", "nature") != net.cfg.pixel_encoder
    current = net.state_dict()
    if adds_memory:
        # Even a previously used target must start from the base policy function.
        name = "memory.read_projection.weight"
        current[name] = torch.zeros_like(current[name])
    transferred, fresh, removed = [], [], []
    for name, tensor in source.items():
        if (name.startswith(("v.", "aux.")) or (removes_map and name.startswith("enc.map."))
                or (swaps_pixels and name.startswith("enc.pix"))):
            removed.append(name)
            continue
        if name not in current:
            raise ValueError(f"unexpected source parameter {name}")
        if name == "enc.out.0.weight" and removes_map:
            width = sum(Encoder.FUSION_WIDTHS)
            if tensor.shape != (net.cfg.embed, width + 128):
                raise ValueError("legacy map fusion shape mismatch")
            tensor = tensor[:, :width]
        if tensor.shape != current[name].shape:
            raise ValueError(f"shape mismatch: {name}")
        current[name] = tensor
        transferred.append(name)
    for name in current:
        if name not in transferred:
            if not (name.startswith(("v.", "enc.pix")) or
                    (adds_memory and name.startswith("memory."))):
                raise ValueError(f"missing required parameter {name}")
            if name.startswith("enc.pix") and source_cfg.get("use_pixels", False) and not swaps_pixels:
                raise ValueError(f"incomplete pixel checkpoint: {name}")
            fresh.append(name)
    net.load_state_dict(current, strict=True)
    return {"function_preserving": not (removes_map or swaps_pixels), "transferred": transferred, "fresh": fresh,
            "removed": removed, "requires_retraining": removes_map or adds_memory or swaps_pixels}
