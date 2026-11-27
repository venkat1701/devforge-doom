"""Shared observation encoding. All temporal/spatial inference lives in the learned GRU."""
from __future__ import annotations

import numpy as np

from .actions import N_ACTIONS

# Unit conversion only; these constants do not classify observations or choose actions.
VITALS_SCALE = np.array([100, 100, 7, 50, 200, 50, 50, 300, 1, 1, 1, 1, 1, 1, 8, 8, 8], np.float32)
PIX_H, PIX_W = 60, 80
FEATURE_SCHEMA = "official-arrays-rgbd-v2-no-geometry"


def pixels(obs: dict) -> np.ndarray:
    rgb = np.asarray(obs["rgb"])[::2, ::2]
    depth = np.asarray(obs["depth"])[::2, ::2]
    return np.concatenate([rgb.transpose(2, 0, 1), depth[None]], 0).astype(np.uint8)


def obs_arrays(obs: dict, prev_action: int) -> dict[str, np.ndarray]:
    return {
        "dist": np.asarray(obs["rays"]["distance"], np.float32),
        "hit": np.asarray(obs["rays"]["hit_type"], np.int64),
        "ent": np.asarray(obs["visible_entities"]["features"], np.float32),
        "etype": np.asarray(obs["visible_entities"]["entity_type"], np.int64),
        "emask": np.asarray(obs["visible_entities"]["mask"], np.float32),
        "vit": np.asarray(obs["vitals"], np.float32) / VITALS_SCALE,
        "prev": np.asarray(prev_action, np.int64),
    }


class AgentSideFeatures:
    """Stateless observation adapter; previous action is explicitly supplied by the caller."""

    def __init__(self, use_map: bool = False, use_pixels: bool = True):
        if use_map:
            raise ValueError("heuristic map preprocessing has been removed")
        self.use_pixels = use_pixels

    def _features(self, obs, prev_action):
        if not 0 <= prev_action < N_ACTIONS:
            raise ValueError("previous action outside the action vocabulary")
        f = obs_arrays(obs, prev_action)
        if self.use_pixels:
            f["pix"] = pixels(obs)
        return f

    def reset(self, obs):
        return self._features(obs, 0)

    def step(self, action_idx: int, obs):
        return self._features(obs, action_idx)
