"""Competition entry point: learned observation encoders and recurrent memory only."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
sys.path.insert(0, str(SUBMISSION_DIR))

from doom_model import ACTIONS, AgentSideFeatures  # noqa: E402
from doom_model.artifacts import load_policy  # noqa: E402

# MultiBinary(9) button order of the official action contract (docs/CONTRACT.md; doom_env.spaces.BUTTONS).
BUTTONS = ("MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT",
           "ATTACK", "USE", "SELECT_NEXT_WEAPON")


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.action_space = action_space
        # CPU avoids CUDA allocation; the actual exported model must pass the remote validator.
        self.device = torch.device("cpu")
        torch.set_num_threads(1)
        self.load_weights(WEIGHTS_DIR)
        self.gen = torch.Generator()                    # own RNG: sampling never touches global state
        self.reset()

    def load_weights(self, weights_dir) -> None:
        """Build the network from <dir>/model_config.json and load <dir>/model.pt (also used by
        training/doomrl/evaluate.py to evaluate checkpoints through this exact class)."""
        weights_dir = Path(weights_dir)
        self.net, self.cfg, self.manifest = load_policy(weights_dir)
        self.greedy = self.cfg.head == "q"
        # Inference temperature for the learned policy (1.0 = sample as trained, 0 = most likely action).
        # Optional weights_dir/inference.json {"temperature": t}; DOOM_AGENT_TEMPERATURE overrides (experiments).
        self.temperature = 1.0
        cfg_file = weights_dir / "inference.json"
        if cfg_file.exists():
            self.temperature = float(json.loads(cfg_file.read_text()).get("temperature", 1.0))
        if os.environ.get("DOOM_AGENT_TEMPERATURE"):
            self.temperature = float(os.environ["DOOM_AGENT_TEMPERATURE"])
        if not (self.temperature >= 0 and np.isfinite(self.temperature)):
            raise ValueError("temperature must be a finite non-negative number")
        # Fixed action-vocabulary restriction (not state-dependent): combinations pressing any listed button are never
        # chosen. inference.json {"masked_buttons": [...]}; DOOM_AGENT_MASK="A,B" overrides (experiments).
        masked = []
        if cfg_file.exists():
            masked = list(json.loads(cfg_file.read_text()).get("masked_buttons", []))
        if os.environ.get("DOOM_AGENT_MASK") is not None:
            masked = [b for b in os.environ["DOOM_AGENT_MASK"].split(",") if b]
        unknown = set(masked) - set(BUTTONS)
        if unknown:
            raise ValueError(f"unknown buttons to mask: {sorted(unknown)}")
        cols = [BUTTONS.index(b) for b in masked]
        self.action_mask = torch.tensor([bool(np.any(ACTIONS[i][cols])) if cols else False for i in range(len(ACTIONS))])
        if bool(self.action_mask.all()):
            raise ValueError("masking leaves no action")
        self.feats = AgentSideFeatures(use_map=self.cfg.use_map, use_pixels=self.cfg.use_pixels)

    def reset(self) -> None:
        # Reseed per map: a map's action samples never depend on which maps this process played before
        # (review fix: evaluation results changed with how jobs were split across workers).
        self.gen.manual_seed(0)
        self.h = self.net.init_state(1)
        self.t = -1
        self.last_action = 0

    @torch.no_grad()
    def act(self, observation: dict) -> np.ndarray:
        self.t += 1
        f = self.feats.reset(observation) if self.t == 0 else self.feats.step(self.last_action, observation)
        o = {k: torch.from_numpy(np.asarray(v))[None, None] for k, v in f.items()}
        starts = torch.tensor([[1.0 if self.t == 0 else 0.0]])
        out = self.net.policy(o, self.h, starts)
        self.h = out["h"]
        if bool(self.action_mask.any()):
            key = "q" if self.greedy else "logits"
            out[key] = out[key].masked_fill(self.action_mask, float("-inf"))
        scores = out["q"] if self.greedy else out["logits"]
        if not torch.isfinite(scores[..., ~self.action_mask]).all():
            raise ValueError("nonfinite policy output")
        if self.greedy:
            a = int(out["q"][0, 0].argmax())
        elif self.temperature == 0:
            a = int(out["logits"][0, 0].argmax())
        else:
            probs = (out["logits"][0, 0] / self.temperature).softmax(-1)
            a = int(torch.multinomial(probs, 1, generator=self.gen))
        self.last_action = a
        return ACTIONS[a].astype(np.int8).copy()
