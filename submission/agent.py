"""Evaluation entry point. The scorer instantiates DoomAgent from this file (docs/CONTRACT.md §6).

Every action comes from the trained network (docs/RULES.md §1). The code around it only encodes
observations (odometry and the ray-built map are deterministic preprocessing of rays, velocities and
the agent's own previous buttons) and maps the chosen discrete action to the 9 buttons.

Weights: submission/weights/model.pt + model_config.json (written by training/doomrl/train.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
sys.path.insert(0, str(SUBMISSION_DIR))

from doom_model import ACTIONS, AgentNet, AgentSideFeatures, ModelConfig  # noqa: E402


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.action_space = action_space
        # Batch-1 inference of a ~1M-parameter net is ~0.5 ms on one CPU thread (training/probes/latency.py),
        # well inside the 40 ms/step budget; CPU avoids CUDA start-up time and the 2 GB VRAM cap.
        self.device = torch.device("cpu")
        torch.set_num_threads(1)
        self.load_weights(WEIGHTS_DIR)
        self.gen = torch.Generator()                    # own RNG: sampling never touches global state
        self.reset()

    def load_weights(self, weights_dir) -> None:
        """Build the network from <dir>/model_config.json and load <dir>/model.pt (also used by
        training/doomrl/evaluate.py to evaluate checkpoints through this exact class)."""
        weights_dir = Path(weights_dir)
        cfg_path, w_path = weights_dir / "model_config.json", weights_dir / "model.pt"
        # Fail closed (review fix): never silently run an untrained network.
        if not (cfg_path.is_file() and w_path.is_file()):
            raise FileNotFoundError(f"trained weights missing: need {cfg_path} and {w_path}")
        self.cfg = ModelConfig.load(cfg_path)
        self.net = AgentNet(self.cfg).eval()
        state = torch.load(w_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        # only heads the agent never evaluates may be missing (critic / aux); never core weights
        bad = [k for k in missing if not k.startswith(("v.", "aux."))]
        if bad or unexpected:
            raise RuntimeError(f"weights do not match config: missing={bad} unexpected={unexpected}")
        self.greedy = self.cfg.head == "q"
        self.feats = AgentSideFeatures(use_map=self.cfg.use_map, use_pixels=self.cfg.use_pixels)

    def reset(self) -> None:
        # Reseed per map: a map's action samples never depend on which maps this process played before
        # (review fix: evaluation results changed with how jobs were split across workers).
        self.gen.manual_seed(0)
        self.h = self.net.init_state(1)
        self.t = -1
        self.last_action = 0
        self.bank_h, self.bank_pose, self.bank_t = [], [], []

    @torch.no_grad()
    def act(self, observation: dict) -> np.ndarray:
        self.t += 1
        f = self.feats.reset(observation) if self.t == 0 else self.feats.step(self.last_action, observation)
        o = {k: torch.from_numpy(np.asarray(v))[None, None] for k, v in f.items()}
        starts = torch.tensor([[1.0 if self.t == 0 else 0.0]])
        mem = None
        if self.cfg.use_episodic:
            H = self.cfg.hidden
            n = len(self.bank_h)
            mem = {
                "t_idx": torch.tensor([[self.t]]),
                "bank_h": torch.stack(self.bank_h)[:, None] if n else torch.zeros(0, 1, H),
                "bank_pose": torch.from_numpy(np.stack(self.bank_pose))[:, None] if n else torch.zeros(0, 1, 3),
                "bank_t": torch.tensor(self.bank_t)[:, None] if n else torch.zeros(0, 1, dtype=torch.long),
                "n_valid": torch.tensor([[n]]),
            }
        out = self.net(o, self.h, starts, mem)
        self.h = out["h"]
        if self.cfg.use_episodic and self.t % self.cfg.episodic_every == 0 and len(self.bank_h) < self.cfg.episodic_slots:
            self.bank_h.append(out["core"][0, 0].clone()); self.bank_pose.append(f["pose"]); self.bank_t.append(self.t)
        if self.greedy:
            a = int(out["q"][0, 0].argmax())
        else:
            a = int(torch.multinomial(out["logits"][0, 0].softmax(-1), 1, generator=self.gen))
        self.last_action = a
        return ACTIONS[a].astype(np.int8).copy()
