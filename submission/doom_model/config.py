"""Model configuration, saved as JSON next to the weights so the agent rebuilds the same network."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class ModelConfig:
    hidden: int = 512                 # GRU width (Sample Factory App. A.1.3 / Fig. A.1: GRU 512)
    embed: int = 512                  # per-step embedding width (Fig. A.1: FC 512 before the GRU)
    use_map: bool = True              # egocentric crops of the ray-built grid map (Neural Map / ANS style)
    use_episodic: bool = False        # attention over past GRU states with relative-pose keys
    use_pixels: bool = False          # CNN on rgb+depth downsampled to 60x80 (exit signs, switches, key colours)
    episodic_every: int = 4           # write interval (steps)
    episodic_slots: int = 288         # max slots per episode (>= 1050 / 4)
    head: str = "actor_critic"        # "actor_critic" (PPO) or "q" (DRQN, dueling)
    privileged_dim: int = 0           # >0: asymmetric critic input size (training only)
    aux: bool = True                  # training-only auxiliary heads (revisit, home vector)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text()))
