"""Model configuration, saved as JSON next to the weights so the agent rebuilds the same network."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class ModelConfig:
    hidden: int = 512                 # GRU width (Sample Factory App. A.1.3 / Fig. A.1: GRU 512)
    embed: int = 512                  # per-step embedding width (Fig. A.1: FC 512 before the GRU)
    use_map: bool = False             # legacy flag: true is rejected
    use_episodic: bool = False        # legacy pose-conditioned memory is rejected
    use_pixels: bool = True           # CNN on rgb+depth downsampled to 60x80
    episodic_every: int = 4           # write interval (steps)
    episodic_slots: int = 288         # max slots per episode (>= 1050 / 4)
    head: str = "actor_critic"        # "actor_critic" (PPO) or "q" (DRQN, dueling)
    privileged_dim: int = 0           # >0: asymmetric critic input size (training only)
    aux: bool = False                 # retired geometric auxiliary targets
    memory_slots: int = 0             # opt-in learned latent slots; zero retains the original GRU
    memory_dim: int = 64              # width of each latent slot (not a spatial representation)
    pixel_encoder: str = "nature"     # "nature" (3 strided convs) or "impala" (Espeholt et al. 2018 residual CNN)

    def __post_init__(self):
        if self.use_map or self.use_episodic or self.aux:
            raise ValueError("geometric map, pose memory and auxiliary labels are retired; migrate explicitly")
        if any(type(x) is not int or x <= 0 for x in (self.hidden, self.embed)):
            raise ValueError("hidden and embed must be positive integers")
        if type(self.memory_slots) is not int or self.memory_slots < 0:
            raise ValueError("memory_slots must be a nonnegative integer")
        if type(self.memory_dim) is not int or self.memory_dim <= 0:
            raise ValueError("memory_dim must be a positive integer")
        if type(self.privileged_dim) is not int or self.privileged_dim < 0:
            raise ValueError("privileged_dim must be a nonnegative integer")
        if self.pixel_encoder not in ("nature", "impala"):
            raise ValueError("pixel_encoder must be 'nature' or 'impala'")
        if self.head not in ("actor_critic", "q"):
            raise ValueError("unsupported policy head")
        if any(type(x) is not bool for x in (self.use_map, self.use_episodic, self.use_pixels, self.aux)):
            raise ValueError("feature flags must be booleans")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text()))
