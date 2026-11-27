"""Observation encoders -> optional latent slots + GRU -> policy and training-only critic."""
from __future__ import annotations

import torch
import torch.nn as nn

from .actions import N_ACTIONS
from .config import ModelConfig
from .features import PIX_H, PIX_W
from .memory import LatentSlotMemory


class _Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(torch.relu(self.c1(torch.relu(x))))


def impala_cnn(in_channels, channels=(16, 32, 32)):
    """IMPALA residual CNN (Espeholt et al. 2018, Fig. 3): per stack conv3x3 -> maxpool 3x3/2 -> 2 residual blocks."""
    layers, c = [], in_channels
    for o in channels:
        layers += [nn.Conv2d(c, o, 3, padding=1), nn.MaxPool2d(3, 2, padding=1), _Residual(o), _Residual(o)]
        c = o
    return nn.Sequential(*layers, nn.ReLU(), nn.Flatten())


class Encoder(nn.Module):
    # Named feature layout is also used by the explicit legacy-checkpoint migration.
    FUSION_WIDTHS = (256, 64, 64, 64)  # rays, entity mean, entity max, vitals/action

    def __init__(self, dim: int, use_pixels: bool = True, pixel_encoder: str = "nature"):
        super().__init__()

        def c1(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2, padding_mode="circular"), nn.ReLU())

        self.rays = nn.Sequential(c1(7, 32), c1(32, 64), nn.Conv1d(64, 64, 4, stride=4), nn.ReLU(),
                                  nn.Flatten(), nn.Linear(64 * 16, 256), nn.ReLU())
        self.etype = nn.Embedding(12, 8)
        self.ent = nn.Sequential(nn.Linear(13, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.prev = nn.Embedding(N_ACTIONS, 16)
        self.vit = nn.Sequential(nn.Linear(17 + 16, 64), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(sum(self.FUSION_WIDTHS), dim), nn.ReLU())
        self.use_pixels = use_pixels
        if use_pixels:
            if pixel_encoder == "impala":
                self.pix = impala_cnn(4)
            else:
                self.pix = nn.Sequential(
                    nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                    nn.Conv2d(64, 128, 3, 2), nn.ReLU(), nn.Flatten())
            with torch.no_grad():
                n_flat = self.pix(torch.zeros(1, 4, PIX_H, PIX_W)).shape[1]
            self.pix_proj = nn.Linear(n_flat, dim)
            nn.init.zeros_(self.pix_proj.weight)
            nn.init.zeros_(self.pix_proj.bias)

    def forward(self, o):
        lead = o["dist"].shape[:-1]

        def flat(x, *tail):
            return x.reshape(-1, *tail)

        hit = nn.functional.one_hot(flat(o["hit"], 64), 6).float().transpose(1, 2)
        r = self.rays(torch.cat([flat(o["dist"], 64)[:, None], hit], 1))
        e = self.ent(torch.cat([flat(o["ent"], 32, 5).clamp(-5, 5),
                                self.etype(flat(o["etype"], 32))], -1))
        m = flat(o["emask"], 32)[..., None]
        e_mean = (e * m).sum(1) / m.sum(1).clamp(min=1)
        e_max = (e * m - (1 - m) * 1e4).max(1).values * (m.sum(1) > 0)
        v = self.vit(torch.cat([flat(o["vit"], 17).clamp(-10, 10),
                                self.prev(o["prev"].reshape(-1))], -1))
        pre = self.out[0](torch.cat([r, e_mean, e_max, v], -1))
        if self.use_pixels:
            pre = pre + self.pix_proj(self.pix(flat(o["pix"], 4, PIX_H, PIX_W).float() / 255.0))
        return self.out[1](pre).reshape(*lead, -1)


class AgentNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = Encoder(cfg.embed, cfg.use_pixels, cfg.pixel_encoder)
        self.gru = nn.GRU(cfg.embed, cfg.hidden)
        if cfg.head == "actor_critic":
            self.pi = nn.Linear(cfg.hidden, N_ACTIONS)
            self.v = nn.Sequential(nn.Linear(cfg.hidden + cfg.privileged_dim, 256), nn.ReLU(),
                                   nn.Linear(256, 1))
        else:
            self.adv = nn.Sequential(nn.Linear(cfg.hidden, 256), nn.ReLU(), nn.Linear(256, N_ACTIONS))
            self.val = nn.Sequential(nn.Linear(cfg.hidden, 256), nn.ReLU(), nn.Linear(256, 1))
        # Instantiate after all original layers: disabled models keep exactly the
        # original parameter names, shapes and seeded initialization.
        if cfg.memory_slots:
            self.memory = LatentSlotMemory(cfg.embed, cfg.memory_slots, cfg.memory_dim)

    def init_state(self, n: int, device=None):
        """Zero packed state [1, N, hidden + memory_slots * memory_dim].

        The prefix is GRU state, followed by flattened [slots, memory_dim]
        contents. PPO can slice h[:, envs] and detach().clone() unchanged.
        With memory disabled this is exactly the original GRU state layout.
        """
        width = self.cfg.hidden + self.cfg.memory_slots * self.cfg.memory_dim
        return self.gru.weight_ih_l0.new_zeros(1, n, width, device=device)

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        self.gru.flatten_parameters()
        return out

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self.gru.flatten_parameters()
        return result

    def core(self, x, h, starts):
        """Return [T, N, hidden] outputs and packed state, resetting before each episode."""
        if self.cfg.memory_slots:
            return self._memory_core(x, h, starts)
        cuts = (starts[1:].sum(1) > 0).nonzero().flatten().add(1).tolist()
        bounds = [0] + cuts + [x.shape[0]]
        outs = []
        for t0, t1 in zip(bounds[:-1], bounds[1:]):
            h = h * (1 - starts[t0])[None, :, None]
            o, h = self.gru(x[t0:t1], h)
            outs.append(o)
        return torch.cat(outs), h

    def _memory_core(self, x, h, starts):
        n = x.shape[1]
        width = self.cfg.hidden + self.cfg.memory_slots * self.cfg.memory_dim
        if h.shape != (1, n, width):
            raise ValueError(f"packed memory state must have shape (1, {n}, {width})")
        recurrent = h[:, :, :self.cfg.hidden].contiguous()
        slots = h[0, :, self.cfg.hidden:].reshape(n, self.cfg.memory_slots, self.cfg.memory_dim)
        outs = []
        for t in range(x.shape[0]):
            reset = starts[t].bool()
            recurrent = recurrent.masked_fill(reset[None, :, None], 0)
            slots = slots.masked_fill(reset[:, None, None], 0)
            residual, slots = self.memory(x[t], slots)
            output, recurrent = self.gru((x[t] + residual).unsqueeze(0), recurrent)
            outs.append(output)
        packed = torch.cat([recurrent, slots.reshape(1, n, -1)], dim=-1)
        return torch.cat(outs), packed

    def policy(self, o, h, starts):
        hs, h = self.core(self.enc(o), h, starts)
        out = {"h": h, "core": hs}
        if self.cfg.head == "actor_critic":
            out["logits"] = self.pi(hs)
        else:
            a = self.adv(hs)
            out["q"] = self.val(hs) + a - a.mean(-1, keepdim=True)
        return out

    def forward(self, o, h, starts, mem=None, privileged=None):
        if mem is not None:
            raise ValueError("pose-conditioned memory is not supported")
        out = self.policy(o, h, starts)
        if self.cfg.head == "actor_critic":
            z = out["core"]
            if self.cfg.privileged_dim:
                if privileged is None:
                    raise ValueError("training critic requires privileged inputs; use policy() for inference")
                z = torch.cat([z, privileged], -1)
            out["value"] = self.v(z)[..., 0]
        return out
