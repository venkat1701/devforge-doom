"""Agent network: per-step encoder -> GRU core -> optional episodic read -> heads.

  encoder   rays (circular 1D conv; the 64 rays wrap around 360 degrees), entities (shared MLP + masked
            mean/max pool over the 32 slots, permutation invariant), vitals + previous action (MLP),
            optional map crops (small CNN)
  core      single-layer GRU (Sample Factory reports GRU matching LSTM sample efficiency at lower
            cost on VizDoom; see training/docs/DESIGN.md)
  episodic  optional attention over past GRU states written every k steps, keys/values include the
            dead-reckoned pose of the memory relative to the current pose (read is not fed back into
            the GRU, so stored detached banks give the same forward pass at training time)
  heads     actor_critic: policy logits + value (value may take privileged inputs: asymmetric
            actor-critic, training only) | q: dueling Q (V + A - mean A)
            aux (training only): revisit logit + ego vector to the episode start (privileged labels)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .actions import N_ACTIONS
from .config import ModelConfig
from .features import CROP, MAP_CH, PIX_H, PIX_W


class Encoder(nn.Module):
    def __init__(self, dim: int, use_map: bool, use_pixels: bool = False):
        super().__init__()
        def c1(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2, padding_mode="circular"), nn.ReLU())
        self.rays = nn.Sequential(c1(7, 32), c1(32, 64), nn.Conv1d(64, 64, 4, stride=4), nn.ReLU(), nn.Flatten(),
                                  nn.Linear(64 * 16, 256), nn.ReLU())
        self.etype = nn.Embedding(12, 8)
        self.ent = nn.Sequential(nn.Linear(13, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.prev = nn.Embedding(N_ACTIONS, 16)
        self.vit = nn.Sequential(nn.Linear(17 + 16, 64), nn.ReLU())
        self.use_map = use_map
        map_out = 0
        if use_map:
            self.map = nn.Sequential(
                nn.Conv2d(2 * MAP_CH, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * (CROP // 8) ** 2, 128), nn.ReLU())
            map_out = 128
        self.out = nn.Sequential(nn.Linear(256 + 128 + 64 + map_out, dim), nn.ReLU())
        self.use_pixels = use_pixels
        if use_pixels:
            # Conv stack of Sample Factory's VizDoom encoder (App. A.1.3 / Fig. A.1: 8x8/4, 4x4/2, 3x3/2,
            # 32/64/128 filters) on rgb+depth at 60x80. Its projection into the embedding is ZERO-initialised
            # and added before the encoder's ReLU, so a network warm-started from a no-pixel checkpoint
            # computes exactly the same function until training moves the projection.
            self.pix = nn.Sequential(
                nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2), nn.ReLU(), nn.Flatten())
            with torch.no_grad():
                n_flat = self.pix(torch.zeros(1, 4, PIX_H, PIX_W)).shape[1]
            self.pix_proj = nn.Linear(n_flat, dim)
            nn.init.zeros_(self.pix_proj.weight); nn.init.zeros_(self.pix_proj.bias)

    def forward(self, o: dict[str, torch.Tensor]) -> torch.Tensor:
        lead = o["dist"].shape[:-1]
        f = lambda x, *tail: x.reshape(-1, *tail)
        hit = nn.functional.one_hot(f(o["hit"], 64), 6).float().transpose(1, 2)
        r = self.rays(torch.cat([f(o["dist"], 64)[:, None], hit], 1))
        e = self.ent(torch.cat([f(o["ent"], 32, 5).clamp(-5, 5), self.etype(f(o["etype"], 32))], -1))
        m = f(o["emask"], 32)[..., None]
        e_mean = (e * m).sum(1) / m.sum(1).clamp(min=1)
        e_max = (e * m - (1 - m) * 1e4).max(1).values * (m.sum(1) > 0)
        v = self.vit(torch.cat([f(o["vit"], 17).clamp(-10, 10), self.prev(o["prev"].reshape(-1))], -1))
        parts = [r, e_mean, e_max, v]
        if self.use_map:
            parts.append(self.map(f(o["crops"], 2 * MAP_CH, CROP, CROP).float() / 255.0))
        pre = self.out[0](torch.cat(parts, -1))
        if self.use_pixels:
            pre = pre + self.pix_proj(self.pix(f(o["pix"], 4, PIX_H, PIX_W).float() / 255.0))
        return self.out[1](pre).reshape(*lead, -1)


class EpisodicReader(nn.Module):
    """Attention over a bank of past GRU states. q = Wq h_t; k = Wk h_m + Wr rel(t, m); v likewise.
    rel = [fwd, right, distance] of the memory's pose in the current ego frame (/1024) + age (/1000)."""

    def __init__(self, hidden: int, dk: int = 64, dv: int = 128):
        super().__init__()
        self.q = nn.Linear(hidden, dk)
        self.k_h, self.k_r = nn.Linear(hidden, dk), nn.Linear(4, dk, bias=False)
        self.v_h, self.v_r = nn.Linear(hidden, dv), nn.Linear(4, dv, bias=False)
        self.dk, self.dv = dk, dv

    def forward(self, h, pose, t_idx, bank_h, bank_pose, bank_t, n_valid, start=None):
        """h: (T,B,H); pose: (T,B,3); t_idx: (T,B) step index in episode;
        bank_h: (M,B,H) detached; bank_pose: (M,B,3); bank_t: (M,B);
        slots usable at step t: [start, start + n_valid) (start=0 at evaluation; during training a
        rollout's bank can hold slots of several consecutive episodes)."""
        M = bank_h.shape[0]
        if M == 0:
            return h.new_zeros(*h.shape[:2], self.dv)
        dx = bank_pose[None, :, :, 0] - pose[:, None, :, 0]
        dy = bank_pose[None, :, :, 1] - pose[:, None, :, 1]
        th = pose[:, None, :, 2]
        fx = (dx * th.cos() + dy * th.sin()) / 1024
        fy = (dx * th.sin() - dy * th.cos()) / 1024
        age = (t_idx[:, None, :] - bank_t[None]).float() / 1000
        rel = torch.stack([fx, fy, (fx ** 2 + fy ** 2 + 1e-12).sqrt(), age], -1)        # (T,M,B,4)
        k = self.k_h(bank_h)[None] + self.k_r(rel)
        att = (self.q(h)[:, None] * k).sum(-1) / self.dk ** 0.5                         # (T,M,B)
        idx = torch.arange(M, device=h.device)[None, :, None]
        st = torch.zeros_like(n_valid) if start is None else start
        valid = (idx >= st[:, None, :]) & (idx < (st + n_valid)[:, None, :])
        w = att.masked_fill(~valid, -1e9).softmax(1) * valid
        return (w[..., None] * (self.v_h(bank_h)[None] + self.v_r(rel))).sum(1)         # (T,B,dv)


class AgentNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = Encoder(cfg.embed, cfg.use_map, cfg.use_pixels)
        self.gru = nn.GRU(cfg.embed, cfg.hidden)
        self.epi = EpisodicReader(cfg.hidden) if cfg.use_episodic else None
        z = cfg.hidden + (self.epi.dv if self.epi else 0)
        if cfg.head == "actor_critic":
            self.pi = nn.Linear(z, N_ACTIONS)
            self.v = nn.Sequential(nn.Linear(z + cfg.privileged_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        elif cfg.head == "q":
            self.adv = nn.Sequential(nn.Linear(z, 256), nn.ReLU(), nn.Linear(256, N_ACTIONS))
            self.val = nn.Sequential(nn.Linear(z, 256), nn.ReLU(), nn.Linear(256, 1))
        else:
            raise ValueError(cfg.head)
        self.aux = nn.Linear(z, 3) if cfg.aux else None

    def init_state(self, n: int, device=None) -> torch.Tensor:
        return torch.zeros(1, n, self.cfg.hidden, device=device)

    def _apply(self, fn, *args, **kwargs):
        # Keep cuDNN GRU weights in one contiguous chunk after .to()/deepcopy/load_state_dict; otherwise
        # PyTorch re-compacts them on every call ("RNN module weights are not part of single contiguous
        # chunk of memory" warning seen on the T4 for the DRQN target network).
        out = super()._apply(fn, *args, **kwargs)
        self.gru.flatten_parameters()
        return out

    def load_state_dict(self, *args, **kwargs):
        res = super().load_state_dict(*args, **kwargs)
        self.gru.flatten_parameters()
        return res

    def __deepcopy__(self, memo):
        import copy as _copy
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, _copy.deepcopy(v, memo))
        new.gru.flatten_parameters()
        return new

    def core(self, x, h, starts):
        """x: (T,B,D); starts: (T,B) 1 where obs t begins a new episode (state reset before step t)."""
        # Split the sequence only at timesteps where SOME env starts an episode and run the cuDNN GRU on
        # each segment. Inside a segment no state is reset except at its first step, so this is exactly
        # the per-step loop (reset h where starts[t]==1, then step), with ~#resets+1 GRU calls instead of T.
        # Equivalence: training/tests/test_core_segments.py.
        cuts = (starts[1:].sum(1) > 0).nonzero().flatten().add(1).tolist()
        bounds = [0] + cuts + [x.shape[0]]
        outs = []
        for t0, t1 in zip(bounds[:-1], bounds[1:]):
            h = h * (1 - starts[t0])[None, :, None]
            o, h = self.gru(x[t0:t1], h)
            outs.append(o)
        return torch.cat(outs), h

    def forward(self, o, h, starts, mem=None, privileged=None):
        """o: dict of (T,B,...) tensors; returns dict with logits/value or q, aux, features, new h."""
        x = self.enc(o)
        hs, h = self.core(x, h, starts)
        z = hs
        if self.epi is not None:
            z = torch.cat([hs, self.epi(hs, o["pose"], mem["t_idx"], mem["bank_h"], mem["bank_pose"],
                                        mem["bank_t"], mem["n_valid"], mem.get("start"))], -1)
        out = {"h": h, "core": hs}
        if self.cfg.head == "actor_critic":
            out["logits"] = self.pi(z)
            if self.cfg.privileged_dim == 0:
                out["value"] = self.v(z)[..., 0]
            elif privileged is not None:          # asymmetric critic: training only
                out["value"] = self.v(torch.cat([z, privileged], -1))[..., 0]
        else:
            a = self.adv(z)
            out["q"] = self.val(z) + a - a.mean(-1, keepdim=True)
        if self.aux is not None:
            out["aux"] = self.aux(z)
        return out
