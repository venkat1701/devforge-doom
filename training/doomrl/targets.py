"""Learning targets, shared by the learners and cross-checked by training/probes/test_targets.py against
independent ports of reference code (rlax / Acme R2D2 for the Q targets, CleanRL for GAE)."""

from __future__ import annotations

import torch


def gae(rews, vals, dones, v_last, gamma, lam):
    """GAE (Schulman et al. 2016). dones[t] = episode ended after step t (no bootstrap past it)."""
    adv = torch.zeros_like(rews); last = torch.zeros_like(v_last)
    for t in reversed(range(len(rews))):
        nv = v_last if t == len(rews) - 1 else vals[t + 1]
        delta = rews[t] + gamma * nv * (1 - dones[t]) - vals[t]
        last = delta + gamma * lam * (1 - dones[t]) * last
        adv[t] = last
    return adv


def h_fn(x, eps=1e-3):
    """Invertible value rescaling = rlax.signed_hyperbolic (eps 1e-3), used by Acme R2D2."""
    return x.sign() * ((x.abs() + 1).sqrt() - 1) + eps * x


def h_inv(x, eps=1e-3):
    """Inverse = rlax.signed_parabolic."""
    return x.sign() * ((((1 + 4 * eps * (x.abs() + 1 + eps)).sqrt() - 1) / (2 * eps)) ** 2 - 1)


def nstep_double_q_target(q_on, q_tg, rew, done, ts, nstep, gamma, rescale):
    """n-step double-DQN target for steps ts; semantics of rlax.transformed_n_step_q_learning as called
    by Acme R2D2: online argmax selects, target net evaluates (van Hasselt et al. 2016, Double DQN)."""
    a_star = q_on.detach().argmax(-1)
    q_next = q_tg.gather(-1, a_star[..., None])[..., 0]
    if rescale:
        q_next = h_inv(q_next)
    G = torch.zeros(len(ts), rew.shape[1], device=rew.device, dtype=rew.dtype)
    alive = torch.ones_like(G)
    for k in range(nstep):
        G = G + alive * (gamma ** k) * rew[ts + k]
        alive = alive * (1 - done[ts + k])
    target = G + alive * (gamma ** nstep) * q_next[ts + nstep]
    return h_fn(target) if rescale else target
