"""Cross-check rl_algos.nstep_double_q_target against an independent numpy port of the reference
code in DeepMind's rlax / Acme R2D2 (ported from the verbatim source):

  rlax.signed_hyperbolic / signed_parabolic (eps=1e-3)
  rlax.n_step_bootstrapped_returns (lambda=1)
  acme r2d2 loss call: transformed_n_step_q_learning(online_q[:-1], a[:-1], target_q[1:],
                       selector_actions[1:], rewards[:-1], discounts[:-1]),
                       discounts = data.discount * gamma (data.discount = 0 at episode end)

    uv run python training/probes/test_targets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_algos as R  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doomrl import targets as P  # noqa: E402  (production copies used by the learners)


def signed_hyperbolic(x, eps=1e-3):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + eps * x


def signed_parabolic(x, eps=1e-3):
    z = np.sqrt(1 + 4 * eps * (eps + 1 + np.abs(x))) / 2 / eps - 1 / 2 / eps
    return np.sign(x) * (np.square(z) - 1)


def n_step_bootstrapped_returns(r_t, discount_t, v_t, n):
    """Port of rlax.n_step_bootstrapped_returns with lambda_t = 1."""
    seq_len = r_t.shape[0]
    lambda_t = np.ones_like(discount_t)
    pad_size = min(n - 1, seq_len)
    targets = np.concatenate([v_t[n - 1:], np.array([v_t[-1]] * pad_size)])
    r_t = np.concatenate([r_t, np.zeros(n - 1)])
    discount_t = np.concatenate([discount_t, np.ones(n - 1)])
    lambda_t = np.concatenate([lambda_t, np.ones(n - 1)])
    v_t = np.concatenate([v_t, np.array([v_t[-1]] * (n - 1))])
    for i in reversed(range(n)):
        r_ = r_t[i:i + seq_len]
        discount_ = discount_t[i:i + seq_len]
        lambda_ = lambda_t[i:i + seq_len]
        v_ = v_t[i:i + seq_len]
        targets = r_ + discount_ * ((1. - lambda_) * v_ + lambda_ * targets)
    return targets


def reference_targets(q_on, q_tg, rew, done, n, gamma, rescale):
    """Per-sequence targets for steps 0..L-2 following the Acme call convention."""
    L, B, _ = q_on.shape
    out = np.zeros((L - 1, B))
    for b in range(B):
        selector = q_on[1:, b].argmax(-1)                       # online argmax at t+1
        v = q_tg[1:, b][np.arange(L - 1), selector]             # target-net value of s_{t+1}
        if rescale:
            v = signed_parabolic(v)
        disc = gamma * (1.0 - done[:-1, b])                     # discount after transition t
        tgt = n_step_bootstrapped_returns(rew[:-1, b], disc, v, n)
        out[:, b] = signed_hyperbolic(tgt) if rescale else tgt
    return out


def check(rescale, mutate=None, seed=0):
    rng = np.random.default_rng(seed)
    L, B, A, n, gamma, burn = 80, 6, 56, 5, 0.99, 40
    q_on = rng.normal(size=(L, B, A)) * 3
    q_tg = rng.normal(size=(L, B, A)) * 3
    rew = rng.normal(size=(L, B)) * 2
    done = (rng.random((L, B)) < 0.05).astype(float)            # episode ends inside sequences
    ref = reference_targets(q_on, q_tg, rew, done, n, gamma, rescale)
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    ts = torch.arange(burn, L - n)
    fn = mutate or R.nstep_double_q_target
    mine = fn(t(q_on), t(q_tg), t(rew), t(done), ts, n, gamma, rescale).numpy()
    # rlax pads the last n-1 steps; compare where no padding is involved (t + n <= L - 1)
    return np.abs(mine - ref[burn:L - n]).max()


def off_by_one(q_on, q_tg, rew, done, ts, nstep, gamma, rescale):
    """Deliberately wrong: bootstraps from s_{t+n-1}. The test must catch this."""
    a_star = q_on.argmax(-1)
    q_next = q_tg.gather(-1, a_star[..., None])[..., 0]
    q_next = R.h_inv(q_next) if rescale else q_next
    G = torch.zeros(len(ts), rew.shape[1], dtype=rew.dtype); alive = torch.ones_like(G)
    for k in range(nstep):
        G = G + alive * (gamma ** k) * rew[ts + k]
        alive = alive * (1 - done[ts + k])
    target = G + alive * (gamma ** nstep) * q_next[ts + nstep - 1]
    return R.h_fn(target) if rescale else target


def cleanrl_gae(rewards, values, dones_cleanrl, next_value, next_done, gamma, lam):
    """Port of CleanRL ppo.py GAE. dones_cleanrl[t] = 'obs t starts a new episode' convention."""
    T = len(rewards); adv = np.zeros_like(rewards); last = 0
    for t in reversed(range(T)):
        if t == T - 1:
            nnt, nv = 1.0 - next_done, next_value
        else:
            nnt, nv = 1.0 - dones_cleanrl[t + 1], values[t + 1]
        delta = rewards[t] + gamma * nv * nnt - values[t]
        adv[t] = last = delta + gamma * lam * nnt * last
    return adv


def check_gae(seed=0):
    rng = np.random.default_rng(seed)
    T, N = 128, 16
    r, v, vl = rng.normal(size=(T, N)), rng.normal(size=(T, N)), rng.normal(size=N)
    ended = (rng.random((T, N)) < 0.03).astype(float)          # mine: ended after step t
    starts = np.concatenate([np.zeros((1, N)), ended[:-1]])      # CleanRL: dones[t] = ended after t-1
    ref = np.stack([cleanrl_gae(r[:, i], v[:, i], starts[:, i], vl[i], ended[-1, i], 0.99, 0.95)
                    for i in range(N)], 1)
    mine = R.gae(torch.tensor(r), torch.tensor(v), torch.tensor(ended), torch.tensor(vl), 0.99, 0.95).numpy()
    return np.abs(mine - ref).max()


if __name__ == "__main__":
    ok = True
    for rescale in (False, True):
        errs = [check(rescale, seed=s) for s in range(20)]
        print(f"rescale={rescale}: max |mine - rlax port| over 20 random batches = {max(errs):.2e}")
        ok &= max(errs) < 1e-8
        bad = check(rescale, mutate=off_by_one)
        print(f"   mutant (bootstrap s_(t+n-1)) error = {bad:.2e}  -> {'caught' if bad > 1e-3 else 'NOT caught'}")
        ok &= bad > 1e-3
    x = np.linspace(-1e4, 1e4, 2001)
    rt = R.h_inv(R.h_fn(torch.tensor(x, dtype=torch.float64))).numpy()
    print(f"h_inv(h(x)) max rel err (float64, |x|<=1e4): {np.max(np.abs(rt - x) / np.maximum(1, np.abs(x))):.2e}")
    print(f"h vs rlax signed_hyperbolic: {np.abs(R.h_fn(torch.tensor(x)).numpy() - signed_hyperbolic(x)).max():.2e}; "
          f"h_inv vs signed_parabolic: {np.abs(R.h_inv(torch.tensor(x)).numpy() - signed_parabolic(x)).max():.2e}")
    e = max(check_gae(s) for s in range(20))
    print(f"GAE: max |mine - CleanRL port| over 20 random rollouts = {e:.2e}")
    ok &= e < 1e-10
    # production module (training/doomrl/targets.py) must match the same references
    for rescale in (False, True):
        e = max(check(rescale, mutate=P.nstep_double_q_target, seed=s) for s in range(20))
        print(f"doomrl.targets n-step (rescale={rescale}) vs rlax port: {e:.2e}"); ok &= e < 1e-8
    saved = R.gae; R.gae = P.gae
    e = max(check_gae(s) for s in range(20)); R.gae = saved
    print(f"doomrl.targets GAE vs CleanRL port: {e:.2e}"); ok &= e < 1e-10
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
