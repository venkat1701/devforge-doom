"""Probe H-alg: which RL algorithm learns fastest with our encoder, under the same budget?

Arms (same Encoder, same 56-way action set, same shaped reward, same env-step budget):
  ppo     recurrent PPO (GRU core, truncated BPTT over 32-step chunks with stored states)
  drqn    recurrent DQN, R2D2-lite: sequence replay with stored state + burn-in, double DQN,
          n-step returns, dueling head, per-actor epsilons (Ape-X style). No prioritization.
  dqn     DQN with a feed-forward core over the last 4 step embeddings (frame stacking),
          otherwise identical to drqn.

Train on MAP01-16 (one env per map), evaluate on unseen MAP17-24.

    uv run --with torch==2.15.0 python training/probes/rl_algos.py --algo ppo --steps 1000000 --out <dir>
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import common
from common import (tt, ParallelShapedVecEnv, N_ACTIONS, TEST_MAPS, TRAIN_MAPS, Encoder, GRUCore, ShapedVecEnv, Stack4Core,
                    summarize, to_torch)

GAMMA = 0.99


class SimNorm(nn.Module):
    """Simplicial embedding (Obando-Ceron et al., ICLR 2026): softmax(z / tau) within each of L groups of V dims."""

    def __init__(self, groups: int, dim: int, tau: float = 1.0):
        super().__init__()
        self.L, self.V, self.tau = groups, dim, tau

    def forward(self, x):
        shp = x.shape
        return torch.softmax(x.view(*shp[:-1], self.L, self.V) / self.tau, -1).view(*shp)


class Net(nn.Module):
    def __init__(self, core: str, dueling: bool, sem: tuple[int, int, float] | None = None):
        super().__init__()
        self.enc = Encoder(256)
        self.core = GRUCore(256) if core == "gru" else Stack4Core(256)
        self.dueling = dueling
        if dueling:
            self.adv = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, N_ACTIONS))
            self.val = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        else:
            if sem is None:
                self.pi = nn.Linear(256, N_ACTIONS)
            else:   # paper's PPO setup: SEM block (Linear -> LayerNorm -> SimNorm) in the actor only, before the final linear
                L, V, tau = sem
                self.pi = nn.Sequential(nn.Linear(256, L * V), nn.LayerNorm(L * V), SimNorm(L, V, tau),
                                        nn.Linear(L * V, N_ACTIONS))
            self.v = nn.Linear(256, 1)

    def forward(self, obs_seq, h, starts=None):
        """obs_seq: dict of (T, N, ...) tensors. Returns (out, value_or_None, h)."""
        x = self.enc(obs_seq)
        if starts is not None and not bool(starts[1:].any()):
            h = h * (1 - starts[0])[None, :, None]
            z, h = self.core(x, h)
        else:
            z, h = self.core(x, h, starts)
        if self.dueling:
            a = self.adv(z)
            return self.val(z) + a - a.mean(-1, keepdim=True), None, h
        return self.pi(z), self.v(z)[..., 0], h


def add_time(b):
    return {k: v[None] for k, v in b.items()}


def stack_steps(steps):
    """list over t of dict (N, ...) -> dict (T, N, ...)"""
    return {k: np.stack([s[k] for s in steps]) for k in steps[0]}


# ----------------------------------------------------------------------------- evaluation
def evaluate(net, algo, seeds=(11, 12), ticks=2100):
    net.eval()
    venv = ParallelShapedVecEnv(TEST_MAPS, ticks=ticks, seed=0)
    eps = []
    rng = np.random.default_rng(0)
    for sd in seeds:
        venv.reseed(sd)
        h = net.core.init_state(venv.n).to(common.DEV)
        alive = np.ones(venv.n, bool)
        start = torch.ones(1, venv.n, device=common.DEV)
        while alive.any():
            with torch.no_grad():
                out, _, h = net(add_time(to_torch(venv.observe())), h, start)
            if algo == "ppo":
                a = Categorical(logits=out[0]).sample().cpu().numpy()
            else:
                a = out[0].argmax(-1).cpu().numpy()
                rnd = rng.random(venv.n) < 0.01
                a[rnd] = rng.integers(N_ACTIONS, size=rnd.sum())
            _, done = venv.step(a)
            start = tt(done)[None]
            for e in venv.pop_finished():
                idx = TEST_MAPS.index(e["map"])
                if alive[idx]:
                    eps.append(e); alive[idx] = False
    venv.close()
    net.train()
    return summarize(eps)


# ----------------------------------------------------------------------------- PPO
def gae(rews, vals, dones, v_last, gamma, lam):
    """GAE (Schulman et al. 2016). dones[t] = episode ended after step t (no bootstrap past it)."""
    adv = torch.zeros_like(rews); last = torch.zeros_like(v_last)
    for t in reversed(range(len(rews))):
        nv = v_last if t == len(rews) - 1 else vals[t + 1]
        delta = rews[t] + gamma * nv * (1 - dones[t]) - vals[t]
        last = delta + gamma * lam * (1 - dones[t]) * last
        adv[t] = last
    return adv


def train_ppo(args, log):
    venv = ParallelShapedVecEnv(TRAIN_MAPS, seed=args.seed)
    net = Net("gru", dueling=False, sem=(args.sem_groups, args.sem_dim, args.sem_tau) if args.sem else None).to(common.DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
    T, CH = 128, 32
    h = net.core.init_state(venv.n).to(common.DEV)
    start = np.ones(venv.n, np.float32)
    steps, next_eval = 0, 0
    while steps < args.steps:
        obs_l, act_l, logp_l, val_l, rew_l, done_l, start_l, h_l = [], [], [], [], [], [], [], []
        for t in range(T):
            ob = venv.observe()
            h_l.append(h[0].clone())
            with torch.no_grad():
                logits, v, h = net(add_time(to_torch(ob)), h, tt(start)[None])
            dist = Categorical(logits=logits[0])
            a = dist.sample()
            r, d = venv.step(a.cpu().numpy())
            obs_l.append(ob); act_l.append(a); logp_l.append(dist.log_prob(a)); val_l.append(v[0])
            rew_l.append(tt(r)); done_l.append(tt(d)); start_l.append(tt(start))
            start = d.copy()
        steps += T * venv.n
        with torch.no_grad():
            _, v_last, _ = net(add_time(to_torch(venv.observe())), h, tt(start)[None])
        vals = torch.stack(val_l); rews = torch.stack(rew_l); dones = torch.stack(done_l)
        adv = gae(rews, vals, dones, v_last[0], GAMMA, 0.95)
        ret = adv + vals
        obs = to_torch(stack_steps(obs_l)); acts = torch.stack(act_l); old_logp = torch.stack(logp_l)
        starts = torch.stack(start_l); hs = torch.stack(h_l)
        seqs = [(n, c) for n in range(venv.n) for c in range(0, T, CH)]
        for _ in range(4):
            np.random.shuffle(seqs)
            for mb in np.array_split(np.array(seqs), 4):
                ns, cs = mb[:, 0], mb[:, 1]
                tidx = tt(cs[None, :] + np.arange(CH)[:, None]); ns = tt(ns); cs = tt(cs)   # (CH, B)
                o = {k: v[tidx, ns[None, :]] for k, v in obs.items()}
                h0 = hs[cs, ns][None]
                logits, v, _ = net(o, h0, starts[tidx, ns[None, :]])
                dist = Categorical(logits=logits)
                lp = dist.log_prob(acts[tidx, ns[None, :]])
                A = adv[tidx, ns[None, :]]; A = (A - A.mean()) / (A.std() + 1e-8)
                ratio = (lp - old_logp[tidx, ns[None, :]]).exp()
                pl = -torch.min(ratio * A, ratio.clamp(0.8, 1.2) * A).mean()
                vl = 0.5 * (v - ret[tidx, ns[None, :]]).pow(2).mean()
                loss = pl + 0.5 * vl - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        next_eval = log(steps, venv, net, next_eval)
    venv.close()
    return net


# ----------------------------------------------------------------------------- DRQN / DQN
class SeqReplay:
    """Sequence replay. prioritized=True: R2D2 proportional priorities (alpha 0.9, IS beta 0.6,
    priority = 0.9 max|td| + 0.1 mean|td| over the trained part of the sequence)."""

    def __init__(self, cap_seqs, prioritized=False, alpha=0.9, beta=0.6):
        self.cap, self.data, self.pos = cap_seqs, [], 0
        self.prioritized, self.alpha, self.beta = prioritized, alpha, beta
        self.prio = np.zeros(cap_seqs, np.float64)
        self.last_idx = None

    def add(self, seq):
        i = len(self.data) if len(self.data) < self.cap else self.pos
        if len(self.data) < self.cap:
            self.data.append(seq)
        else:
            self.data[i] = seq
        self.prio[i] = self.prio[:len(self.data)].max() if len(self.data) > 1 else 1.0
        self.pos = (self.pos + 1) % self.cap

    def update(self, td_abs):
        """td_abs: (T_train, B) tensor of |td| for the last sampled batch."""
        if self.prioritized and self.last_idx is not None:
            p = 0.9 * td_abs.max(0).values + 0.1 * td_abs.mean(0)
            self.prio[self.last_idx] = p.detach().cpu().numpy() + 1e-6

    def sample(self, b):
        n = len(self.data)
        if self.prioritized:
            pr = self.prio[:n] ** self.alpha; pr /= pr.sum()
            idx = np.random.choice(n, size=b, p=pr)
            w = (n * pr[idx]) ** (-self.beta); w = w / w.max()
        else:
            idx = np.random.randint(n, size=b); w = np.ones(b)
        self.last_idx = idx
        seqs = [self.data[i] for i in idx]
        obs = {k: tt(np.stack([s["obs"][k] for s in seqs], 1).astype(
            np.int64 if k in ("hit", "etype", "prev") else np.float32)) for k in seqs[0]["obs"]}
        cat = lambda k: tt(np.stack([s[k] for s in seqs], 1))
        return obs, cat("act").long(), cat("rew").float(), cat("done").float(), cat("start").float(), \
            torch.cat([s["h0"] for s in seqs], 1).to(common.DEV), tt(w.astype(np.float32))


def h_fn(x, eps=1e-3):
    """R2D2 invertible value rescaling."""
    return x.sign() * ((x.abs() + 1).sqrt() - 1) + eps * x


def h_inv(x, eps=1e-3):
    return x.sign() * ((((1 + 4 * eps * (x.abs() + 1 + eps)).sqrt() - 1) / (2 * eps)) ** 2 - 1)


def nstep_double_q_target(q_on, q_tg, rew, done, ts, nstep, gamma, rescale):
    """n-step double-DQN target for steps ts (R2D2 / rlax transformed_n_step_q_learning semantics).

    q_on, q_tg: (L, B, A) online / target Q over the sequence; rew, done: (L, B), where rew[t] and
    done[t] belong to the transition taken at step t. Bootstrap state s_{t+n} is evaluated by the
    target net at the online argmax (van Hasselt et al. 2016). rescale: Q lives in h-space.
    """
    a_star = q_on.detach().argmax(-1)
    q_next = q_tg.gather(-1, a_star[..., None])[..., 0]
    if rescale:
        q_next = h_inv(q_next)
    G = torch.zeros(len(ts), rew.shape[1], device=rew.device)
    alive = torch.ones_like(G)
    for k in range(nstep):
        G = G + alive * (gamma ** k) * rew[ts + k]
        alive = alive * (1 - done[ts + k])
    target = G + alive * (gamma ** nstep) * q_next[ts + nstep]
    return h_fn(target) if rescale else target


def train_dqn(args, log, core):
    if args.paper:
        # Paper-faithful settings: R2D2 (seq 80, burn-in 40, overlap 40, prioritized replay, value
        # rescaling, batch 64 -> 32 here for compute), DQN/Arnold (replay ~1M steps, epsilon annealed
        # 1.0 -> 0.1 over 1M steps, learning starts 50k), target sync every 2500 updates.
        L, BURN, NSTEP, OVERLAP, B = 80, 40, 5, 40, 32
        cap_steps, learn_start, tgt_every = 1_000_000, 50_000, 2500
        # replay ratio ~8 (Arnold): B*(L-BURN-NSTEP)=1120 trained transitions per update -> one
        # update per ~140 env steps = every 9 vector steps of 16 envs (ratio 1120/144 = 7.8)
        update_every = 9
    else:
        L, BURN, NSTEP, OVERLAP, B = 40, 8, 5, 20, 16
        cap_steps, learn_start, tgt_every = 100_000, 20_000, 1000
        update_every = 4
    if args.learn_start is not None:
        learn_start = args.learn_start
    venv = ParallelShapedVecEnv(TRAIN_MAPS, seed=args.seed)
    net = Net(core, dueling=True).to(common.DEV)
    tgt = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, eps=1e-3)
    replay = SeqReplay(cap_steps // (L - OVERLAP), prioritized=args.paper)
    eps_i = 0.4 ** (1 + 7 * np.arange(venv.n) / (venv.n - 1))
    h = net.core.init_state(venv.n).to(common.DEV)
    start = np.ones(venv.n, np.float32)
    bufs = [[] for _ in range(venv.n)]
    steps, next_eval, updates, vstep = 0, 0, 0, 0
    while steps < args.steps:
        ob = venv.observe()
        h_before = h.clone()
        with torch.no_grad():
            q, _, h = net(add_time(to_torch(ob)), h, tt(start)[None])
        a = q[0].argmax(-1).cpu().numpy()
        if args.paper:
            eps_i = max(0.1, 1.0 - 0.9 * steps / 1_000_000)
        rnd = np.random.random(venv.n) < eps_i
        a[rnd] = np.random.randint(N_ACTIONS, size=rnd.sum())
        r, d = venv.step(a)
        for i in range(venv.n):
            small = {k: (v[i].astype(np.int8) if k in ("hit", "etype") else v[i]) for k, v in ob.items()}
            bufs[i].append((small, a[i], r[i], d[i], start[i], h_before[:, i:i + 1].cpu()))
            if len(bufs[i]) == L:
                s = bufs[i]
                replay.add({"obs": {k: np.stack([x[0][k] for x in s]) for k in s[0][0]},
                            "act": np.array([x[1] for x in s]), "rew": np.array([x[2] for x in s], np.float32),
                            "done": np.array([x[3] for x in s], np.float32),
                            "start": np.array([x[4] for x in s], np.float32), "h0": s[0][5]})
                bufs[i] = s[L - OVERLAP:]
        start = d.copy()
        steps += venv.n; vstep += 1
        if steps >= learn_start and vstep % update_every == 0:
            obs, act, rew, done, st, h0, isw = replay.sample(B)
            q_on, _, _ = net(obs, h0, st)
            ts = torch.arange(BURN, L - NSTEP, device=common.DEV)
            with torch.no_grad():
                q_tg, _, _ = tgt(obs, h0, st)
                target = nstep_double_q_target(q_on, q_tg, rew, done, ts, NSTEP, GAMMA, rescale=args.paper)
            pred = q_on[ts].gather(-1, act[ts][..., None])[..., 0]
            td = pred - target                                                      # (T_train, B)
            if args.paper:   # Acme R2D2: 0.5 * sum_t td^2 per sequence, IS-weighted batch mean
                loss = (isw * 0.5 * td.pow(2).sum(0)).mean()
            else:
                loss = (nn.functional.smooth_l1_loss(pred, target, reduction="none") * isw[None]).mean()
            replay.update(td.abs())
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
            updates += 1
            if updates % tgt_every == 0:
                tgt.load_state_dict(net.state_dict())
        next_eval = log(steps, venv, net, next_eval)
    venv.close()
    return net


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=("ppo", "drqn", "dqn"), required=True)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--eval_every", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--learn_start", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--paper", action="store_true", help="dqn/drqn: R2D2/DQN/Arnold paper settings")
    ap.add_argument("--sem", action="store_true", help="PPO only: simplicial embedding in the actor head")
    ap.add_argument("--sem_groups", type=int, default=128)
    ap.add_argument("--sem_dim", type=int, default=4)
    ap.add_argument("--sem_tau", type=float, default=1.0)
    args = ap.parse_args()
    assert not args.sem or args.algo == "ppo", "--sem is only implemented for --algo ppo"
    tag = args.algo + ("_sem" if args.sem else "") + ("_paper" if args.paper else "")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    common.set_device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"{tag}_s{args.seed}.jsonl", "w")
    t0 = time.time()
    train_eps: list[dict] = []
    last_print = [0]

    def log(steps, venv, net, next_eval):
        train_eps.extend(venv.pop_finished())
        if steps - last_print[0] >= 50_000:
            last_print[0] = steps
            rec = {"algo": tag, "steps": steps, "minutes": (time.time() - t0) / 60,
                   "train": summarize(train_eps[-64:])}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if steps >= next_eval:
            rec = {"algo": tag, "steps": steps, "minutes": (time.time() - t0) / 60,
                   "eval_unseen": evaluate(net, args.algo)}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
            return next_eval + args.eval_every
        return next_eval

    net = train_ppo(args, log) if args.algo == "ppo" else train_dqn(args, log, "gru" if args.algo == "drqn" else "stack4")
    rec = {"algo": tag, "steps": args.steps, "minutes": (time.time() - t0) / 60, "final": True,
           "eval_unseen": evaluate(net, args.algo)}
    print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n")
    torch.save(net.state_dict(), out / f"{tag}_s{args.seed}.pt")


if __name__ == "__main__":
    main()
