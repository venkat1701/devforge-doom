"""Recurrent DQN (R2D2-style learner) for the PPO vs DRQN decision.

Learner = Acme R2D2 loss, read verbatim (training/docs/REFERENCES.md, item 10):
  * replayed sequences start from the STORED recurrent state; burn-in unrolls online and target nets
    from it; gradients flow through the online burn-in (Acme applies no stop_gradient there)
  * n-step double-Q target with value rescaling (rlax transformed_n_step_q_learning semantics);
    tested against an rlax port in training/probes/test_targets.py
  * loss per sequence = 0.5 * sum_t td^2; importance weights (1/p)^beta normalised by the batch max
  * priorities = 0.9 * max|td| + 0.1 * mean|td| (max_priority_weight 0.9), priority exponent 0.9
Acting/replay from Arnold (Lample & Chaplot 2017, the ViZDoom DRQN): replay of ~1M steps, gamma 0.99,
experiences sampled ~8 times on average. Exploration: --epsilon arnold (1 -> 0.1 linearly over the first
1M steps, then 0.1) or --epsilon apex (fixed per-actor epsilons, as in Ape-X / R2D2 distributed actors;
formula marked in DESIGN.md as needing source confirmation).

Not reproduced (stated so nobody mistakes this for the full papers): Arnold's separate navigation network
and its enemy-presence auxiliary loss (our observation already contains entities); R2D2's 256 actors.
Episodic memory is PPO-only (replay would need per-sequence banks).
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import numpy as np
import torch

from doom_model import AgentNet, ModelConfig

from .remote_envs import make_envs
from .envs import ProcEnvs, ram_stats, stack_feats
from .targets import nstep_double_q_target


STORE_DTYPE = {"dist": np.float32, "hit": np.int8, "ent": np.float32, "etype": np.int8, "emask": np.uint8,
               "vit": np.float32, "prev": np.uint8, "crops": np.uint8, "pose": np.float32}


def total_ram_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


class RingSeqReplay:
    """Prioritised sequence replay that stores every step ONCE (review fix: overlapping sequences used to
    duplicate every step and keep per-sequence Python objects, ~14 GiB at 1M steps with map crops).

    Per env: a ring of `cap` steps in preallocated arrays. Sequence k of env i = steps [k*P, k*P+L) of that
    env's stream (P = period = L - overlap, as Acme's `sequence_period`), with the recurrent state stored
    at its start (h0[i, k % nseq]). A sequence is sampleable once fully written and is invalidated (priority
    0) as soon as any of its steps is overwritten. Priorities/IS weights as in Acme R2D2 (see module doc)."""

    def __init__(self, n_envs, capacity_steps, L, P, hidden, example: dict, alpha, beta, eta,
                 ram_fraction=0.4):
        self.n, self.L, self.P = n_envs, L, P
        self.cap = max(L, (capacity_steps // n_envs) // P * P)
        self.nseq = self.cap // P
        self.alpha, self.beta, self.eta = alpha, beta, eta
        per_step = sum(np.dtype(STORE_DTYPE[k]).itemsize * int(np.prod(v.shape[1:])) for k, v in example.items())
        per_step += 1 + 4 + 1 + 1                                     # act, rew, done, start
        total = n_envs * self.cap * per_step + n_envs * self.nseq * hidden * 4
        budget = ram_fraction * total_ram_bytes()
        print(f"[replay] {n_envs} envs x {self.cap:,} steps x {per_step:,} B/step = {total / 2**30:.2f} GiB "
              f"(budget {budget / 2**30:.1f} GiB = {ram_fraction:.0%} of RAM)", flush=True)
        if total > budget:
            raise MemoryError(f"replay needs {total / 2**30:.1f} GiB > budget {budget / 2**30:.1f} GiB; "
                              "lower drqn.capacity_steps or raise drqn.replay_ram_fraction")
        self.obs = {k: np.zeros((n_envs, self.cap, *v.shape[1:]), STORE_DTYPE[k]) for k, v in example.items()}
        self.act = np.zeros((n_envs, self.cap), np.uint8)
        self.rew = np.zeros((n_envs, self.cap), np.float32)
        self.done = np.zeros((n_envs, self.cap), np.uint8)
        self.start = np.zeros((n_envs, self.cap), np.uint8)
        self.h0 = np.zeros((n_envs, self.nseq, hidden), np.float32)
        self.w = np.zeros(n_envs, np.int64)                           # steps written per env
        self.prio = np.zeros((n_envs, self.nseq), np.float64)
        self.slot_k = np.full((n_envs, self.nseq), -1, np.int64)
        self.max_prio = 1.0
        self.last = None

    def add_step(self, i, obs_i: dict, a, r, d, st, h_before: np.ndarray):
        s = int(self.w[i]); pos = s % self.cap
        if s >= self.cap:                                             # invalidate sequences losing a step
            lo, hi = s - self.cap - self.L + 1, s - self.cap
            for k in range(max(0, -(-lo // self.P)), hi // self.P + 1):
                slot = k % self.nseq
                if self.slot_k[i, slot] == k:
                    self.prio[i, slot] = 0.0; self.slot_k[i, slot] = -1
        for k, v in obs_i.items():
            self.obs[k][i, pos] = v
        self.act[i, pos], self.rew[i, pos], self.done[i, pos], self.start[i, pos] = a, r, d, st
        if s % self.P == 0:
            self.h0[i, (s // self.P) % self.nseq] = h_before
        self.w[i] = s + 1
        s0 = s + 1 - self.L                                           # a sequence just completed?
        if s0 >= 0 and s0 % self.P == 0:
            k = s0 // self.P; slot = k % self.nseq
            self.slot_k[i, slot] = k; self.prio[i, slot] = self.max_prio

    def __len__(self):
        return int((self.prio > 0).sum())

    def sample(self, b, dev):
        flat = self.prio.ravel() ** self.alpha
        p = flat / flat.sum()
        idx = np.random.choice(len(flat), size=b, p=p)
        w = (1.0 / (p[idx] + 1e-6)) ** self.beta                       # Acme: (1/p)^beta, / max
        w /= w.max()
        env, slot = idx // self.nseq, idx % self.nseq
        s0 = self.slot_k[env, slot] * self.P
        pos = (s0[None, :] + np.arange(self.L)[:, None]) % self.cap       # (L, B)
        self.last = (env, slot)
        e = env[None, :]
        obs = {}
        for k, arr in self.obs.items():
            x = torch.as_tensor(arr[e, pos], device=dev)
            obs[k] = x.long() if k in ("hit", "etype", "prev") else (x if k == "crops" else x.float())
        t = lambda a, dt: torch.as_tensor(a[e, pos], dtype=dt, device=dev)  # noqa: E731
        h0 = torch.as_tensor(self.h0[env, (s0 // self.P) % self.nseq], device=dev)[None]
        return (obs, t(self.act, torch.long), t(self.rew, torch.float32), t(self.done, torch.float32),
                t(self.start, torch.float32), h0, torch.as_tensor(w, dtype=torch.float32, device=dev))

    def update(self, td_abs):
        pr = (self.eta * td_abs.max(0).values + (1 - self.eta) * td_abs.mean(0)).detach().cpu().numpy() + 1e-6
        env, slot = self.last
        self.prio[env, slot] = pr
        self.max_prio = max(self.max_prio, float(pr.max()))


def train(cfg: dict, out_dir: Path, log) -> AgentNet:
    dev = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    q = cfg["drqn"]
    mcfg = ModelConfig(**cfg["model"], head="q", privileged_dim=0)
    assert not mcfg.use_episodic, "episodic memory is implemented for PPO only"
    net = AgentNet(mcfg).to(dev)
    tgt = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=q["lr"], eps=q.get("adam_eps", 1e-3))
    N = cfg["num_envs"]
    L, BURN, NSTEP, OVERLAP, B = q["seq_len"], q["burn_in"], q["n_step"], q["overlap"], q["batch_seqs"]
    trained = L - BURN - NSTEP
    # replay ratio: each stored step is trained on ~q["replay_ratio"] times (Arnold: 8)
    update_every = max(1, round(B * trained / (q["replay_ratio"] * N)))
    envs = make_envs(cfg, N, mcfg.use_map)      # local process pool or a remote env server
    outs = envs.reset()
    replay = RingSeqReplay(N, q["capacity_steps"], L, L - OVERLAP, mcfg.hidden, stack_feats(outs),
                           q["priority_exponent"], q["is_exponent"], q["max_priority_weight"],
                           q.get("replay_ram_fraction", 0.4))
    apex_eps = 0.4 ** (1 + 7 * np.arange(N) / max(1, N - 1))
    h = net.init_state(N, dev)
    start = np.ones(N, np.float32)
    steps, vstep, updates, t0 = 0, 0, 0, time.time()
    while steps < cfg["total_steps"]:
        f = stack_feats(outs)
        o_t = {k: torch.as_tensor(v, device=dev)[None] for k, v in f.items()}
        o_t = {k: (v.long() if k in ("hit", "etype", "prev") else v if v.dtype == torch.uint8 else v.float())
               for k, v in o_t.items()}
        h_before = h.clone()
        with torch.no_grad():
            out = net(o_t, h, torch.as_tensor(start, device=dev)[None])
        h = out["h"]
        a = out["q"][0].argmax(-1).cpu().numpy()
        eps = (np.full(N, max(0.1, 1.0 - 0.9 * steps / 1_000_000)) if q["epsilon"] == "arnold" else apex_eps)
        rnd = np.random.random(N) < eps
        a[rnd] = np.random.randint(out["q"].shape[-1], size=rnd.sum())
        outs = envs.step(a)
        r = np.array([o["r"] for o in outs], np.float32)
        d = np.array([o["done"] for o in outs], np.float32)
        for o in outs:
            if "episode" in o:
                log.episode(o["episode"], steps)
        hb = h_before[0].cpu().numpy()
        for i in range(N):
            replay.add_step(i, {k: v[i] for k, v in f.items()}, a[i], r[i], d[i], start[i], hb[i])
        start = d.copy()
        steps += N; vstep += 1
        if vstep % 64 == 0:
            envs.set_global_step(steps)
        if steps >= q["learn_start"] and len(replay) >= B and vstep % update_every == 0:
            obs, act, rew, done, st, h0, isw = replay.sample(B, dev)
            q_on = net(obs, h0, st)["q"]
            ts = torch.arange(BURN, L - NSTEP, device=dev)
            with torch.no_grad():
                q_tg = tgt(obs, h0, st)["q"]
                target = nstep_double_q_target(q_on, q_tg, rew, done, ts, NSTEP, q["gamma"], rescale=q["rescale"])
            pred = q_on[ts].gather(-1, act[ts][..., None])[..., 0]
            td = pred - target
            loss = (isw * 0.5 * td.pow(2).sum(0)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), q.get("max_grad_norm", 10.0)); opt.step()
            replay.update(td.abs())
            updates += 1
            if updates % q["target_update"] == 0:
                tgt.load_state_dict(net.state_dict())
            if updates % 200 == 0 or updates == 1:
                log.update(steps, {"loss": loss.item(), "eps": float(np.mean(eps)), "sps": steps / (time.time() - t0),
                                   "replay_seqs": len(replay), **ram_stats(envs)})
        if steps % cfg.get("save_every_steps", 250_000) < N:
            save(net, mcfg, out_dir, steps)
    envs.close()
    save(net, mcfg, out_dir, steps)
    return net


def save(net, mcfg, out_dir: Path, steps: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items() if not k.startswith("aux.")}
    ModelConfig(**{**mcfg.__dict__, "aux": False}).save(out_dir / "model_config.json")
    torch.save(state, out_dir / "model.pt")
    torch.save(state, out_dir / f"model_{steps}.pt")
