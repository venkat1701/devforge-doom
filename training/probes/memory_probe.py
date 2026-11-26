"""Probe H-mem: which state representation preserves game state on UNSEEN maps?

Supervised, offline (no RL), on explorer trajectories from collect.py. Agent-visible inputs only:
rays (distance + hit type), ego velocity, previous buttons. Labels from privileged_state.

Targets that need memory:
  revisit   current 64-unit cell was visited >= 30 steps ago (binary)
  home      start position in the current ego frame (regression, map units)
  door      last seen door hit point (not visible for >= 20 steps), in the ego frame (regression)

Arms (same encoder, same heads):
  gru        encoder -> GRU-256 -> heads, trained on the targets
  gru_pred   encoder -> GRU-256 trained ONLY to predict the next ray scan (world-model style,
             self-supervised), then frozen; linear heads fit on top (does prediction alone keep state?)
  gru_epi    GRU + causal attention over past GRU states (every 4th step), keys carry the
             dead-reckoned relative pose of each memory (episodic embedding buffer)
  gru_map    GRU + egocentric crops of a ray-built grid map (dead-reckoned pose, rule odometry)

    uv run --with torch==2.15.0 python training/probes/memory_probe.py <train dir> <test dir>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import odometry as O  # noqa: E402

THRUST = np.array([25 * 2048 / 65536, 24 * 2048 / 65536])
GRID, CELLM, CROP, MCH = 192, 48.0, 24, 5        # map: 192x192 cells of 48 units; 5 channels
EVERY = 4                                        # episodic memory write interval
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)


# ----------------------------------------------------------------------------- preprocessing
def dead_reckon(z):
    """Agent-side pose (x, y, theta) in the episode frame (start = origin, heading 0)."""
    turns = O.tic_turns(z["action"][:-1])
    est, _ = O.estimate_disp(z, turns, THRUST, 0.5, use_rays=False)
    d0, d1 = z["dist"][:-1] * 1024, z["dist"][1:] * 1024
    shift = turns.sum(1) / (360 / 64)
    idx = (np.arange(64)[None] + shift[:, None]) % 64
    lo = np.floor(idx).astype(int); fr = idx - lo; rows = np.arange(len(d1))[:, None]
    d1s = (1 - fr) * d1[rows, lo] + fr * d1[rows, (lo + 1) % 64]
    est = np.where((np.abs(d1s - d0).mean(1) < 0.25)[:, None], 0.0, est)
    th = np.radians(np.concatenate([[0], np.cumsum(turns.sum(1))]))
    c, s = np.cos(th[:-1]), np.sin(th[:-1])
    x = np.concatenate([[0], np.cumsum(est[:, 0] * c + est[:, 1] * s)])
    y = np.concatenate([[0], np.cumsum(est[:, 0] * s - est[:, 1] * c)])
    return np.stack([x, y, th], 1)


def ego(dx, dy, h):
    c, s = np.cos(h), np.sin(h)
    return np.stack([dx * c + dy * s, dx * s - dy * c], -1)


def targets(z):
    n = len(z["x"])
    h = np.radians(z["angle"])
    cell = np.stack([z["x"] // 64, z["y"] // 64], 1).astype(int)
    first = {}
    revisit = np.zeros(n, np.float32)
    for t in range(n):
        k = (cell[t, 0], cell[t, 1])
        if k in first and t - first[k] >= 30:
            revisit[t] = 1
        first.setdefault(k, t)
    home = ego(z["x"][0] - z["x"], z["y"][0] - z["y"], h).astype(np.float32)
    # last seen door hit point (world), valid if no door visible now and seen >= 20 steps ago
    door = np.zeros((n, 2), np.float32); door_ok = np.zeros(n, np.float32)
    last, last_t = None, -1
    ray_ang = np.arange(64) * (2 * np.pi / 64)
    for t in range(n):
        dm = np.isin(z["hit"][t], (2, 3))
        if dm.any():
            i = np.flatnonzero(dm)[np.argmin(z["dist"][t][dm])]
            r = z["dist"][t][i] * 1024
            last = (z["x"][t] + np.cos(h[t] + ray_ang[i]) * r, z["y"][t] + np.sin(h[t] + ray_ang[i]) * r)
            last_t = t
        elif last is not None and t - last_t >= 20:
            door[t] = ego(last[0] - z["x"][t], last[1] - z["y"][t], h[t]); door_ok[t] = 1
    return revisit, home, door, door_ok


def map_crops(z, pose):
    """Per step: egocentric local + coarse crops of the map built so far (uint8, (n, 2*MCH, CROP, CROP))."""
    n = len(z["x"])
    grid = np.zeros((MCH, GRID, GRID), np.uint8)
    ray_ang = np.arange(64) * (2 * np.pi / 64)
    out = np.zeros((n, 2 * MCH, CROP, CROP), np.uint8)
    u = (np.arange(CROP) - CROP / 2 + 0.5)
    gu, gv = np.meshgrid(u, u)                  # gv rows = forward, gu cols = right
    tocell = lambda wx, wy: ((wx / CELLM + GRID / 2).astype(int).clip(0, GRID - 1),
                             (wy / CELLM + GRID / 2).astype(int).clip(0, GRID - 1))
    for t in range(n):
        x, y, th = pose[t]
        ang = th + ray_ang; r = z["dist"][t] * 1024
        for k in np.linspace(0, 1, 12, endpoint=False):
            cx, cy = tocell(x + np.cos(ang) * r * k, y + np.sin(ang) * r * k)
            grid[0, cy, cx] = 255
        cx, cy = tocell(x + np.cos(ang) * r, y + np.sin(ang) * r)
        hit = z["hit"][t]
        for ch, types in ((1, (1,)), (2, (2,)), (3, (3,))):
            m = np.isin(hit, types) & (z["dist"][t] < 1)
            grid[ch, cy[m], cx[m]] = 255
        px, py = tocell(np.array([x]), np.array([y])); grid[4, py, px] = 255
        for j, scale in enumerate((1.0, 6.0)):          # local: 1 cell/px, coarse: 6 cells/px
            fwd, right = -gv[::-1] * scale * CELLM, gu * scale * CELLM   # top row = ahead
            wx = x + fwd * np.cos(th) + right * np.sin(th)
            wy = y + fwd * np.sin(th) - right * np.cos(th)
            cx, cy = tocell(wx, wy)
            out[t, j * MCH:(j + 1) * MCH] = grid[:, cy, cx]
    return out


def load(d: Path, with_map: bool):
    eps = []
    for f in sorted(d.glob("*.npz")):
        z = dict(np.load(f))
        if len(z["x"]) < 60:
            continue
        pose = dead_reckon(z)
        rev, home, door, door_ok = targets(z)
        prev = np.concatenate([np.zeros((1, 9)), z["action"][:-1]]).astype(np.float32)
        ep = {
            "dist": z["dist"].astype(np.float32), "hit": z["hit"].astype(np.int64),
            "vel": (z["vel"] / 8).astype(np.float32), "prev": prev,
            "rpose": pose.astype(np.float32),
            "revisit": rev, "home": home / 1024, "door": door / 1024, "door_ok": door_ok,
            "next_dist": np.concatenate([z["dist"][1:], z["dist"][-1:]]).astype(np.float32),
        }
        if with_map:
            ep["crops"] = map_crops(z, pose)
        eps.append(ep)
    return eps


# ----------------------------------------------------------------------------- model
class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        def c1(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2, padding_mode="circular"), nn.ReLU())
        self.rays = nn.Sequential(c1(7, 32), c1(32, 64), nn.Conv1d(64, 64, 4, stride=4), nn.ReLU(),
                                  nn.Flatten(), nn.Linear(64 * 16, 192), nn.ReLU())
        self.rest = nn.Sequential(nn.Linear(2 + 9, 64), nn.ReLU())

    def forward(self, b):
        T, B = b["dist"].shape[:2]
        hit = nn.functional.one_hot(b["hit"].reshape(-1, 64), 6).float().transpose(1, 2)
        r = self.rays(torch.cat([b["dist"].reshape(-1, 1, 64), hit], 1)).reshape(T, B, -1)
        return torch.cat([r, self.rest(torch.cat([b["vel"], b["prev"]], -1))], -1)   # (T, B, 256)


class MapEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2 * MCH, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 9, 128), nn.ReLU())

    def forward(self, crops):
        T, B = crops.shape[:2]
        return self.net(crops.reshape(T * B, *crops.shape[2:]).float() / 255).reshape(T, B, -1)


class Model(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.enc = Enc()
        extra = 128 if arm == "gru_map" else 0
        self.map = MapEnc() if arm == "gru_map" else None
        self.gru = nn.GRU(256 + extra, 256)
        feat = 256
        if arm == "gru_epi":
            self.q = nn.Linear(256, 64)
            self.k_h, self.k_r = nn.Linear(256, 64), nn.Linear(4, 64, bias=False)
            self.v_h, self.v_r = nn.Linear(256, 128), nn.Linear(4, 128, bias=False)
            feat = 256 + 128
        self.pred = nn.Linear(256, 64) if arm == "gru_pred" else None
        self.heads = nn.Linear(feat, 1 + 2 + 2)
        self.feat = feat

    def features(self, b):
        x = self.enc(b)
        if self.map is not None:
            x = torch.cat([x, self.map(b["crops"])], -1)
        h, _ = self.gru(x)                                         # (T, B, 256)
        if self.arm != "gru_epi":
            return h
        T, B, _ = h.shape
        mem_idx = torch.arange(0, T, EVERY, device=h.device)
        hm = h[mem_idx]                                            # (M, B, 256) stored embeddings
        kh, vh = self.k_h(hm), self.v_h(hm)                        # (M, B, 64), (M, B, 128)
        p = b["rpose"]                                             # (T, B, 3) agent-side pose
        q = self.q(h)                                              # (T, B, 64)
        reads = []
        for t0 in range(0, T, 128):                                # chunk over time to bound memory
            t1 = min(T, t0 + 128); tt_ = torch.arange(t0, t1, device=h.device)
            dx = p[mem_idx][None, :, :, 0] - p[t0:t1, None, :, 0]  # (Tc, M, B)
            dy = p[mem_idx][None, :, :, 1] - p[t0:t1, None, :, 1]
            th = p[t0:t1, None, :, 2]
            fx, fy = (dx * th.cos() + dy * th.sin()) / 1024, (dx * th.sin() - dy * th.cos()) / 1024
            age = (tt_[:, None] - mem_idx[None, :]).float()[..., None].expand(-1, -1, B) / 1000
            rel = torch.stack([fx, fy, (fx ** 2 + fy ** 2).sqrt(), age], -1)          # (Tc, M, B, 4)
            att = (q[t0:t1, None] * (kh[None] + self.k_r(rel))).sum(-1) / 8          # (Tc, M, B)
            causal = (mem_idx[None, :] < tt_[:, None])[..., None]                    # strictly past
            w = att.masked_fill(~causal, -1e9).softmax(1) * causal
            reads.append((w[..., None] * (vh[None] + self.v_r(rel))).sum(1))         # (Tc, B, 128)
        read = torch.cat(reads)
        return torch.cat([h, read], -1)

    def forward(self, b):
        f = self.features(b)
        return f, self.heads(f)


def batchify(eps, keys):
    T = max(len(e["dist"]) for e in eps)
    out = {}
    for k in keys:
        arrs = []
        for e in eps:
            a = e[k]; pad = T - len(a)
            arrs.append(np.concatenate([a, np.zeros((pad, *a.shape[1:]), a.dtype)]) if pad else a)
        out[k] = torch.from_numpy(np.stack(arrs, 1)).to(DEV)
    mask = np.zeros((T, len(eps)), np.float32)
    for i, e in enumerate(eps):
        mask[:len(e["dist"]), i] = 1
    out["mask"] = torch.from_numpy(mask).to(DEV)
    return out


def losses(out, b):
    m = b["mask"]
    rev = nn.functional.binary_cross_entropy_with_logits(out[..., 0], b["revisit"], reduction="none")
    home = (out[..., 1:3] - b["home"]).abs().sum(-1)
    door = (out[..., 3:5] - b["door"]).abs().sum(-1) * b["door_ok"]
    return ((rev + home + door) * m).sum() / m.sum()


KEYS = ["dist", "hit", "vel", "prev", "rpose", "revisit", "home", "door", "door_ok", "next_dist"]


def train(arm, tr, epochs, bs=8):
    model = Model(arm).to(DEV)
    keys = KEYS + (["crops"] if arm == "gru_map" else [])
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(epochs):
        perm = np.random.permutation(len(tr))
        for i in range(0, len(perm), bs):
            b = batchify([tr[j] for j in perm[i:i + bs]], keys)
            f, out = model(b)
            if arm == "gru_pred":     # self-supervised next-scan prediction only
                loss = ((model.pred(f) - b["next_dist"]).abs().mean(-1) * b["mask"]).sum() / b["mask"].sum()
            else:
                loss = losses(out, b)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if arm == "gru_pred":             # freeze, then fit linear heads on frozen features
        for p in model.parameters():
            p.requires_grad_(False)
        model.heads.weight.requires_grad_(True); model.heads.bias.requires_grad_(True)
        opt = torch.optim.Adam(model.heads.parameters(), 3e-3)
        with torch.no_grad():
            cache = []
            for i in range(0, len(tr), bs):
                b = batchify(tr[i:i + bs], keys); cache.append((model.features(b), b))
        for _ in range(60):
            for f, b in cache:
                loss = losses(model.heads(f), b)
                opt.zero_grad(); loss.backward(); opt.step()
    return model, keys


def evaluate(model, keys, te):
    res = {"rev_acc": [], "home_err": [], "door_err": []}
    buckets = {"0-250": (0, 250), "250-500": (250, 500), "500+": (500, 10_000)}
    per = {k: {m: [] for m in res} for k in buckets}
    with torch.no_grad():
        for e in te:
            b = batchify([e], keys)
            _, out = model(b)
            out = out[:, 0].cpu().numpy()
            n = len(e["dist"]); t = np.arange(n)
            rev_ok = ((out[:, 0] > 0) == (e["revisit"] > 0.5)).astype(float)
            home_err = np.linalg.norm(out[:, 1:3] - e["home"], axis=1) * 1024
            door_err = np.where(e["door_ok"] > 0, np.linalg.norm(out[:, 3:5] - e["door"], axis=1) * 1024, np.nan)
            for name, (lo, hi) in buckets.items():
                sel = (t >= lo) & (t < hi)
                if sel.any():
                    # balanced revisit accuracy within bucket
                    pos, neg = sel & (e["revisit"] > 0.5), sel & (e["revisit"] < 0.5)
                    if pos.any() and neg.any():
                        per[name]["rev_acc"].append(0.5 * (rev_ok[pos].mean() + rev_ok[neg].mean()))
                    per[name]["home_err"].append(np.nanmedian(home_err[sel]))
                    if np.isfinite(door_err[sel]).any():
                        per[name]["door_err"].append(np.nanmedian(door_err[sel]))
    return {bk: {m: (float(np.mean(v)) if v else float("nan")) for m, v in d.items()} for bk, d in per.items()}


def main() -> None:
    t0 = time.time()
    tr = load(Path(sys.argv[1]), with_map=True)
    te = load(Path(sys.argv[2]), with_map=True)
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    print(f"train eps {len(tr)} ({sum(len(e['dist']) for e in tr):,} steps, MAP01-16)  "
          f"test eps {len(te)} (MAP17-24 unseen)  prep {time.time() - t0:.0f}s")
    rate = np.mean(np.concatenate([e["revisit"] for e in te]))
    print(f"test revisit base rate {rate:.2f}; door-recall labels on "
          f"{np.mean(np.concatenate([e['door_ok'] for e in te])):.0%} of steps")
    print("\n   arm        | revisit balanced acc       | home error (units, median)  | door recall error (units)")
    print("              |  0-250   250-500  500+     |  0-250  250-500   500+      |  0-250  250-500   500+")
    for arm in ("gru", "gru_pred", "gru_epi", "gru_map"):
        t1 = time.time()
        model, keys = train(arm, tr, epochs)
        r = evaluate(model, keys, te)
        f = lambda m, fmt: "  ".join(fmt.format(r[b][m]) for b in ("0-250", "250-500", "500+"))
        print(f"   {arm:10s} |  {f('rev_acc', '{:5.2f} ')}    |  {f('home_err', '{:6.0f}')}     |  "
              f"{f('door_err', '{:6.0f}')}   ({time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
