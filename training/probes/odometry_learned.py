"""Probe H-odo-L: a small learned odometry net beats hand rules on UNSEEN maps.

Inputs are agent-visible only (vitals velocity, own previous action, ray scans at t and t+1).
Heading change comes from the exact Doom turn schedule (H-odo-1); the net predicts the ego-frame
displacement. Labels come from privileged_state (training only). Train on MAP01-16, test on 17-24.

    uv run --with torch==2.15.0 python training/probes/odometry_learned.py <train dir> <test dir>
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
STATIC = (1, 2, 3)   # wall, door, locked door: hits that don't move on their own (doors mostly)


def step_inputs(z):
    turns = O.tic_turns(z["action"][:-1])
    n = len(turns)
    d0 = z["dist"][:-1]
    shift = turns.sum(1) / (360.0 / 64)
    idx = (np.arange(64)[None, :] + shift[:, None]) % 64
    lo = np.floor(idx).astype(int); fr = idx - lo
    rows = np.arange(n)[:, None]
    d1 = z["dist"][1:]
    d1s = (1 - fr) * d1[rows, lo] + fr * d1[rows, (lo + 1) % 64]
    h0 = np.isin(z["hit"][:-1], STATIC).astype(np.float32)
    h1 = np.isin(z["hit"][1:], STATIC)[rows, np.rint(idx).astype(int) % 64].astype(np.float32)
    rays = np.stack([d0, d1s, (d1s - d0) * 16.0, h0, h1], 1).astype(np.float32)   # (n, 5, 64)

    a = z["action"][:-1].astype(np.float32)
    tot = np.radians(turns.sum(1))
    vn = z["vel"][1:]
    c, s = np.cos(tot), np.sin(tot)
    vn_t = np.stack([vn[:, 0] * c + vn[:, 1] * s, -vn[:, 0] * s + vn[:, 1] * c], 1)
    free = O.physics_features(z, turns) @ np.array([1.0, *THRUST])
    still = np.abs(d1s - d0).mean(1) * 1024.0 < 0.25
    free = np.where(still[:, None], 0.0, free)          # base = physics + ray-stillness rule
    scal = np.concatenate([
        z["vel"][:-1] / 8.0, vn_t / 8.0, a[:, :4], turns / 3.6, free / 32.0,
    ], 1).astype(np.float32)                                                          # (n, 14)
    return rays, scal, free, turns


def dataset(d: Path):
    R, S, Y, eps = [], [], [], []
    for f in sorted(d.glob("*.npz")):
        z = dict(np.load(f)); z["name"] = f.stem
        if len(z["x"]) < 3:
            continue
        rays, scal, free, turns = step_inputs(z)
        y = O.truth(z)[1].astype(np.float32)
        # drop the death step (heading/position of a dying player are not meaningful)
        keep = np.ones(len(y), bool)
        if z["health"][-1] <= 0:
            keep[-1] = False
        R.append(rays[keep]); S.append(scal[keep]); Y.append(y[keep])
        eps.append((z, rays, scal, free, turns))
    return np.concatenate(R), np.concatenate(S), np.concatenate(Y), eps


class OdoNet(nn.Module):
    """Circular 1D conv over the two aligned ray scans + MLP over velocity/buttons -> (fwd, right)."""

    def __init__(self):
        super().__init__()
        def conv(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2, padding_mode="circular"), nn.ReLU())
        self.rays = nn.Sequential(conv(5, 32), conv(32, 32), nn.AdaptiveAvgPool1d(8), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(32 * 8 + 14, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                  nn.Linear(64, 3))

    def forward(self, rays, scal, free):
        out = self.head(torch.cat([self.rays(rays), scal], 1))
        gate = 2.0 * torch.sigmoid(out[:, :1])    # scale on the rule estimate, 1.0 at init
        return gate * free + out[:, 1:]             # plus a learned residual


def main() -> None:
    torch.manual_seed(0)
    Rtr, Str, Ytr, _ = dataset(Path(sys.argv[1]))
    _, _, _, test_eps = dataset(Path(sys.argv[2]))
    print(f"train steps: {len(Ytr):,}  (maps MAP01-16)   test episodes: {len(test_eps)} (maps MAP17-24, unseen)")

    # physics estimate is part of the scalar input (cols 10:12, scaled by 1/32)
    free_tr = Str[:, 10:12] * 32.0
    net = OdoNet()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    Rt, St, Ft, Yt = map(torch.from_numpy, (Rtr, Str, free_tr.astype(np.float32), Ytr))
    t0 = time.time()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40)
    for epoch in range(40):
        perm = torch.randperm(len(Yt))
        tot = 0.0
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = nn.functional.smooth_l1_loss(net(Rt[b], St[b], Ft[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if epoch % 10 == 9:
            print(f"   epoch {epoch + 1:2d}  train loss {tot / len(Yt):.4f}")
    print(f"   trained in {time.time() - t0:.0f}s, params={sum(p.numel() for p in net.parameters()):,}")
    net.eval()

    methods = {}
    print("\nPer-step |displacement error| (map units/step) on unseen maps, and integrated drift")
    rows = {k: {"step": [], 100: [], 250: [], 500: [], 1000: []} for k in ("physics", "physics+rays", "learned")}
    for z, rays, scal, free, turns in test_eps:
        y = O.truth(z)[1]
        n = len(y)
        est_phys, _ = O.estimate_disp(z, turns, THRUST, 0.5, use_rays=False)
        with torch.no_grad():
            est_l = net(torch.from_numpy(rays), torch.from_numpy(scal),
                        torch.from_numpy(free.astype(np.float32))).numpy()
        # physics+rays with the mean residual variant (first version of the rule)
        d0, d1 = z["dist"][:-1] * 1024, rays[:, 1] * 1024
        still = np.abs(d1 - d0).mean(1) < 0.25
        est_pr = np.where(still[:, None], 0.0, est_phys)
        for k, est in (("physics", est_phys), ("physics+rays", est_pr), ("learned", est_l)):
            e = np.linalg.norm(est - y, axis=1)
            if z["health"][-1] <= 0:
                e = e[:-1]; est = est[:-1]
            rows[k]["step"].append(e)
            th = np.radians(z["angle"][0] + np.concatenate([[0], np.cumsum(turns.sum(1))]))[:len(est) + 1]
            c, s = np.cos(th[:-1]), np.sin(th[:-1])
            px = z["x"][0] + np.concatenate([[0], np.cumsum(est[:, 0] * c + est[:, 1] * s)])
            py = z["y"][0] + np.concatenate([[0], np.cumsum(est[:, 0] * s - est[:, 1] * c)])
            perr = np.hypot(px - z["x"][:len(px)], py - z["y"][:len(py)])
            for h in (100, 250, 500, 1000):
                if len(perr) > h:
                    rows[k][h].append(perr[h])
    print("   method          mean/step   drift median (max) @100      @250        @500        @1000   [n eps]")
    for k, r in rows.items():
        e = np.concatenate(r["step"])
        cells = "  ".join(f"{np.median(r[h]):5.0f} ({np.max(r[h]):4.0f})" if r[h] else "     -     " for h in (100, 250, 500, 1000))
        print(f"   {k:14s}  {e.mean():8.3f}    {cells}   [{', '.join(str(len(r[h])) for h in (100, 250, 500, 1000))}]")

    # inference cost on one CPU thread (the evaluation budget is 40 ms per step for everything)
    torch.set_num_threads(1)
    r1, s1, f1 = (torch.from_numpy(x[:1]) for x in (rays, scal, free.astype(np.float32)))
    with torch.no_grad():
        for _ in range(20):
            net(r1, s1, f1)
        t0 = time.perf_counter()
        for _ in range(500):
            net(r1, s1, f1)
    print(f"\n   OdoNet latency, 1 CPU thread, batch 1: {(time.perf_counter() - t0) / 500 * 1000:.3f} ms")


if __name__ == "__main__":
    main()
