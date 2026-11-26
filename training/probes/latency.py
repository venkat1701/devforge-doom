"""Probe H-lat: the full proposed agent fits the 40 ms/step budget on ONE CPU thread.

Times a forward pass of the candidate architecture (untrained weights, realistic shapes), including
the numpy map write and egocentric crop, at batch 1 with torch.set_num_threads(1), as at evaluation.

    uv run --with torch==2.15.0 python training/probes/latency.py
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

torch.set_num_threads(1)

GRID, CELL, CROP, CH = 128, 64.0, 32, 8
RAY_ANGLES = np.arange(64) * (2 * np.pi / 64)


class Agent(nn.Module):
    def __init__(self, hidden=256, slots=64):
        super().__init__()
        def c1(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2, padding_mode="circular"), nn.ReLU())
        self.rays = nn.Sequential(c1(7, 32), c1(32, 64), c1(64, 64), nn.AdaptiveAvgPool1d(4), nn.Flatten())
        self.ent_type = nn.Embedding(12, 8)
        self.ent = nn.Sequential(nn.Linear(13, 64), nn.ReLU(), nn.Linear(64, 64))
        self.ent_attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.vit = nn.Sequential(nn.Linear(17 + 9, 64), nn.ReLU())
        def c2(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), nn.ReLU())
        self.map = nn.Sequential(c2(2 * CH, 32, 2), c2(32, 64, 2), c2(64, 64, 2), nn.Flatten(),
                                 nn.Linear(64 * 4 * 4, 256), nn.ReLU())
        self.gru = nn.GRUCell(256 + 64 + 64 + 256 + 64, hidden)
        self.q = nn.Linear(hidden, 64)
        self.mem_attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.pi = nn.Linear(hidden + 64, 3 + 3 + 3 + 2 + 2 + 2)
        self.v = nn.Linear(hidden + 64, 1)
        self.mem = torch.zeros(1, slots, 64)
        self.prev_read = torch.zeros(1, 64)

    def forward(self, rays, ents, etype, emask, vit, crops, h):
        r = self.rays(rays)
        e = self.ent(torch.cat([ents, self.ent_type(etype)], -1))
        pooled, _ = self.ent_attn(e.mean(1, keepdim=True), e, e, key_padding_mask=~emask)
        x = torch.cat([r, pooled[:, 0], self.vit(vit), self.map(crops), self.prev_read], 1)
        h = self.gru(x, h)
        read, _ = self.mem_attn(self.q(h)[:, None], self.mem, self.mem)
        self.prev_read = read[:, 0]
        z = torch.cat([h, read[:, 0]], 1)
        return self.pi(z), self.v(z), h


def write_map(grid, pose, dist, hit):
    """Trace 64 rays into the allocentric grid (numpy, no learning)."""
    x, y, th = pose
    ang = th + RAY_ANGLES
    r = dist * 1024.0
    for k in np.linspace(0.0, 1.0, 16, endpoint=False):     # free space along each ray
        gx = ((x + np.cos(ang) * r * k) / CELL + GRID / 2).astype(int) % GRID
        gy = ((y + np.sin(ang) * r * k) / CELL + GRID / 2).astype(int) % GRID
        grid[0, gy, gx] = 1.0
    gx = ((x + np.cos(ang) * r) / CELL + GRID / 2).astype(int) % GRID
    gy = ((y + np.sin(ang) * r) / CELL + GRID / 2).astype(int) % GRID
    hitm = dist < 1.0
    for t in range(1, 6):
        m = hitm & (hit == t)
        grid[t, gy[m], gx[m]] = 1.0
    grid[6, int(y / CELL + GRID / 2) % GRID, int(x / CELL + GRID / 2) % GRID] = 1.0


def crops(grid_t, pose):
    """Egocentric rotated crops: fine local (CROP cells) and coarse whole-grid, via grid_sample."""
    x, y, th = pose
    cx, cy = x / CELL / (GRID / 2), y / CELL / (GRID / 2)
    out = []
    for scale in (CROP / GRID, 1.0):
        c, s = np.cos(th - np.pi / 2), np.sin(th - np.pi / 2)
        theta = torch.tensor([[[c * scale, -s * scale, cx], [s * scale, c * scale, cy]]], dtype=torch.float32)
        g = Fn.affine_grid(theta, (1, CH, CROP, CROP), align_corners=False)
        out.append(Fn.grid_sample(grid_t, g, align_corners=False))
    return torch.cat(out, 1)


def main() -> None:
    rng = np.random.default_rng(0)
    net = Agent().eval()
    grid = np.zeros((CH, GRID, GRID), np.float32)
    h = torch.zeros(1, 256)
    n_params = sum(p.numel() for p in net.parameters())
    times = {"map write": [], "crops": [], "network": [], "total": []}
    with torch.no_grad():
        for step in range(600):
            dist = rng.random(64).astype(np.float32)
            hit = rng.integers(0, 6, 64)
            pose = (rng.normal() * 500, rng.normal() * 500, rng.random() * 6.28)
            t0 = time.perf_counter()
            write_map(grid, pose, dist, hit)
            t1 = time.perf_counter()
            cr = crops(torch.from_numpy(grid)[None], pose)
            t2 = time.perf_counter()
            rays = torch.from_numpy(np.concatenate([dist[None], np.eye(6, dtype=np.float32)[hit].T], 0))[None]
            ents = torch.randn(1, 32, 5)
            et = torch.randint(0, 12, (1, 32))
            em = torch.zeros(1, 32, dtype=torch.bool); em[0, :5] = True
            vit = torch.randn(1, 26)
            logits, v, h = net(rays, ents, et, em, vit, cr, h)
            t3 = time.perf_counter()
            if step >= 100:   # skip warm-up
                times["map write"].append(t1 - t0); times["crops"].append(t2 - t1)
                times["network"].append(t3 - t2); times["total"].append(t3 - t0)
    print(f"H-lat  candidate agent, {n_params:,} params, 1 CPU thread, batch 1 (500 steps after warm-up)")
    for k, v in times.items():
        v = np.array(v) * 1000
        print(f"   {k:10s} mean {v.mean():6.3f} ms   p99 {np.percentile(v, 99):6.3f} ms")
    print(f"   budget: 40 ms/step -> headroom x{40 / (np.mean(times['total']) * 1000):.0f}")
    print(f"   weights fp32: {n_params * 4 / 1e6:.1f} MB (limit: 1 GB repo, 2 GB VRAM)")


if __name__ == "__main__":
    main()
