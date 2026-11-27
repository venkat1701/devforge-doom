"""Procedural small Doom maps (UDMF, zdoom namespace) for training and held-out evaluation.

Why: the hidden evaluation maps have 2,100-4,200 tic caps (60-120 s) and four tiers (combat; multi-room
exploration with doors; keycard -> locked door -> exit; full gauntlet). Full Freedoom levels cannot be
finished in that time, so an agent trained only on them almost never experiences an exit or a key.
These maps follow the conventions of the template's maps/two_rooms.wad (Door_Raise special 12 with
tag 0 on both faces of a door sector, BIGDOOR2/DOORTRAK textures, a GATE3 exit pad ringed by
walk-over Exit_Normal (243) lines), so the wrapper's door/lock parser (env/doom_env/mapinfo.py) reads them.

Layout: rooms on a grid joined by a random spanning tree. Each connection is corridor | door | corridor
(or an open corridor). The exit room is the room farthest from the start. A locked door
(Door_LockedRaise, special 13) sits on the start->exit path and its keycard is placed in a room reachable
without crossing it, preferably off the path, so the agent must explore a branch and come back.
ZDoom builds nodes itself for UDMF, so no node builder is needed.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import struct

G = 576                     # grid pitch (map units)
WALLS = ["STARTAN2", "STONE2", "BROWN1", "GRAY1", "BRICK1", "TEKGREN2", "STARG3", "COMPBLUE", "ICKWALL3"]
FLOORS = ["FLOOR0_1", "FLOOR4_8", "FLOOR5_1", "FLAT14", "FLAT5_4", "FLOOR7_1", "MFLR8_1"]
CEILS = ["CEIL1_1", "CEIL3_5", "FLAT1", "CEIL5_1"]
MONSTERS = {"zombie": 3004, "shotgun": 9, "imp": 3001, "demon": 3002}
ITEMS = [2011, 2012, 2007, 2008, 2014, 2015]   # stimpack, medikit, clip, shells, health bonus, armor bonus
KEYS = [(13, 1), (5, 2), (6, 3)]               # (keycard thing, ZDoom lock number): red, blue, yellow
TIERS = {1: "combat", 2: "multiroom", 3: "keydoor", 4: "gauntlet"}


@dataclass
class Room:
    gx: int
    gy: int
    x0: float = 0
    x1: float = 0
    y0: float = 0
    y1: float = 0
    floor: str = "FLOOR0_1"
    ceil_h: int = 128
    sector: int = -1
    openings: dict = field(default_factory=lambda: {"L": [], "R": [], "D": [], "U": []})


class Builder:
    def __init__(self):
        self.verts, self.vindex, self.sectors, self.sides, self.lines, self.things = [], {}, [], [], [], []

    def v(self, x, y):
        k = (float(x), float(y))
        if k not in self.vindex:
            self.vindex[k] = len(self.verts)
            self.verts.append(k)
        return self.vindex[k]

    def sector(self, floor, ceil, hf, hc, light):
        self.sectors.append(dict(texturefloor=floor, textureceiling=ceil, heightfloor=hf, heightceiling=hc, lightlevel=light))
        return len(self.sectors) - 1

    def side(self, sector, top="-", mid="-", bottom="-"):
        self.sides.append(dict(sector=sector, texturetop=top, texturemiddle=mid, texturebottom=bottom))
        return len(self.sides) - 1

    def edge(self, a, b, front, front_dir, back=None, front_tex=None, back_tex=None, **extra):
        """Axis-aligned segment a-b. front_dir: side of the segment where the FRONT sector lies
        ('L','R' for vertical segments; 'D','U' for horizontal). Doom's front side is the right of v1->v2."""
        (ax, ay), (bx, by) = a, b
        if ax == bx:                       # vertical: going up (+y) has +x on its right
            lo, hi = sorted([ay, by])
            p1, p2 = ((ax, lo), (ax, hi)) if front_dir == "R" else ((ax, hi), (ax, lo))
        else:                              # horizontal: going +x has -y on its right
            lo, hi = sorted([ax, bx])
            p1, p2 = ((lo, ay), (hi, ay)) if front_dir == "D" else ((hi, ay), (lo, ay))
        ft = front_tex or {}
        line = dict(v1=self.v(*p1), v2=self.v(*p2), sidefront=self.side(front, **ft))
        if back is not None:
            line["sideback"] = self.side(back, **(back_tex or {}))
            line["twosided"] = True
        else:
            line["blocking"] = True
        line.update(extra)
        self.lines.append(line)

    def thing(self, x, y, type_, angle=0):
        self.things.append(dict(x=float(x), y=float(y), angle=angle, type=type_, skill1=True, skill2=True,
                                skill3=True, skill4=True, skill5=True, single=True))

    def textmap(self):
        def block(name, d):
            items = []
            for k, val in d.items():
                if isinstance(val, bool):
                    items.append(f"{k} = {'true' if val else 'false'};")
                elif isinstance(val, str):
                    items.append(f'{k} = "{val}";')
                else:
                    items.append(f"{k} = {val};")
            return f"{name} {{ " + " ".join(items) + " }"
        out = ['namespace = "zdoom";']
        out += [block("vertex", dict(x=x, y=y)) for x, y in self.verts]
        out += [block("sector", s) for s in self.sectors]
        out += [block("sidedef", s) for s in self.sides]
        out += [block("linedef", l) for l in self.lines]
        out += [block("thing", t) for t in self.things]
        return "\n".join(out) + "\n"


def spanning_tree(rng, w, h, n):
    """Random connected set of n grid cells grown from (0,0) and the tree edges that connect them."""
    cells, edges = [(0, 0)], []
    while len(cells) < n:
        c = rng.choice(cells)
        nb = [(c[0] + dx, c[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nb = [q for q in nb if 0 <= q[0] < w and 0 <= q[1] < h and q not in cells]
        if nb:
            q = rng.choice(nb)
            cells.append(q)
            edges.append((c, q))
    return cells, edges


def generate(seed: int, tier: int):
    rng = random.Random(seed)
    b = Builder()
    wall, ceil_t, light = rng.choice(WALLS), rng.choice(CEILS), rng.choice([144, 160, 176, 192, 208])
    n_rooms = {1: rng.randint(1, 2), 2: rng.randint(3, 6), 3: rng.randint(3, 6), 4: rng.randint(5, 8)}[tier]
    cells, edges = spanning_tree(rng, 4, 3, n_rooms)
    rooms = {}
    for (gx, gy) in cells:
        big = tier == 1
        hw = rng.choice([224, 256]) if big else rng.choice([128, 160, 192, 224])
        hh = rng.choice([224, 256]) if big else rng.choice([128, 160, 192, 224])
        if big:
            hw = hh = min(hw, G // 2 - 48)
        cx, cy = gx * G, gy * G
        r = Room(gx, gy, cx - hw, cx + hw, cy - hh, cy + hh, rng.choice(FLOORS), rng.choice([128, 144, 160, 192]))
        r.sector = b.sector(r.floor, ceil_t, 0, r.ceil_h, light)
        rooms[(gx, gy)] = r
    adj = {c: [] for c in cells}
    for a, c in edges:
        adj[a].append(c)
        adj[c].append(a)
    start = cells[0]
    dist, parent, q = {start: 0}, {start: None}, deque([start])
    while q:
        c = q.popleft()
        for d in adj[c]:
            if d not in dist:
                dist[d], parent[d] = dist[c] + 1, c
                q.append(d)
    exit_cell = max(cells, key=lambda c: (dist[c], c)) if len(cells) > 1 else start
    path = []
    c = exit_cell
    while parent[c] is not None:
        path.append((parent[c], c))
        c = parent[c]
    path.reverse()
    # locked doors on the start->exit path; their keys in rooms reachable before them
    locks = {}
    n_locks = {1: 0, 2: 0, 3: 1, 4: min(2, len(path))}[tier]
    if n_locks and path:
        chosen = sorted(rng.sample(range(len(path)), n_locks)) if len(path) >= n_locks else list(range(len(path)))
        keys = rng.sample(KEYS, len(chosen))
        for i, k in zip(chosen, keys):
            locks[frozenset(path[i])] = k
    # build connections
    key_rooms = []
    # layout for training-only route shaping (no RNG draws here: the WAD bytes must not change)
    layout = {"rooms": {f"{c[0]},{c[1]}": [r.x0, r.y0, r.x1, r.y1] for c, r in rooms.items()},
              "links": [], "keys": [], "start_cell": list(start), "exit_cell": list(exit_cell)}
    for a, c in edges:
        ra, rc = rooms[a], rooms[c]
        if ra.gx > rc.gx or ra.gy > rc.gy:
            ra, rc = rc, ra
        lock = locks.get(frozenset((a, c)))
        door = lock is not None or (tier != 1 and rng.random() < 0.75)
        width = rng.choice([64, 96, 128])
        depth = rng.choice([16, 24])
        special = dict(special=13, arg0=0, arg1=16, arg2=150, arg3=lock[1], playeruse=True, repeatspecial=True) if lock \
            else dict(special=12, arg0=0, arg1=16, arg2=150, playeruse=True, monsteruse=True, repeatspecial=True)
        door_tex = {"top": {1: "DOORRED", 2: "DOORBLU", 3: "DOORYEL"}[lock[1]]} if lock else {"top": "BIGDOOR2"}
        horizontal = ra.gy == rc.gy          # ra is left of (or below) rc
        if horizontal:
            lo, hi = max(ra.y0, rc.y0) + 16, min(ra.y1, rc.y1) - 16 - width
            ca, cb = ra.x1, rc.x0
        else:
            lo, hi = max(ra.x0, rc.x0) + 16, min(ra.x1, rc.x1) - 16 - width
            ca, cb = ra.y1, rc.y0
        o0 = rng.choice(range(int(lo), int(hi) + 1, 16)) if hi >= lo else int((max(lo, hi) + min(lo, hi)) / 2)
        o1 = o0 + width
        ra.openings["R" if horizontal else "U"].append((o0, o1))
        rc.openings["L" if horizontal else "D"].append((o0, o1))
        m = (ca + cb) / 2
        mid = (o0 + o1) / 2
        layout["links"].append({"a": [ra.gx, ra.gy], "b": [rc.gx, rc.gy], "lock": lock[1] if lock else 0,
                                "portal": [m, mid] if horizontal else [mid, m],
                                "rect": [ca, o0, cb, o1] if horizontal else [o0, ca, o1, cb]})
        segs = [(ca, m - depth / 2, "c"), (m - depth / 2, m + depth / 2, "d"), (m + depth / 2, cb, "c")] if door \
            else [(ca, cb, "c")]
        seq = [(ra.sector, "room")]
        for s0, s1, kind in segs:
            hc = 0 if kind == "d" else min(ra.ceil_h, rc.ceil_h, 128)
            sec = b.sector("FLAT20" if kind == "d" else ra.floor, ceil_t, 0, hc, light)
            seq.append((sec, kind))
            tex = {"mid": "DOORTRAK" if kind == "d" else wall}
            if horizontal:   # side walls at y=o0 (sector above it) and y=o1 (sector below it)
                b.edge((s0, o0), (s1, o0), sec, "U", front_tex=tex)
                b.edge((s0, o1), (s1, o1), sec, "D", front_tex=tex)
            else:            # side walls at x=o0 (sector right of it) and x=o1 (sector left of it)
                b.edge((o0, s0), (o0, s1), sec, "R", front_tex=tex)
                b.edge((o1, s0), (o1, s1), sec, "L", front_tex=tex)
        seq.append((rc.sector, "room"))
        bounds = [ca] + [s1 for _, s1, _ in segs]
        for i, pos in enumerate(bounds):        # boundary between seq[i] and seq[i+1]
            (sa, ka), (sb, kb) = seq[i], seq[i + 1]
            a_, b_ = ((pos, o0), (pos, o1)) if horizontal else ((o0, pos), (o1, pos))
            near, far = ("L", "R") if horizontal else ("D", "U")
            if kb == "d":      # entering the door: front faces the player approaching from seq[i]
                b.edge(a_, b_, sa, near, back=sb, front_tex=door_tex, **special)
            elif ka == "d":    # leaving the door: front faces the player approaching from seq[i+1]
                b.edge(a_, b_, sb, far, back=sa, front_tex=door_tex, **special)
            else:
                b.edge(a_, b_, sa, near, back=sb, front_tex={"top": wall}, back_tex={"top": wall})
        if lock:
            key_rooms.append((frozenset((a, c)), lock))
    # room walls, split around openings
    for r in rooms.values():
        for side, (fixed, lo, hi, d) in {"L": (r.x0, r.y0, r.y1, "R"), "R": (r.x1, r.y0, r.y1, "L"),
                                         "D": (r.y0, r.x0, r.x1, "U"), "U": (r.y1, r.x0, r.x1, "D")}.items():
            cuts = sorted(r.openings[side])
            pos = lo
            for o0, o1 in cuts + [(hi, hi)]:
                if o0 > pos:
                    a_, b_ = ((fixed, pos), (fixed, o0)) if side in "LR" else ((pos, fixed), (o0, fixed))
                    b.edge(a_, b_, r.sector, d, front_tex={"mid": wall})
                pos = max(pos, o1)
    # keys: reachable from start without crossing their own (or any later) locked edge
    for edge_set, (key_thing, _lock) in key_rooms:
        blocked = {e for e in locks}
        seen, q = {start}, deque([start])
        while q:
            c = q.popleft()
            for d in adj[c]:
                if d not in seen and frozenset((c, d)) not in blocked:
                    seen.add(d)
                    q.append(d)
        on_path = {c for e in path for c in e}
        options = [c for c in seen if c not in on_path] or [c for c in seen if c != start] or [start]
        kr = rooms[rng.choice(sorted(options))]
        kx = rng.uniform(kr.x0 + 48, kr.x1 - 48)
        ky = rng.uniform(kr.y0 + 48, kr.y1 - 48)
        b.thing(kx, ky, key_thing)
        layout["keys"].append({"lock": _lock, "pos": [kx, ky], "cell": [kr.gx, kr.gy]})
    # exit pad in the exit room (GATE3, walk-over Exit_Normal on all four edges)
    er = rooms[exit_cell]
    s = 64
    if exit_cell == start:
        px, py = er.x1 - 96, er.y1 - 96
    else:
        px, py = rng.uniform(er.x0 + 80, er.x1 - 80), rng.uniform(er.y0 + 80, er.y1 - 80)
    px, py = round(px / 16) * 16, round(py / 16) * 16
    pad = b.sector("GATE3", ceil_t, 0, er.ceil_h, 255)
    ex = dict(special=243, playercross=True)
    b.edge((px - s / 2, py - s / 2), (px - s / 2, py + s / 2), er.sector, "L", back=pad, **ex)
    b.edge((px + s / 2, py - s / 2), (px + s / 2, py + s / 2), er.sector, "R", back=pad, **ex)
    b.edge((px - s / 2, py - s / 2), (px + s / 2, py - s / 2), er.sector, "D", back=pad, **ex)
    b.edge((px - s / 2, py + s / 2), (px + s / 2, py + s / 2), er.sector, "U", back=pad, **ex)
    # player start, monsters, items
    sr = rooms[start]
    b.thing(sr.x0 + 64, (sr.y0 + sr.y1) / 2, 1, angle=rng.choice([0, 90, 270]))
    layout["exit"] = [px, py]
    layout["start"] = [sr.x0 + 64, (sr.y0 + sr.y1) / 2]
    n_mon = {1: rng.randint(3, 8), 2: rng.randint(1, 4), 3: rng.randint(1, 4), 4: rng.randint(4, 9)}[tier]
    pool = ["zombie", "zombie", "shotgun", "imp", "imp", "demon"]
    for _ in range(n_mon):
        c = rng.choice(cells if tier == 1 else [c for c in cells if c != start] or cells)
        r = rooms[c]
        mx, my = rng.uniform(r.x0 + 40, r.x1 - 40), rng.uniform(r.y0 + 40, r.y1 - 40)
        if c == start and abs(mx - (sr.x0 + 64)) < 160:
            mx = sr.x1 - 48
        b.thing(mx, my, MONSTERS[rng.choice(pool)], angle=rng.choice([0, 90, 180, 270]))
    for _ in range(rng.randint(1, 4)):
        r = rooms[rng.choice(cells)]
        b.thing(rng.uniform(r.x0 + 32, r.x1 - 32), rng.uniform(r.y0 + 32, r.y1 - 32), rng.choice(ITEMS))
    meta = {"seed": seed, "tier": tier, "kind": TIERS[tier], "rooms": len(cells), "path_len": len(path),
            "locks": len(locks), "monsters": n_mon, "doors": sum(1 for l in b.lines if l.get("special") in (12, 13)) // 2,
            "layout": layout}
    return b.textmap(), meta


def write_wad(path: Path, maps: list[tuple[str, str]]):
    lumps = []
    for name, text in maps:
        lumps += [(name, b""), ("TEXTMAP", text.encode()), ("ENDMAP", b"")]
    data, directory, off = b"", b"", 12
    for name, payload in lumps:
        directory += struct.pack("<II8s", off, len(payload), name.encode().ljust(8, b"\0"))
        data += payload
        off += len(payload)
    path.write_bytes(b"PWAD" + struct.pack("<II", len(lumps), 12 + len(data)) + data + directory)


def build(out_dir: Path, split: str, count: int, seed0: int, per_wad: int = 30):
    out_dir.mkdir(parents=True, exist_ok=True)
    tiers = [1, 2, 2, 3, 3, 3, 4, 4]   # weight toward doors and keys (tiers 2-4 carry more score weight)
    entries, metas = [], []
    for w in range((count + per_wad - 1) // per_wad):
        maps = []
        for i in range(min(per_wad, count - w * per_wad)):
            k = w * per_wad + i
            tier = tiers[k % len(tiers)]
            text, meta = generate(seed0 + k, tier)
            name = f"MAP{i + 1:02d}"
            maps.append((name, text))
            meta.update(wad=f"{split}_{w:02d}.wad", map=name)
            metas.append(meta)
        write_wad(out_dir / f"{split}_{w:02d}.wad", maps)
    (out_dir / f"{split}_maps.json").write_text(json.dumps(metas, indent=1))
    return metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "maps_gen"))
    a = ap.parse_args()
    out = Path(a.out)
    # disjoint seed ranges per split; dev/test are never trained on
    build(out, "train", 240, 100_000)
    build(out, "dev", 24, 200_000)
    build(out, "test", 24, 300_000)


if __name__ == "__main__":
    main()
