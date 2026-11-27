"""Per-episode progress in the competition's scoring categories, from training-only privileged state.

docs/EVALUATION.md scores each map on: exit reached (largest; small bonus for remaining health and time),
progression (keycards picked up, doors opened, locked doors unlocked), exploration (unique areas visited)
and combat (kills; small penalties for damage taken and dying early). The scorer's exact areas and
weights are private, so this tracker is a proxy:
  * exploration  = distinct CELL x CELL map-unit cells the player has stood in this episode
  * door opened  = a door sector whose ceiling rose more than DOOR_OPEN_RISE above its episode-start height
                   (door_lock > 0 marks a locked door; opening one counts as "unlocked")
Positions and sector heights come from env.privileged_state(), which the wrapper documents as ground truth
"for scoring and reward shaping. Never part of the agent's observation." The policy never sees any of it;
it only shapes the training reward and the offline evaluation metrics.
"""
from __future__ import annotations

import numpy as np

CELL = 128.0
DOOR_OPEN_RISE = 24.0

# Training reward = official wrapper reward (+1 exit, -1 death, +0.1 per kill) + these terms.
PROGRESS_V1 = {
    "name": "progress_v1",
    "cell": 0.025,        # per newly visited cell (exploration); smoke run showed 3-9 cells/episode, revisit ratio ~0.97
    "key": 0.5,           # per keycard picked up (progression)
    "door": 0.1,          # per ordinary door opened (progression)
    "unlock": 0.5,        # per locked door opened (progression)
    "kill": 0.05,         # extra per kill, on top of the wrapper's +0.1 (combat; already learned, kept small)
    "damage": -0.002,     # per hit point of damage taken (combat penalty)
    "exit": 2.0,          # extra on exit, on top of the wrapper's +1 (exit is the largest category)
    "exit_health": 0.25,  # x remaining health / 100, on exit
    "exit_time": 0.25,    # x unused fraction of the tick cap, on exit
}
# v2: everything in v1, plus a penalty for ammo spent while no enemy is in the agent's own view (measured:
# 62-89% of shots fired with no enemy visible; a third to two thirds of held-out episodes ended with 0 ammo),
# and a larger locked-door reward (keys were picked up on held-out key maps but locked doors almost never opened).
PROGRESS_V2 = dict(PROGRESS_V1, name="progress_v2", unlock=1.0, blind_ammo=-0.01, aim_cone_deg=None)
# v3: the ammo penalty applies unless an enemy is inside a +-8 degree aim cone (v2 excused any visible enemy, and -
# a bug - also enemy ray hits BEHIND the agent: the 64 rays cover 360 degrees). Measured on held-out maps: only ~1/3 of
# shots with an enemy visible were within +-6 degrees of it; ~40% of episodes ended with no ammo.
PROGRESS_V3 = dict(PROGRESS_V2, name="progress_v3", aim_cone_deg=8.0)
SHAPINGS = {"progress_v1": PROGRESS_V1, "progress_v2": PROGRESS_V2, "progress_v3": PROGRESS_V3}
FOV_HALF_DEG = 45.0
# v4: v3 + potential-based route shaping (Ng, Harada & Russell 1999) on generated maps, whose layout is known:
# F = PHI(s') - PHI(s), PHI = -route_distance_to_next_required_objective / ROUTE_SCALE. Objectives in order:
# the keycard for the first locked door still between the agent and the exit, then the exit. Freedoom maps have
# no layout, so the term is 0 there. Shaping uses gamma = 1 (not the PPO 0.99): with 0.99, standing still far
# from the goal would earn (1 - gamma) * |PHI| every step.
PROGRESS_V4 = dict(PROGRESS_V3, name="progress_v4", route=1.0)
SHAPINGS["progress_v4"] = PROGRESS_V4
ROUTE_SCALE = 2000.0
# v5: v4 + stronger fire discipline. Held-out data: ATTACK pressed on ~50-56% of steps, ~half with no enemy in view;
# most presses spend no ammo (weapon refire delay, fists), so the per-round penalty never saw them. Adds a per-PRESS
# penalty when no enemy is inside the +-8 degree aim cone, and triples the per-round off-target penalty.
PROGRESS_V5 = dict(PROGRESS_V4, name="progress_v5", blind_ammo=-0.03, blind_press=-0.003)
SHAPINGS["progress_v5"] = PROGRESS_V5
# v6: v5 + weapon discipline + time cost. Held-out data: 41-53% of steps on fists while holding bullets (punching spends
# no ammo, so it dodged the per-round penalty; the pistol always exists), and 10-17% of key-map episodes ping-pong
# between two rooms (time was free). Both terms use the agent's own vitals.
PROGRESS_V6 = dict(PROGRESS_V5, name="progress_v6", fist_with_ammo=-0.005, step=-0.001)
SHAPINGS["progress_v6"] = PROGRESS_V6
# v7: v6 with much stronger trigger discipline (held-out: ATTACK still pressed on ~50% of steps, half with no enemy in view).
PROGRESS_V7 = dict(PROGRESS_V6, name="progress_v7", blind_press=-0.01, blind_ammo=-0.05)
SHAPINGS["progress_v7"] = PROGRESS_V7
CARD_INDEX = {1: 2, 2: 0, 3: 1}   # ZDoom lock number -> index of that keycard in privileged_state()["keys"]


class RouteGuide:
    """Route distance on a generated map's room tree (rooms joined by door/corridor links through portals)."""

    def __init__(self, layout):
        self.rooms = {tuple(int(v) for v in k.split(",")): r for k, r in layout["rooms"].items()}
        self.links = [dict(l, a=tuple(l["a"]), b=tuple(l["b"])) for l in layout["links"]]
        self.adj = {c: [] for c in self.rooms}
        for i, l in enumerate(self.links):
            self.adj[l["a"]].append((l["b"], i))
            self.adj[l["b"]].append((l["a"], i))
        self.exit, self.exit_cell = tuple(layout["exit"]), tuple(layout["exit_cell"])
        self.keys = {k["lock"]: (tuple(k["pos"]), tuple(k["cell"])) for k in layout["keys"]}

    @staticmethod
    def _inside(rect, x, y, margin=8.0):
        return rect[0] - margin <= x <= rect[2] + margin and rect[1] - margin <= y <= rect[3] + margin

    def _locate(self, x, y):
        """Rooms the point is in (a corridor point counts as being at its link's portal, next to both rooms)."""
        for c, r in self.rooms.items():
            if self._inside(r, x, y, 0.0):
                return [(c, (x, y))]
        for l in self.links:
            if self._inside(l["rect"], x, y):
                return [(l["a"], (x, y)), (l["b"], (x, y))]
        best = min(self.rooms, key=lambda c: (max(self.rooms[c][0] - x, 0, x - self.rooms[c][2]) ** 2
                                              + max(self.rooms[c][1] - y, 0, y - self.rooms[c][3]) ** 2))
        return [(best, (x, y))]

    def _path(self, src, dst):
        prev, queue = {src: None}, [src]
        for c in queue:
            for d, i in self.adj[c]:
                if d not in prev:
                    prev[d] = (c, i)
                    queue.append(d)
        links, c = [], dst
        while prev.get(c):
            c, i = prev[c]
            links.append(i)
        return links[::-1]

    def _distance(self, starts, goal_cell, goal_xy):
        best = float("inf")
        for cell, xy in starts:
            d, p = 0.0, xy
            for i in self._path(cell, goal_cell):
                q = self.links[i]["portal"]
                d += float(np.hypot(q[0] - p[0], q[1] - p[1]))
                p = q
            best = min(best, d + float(np.hypot(goal_xy[0] - p[0], goal_xy[1] - p[1])))
        return best

    def remaining(self, x, y, keys_held):
        """Length of the whole remaining route: via the keycard of each locked door still ahead, in order, to the exit.
        Continuous when a key is picked up (the route from the key onward is unchanged), so the shaping never
        penalizes collecting a required key."""
        held = {lock for lock, idx in CARD_INDEX.items() if keys_held[idx] > 0}
        starts, total = self._locate(x, y), 0.0
        for _ in range(len(self.keys) + 1):
            here = starts[0][0]
            nxt = next((self.links[i]["lock"] for i in self._path(here, self.exit_cell)
                        if self.links[i]["lock"] and self.links[i]["lock"] not in held and self.links[i]["lock"] in self.keys), None)
            if nxt is None:
                return total + self._distance(starts, self.exit_cell, self.exit)
            pos, cell = self.keys[nxt]
            total += self._distance(starts, cell, pos)
            held.add(nxt)
            starts = [(cell, pos)]
        return total + self._distance(starts, self.exit_cell, self.exit)

    def potential(self, p):
        return -self.remaining(p["x"], p["y"], np.asarray(p["keys"])) / ROUTE_SCALE


def enemy_in_view(obs, cone_deg=None):
    """Whether the agent's OWN observation shows an enemy within +-cone_deg of where it faces.
    cone_deg=None reproduces progress_v2's definition: any visible enemy entity or ANY enemy ray hit (rays cover 360 deg).
    Entities carry (distance, sin, cos) of the relative angle; ray k points k * 360/NUM_RAYS degrees from the heading."""
    from doom_env.spaces import ENTITY_TYPES
    ents = obs.get("visible_entities") or {}
    et, mask = np.asarray(ents.get("entity_type", [])), np.asarray(ents.get("mask", [])) > 0
    enemy = np.array([t.startswith("enemy") for t in ENTITY_TYPES])
    rays = np.asarray((obs.get("rays") or {}).get("hit_type", []))
    if et.size:
        hit = mask & enemy[np.clip(et, 0, len(enemy) - 1)]
        if cone_deg is not None and hit.any():
            f = np.asarray(ents["features"])
            bearing = np.degrees(np.arctan2(f[:, 1], f[:, 2]))
            hit &= np.abs(bearing) <= cone_deg
        if bool(np.any(hit)):
            return True
    if cone_deg is None:
        return bool(np.any(rays == 4))
    if not rays.size:
        return False
    ang = (np.arange(rays.size) * 360.0 / rays.size + 180.0) % 360.0 - 180.0
    return bool(np.any((rays == 4) & (np.abs(ang) <= cone_deg)))


def total_ammo(obs):
    """Bullets + shells + rockets + cells from the observation's vitals (indices 4-7)."""
    return float(np.sum(np.asarray(obs["vitals"])[4:8]))


def resolve_shaping(name):
    """None/'official-wrapper' -> None (native reward only); a known name -> its weights; anything else refused."""
    if name in (None, "official-wrapper"):
        return None
    if isinstance(name, str) and name in SHAPINGS:
        return SHAPINGS[name]
    raise ValueError(f"unknown reward shaping {name!r}; known: {sorted(SHAPINGS)}")


def _cell(p):
    return int(np.floor(p["x"] / CELL)), int(np.floor(p["y"] / CELL))


class EpisodeProgress:
    """Tracks one episode. Construct with the privileged state right after reset(); call update() after
    every step with the new privileged state; it returns the increments since the previous call."""

    def __init__(self, p0):
        self.cells = {_cell(p0)}
        self.keys = float(np.sum(p0["keys"]))
        self.kills0 = self.kills = p0["kills"]
        self.damage0 = self.damage = float(p0["damage_taken"])
        ceil, lock = p0.get("sector_ceiling"), p0.get("door_lock")
        self.ceil0 = None if ceil is None else np.asarray(ceil, np.float64).copy()
        self.lock = None if lock is None else np.asarray(lock, np.int64).copy()
        self.opened, self.unlocked = set(), set()
        self.exit = self.dead = False
        self.health = float(p0["health"])
        self.tic_fraction = 0.0
        self.xy = (float(p0["x"]), float(p0["y"]))
        self.distance, self.steps = 0.0, 0
        self.components = {k: 0.0 for k in ("cell", "key", "door", "unlock", "kill", "damage", "exit", "blind_ammo", "route",
                                            "blind_press", "fist_with_ammo", "step")}
        self.blind_presses = 0
        self.ammo_spent = self.blind_ammo = 0.0

    def update(self, p):
        d = {"cells": 0, "keys": 0.0, "doors": 0, "unlocks": 0, "kills": 0, "damage": 0.0}
        xy = (float(p["x"]), float(p["y"]))
        self.distance += float(np.hypot(xy[0] - self.xy[0], xy[1] - self.xy[1]))
        self.xy = xy
        self.steps += 1
        c = _cell(p)
        if c not in self.cells:
            self.cells.add(c)
            d["cells"] = 1
        keys = float(np.sum(p["keys"]))
        d["keys"], self.keys = max(0.0, keys - self.keys), max(keys, self.keys)
        d["kills"], self.kills = p["kills"] - self.kills, p["kills"]
        dmg = float(p["damage_taken"])
        d["damage"], self.damage = max(0.0, dmg - self.damage), max(dmg, self.damage)
        ceil = p.get("sector_ceiling")
        if self.ceil0 is not None and self.lock is not None and ceil is not None and len(ceil) == len(self.ceil0):
            rose = (np.asarray(ceil) - self.ceil0 > DOOR_OPEN_RISE) & (self.lock >= 0)
            for s in np.flatnonzero(rose).tolist():
                if s in self.opened:
                    continue
                self.opened.add(s)
                if self.lock[s] > 0:
                    self.unlocked.add(s)
                    d["unlocks"] += 1
                else:
                    d["doors"] += 1
        self.exit, self.dead = bool(p["exit_reached"]), bool(p["dead"])
        self.health = float(p["health"])
        self.tic_fraction = min(1.0, p["tic"] / max(1, p["episode_ticks"]))
        return d

    def tick(self, obs, w):
        """Per-step costs from the agent's own vitals: on fists (weapon slot 1) while holding bullets; plain time."""
        r = 0.0
        if w.get("fist_with_ammo") and int(np.asarray(obs["vitals"])[2]) == 1 and float(np.asarray(obs["vitals"])[4]) > 0:
            self.components["fist_with_ammo"] += w["fist_with_ammo"]
            r += w["fist_with_ammo"]
        if w.get("step"):
            self.components["step"] += w["step"]
            r += w["step"]
        return float(r)

    def pressed(self, attack, enemy_seen, w):
        """ATTACK pressed on this decision with no enemy in the aim cone (whether or not ammo was spent)."""
        if not attack or enemy_seen or not w.get("blind_press"):
            return 0.0
        self.blind_presses += 1
        self.components["blind_press"] += w["blind_press"]
        return float(w["blind_press"])

    def fired(self, spent, enemy_seen, w):
        """Ammo spent on one step (from the agent's vitals) and whether an enemy was in its view when acting."""
        spent = max(0.0, float(spent))
        self.ammo_spent += spent
        if enemy_seen or spent <= 0:
            return 0.0
        self.blind_ammo += spent
        r = w.get("blind_ammo", 0.0) * spent
        self.components["blind_ammo"] += r
        return float(r)

    def bonus(self, d, w):
        """Shaping added to the wrapper reward for one step's increments d (from update())."""
        parts = {"cell": w["cell"] * d["cells"], "key": w["key"] * d["keys"], "door": w["door"] * d["doors"],
                 "unlock": w["unlock"] * d["unlocks"], "kill": w["kill"] * d["kills"], "damage": w["damage"] * d["damage"],
                 "exit": (w["exit"] + w["exit_health"] * max(0.0, self.health) / 100.0
                          + w["exit_time"] * (1.0 - self.tic_fraction)) if self.exit else 0.0}
        for k, v in parts.items():
            self.components[k] += v
        return float(sum(parts.values()))

    def summary(self):
        return {"cells": len(self.cells), "keys_picked": self.keys, "doors_opened": len(self.opened) - len(self.unlocked),
                "locked_opened": len(self.unlocked), "kills": self.kills - self.kills0,
                "damage": self.damage - self.damage0, "exit": self.exit, "dead": self.dead,
                "distance": self.distance,
                # fraction of steps spent in cells already visited this episode (1.0 = never found a new area)
                "revisit_ratio": 1.0 - (len(self.cells) - 1) / max(1, self.steps),
                "ammo_spent": self.ammo_spent, "blind_ammo": self.blind_ammo, "blind_presses": self.blind_presses,
                "shaping": dict(self.components)}
