"""Accelerated, output-identical replacements for doom_env.sensors.LevelGeometry.update / Sensors.compute.

TRAINING ONLY. Evaluation always runs the official wrapper; this module exists to make training
steps cheaper while producing bit-identical observations (checked by test_equivalence.py).

How exactness is kept:
- The hot O(rays x lines) and O(targets x lines) loops (`_ray_line_t`, `_segments_blocked`, the
  `nearby` / blocking filters) and the per-line blocking update run as numba kernels that evaluate
  the *same* float64 expressions in the same order as the numpy code (no fastmath, so LLVM does
  not contract a*b+c into FMA). Argmin ties keep the first index, like np.argmin.
- Everything whose rounding could depend on the library (np.cos/np.sin/np.arctan2/np.hypot/%,
  the BLAS matmul `dirs @ c.T`, float32 casts) is still computed by numpy, with the official
  expressions, so it rounds identically on every platform.
- Static geometry is built by the official LevelGeometry constructor; KeyTracker is the official
  one (the wrapper owns it); the classification cache is the official module-level _CLASS_CACHE.

If numba is unavailable, FastLevelGeometry / FastSensors fall back to the official methods (with a
warning), so the env still works, just without the speedup.
"""

from __future__ import annotations

import math
import warnings
from operator import attrgetter

import numpy as np

from doom_env import spaces as S
from doom_env import sensors as _off
from doom_env.sensors import (
    DEFAULT_RADIUS,
    ENTITY_MAX_DISTANCE,
    HALF_FOV,
    MAX_STEP_UP,
    MONSTER_RADIUS,
    PLAYER_HEIGHT,
    RAY_DOOR,
    RAY_LOCKED,
    RAY_NONE,
    RAY_WALL,
    TYPE_ID,
    LevelGeometry,
    Sensors,
    classify,
    ego_rotate,
)

try:
    import numba as _nb

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without the train group
    _nb = None
    HAVE_NUMBA = False

_NUM_RAYS = S.NUM_RAYS
_RAY_RANGE = S.RAY_RANGE
_MAX_ENTITIES = S.MAX_ENTITIES
_NUM_FEATS = len(S.ENTITY_FEATURES)
_KEY_TYPE = _off._KEY_TYPE
_RAY_CODE_BY_TYPE = _off._RAY_CODE_BY_TYPE
_RAY_ANGLES = _off._RAY_ANGLES
_CLASS_CACHE = _off._CLASS_CACHE  # shared with the official Sensors: same first-seen semantics
_T_MAX = 1.0 - 1e-6  # same constant as in _segments_blocked

_get_floor = attrgetter("floor_height")
_get_ceil = attrgetter("ceiling_height")
_get_name = attrgetter("name")
_get_px = attrgetter("position_x")
_get_py = attrgetter("position_y")
_MONSTER_INFO: dict[str, tuple[int, float]] = {}


def _classify_info(name: str, category: str) -> tuple[int, float]:
    """(type index, ray radius) exactly as built inside the official Sensors.compute loop."""
    kind = None if category == "Self" else classify(name, category)
    return (TYPE_ID[kind], MONSTER_RADIUS.get(name, DEFAULT_RADIUS)) if kind else (0, 0.0)


# ------------------------------------------------------------------------------------------ kernels
if HAVE_NUMBA:
    _jit = _nb.njit(cache=True, nogil=True, fastmath=False, error_model="numpy", boundscheck=False)

    @_jit
    def _update_blocks(floor, ceil, sec_a, sec_b, one_sided, passable_same, flag_blocking, door_lock):
        """Mirror of LevelGeometry.update's vectorised part, one line at a time."""
        n = sec_a.shape[0]
        move_block = np.empty(n, np.bool_)
        sight_block = np.empty(n, np.bool_)
        hit = np.empty(n, np.int64)
        for i in range(n):
            a = sec_a[i]
            osd = one_sided[i]
            b = a if osd else sec_b[i]
            fa = floor[a]
            fb = floor[b]
            ca = ceil[a]
            cb = ceil[b]
            opening = (ca if ca < cb else cb) - (fa if fa > fb else fb)
            step = abs(fa - fb)
            solid = osd and not passable_same[i]
            move_block[i] = solid or ((not osd) and (flag_blocking[i] or opening < PLAYER_HEIGHT or step > MAX_STEP_UP))
            sight_block[i] = solid or ((not osd) and opening <= 0)
            h = RAY_WALL
            if (not osd) and opening < PLAYER_HEIGHT:
                la = door_lock[a]
                lb = door_lock[b]
                lock = la if la > lb else lb
                if lock == 0:
                    h = RAY_DOOR
                elif lock > 0:
                    h = RAY_LOCKED
            hit[i] = h
        return move_block, sight_block, hit

    @_jit
    def _cast_rays(px, py, rng, dirs, p1, d, bb_min, bb_max, move_block, hit_type, out_dist, out_hit):
        """Rays vs movement-blocking lines within `rng` (bbox test as in LevelGeometry.nearby).

        Same arithmetic as _ray_line_t; lines are visited in ascending index order and a ray's
        best hit is replaced only on a strictly smaller t, which is np.argmin's first-minimum rule.
        """
        nr = dirs.shape[0]
        best_j = np.full(nr, -1, np.int64)
        for r in range(nr):
            out_dist[r] = np.inf
        xhi = px + rng
        xlo = px - rng
        yhi = py + rng
        ylo = py - rng
        for j in range(p1.shape[0]):
            if not (bb_min[j, 0] <= xhi and bb_max[j, 0] >= xlo and bb_min[j, 1] <= yhi and bb_max[j, 1] >= ylo):
                continue
            if not move_block[j]:
                continue
            dx = d[j, 0]
            dy = d[j, 1]
            wx = p1[j, 0] - px
            wy = p1[j, 1] - py
            tnum = wx * dy - wy * dx
            for r in range(nr):
                rx = dirs[r, 0]
                ry = dirs[r, 1]
                denom = rx * dy - ry * dx
                if not (abs(denom) > 1e-9):
                    continue
                t = tnum / denom
                if not (t > 1e-6):
                    continue
                if not (t < out_dist[r]):
                    continue
                u = (wx * ry - wy * rx) / denom
                if u >= 0.0 and u <= 1.0:
                    out_dist[r] = t
                    best_j[r] = j
        for r in range(nr):
            j = best_j[r]
            out_hit[r] = hit_type[j] if j >= 0 else RAY_NONE

    @_jit
    def _rays_vs_entities(tc, c, rad, code, dist, hit):
        """Mirror of the 'rays also stop at monsters and pickups' block (tc = dirs @ c.T from numpy)."""
        nr, k = tc.shape
        cc = np.empty(k)
        r2 = np.empty(k)
        for q in range(k):
            cc[q] = c[q, 0] * c[q, 0] + c[q, 1] * c[q, 1]
            r2[q] = rad[q] * rad[q]
        for r in range(nr):
            tb = np.inf
            kb = 0
            for q in range(k):
                tcq = tc[r, q]
                d2 = cc[q] - tcq * tcq
                if tcq > 0 and d2 <= r2[q]:
                    s = r2[q] - d2
                    s = s if s >= 0.0 else 0.0  # np.maximum(s, 0.0) (keeps -0.0 like numpy)
                    th = tcq - np.sqrt(s)
                    th = th if th >= 0.0 else 0.0  # np.maximum(th, 0.0)
                    if th < tb:
                        tb = th
                        kb = q
            if tb < dist[r]:
                dist[r] = tb
                hit[r] = code[kb]

    @_jit
    def _los_blocked(px, py, fx, fy, rng, tx, ty, p1, p2, d, bb_min, bb_max, sight_block, cand):
        """Mirror of the sight-line filter + _segments_blocked. Returns a bool per target."""
        m = 0
        xhi = px + rng
        xlo = px - rng
        yhi = py + rng
        ylo = py - rng
        for j in range(p1.shape[0]):
            if not (bb_min[j, 0] <= xhi and bb_max[j, 0] >= xlo and bb_min[j, 1] <= yhi and bb_max[j, 1] >= ylo):
                continue
            if not sight_block[j]:
                continue
            f1 = (p1[j, 0] - px) * fx + (p1[j, 1] - py) * fy
            f2 = (p2[j, 0] - px) * fx + (p2[j, 1] - py) * fy
            if f1 > 0 or f2 > 0:  # np.maximum(f1, f2) > 0
                cand[m] = j
                m += 1
        nt = tx.shape[0]
        out = np.zeros(nt, np.bool_)
        for q in range(nt):
            rx = tx[q] - px
            ry = ty[q] - py
            for i in range(m):
                j = cand[i]
                dx = d[j, 0]
                dy = d[j, 1]
                denom = rx * dy - ry * dx
                if not (abs(denom) > 1e-9):
                    continue
                wx = p1[j, 0] - px
                wy = p1[j, 1] - py
                t = (wx * dy - wy * dx) / denom
                if not (t > 1e-6 and t < _T_MAX):
                    continue
                u = (wx * ry - wy * rx) / denom
                if u >= 0.0 and u <= 1.0:
                    out[q] = True
                    break
        return out


# ------------------------------------------------------------------------------------------ classes
class FastLevelGeometry(LevelGeometry):
    """LevelGeometry with a numba per-step update. Static data comes from the official constructor."""

    @classmethod
    def from_official(cls, geo: LevelGeometry) -> "FastLevelGeometry":
        fast = cls.__new__(cls)
        fast.__dict__.update(geo.__dict__)
        fast._prepare()
        return fast

    def __init__(self, sectors, mapinfo):
        super().__init__(sectors, mapinfo)
        self._prepare()

    def _prepare(self) -> None:
        c = np.ascontiguousarray
        self._p1 = c(self.p1, np.float64)
        self._p2 = c(self.p2, np.float64)
        self._d = c(self.p2 - self.p1, np.float64)  # same values as the per-call p2[idx] - p1[idx]
        self._bb_min = c(self.lines_bb_min, np.float64)
        self._bb_max = c(self.lines_bb_max, np.float64)
        self._sec_a = c(self.sec_a, np.int64)
        self._sec_b = c(self.sec_b, np.int64)
        self._one_sided = c(self.one_sided, np.bool_)
        self._passable = c(self.passable_same_sector, np.bool_)
        self._flag = c(self.flag_blocking, np.bool_)
        self._door_lock = c(self.door_lock, np.int64)
        self._cand = np.empty(len(self._p1), np.int64)
        # The kernels skip bounds checks: only use them when every sector index is in range
        # (the official numpy code would raise / wrap otherwise, so defer to it).
        n_sec = len(self.door_lock)
        two = ~self._one_sided
        self.use_kernels = HAVE_NUMBA and (
            len(self._sec_a) == 0
            or (
                self._sec_a.min() >= 0 and self._sec_a.max() < n_sec
                and (not two.any() or (self._sec_b[two].min() >= 0 and self._sec_b[two].max() < n_sec))
            )
        )
        if not HAVE_NUMBA:
            warnings.warn("numba not installed: FastDoomEnv falls back to the official sensors", RuntimeWarning, stacklevel=3)

    def update(self, sectors) -> None:
        if not self.use_kernels:
            return LevelGeometry.update(self, sectors)
        n = len(self.door_lock)
        if self.ordered:
            m = len(sectors)
            floor = np.fromiter(map(_get_floor, sectors), np.float64, m)
            ceil = np.fromiter(map(_get_ceil, sectors), np.float64, m)
        else:
            floor = np.zeros(n, np.float64)
            ceil = np.zeros(n, np.float64)
            for s in sectors:
                floor[s.id], ceil[s.id] = s.floor_height, s.ceiling_height
        self.floor, self.ceil = floor, ceil
        self.move_block, self.sight_block, self.hit_type = _update_blocks(
            floor, ceil, self._sec_a, self._sec_b, self._one_sided, self._passable, self._flag, self._door_lock)


class FastSensors(Sensors):
    def compute(self, px, py, heading, pvel, objects):
        geo = self.geo
        if not getattr(geo, "use_kernels", False):
            return Sensors.compute(self, px, py, heading, pvel, objects)
        px = float(px)
        py = float(py)
        origin = np.array([px, py], np.float64)

        # ---- rays vs blocking lines (numba) ---------------------------------------------
        angles = heading + _RAY_ANGLES
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        dist = np.empty(_NUM_RAYS, np.float64)
        hit = np.empty(_NUM_RAYS, np.int64)
        _cast_rays(px, py, _RAY_RANGE, dirs, geo._p1, geo._d, geo._bb_min, geo._bb_max,
                   geo.move_block, geo.hit_type, dist, hit)

        # ---- entities: same classification rule and the same shared cache as the official loop
        # Attribute reads on vizdoom objects cost ~0.1 us each (pybind), so they are batched with
        # map() and velocities are only read for the (at most 32) entities that end up visible.
        # The state is an immutable snapshot, so late reads return the same values.
        cache = _CLASS_CACHE
        objects = objects if isinstance(objects, list) else list(objects)
        names = list(map(_get_name, objects))
        infos = list(map(cache.get, names))
        if None in infos:
            mon = _MONSTER_INFO
            for i, info in enumerate(infos):
                if info is None:
                    name = names[i]
                    category = objects[i].category
                    if category == "Monster":
                        # Official: recomputed every time and never put in _CLASS_CACHE (monsters
                        # change name when they die). classify(name, "Monster") is pure, so memoise
                        # it privately, keyed by name.
                        info = mon.get(name)
                        if info is None:
                            info = mon[name] = _classify_info(name, category)
                    else:
                        info = cache[name] = _classify_info(name, category)
                    infos[i] = info
        idx = [i for i, info in enumerate(infos) if info[0]]
        kept = [objects[i] for i in idx]
        keys: dict[int, str] = {}
        for i in idx:
            if infos[i][0] == _KEY_TYPE:
                keys[objects[i].id] = names[i]
        self.present_keys = keys

        feats = np.zeros((_MAX_ENTITIES, _NUM_FEATS), np.float32)
        etype = np.zeros(_MAX_ENTITIES, np.int64)
        mask = np.zeros(_MAX_ENTITIES, np.int8)

        rows = len(kept)
        if rows:
            arr = np.empty((rows, 4), np.float64)  # (type, x, y, radius); official has (type, x, y, vx, vy, radius)
            arr[:, 0] = [infos[i][0] for i in idx]
            arr[:, 1] = list(map(_get_px, kept))
            arr[:, 2] = list(map(_get_py, kept))
            arr[:, 3] = [infos[i][1] for i in idx]
            rel_xy = arr[:, 1:3] - origin
            edist = np.hypot(rel_xy[:, 0], rel_xy[:, 1])
            keep = (edist <= ENTITY_MAX_DISTANCE) & (edist > 1e-6)
            src = None  # filtered row -> index into `kept`
            if not keep.all():
                src = np.flatnonzero(keep)
                arr, rel_xy, edist = arr[keep], rel_xy[keep], edist[keep]
            etypes = arr[:, 0].astype(np.int64)
        if rows and len(arr):
            rel_angle = (np.arctan2(rel_xy[:, 1], rel_xy[:, 0]) - heading + np.pi) % (2 * np.pi) - np.pi
            radius = arr[:, 3]

            ray_code = _RAY_CODE_BY_TYPE[etypes]
            solid = ray_code != RAY_NONE
            if solid.any():
                c = rel_xy[solid]  # (K,2) contiguous copy, exactly as in the official code
                tc = dirs @ c.T  # BLAS: keep numpy so rounding matches the official env
                _rays_vs_entities(np.ascontiguousarray(tc), c, radius[solid], ray_code[solid], dist, hit)

            vis = np.flatnonzero(np.abs(rel_angle) <= HALF_FOV + np.arctan2(radius, edist))
            if len(vis):
                tgt = arr[vis]
                blocked = _los_blocked(
                    px, py, math.cos(heading), math.sin(heading), float(edist[vis].max()),
                    np.ascontiguousarray(tgt[:, 1]), np.ascontiguousarray(tgt[:, 2]),
                    geo._p1, geo._p2, geo._d, geo._bb_min, geo._bb_max, geo.sight_block, geo._cand)
                vis = vis[~blocked]
            if len(vis) > 1:
                vis = vis[np.argsort(edist[vis], kind="stable")][:_MAX_ENTITIES]
            n = len(vis)
            if n:
                vobj = [kept[i] for i in (vis if src is None else src[vis]).tolist()]
                vx = np.array([o.velocity_x for o in vobj], np.float64)
                vy = np.array([o.velocity_y for o in vobj], np.float64)
                fwd, right = ego_rotate(vx - pvel[0], vy - pvel[1], heading)
                ang = rel_angle[vis]
                feats[:n] = np.stack([edist[vis] / _RAY_RANGE, np.sin(ang), np.cos(ang), fwd, right], axis=1)
                etype[:n] = etypes[vis]
                mask[:n] = 1

        in_range = dist <= _RAY_RANGE
        rays = {
            "distance": np.where(in_range, dist / _RAY_RANGE, 1.0).astype(np.float32),
            "hit_type": np.where(in_range, hit, RAY_NONE).astype(np.int64),
        }
        entities = {"features": feats, "entity_type": etype, "mask": mask}
        return rays, entities
