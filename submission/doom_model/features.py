"""Agent-side features computed from observations only: observation arrays, dead-reckoned pose and
an ego-centric grid map. The same class runs inside training env workers and inside DoomAgent, so
the network sees identical inputs in training and at evaluation.

Odometry (validated in training/probes/odometry.py, results in training/probes/README.md):
  * heading: vanilla Doom turn schedule. Each tic with a turn key down turns 320/65536 of a circle for
    the first 5 tics a key is held, then 640/65536 ("turnheld" counts tics with ANY turn key down).
    Matched ground truth on 99.96% of 11,301 steps.
  * displacement: vanilla per-tic physics (thrust 25*2048/65536 forward, 24*2048/65536 strafe, move,
    friction 0xE800/0x10000) from the observed ego velocity; on a detected collision fall back to the
    mean of the observed velocities; if the ray scan did not change (after undoing the turn) the player
    did not translate (Doom keeps velocity while pushing into a wall).
    Median drift on unseen maps: 28 / 54 / 138 map units after 250 / 500 / 1000 steps.
Map (as in training/probes/memory_probe.py): 192x192 cells of 48 units in an episode frame that starts
at the agent (no global coordinates), channels free / wall / door / locked door / visited, read as two
ego-centric rotated crops (24x24 at 1 cell/px, 24x24 at 6 cells/px).
"""

from __future__ import annotations

import math

import numpy as np

from .actions import ACTIONS

# ---------------------------------------------------------------- observation arrays
VITALS_SCALE = np.array([100, 100, 7, 50, 200, 50, 50, 300, 1, 1, 1, 1, 1, 1, 8, 8, 8], np.float32)


def obs_arrays(obs: dict, prev_action: int) -> dict[str, np.ndarray]:
    """One observation -> flat arrays (no pixels). Batch with np.stack over envs."""
    return {
        "dist": np.asarray(obs["rays"]["distance"], np.float32),
        "hit": np.asarray(obs["rays"]["hit_type"], np.int64),
        "ent": np.asarray(obs["visible_entities"]["features"], np.float32),
        "etype": np.asarray(obs["visible_entities"]["entity_type"], np.int64),
        "emask": np.asarray(obs["visible_entities"]["mask"], np.float32),
        "vit": (np.asarray(obs["vitals"], np.float32) / VITALS_SCALE).astype(np.float32),
        "prev": np.asarray(prev_action, np.int64),
    }


# ---------------------------------------------------------------- odometry
TICS = 4
FRICTION = 0xE800 / 0x10000
SLOW_TURN, FAST_TURN = 320 * 360 / 65536, 640 * 360 / 65536      # degrees per tic
SLOW_TICS = 5
THRUST_F, THRUST_S = 25 * 2048 / 65536, 24 * 2048 / 65536
COLLIDE_THRESH = 0.5          # |predicted - observed| velocity (units/tic)
STILL_THRESH = 0.25           # mean |ray change| (units) below which the player did not move


class Odometry:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = self.y = self.th = 0.0          # episode frame: start = origin, heading 0
        self.turn_held = 0

    def _tic_turns(self, a: np.ndarray) -> np.ndarray:
        out = np.zeros(TICS)
        sign = int(a[4]) - int(a[5])
        for k in range(TICS):
            if a[4] or a[5]:
                out[k] = sign * (SLOW_TURN if self.turn_held < SLOW_TICS else FAST_TURN)
                self.turn_held += 1
            else:
                self.turn_held = 0
        return out

    def step(self, a: np.ndarray, vel0: np.ndarray, vel1: np.ndarray, d0: np.ndarray, d1: np.ndarray):
        """a: 9 buttons pressed between obs t and t+1; vel*: ego (fwd, right) velocity in vitals;
        d*: ray distances (normalized). Updates and returns the pose at t+1."""
        turns = self._tic_turns(a)
        theta = np.radians(np.cumsum(turns))                  # heading offset during each tic
        fwd_in = float(a[0]) - float(a[1])
        side_in = float(a[3]) - float(a[2])
        # free-space physics: per tic, mom += thrust * dir(theta_k); pos += mom; mom *= FRICTION
        m = np.array(vel0, np.float64)
        disp = np.zeros(2)
        for k in range(TICS):
            c, s = math.cos(theta[k]), math.sin(theta[k])
            m = m + THRUST_F * fwd_in * np.array([c, -s]) + THRUST_S * side_in * np.array([s, c])
            disp += m
            m = m * FRICTION
        tot = theta[-1]
        c, s = math.cos(tot), math.sin(tot)
        m_next = np.array([m[0] * c - m[1] * s, m[0] * s + m[1] * c])        # frame t -> t+1
        v1 = np.asarray(vel1, np.float64)
        if np.linalg.norm(m_next - v1) > COLLIDE_THRESH:                    # collision: fallback
            v1_t = np.array([v1[0] * c + v1[1] * s, -v1[0] * s + v1[1] * c])  # frame t+1 -> t
            disp = 0.5 * (np.asarray(vel0, np.float64) + v1_t) * TICS
        # ray stillness: undo the turn by a fractional circular shift, compare scans
        shift = turns.sum() / (360.0 / 64)
        idx = (np.arange(64) + shift) % 64
        lo = np.floor(idx).astype(int)
        fr = idx - lo
        d1s = (1 - fr) * d1[lo] + fr * d1[(lo + 1) % 64]
        if np.abs(d1s - d0).mean() * 1024.0 < STILL_THRESH:
            disp = np.zeros(2)
        c0, s0 = math.cos(self.th), math.sin(self.th)
        self.x += disp[0] * c0 + disp[1] * s0
        self.y += disp[0] * s0 - disp[1] * c0
        self.th += math.radians(turns.sum())
        return self.x, self.y, self.th


# ---------------------------------------------------------------- spatial map
GRID, CELL, CROP, MAP_CH, COARSE = 192, 48.0, 24, 5, 6.0
_RAY_ANG = np.arange(64) * (2 * np.pi / 64)
_U = np.arange(CROP) - CROP / 2 + 0.5
_GU, _GV = np.meshgrid(_U, _U)                 # _GV: row offset (top row negative), _GU: column offset


class SpatialMap:
    def __init__(self):
        self.grid = np.zeros((MAP_CH, GRID, GRID), np.uint8)

    def reset(self):
        self.grid[:] = 0

    @staticmethod
    def _cell(wx, wy):
        """World (episode-frame) coords -> cell indices and an in-bounds mask (no clipping/wrapping:
        clipping would pile out-of-range geometry onto the border as fake walls)."""
        cx = np.floor(np.asarray(wx) / CELL + GRID / 2).astype(int)
        cy = np.floor(np.asarray(wy) / CELL + GRID / 2).astype(int)
        ok = (cx >= 0) & (cx < GRID) & (cy >= 0) & (cy < GRID)
        return cx.clip(0, GRID - 1), cy.clip(0, GRID - 1), ok

    def write(self, pose, dist: np.ndarray, hit: np.ndarray) -> None:
        x, y, th = pose
        ang = th + _RAY_ANG
        r = dist * 1024.0
        g = self.grid
        for k in np.linspace(0, 1, 12, endpoint=False):           # free space along each ray
            cx, cy, ok = self._cell(x + np.cos(ang) * r * k, y + np.sin(ang) * r * k)
            g[0, cy[ok], cx[ok]] = 255
        cx, cy, ok = self._cell(x + np.cos(ang) * r, y + np.sin(ang) * r)
        hitm = (dist < 1) & ok
        for ch, t in ((1, 1), (2, 2), (3, 3)):                     # wall, door, locked door
            m = (hit == t) & hitm
            g[ch, cy[m], cx[m]] = 255
        px, py, ok = self._cell(np.array([x]), np.array([y]))
        if ok[0]:
            g[4, py, px] = 255                                     # visited

    def crops(self, pose) -> np.ndarray:
        x, y, th = pose
        out = np.zeros((2 * MAP_CH, CROP, CROP), np.uint8)
        for j, scale in enumerate((1.0, COARSE)):
            fwd, right = -_GV * scale * CELL, _GU * scale * CELL       # row 0 = farthest ahead
            wx = x + fwd * math.cos(th) + right * math.sin(th)
            wy = y + fwd * math.sin(th) - right * math.cos(th)
            cx, cy, ok = self._cell(wx, wy)
            out[j * MAP_CH:(j + 1) * MAP_CH] = self.grid[:, cy, cx] * ok      # outside the grid reads 0
        return out


# ---------------------------------------------------------------- per-episode feature state
class AgentSideFeatures:
    """Call reset(first_obs) at episode start, then step(action_idx, obs) after every env step.
    Each returns the network input arrays for the current observation."""

    def __init__(self, use_map: bool = True):
        self.use_map = use_map
        self.odo = Odometry()
        self.map = SpatialMap() if use_map else None

    def _features(self, obs, prev_action):
        f = obs_arrays(obs, prev_action)
        f["pose"] = np.array([self.odo.x, self.odo.y, self.odo.th], np.float32)
        if self.use_map:
            self.map.write((self.odo.x, self.odo.y, self.odo.th), f["dist"], f["hit"])
            f["crops"] = self.map.crops((self.odo.x, self.odo.y, self.odo.th))
        self._last = obs
        return f

    def reset(self, obs):
        self.odo.reset()
        if self.use_map:
            self.map.reset()
        return self._features(obs, 0)

    def step(self, action_idx: int, obs):
        prev = self._last
        self.odo.step(ACTIONS[action_idx], prev["vitals"][14:16], obs["vitals"][14:16],
                      np.asarray(prev["rays"]["distance"], np.float64), np.asarray(obs["rays"]["distance"], np.float64))
        return self._features(obs, action_idx)
