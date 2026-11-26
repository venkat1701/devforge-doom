"""Side-by-side equivalence test: official DoomEnv vs FastDoomEnv.

Both envs get the same seed and the same random macro-action sequence (the explorer from
training/probes/collect.py: hold one of 12 macros for 1-12 steps, biased toward moving forward).
Every step compares every observation key, reward/terminated/truncated, info and
privileged_state(). Exits non-zero on any mismatch.

    uv run --group train python training/fastenv/test_equivalence.py            # ~1000 steps/map
    uv run --group train python training/fastenv/test_equivalence.py --steps 300 --maps two_rooms

Comparison rules: integer/bool arrays np.array_equal; float arrays np.allclose(atol=1e-5, rtol=0)
(the max abs diff and whether they were bit-identical are reported); rgb/depth np.array_equal.
On a mismatch the official Sensors are re-run on the fast env's own engine state, which tells a
fast-code bug apart from the two engine instances diverging.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import vizdoom as vd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # training/ -> import fastenv

from doom_env import make_env  # noqa: E402
from doom_env import spaces as S  # noqa: E402
from doom_env.sensors import LevelGeometry, Sensors  # noqa: E402
from fastenv import HAVE_NUMBA, make_fast_env  # noqa: E402
from fastenv.sensors_fast import FastLevelGeometry, FastSensors  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
IWAD = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
TWO_ROOMS = str(REPO / "maps" / "two_rooms.wad")

# Macro actions (MultiBinary(9): F B L R TL TR ATK USE WPN), same table as probes/collect.py.
MACROS = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # forward
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0],  # forward + turn left
    [1, 0, 0, 0, 0, 1, 0, 0, 0],  # forward + turn right
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # turn left
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # turn right
    [0, 0, 1, 0, 0, 0, 0, 0, 0],  # strafe left
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # strafe right
    [0, 1, 0, 0, 0, 0, 0, 0, 0],  # backward
    [1, 0, 0, 0, 0, 0, 1, 0, 0],  # forward + attack
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # use
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # idle
], dtype=np.int8)

# name -> (wad, map, render_buffers)
MAPS = {
    "two_rooms": (TWO_ROOMS, "MAP01", True),
    "MAP01": (IWAD, "MAP01", True),
    "MAP07": (IWAD, "MAP07", False),
    "MAP20": (IWAD, "MAP20", False),
}


class Explorer:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.hold = 0
        self.macro = MACROS[0]

    def __call__(self) -> np.ndarray:
        if self.hold <= 0:
            self.macro, self.hold = MACROS[self.rng.integers(len(MACROS))], int(self.rng.integers(1, 13))
        self.hold -= 1
        return self.macro


class Comparator:
    def __init__(self):
        self.max_float_diff = 0.0
        self.float_bit_identical = True
        self.mismatches: list[str] = []
        self.compared_arrays = 0

    def cmp(self, path: str, a, b) -> bool:
        if isinstance(a, dict) or isinstance(b, dict):
            if not (isinstance(a, dict) and isinstance(b, dict)) or a.keys() != b.keys():
                return self._fail(f"{path}: keys {sorted(a) if isinstance(a, dict) else type(a)} != "
                                  f"{sorted(b) if isinstance(b, dict) else type(b)}")
            ok = True
            for k in a:
                ok &= self.cmp(f"{path}.{k}", a[k], b[k])
            return ok
        if a is None or b is None:
            return True if a is None and b is None else self._fail(f"{path}: {a!r} vs {b!r}")
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            a, b = np.asarray(a), np.asarray(b)
            self.compared_arrays += 1
            if a.dtype != b.dtype or a.shape != b.shape:
                return self._fail(f"{path}: dtype/shape {a.dtype}{a.shape} vs {b.dtype}{b.shape}")
            if np.issubdtype(a.dtype, np.floating):
                fin = np.isfinite(a) & np.isfinite(b)
                diff = float(np.max(np.abs(a[fin].astype(np.float64) - b[fin].astype(np.float64)))) if fin.any() else 0.0
                self.max_float_diff = max(self.max_float_diff, diff)
                if not np.array_equal(a.view(np.uint8), b.view(np.uint8)):  # bitwise (sign of zero, NaN payload)
                    self.float_bit_identical = False
                if not np.allclose(a, b, atol=1e-5, rtol=0.0, equal_nan=True):
                    return self._fail(f"{path}: float max|diff|={diff:.3g}")
                return True
            if not np.array_equal(a, b):
                bad = np.argwhere(a != b)
                return self._fail(f"{path}: {len(bad)} elements differ, first at {bad[0].tolist()}: "
                                  f"{a[tuple(bad[0])]} vs {b[tuple(bad[0])]}")
            return True
        if isinstance(a, float) or isinstance(b, float):
            if not (a == b or (math.isnan(a) and math.isnan(b))):
                self.max_float_diff = max(self.max_float_diff, abs(a - b))
                return self._fail(f"{path}: {a!r} vs {b!r}")
            return True
        return True if (type(a) is type(b) and a == b) else self._fail(f"{path}: {a!r} ({type(a).__name__}) vs {b!r} ({type(b).__name__})")

    def _fail(self, msg: str) -> bool:
        self.mismatches.append(msg)
        return False


def diagnose(fast_env) -> str:
    """Re-run the official sensors on the fast env's current engine state."""
    g = fast_env._game
    if g.is_episode_finished():
        return "episode finished (obs is the cached last obs)"
    state = g.get_state()
    v = np.asarray(state.game_variables, np.float64)
    geo = LevelGeometry(state.sectors, fast_env.mapinfo)
    geo.update(state.sectors)
    heading = math.radians(v[11])
    rays, ents = Sensors(geo).compute(v[8], v[9], heading, (v[12], v[13]), state.objects)
    ref = {"rays": rays, "visible_entities": ents}
    got = {"rays": fast_env._last_obs["rays"], "visible_entities": fast_env._last_obs["visible_entities"]}
    c = Comparator()
    same = c.cmp("same_state", ref, got)
    return ("official sensors on the fast env's state agree with fast obs -> engine instances diverged"
            if same else "official sensors on the SAME state disagree -> fast code bug: " + "; ".join(c.mismatches[:5]))


def run_map(name: str, steps: int, seed: int, episode_ticks: int, render: bool | None) -> dict:
    wad, map_name, default_render = MAPS[name]
    render = default_render if render is None else render
    off = make_env(wad, map_name, episode_ticks=episode_ticks, render_buffers=render)
    fast = make_fast_env(wad, map_name, episode_ticks=episode_ticks, render_buffers=render)
    comp = Comparator()
    explorer = Explorer(seed)
    episodes, ep_seed = 1, seed
    t0 = time.perf_counter()
    try:
        o1, i1 = off.reset(seed=ep_seed)
        o2, i2 = fast.reset(seed=ep_seed)
        ok = comp.cmp("reset.obs", o1, o2) & comp.cmp("reset.info", i1, i2) & comp.cmp(
            "reset.priv", off.privileged_state(), fast.privileged_state())
        if not ok:
            print(f"  [{name}] reset mismatch: {comp.mismatches[:5]} | {diagnose(fast)}")
        geo_src = fast._geometry.source
        kernels = bool(getattr(fast._geometry, "use_kernels", False))
        n_steps = n_vis = n_bad_steps = max_vis = keys_held = 0
        hits = np.zeros(6, np.int64)
        for step in range(steps):
            a = explorer()
            r1 = off.step(a)
            r2 = fast.step(a)
            before = len(comp.mismatches)
            ok = comp.cmp("obs", r1[0], r2[0])
            ok &= comp.cmp("reward", float(r1[1]), float(r2[1]))
            ok &= comp.cmp("terminated", r1[2], r2[2])
            ok &= comp.cmp("truncated", r1[3], r2[3])
            ok &= comp.cmp("info", r1[4], r2[4])
            ok &= comp.cmp("priv", off.privileged_state(), fast.privileged_state())
            n_steps += 1
            nv = int(r1[0]["visible_entities"]["mask"].sum())
            n_vis += nv
            max_vis = max(max_vis, nv)
            hits += np.bincount(r1[0]["rays"]["hit_type"], minlength=6)
            keys_held = max(keys_held, int(r1[0]["vitals"][8:14].sum()))
            if not ok:
                n_bad_steps += 1
                if n_bad_steps <= 3:
                    print(f"  [{name}] step {step} mismatch: {comp.mismatches[before:before + 5]} | {diagnose(fast)}")
            if r1[2] or r1[3]:
                episodes += 1
                ep_seed += 1
                o1, i1 = off.reset(seed=ep_seed)
                o2, i2 = fast.reset(seed=ep_seed)
                if not (comp.cmp("reset.obs", o1, o2) & comp.cmp("reset.info", i1, i2)):
                    n_bad_steps += 1
    finally:
        off.close()
        fast.close()
    return {
        "map": name, "render": render, "geometry": geo_src, "kernels": kernels, "steps": n_steps,
        "episodes": episodes, "bad_steps": n_bad_steps, "max_float_diff": comp.max_float_diff,
        "bit_identical": comp.float_bit_identical, "arrays": comp.compared_arrays,
        "mean_visible": n_vis / max(n_steps, 1), "max_visible": max_vis, "hits": hits, "keys": keys_held,
        "mismatches": comp.mismatches, "secs": time.perf_counter() - t0,
    }


# ------------------------------------------------------------------------------ pure-function fuzz
class _Obj:
    __slots__ = ("id", "name", "category", "position_x", "position_y", "velocity_x", "velocity_y")

    def __init__(self, oid, name, category, x, y, vx, vy):
        self.id, self.name, self.category = oid, name, category
        self.position_x, self.position_y, self.velocity_x, self.velocity_y = x, y, vx, vy


class _Sec:
    __slots__ = ("id", "floor_height", "ceiling_height")

    def __init__(self, sid, f, c):
        self.id, self.floor_height, self.ceiling_height = sid, f, c


def fuzz_map(name: str, samples: int, seed: int) -> dict:
    """Official vs fast LevelGeometry.update + Sensors.compute on the same random inputs.

    The side-by-side run only sees what a random explorer meets. This drives the pure functions
    with adversarial inputs: perturbed sector heights around the 24-unit step / 56-unit opening /
    closed-door thresholds, poses on and near vertices, headings that aim rays at vertices,
    crowds of entities (> MAX_ENTITIES, equal distances, at the player, at 4096 units), for the
    WAD geometry path, the engine geometry path (mapinfo=None) and unordered sector lists.
    """
    wad, map_name, _ = MAPS[name]
    env = make_env(wad, map_name, render_buffers=False)
    rng = np.random.default_rng(seed)
    comp = Comparator()
    bad = full = 0
    hits = np.zeros(6, np.int64)
    try:
        env.reset(seed=seed)
        state = env._game.get_state()
        real_secs = list(state.sectors)
        kinds = sorted({(o.name, o.category) for o in state.objects})
        kinds += [("DoomImpBall", "Projectile"), ("Rocket", "Projectile"), ("RedSkull", "Key"), ("DoomPlayer", "Self")]
        variants = {
            "wad": (real_secs, env.mapinfo),
            "engine": (real_secs, None),
            "unordered": (list(rng.permutation(np.array(real_secs, dtype=object))), env.mapinfo),
        }
        geos = {}
        for vname, (secs, mi) in variants.items():
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    g_off = LevelGeometry(secs, mi)
                    g_fast = FastLevelGeometry.from_official(LevelGeometry(secs, mi))
            geos[vname] = (secs, g_off, g_fast, Sensors(g_off), FastSensors(g_fast))
        base_f = {s.id: s.floor_height for s in real_secs}
        base_c = {s.id: s.ceiling_height for s in real_secs}
        for i in range(samples):
            vname = ("wad", "engine", "unordered")[i % 3]
            secs, g_off, g_fast, s_off, s_fast = geos[vname]
            # --- sector heights
            fake = []
            for s in secs:
                f, c = base_f[s.id], base_c[s.id]
                if rng.random() < 0.3:
                    f = f + rng.choice([0.0, 8.0, 24.0, 24.000001, 23.999999, 25.0, -24.0, -24.5, 100.0])
                    c = f + rng.choice([0.0, -1.0, 55.999, 56.0, 56.001, 72.0, 128.0, c - f])
                fake.append(_Sec(s.id, float(f), float(c)))
            g_off.update(fake)
            g_fast.update(fake)
            ok = comp.cmp("geo", {k: getattr(g_off, k) for k in ("floor", "ceil", "move_block", "sight_block", "hit_type")},
                          {k: getattr(g_fast, k) for k in ("floor", "ceil", "move_block", "sight_block", "hit_type")})
            # --- pose
            j = int(rng.integers(len(g_off.p1)))
            base = (g_off.p1[j], g_off.p2[j], 0.5 * (g_off.p1[j] + g_off.p2[j]))[int(rng.integers(3))]
            r = rng.random()
            off_xy = np.zeros(2) if r < 0.15 else rng.normal(0, 4 if r < 0.4 else 96, 2)
            px, py = float(base[0] + off_xy[0]), float(base[1] + off_xy[1])
            r = rng.random()
            if r < 0.2:
                heading = float(rng.integers(64)) * (2 * np.pi / 64)
            elif r < 0.45:  # aim ray 0 exactly at a nearby vertex
                v = g_off.p1[int(rng.integers(len(g_off.p1)))]
                heading = math.atan2(v[1] - py, v[0] - px)
            else:
                heading = float(rng.uniform(-np.pi, 3 * np.pi))
            pvel = (float(rng.normal(0, 8)), float(rng.normal(0, 8)))
            # --- objects
            objs = []
            n_obj = int(rng.choice([0, 1, 5, 20, 60, 120]))
            crowd = rng.random() < 0.35  # pack entities into the view cone, close by (> 32 visible)
            for k in range(n_obj):
                nm, cat = kinds[int(rng.integers(len(kinds)))]
                r = rng.random()
                if r < 0.05:
                    x, y = px, py  # at the player -> dropped (edist <= 1e-6)
                elif r < 0.10:
                    x, y = px + 4096.0, py  # exactly at the entity range limit
                elif r < 0.25 and objs:  # same position as a previous one -> distance ties
                    x, y = objs[-1].position_x, objs[-1].position_y
                elif crowd:
                    ang, dd = heading + rng.uniform(-0.8, 0.8), rng.uniform(1, 120)
                    x, y = px + dd * math.cos(ang), py + dd * math.sin(ang)
                else:
                    ang, dd = rng.uniform(0, 2 * np.pi), rng.choice([rng.uniform(1, 300), rng.uniform(1, 1500), rng.uniform(1, 4500)])
                    x, y = px + dd * math.cos(ang), py + dd * math.sin(ang)
                objs.append(_Obj(1000 + k, nm, cat, float(x), float(y), float(rng.normal(0, 10)), float(rng.normal(0, 10))))
            ro, eo = s_off.compute(np.float64(px), np.float64(py), heading, pvel, objs)
            rf, ef = s_fast.compute(np.float64(px), np.float64(py), heading, pvel, objs)
            hits += np.bincount(ro["hit_type"], minlength=6)
            full += int(eo["mask"].sum() == S.MAX_ENTITIES)
            ok &= comp.cmp("rays", ro, rf) & comp.cmp("ents", eo, ef) & comp.cmp("keys", s_off.present_keys, s_fast.present_keys)
            if not ok:
                bad += 1
                if bad <= 3:
                    print(f"  [fuzz {name}/{vname}] sample {i} mismatch: {comp.mismatches[-5:]}")
    finally:
        env.close()
    return {"map": name, "samples": samples, "bad": bad, "hits": hits, "full": full,
            "sources": {k: v[1].source + ("/ordered" if v[1].ordered else "/unordered") for k, v in geos.items()}, "max_float_diff": comp.max_float_diff,
            "bit_identical": comp.float_bit_identical, "mismatches": comp.mismatches}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=list(MAPS), choices=list(MAPS))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--episode-ticks", type=int, default=2100)
    ap.add_argument("--render", choices=("default", "on", "off"), default="default")
    ap.add_argument("--fuzz", type=int, default=600, help="pure-function fuzz samples per map (0 = skip)")
    args = ap.parse_args()
    render = {"default": None, "on": True, "off": False}[args.render]
    print(f"numba kernels available: {HAVE_NUMBA}")
    results = [run_map(m, args.steps, args.seed + 100 * i, args.episode_ticks, render) for i, m in enumerate(args.maps)]
    print(f"\n{'map':10s} {'render':6s} {'geom':6s} {'kern':5s} {'steps':>6s} {'eps':>4s} {'vis/step':>8s} "
          f"{'maxvis':>6s} {'keys':>4s} {'arrays':>7s} {'max|dfloat|':>11s} {'bitident':>8s} {'bad':>4s}  ray hits none/wall/door/locked/enemy/item")
    failed = False
    for r in results:
        print(f"{r['map']:10s} {str(r['render']):6s} {r['geometry']:6s} {str(r['kernels']):5s} {r['steps']:6d} "
              f"{r['episodes']:4d} {r['mean_visible']:8.2f} {r['max_visible']:6d} {r['keys']:4d} {r['arrays']:7d} "
              f"{r['max_float_diff']:11.3g} {str(r['bit_identical']):8s} {r['bad_steps']:4d}  {'/'.join(map(str, r['hits']))}")
        if r["bad_steps"] or r["mismatches"]:
            failed = True
            for m in r["mismatches"][:10]:
                print(f"    {m}")
    if args.fuzz:
        print(f"\nfuzz (official vs fast sensors on identical random inputs, {args.fuzz} samples/map):")
        for i, m in enumerate(args.maps):
            fr = fuzz_map(m, args.fuzz, 1000 + i)
            print(f"  {m:10s} samples={fr['samples']:5d} bad={fr['bad']:3d} max|dfloat|={fr['max_float_diff']:.3g} "
                  f"bit_identical={fr['bit_identical']} samples_with_32_visible={fr['full']} "
                  f"ray hits={'/'.join(map(str, fr['hits']))} geometry={fr['sources']}")
            if fr["bad"]:
                failed = True
                for msg in fr["mismatches"][:10]:
                    print(f"    {msg}")
    print("\nRESULT:", "MISMATCH" if failed else "all observations identical")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
