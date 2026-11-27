"""Scoring-category progress tracker and the progress_v1 training reward."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(Path(__file__).parent)]
from doomrl.progress import CELL, PROGRESS_V1, EpisodeProgress, resolve_shaping
from doomrl import envs
from test_native_training import FakeEnv


def state(x=0., y=0., keys=(0, 0, 0), kills=0, dmg=0., ceil=(0., 0., 0.), lock=(-1, 0, 2), exit=False,
          dead=False, health=100., tic=0, cap=2100):
    return {"x": x, "y": y, "keys": np.array(keys, float), "kills": kills, "damage_taken": dmg,
            "sector_ceiling": np.array(ceil), "door_lock": np.array(lock), "exit_reached": exit,
            "dead": dead, "health": health, "tic": tic, "episode_ticks": cap, "items": 0, "timeout": False}


def test_cells_count_new_areas_once_and_spinning_earns_nothing():
    pr = EpisodeProgress(state())
    assert pr.update(state(x=10))["cells"] == 0          # same cell
    assert pr.update(state(x=CELL + 1))["cells"] == 1     # new cell
    assert pr.update(state(x=1))["cells"] == 0            # revisit
    assert pr.update(state(x=1))["cells"] == 0            # standing/spinning in place
    assert pr.summary()["cells"] == 2


def test_doors_unlocks_keys_kills_damage():
    pr = EpisodeProgress(state())
    d = pr.update(state(ceil=(100., 30., 0.)))            # sector 0 not a door; sector 1 ordinary door opened
    assert d["doors"] == 1 and d["unlocks"] == 0
    assert pr.update(state(ceil=(100., 30., 0.)))["doors"] == 0   # counted once
    d = pr.update(state(keys=(1, 0, 0), ceil=(100., 30., 64.), kills=2, dmg=15.))
    assert d["unlocks"] == 1 and d["keys"] == 1 and d["kills"] == 2 and d["damage"] == 15.
    s = pr.summary()
    assert (s["doors_opened"], s["locked_opened"], s["keys_picked"], s["kills"], s["damage"]) == (1, 1, 1., 2, 15.)


def test_small_ceiling_motion_is_not_a_door_opening():
    pr = EpisodeProgress(state())
    assert pr.update(state(ceil=(0., 10., 0.)))["doors"] == 0


def test_bonus_weights_and_exit_is_largest():
    w = PROGRESS_V1
    pr = EpisodeProgress(state())
    d = pr.update(state(x=CELL * 2, keys=(1, 0, 0), kills=1, dmg=10.))
    assert pr.bonus(d, w) == pytest.approx(w["cell"] + w["key"] + w["kill"] + 10 * w["damage"])
    pr = EpisodeProgress(state())
    d = pr.update(state(exit=True, health=50., tic=1050, cap=2100))
    assert pr.bonus(d, w) == pytest.approx(w["exit"] + w["exit_health"] * .5 + w["exit_time"] * .5)
    assert w["exit"] + 1.0 > max(w["key"], w["unlock"], 100 * w["cell"])


def test_resolve_shaping():
    assert resolve_shaping(None) is None and resolve_shaping("official-wrapper") is None
    assert resolve_shaping("progress_v1") is PROGRESS_V1
    with pytest.raises(ValueError):
        resolve_shaping({"cell": .02})
    with pytest.raises(ValueError):
        resolve_shaping("made_up")


class MovingEnv(FakeEnv):
    """FakeEnv that walks one cell per step and reports positions/geometry."""
    def privileged_state(self):
        p = super().privileged_state()
        p.update(x=self.t * CELL, y=0., sector_ceiling=np.zeros(2), door_lock=np.array([-1, 0]))
        return p


def test_runner_adds_shaping_and_logs_native_and_categories(monkeypatch):
    monkeypatch.setattr(envs, "_make_env", lambda *a, **kw: MovingEnv())
    runner = envs.TrainEpisodeRunner(0, 8, False, False, reward="progress_v1")
    runner.reset()
    first, final = runner.step(0), runner.step(0)
    # wrapper reward .125/step (FakeEnv) + one new cell per step; FakeEnv kills = t -> one kill per step
    expected = .125 + PROGRESS_V1["cell"] + PROGRESS_V1["kill"]
    assert first["r"] == pytest.approx(expected) and final["r"] == pytest.approx(expected)
    ep = final["episode"]
    assert ep["native_return"] == pytest.approx(.25) and ep["return"] == pytest.approx(2 * expected)
    assert ep["cells"] == 3 and ep["doors_opened"] == 0 and ep["locked_opened"] == 0


def test_runner_native_default_unchanged(monkeypatch):
    monkeypatch.setattr(envs, "_make_env", lambda *a, **kw: MovingEnv())
    runner = envs.TrainEpisodeRunner(0, 8, False, False)
    runner.reset()
    assert runner.step(0)["r"] == .125
    with pytest.raises(ValueError):
        envs.TrainEpisodeRunner(0, 8, False, False, reward="made_up")


def test_distance_revisit_and_component_breakdown():
    w = PROGRESS_V1
    pr = EpisodeProgress(state())
    pr.bonus(pr.update(state(x=CELL)), w)        # new cell, moved CELL units
    pr.bonus(pr.update(state(x=0.)), w)          # back to the first cell
    pr.bonus(pr.update(state(x=0., kills=1)), w)
    s = pr.summary()
    assert s["distance"] == pytest.approx(2 * CELL)
    assert s["revisit_ratio"] == pytest.approx(1 - 1 / 3)
    assert s["shaping"]["cell"] == pytest.approx(w["cell"]) and s["shaping"]["kill"] == pytest.approx(w["kill"])
    assert s["shaping"]["exit"] == 0.0


def test_v2_keeps_every_v1_term_and_adds_blind_fire_penalty():
    from doomrl.progress import PROGRESS_V2
    for k, v in PROGRESS_V1.items():
        if k not in ("name", "unlock"):
            assert PROGRESS_V2[k] == v
    assert PROGRESS_V2["unlock"] == 1.0 and PROGRESS_V2["blind_ammo"] == -0.01
    pr = EpisodeProgress(state())
    assert pr.fired(5, True, PROGRESS_V2) == 0.0                 # shooting at a visible enemy is free
    assert pr.fired(3, False, PROGRESS_V2) == pytest.approx(-0.03)
    assert pr.fired(0, False, PROGRESS_V2) == 0.0                # no ammo spent, no penalty
    assert pr.fired(-10, False, PROGRESS_V2) == 0.0              # pickups are not negative spending
    s = pr.summary()
    assert s["ammo_spent"] == 8 and s["blind_ammo"] == 3 and s["shaping"]["blind_ammo"] == pytest.approx(-0.03)


def test_enemy_in_view_from_entities_or_rays():
    from doom_env.spaces import ENTITY_TYPES
    from doomrl.progress import enemy_in_view, total_ammo
    enemy = next(i for i, t in enumerate(ENTITY_TYPES) if t.startswith("enemy"))
    other = next(i for i, t in enumerate(ENTITY_TYPES) if not t.startswith("enemy"))
    base = {"rays": {"hit_type": np.ones(64, int)}, "vitals": np.arange(17, dtype=np.float32),
            "visible_entities": {"entity_type": np.full(32, other), "mask": np.zeros(32, np.float32)}}
    assert not enemy_in_view(base)
    seen = dict(base, visible_entities={"entity_type": np.full(32, enemy), "mask": np.r_[1., np.zeros(31)]})
    assert enemy_in_view(seen)
    masked = dict(base, visible_entities={"entity_type": np.full(32, enemy), "mask": np.zeros(32)})
    assert not enemy_in_view(masked)
    ray = dict(base, rays={"hit_type": np.r_[4, np.ones(63, int)]})
    assert enemy_in_view(ray)
    assert total_ammo(base) == 4 + 5 + 6 + 7


class AmmoEnv(FakeEnv):
    """Spends 2 rounds per step with no enemy in view."""
    def reset(self, seed=None):
        o, i = super().reset(seed)
        self.ammo = 50.
        return self._fix(o), i

    def _fix(self, o):
        o["vitals"] = np.zeros(17, np.float32); o["vitals"][4] = self.ammo
        o["rays"]["hit_type"] = np.ones(64, int)
        return o

    def step(self, buttons):
        o, r, term, trunc, info = super().step(buttons)
        self.ammo -= 2
        return self._fix(o), r, term, trunc, info

    def privileged_state(self):
        p = super().privileged_state()
        p.update(x=0., y=0., sector_ceiling=np.zeros(2), door_lock=np.array([-1, 0]))
        return p


def test_runner_v2_penalizes_blind_fire_but_not_on_terminal_step(monkeypatch):
    from doomrl.progress import PROGRESS_V2
    monkeypatch.setattr(envs, "_make_env", lambda *a, **kw: AmmoEnv())
    runner = envs.TrainEpisodeRunner(0, 8, False, False, reward="progress_v2")
    runner.reset()
    first, final = runner.step(0), runner.step(0)
    assert first["r"] == pytest.approx(.125 + PROGRESS_V2["kill"] + 2 * PROGRESS_V2["blind_ammo"])
    assert final["r"] == pytest.approx(.125 + PROGRESS_V2["kill"])     # terminal obs is stale: no ammo term
    assert final["episode"]["blind_ammo"] == 2 and final["episode"]["ammo_spent"] == 2


def _obs(ent_bearings=(), ray_hits=()):
    from doom_env.spaces import ENTITY_TYPES
    enemy = next(i for i, t in enumerate(ENTITY_TYPES) if t.startswith("enemy"))
    f = np.zeros((32, 5), np.float32); et = np.zeros(32, int); m = np.zeros(32, np.float32)
    for i, b in enumerate(ent_bearings):
        f[i] = [.2, np.sin(np.radians(b)), np.cos(np.radians(b)), 0, 0]; et[i] = enemy; m[i] = 1
    ht = np.ones(64, int)
    for k in ray_hits:
        ht[k] = 4
    return {"visible_entities": {"features": f, "entity_type": et, "mask": m}, "rays": {"hit_type": ht}}


def test_aim_cone_uses_entity_bearing_and_ray_angles():
    from doomrl.progress import enemy_in_view
    assert enemy_in_view(_obs(ent_bearings=[5.]), 8.0) and not enemy_in_view(_obs(ent_bearings=[20.]), 8.0)
    assert enemy_in_view(_obs(ent_bearings=[-7.]), 8.0)
    assert enemy_in_view(_obs(ray_hits=[0]), 8.0) and enemy_in_view(_obs(ray_hits=[63]), 8.0)   # 0 and -5.6 deg
    assert not enemy_in_view(_obs(ray_hits=[32]), 8.0)        # directly behind the agent
    assert not enemy_in_view(_obs(ray_hits=[32]), 45.0)
    assert enemy_in_view(_obs(ray_hits=[32]), None)           # v2's legacy definition counted rays behind
    assert enemy_in_view(_obs(ent_bearings=[20.]), None)


def test_v3_only_changes_the_aim_cone():
    from doomrl.progress import PROGRESS_V2, PROGRESS_V3
    assert {k for k in PROGRESS_V3 if PROGRESS_V3[k] != PROGRESS_V2.get(k)} == {"name", "aim_cone_deg"}
    assert PROGRESS_V3["aim_cone_deg"] == 8.0 and PROGRESS_V2["aim_cone_deg"] is None


LAYOUT = {"rooms": {"0,0": [-200, -200, 200, 200], "1,0": [376, -200, 776, 200], "1,1": [376, 376, 776, 776]},
          "links": [{"a": [0, 0], "b": [1, 0], "lock": 1, "portal": [288, 0], "rect": [200, -32, 376, 32]},
                    {"a": [1, 0], "b": [1, 1], "lock": 0, "portal": [576, 288], "rect": [544, 200, 608, 376]}],
          "keys": [], "start_cell": [0, 0], "exit_cell": [1, 1], "exit": [576, 576], "start": [-136, 0]}


def test_route_distance_and_potential_follow_the_path():
    from doomrl.progress import RouteGuide, ROUTE_SCALE
    g = RouteGuide(LAYOUT)
    none = np.zeros(6)
    d0 = g.remaining(-136, 0, none)
    expected = np.hypot(288 + 136, 0) + np.hypot(576 - 288, 288) + np.hypot(0, 576 - 288)
    assert d0 == pytest.approx(expected)
    assert g.remaining(500, 0, none) < d0 and g.remaining(576, 500, none) < g.remaining(500, 0, none)
    assert g.remaining(576, 576, none) == pytest.approx(0.0)
    assert g.potential({"x": -136, "y": 0, "keys": none}) == pytest.approx(-expected / ROUTE_SCALE)
    assert g.remaining(300, 0, none) == pytest.approx(np.hypot(576 - 300, 288) + 288)    # inside the corridor


def test_route_targets_key_first_then_exit():
    from doomrl.progress import RouteGuide
    lay = dict(LAYOUT, keys=[{"lock": 1, "pos": [0, 150], "cell": [0, 0]}])
    g = RouteGuide(lay)
    none, red = np.zeros(6), np.array([0, 0, 1, 0, 0, 0])
    via_key = g.remaining(0, 150, none)                            # standing on the key, card not yet held
    assert g.remaining(0, 0, none) == pytest.approx(150 + via_key)  # route goes to the key first
    assert g.remaining(0, 150, red) == pytest.approx(via_key)      # picking the card up changes nothing
    assert g.remaining(0, 0, red) < g.remaining(0, 0, none)         # once held: straight to the exit


def test_v4_adds_only_route():
    from doomrl.progress import PROGRESS_V3, PROGRESS_V4
    assert {k for k in PROGRESS_V4 if PROGRESS_V4[k] != PROGRESS_V3.get(k)} == {"name", "route"}


def test_generated_maps_have_layouts_and_freedoom_does_not():
    from doomrl import maps
    if not maps.SUITES["gen_train"]:
        pytest.skip("generated maps not built")
    w, m = maps.SUITES["gen_train"][3]
    assert maps.gen_layout(w, m) is not None and maps.gen_layout(*maps.SUITES["train_base"][0]) is None


def test_v5_press_penalty_only_without_target_and_keeps_v4():
    from doomrl.progress import PROGRESS_V4, PROGRESS_V5
    assert {k for k in PROGRESS_V5 if PROGRESS_V5[k] != PROGRESS_V4.get(k)} == {"name", "blind_ammo", "blind_press"}
    pr = EpisodeProgress(state())
    assert pr.pressed(True, True, PROGRESS_V5) == 0.0           # enemy in the aim cone: free
    assert pr.pressed(False, False, PROGRESS_V5) == 0.0         # not firing: free
    assert pr.pressed(True, False, PROGRESS_V5) == pytest.approx(-0.003)
    assert pr.pressed(True, False, PROGRESS_V4) == 0.0          # v4 has no press term
    s = pr.summary()
    assert s["blind_presses"] == 1 and s["shaping"]["blind_press"] == pytest.approx(-0.003)


def test_runner_v5_penalizes_blind_press(monkeypatch):
    from doomrl.progress import PROGRESS_V5
    from doom_model.actions import ACTIONS
    fire = next(i for i, a in enumerate(ACTIONS) if a[6])
    monkeypatch.setattr(envs, "_make_env", lambda *a, **kw: AmmoEnv())
    runner = envs.TrainEpisodeRunner(0, 8, False, False, reward="progress_v5")
    runner.reset()
    first = runner.step(fire)
    assert first["r"] == pytest.approx(.125 + PROGRESS_V5["kill"] + 2 * PROGRESS_V5["blind_ammo"] + PROGRESS_V5["blind_press"])


def test_v6_fist_and_time_costs():
    from doomrl.progress import PROGRESS_V5, PROGRESS_V6
    assert {k for k in PROGRESS_V6 if PROGRESS_V6[k] != PROGRESS_V5.get(k)} == {"name", "fist_with_ammo", "step"}
    pr = EpisodeProgress(state())
    v = np.zeros(17, np.float32)
    v[2], v[4] = 1, 20                      # fists, 20 bullets
    assert pr.tick({"vitals": v}, PROGRESS_V6) == pytest.approx(-0.005 - 0.001)
    v[4] = 0                                # fists with no bullets: only the time cost
    assert pr.tick({"vitals": v}, PROGRESS_V6) == pytest.approx(-0.001)
    v[2], v[4] = 2, 20                      # pistol with bullets
    assert pr.tick({"vitals": v}, PROGRESS_V6) == pytest.approx(-0.001)
    assert pr.tick({"vitals": v}, PROGRESS_V5) == 0.0
    assert pr.summary()["shaping"]["fist_with_ammo"] == pytest.approx(-0.005)


def test_v7_only_strengthens_trigger_discipline():
    from doomrl.progress import PROGRESS_V6, PROGRESS_V7
    assert {k for k in PROGRESS_V7 if PROGRESS_V7[k] != PROGRESS_V6.get(k)} == {"name", "blind_press", "blind_ammo"}
    assert PROGRESS_V7["blind_press"] == -0.01 and PROGRESS_V7["blind_ammo"] == -0.05
