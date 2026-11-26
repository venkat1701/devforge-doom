"""Map suites and curriculum.

Three disjoint suites (review fix: the earlier probes reused MAP17-24 for every decision, which made it
a development set, not a test set):
  train  Freedoom 2 MAP01-16, Freedoom 1 E1M1-E4M9 (36 maps, 2-6 locked doors each on the ones checked),
         and the stock ViZDoom scenarios (combat / navigation drills)
  dev    Freedoom 2 MAP17-24: model selection, ablations, checkpoints
  test   Freedoom 2 MAP25-32: UNTOUCHED until the final report. evaluate.py refuses to run it unless
         --final is passed, and logs every time it is used.

Curriculum (Wu & Tian 2017 won ViZDoom 2016 Track 1 with A3C + curriculum, see training/docs/DESIGN.md):
stages are sampled by global env step; each stage is a weighted pool of maps.
"""

from __future__ import annotations

import os

import vizdoom as vd

_V = os.path.dirname(vd.__file__)
FD1 = os.path.join(_V, "freedoom1.wad")
FD2 = os.path.join(_V, "freedoom2.wad")
SCEN = lambda name: os.path.join(_V, "scenarios", f"{name}.wad")  # noqa: E731

FD2_TRAIN = [(FD2, f"MAP{i:02d}") for i in range(1, 17)]
FD1_ALL = [(FD1, f"E{e}M{m}") for e in range(1, 5) for m in range(1, 10)]
DRILLS = [(SCEN(n), "MAP01") for n in ("defend_the_center", "defend_the_line", "health_gathering",
                                        "my_way_home", "deadly_corridor")]

SUITES = {
    "train": FD2_TRAIN + FD1_ALL + DRILLS,
    "dev": [(FD2, f"MAP{i:02d}") for i in range(17, 25)],
    "test": [(FD2, f"MAP{i:02d}") for i in range(25, 33)],
}

# (start_step, [(maps, weight), ...]): later stages keep earlier material in the mix
CURRICULUM = [
    (0, [(DRILLS, 0.5), (FD2_TRAIN[:6] + FD1_ALL[:6], 0.5)]),
    (2_000_000, [(DRILLS, 0.15), (FD2_TRAIN[:10] + FD1_ALL[:18], 0.85)]),
    (6_000_000, [(DRILLS, 0.05), (FD2_TRAIN + FD1_ALL, 0.95)]),
]


def stage_at(step: int, curriculum=CURRICULUM):
    pools = curriculum[0][1]
    for start, p in curriculum:
        if step >= start:
            pools = p
    return pools


def sample_map(rng, step: int, curriculum=CURRICULUM):
    pools = stage_at(step, curriculum)
    w = [wt for _, wt in pools]
    maps = pools[rng.choice(len(pools), p=[x / sum(w) for x in w])][0]
    return maps[rng.integers(len(maps))]


def assert_disjoint() -> None:
    tr, dv, te = (set(SUITES[k]) for k in ("train", "dev", "test"))
    assert not (tr & dv) and not (tr & te) and not (dv & te), "suites overlap"
    cur = {m for _, pools in CURRICULUM for maps, _ in pools for m in maps}
    assert cur <= tr, "curriculum uses non-train maps"
