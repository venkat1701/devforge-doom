"""Disjoint map suites and uniform sampling from an explicit training manifest."""

from __future__ import annotations

import os
from pathlib import Path
import json

import vizdoom as vd

_V = os.path.dirname(vd.__file__)
FD1 = os.path.join(_V, "freedoom1.wad")
FD2 = os.path.join(_V, "freedoom2.wad")
SCEN = lambda name: os.path.join(_V, "scenarios", f"{name}.wad")  # noqa: E731

FD2_TRAIN = [(FD2, f"MAP{i:02d}") for i in range(1, 17)]
FD1_ALL = [(FD1, f"E{e}M{m}") for e in range(1, 5) for m in range(1, 10)]
DRILLS = [(SCEN(n), "MAP01") for n in ("defend_the_center", "defend_the_line", "health_gathering",
                                        "my_way_home", "deadly_corridor")]

# Procedurally generated small maps (training/mapgen/generate.py; disjoint seed ranges per split).
GEN_DIR = Path(__file__).resolve().parents[1] / "maps_gen"
TWO_ROOMS = (str(Path(__file__).resolve().parents[2] / "maps" / "two_rooms.wad"), "MAP01")


_LAYOUTS = None


def gen_layout(wad, map_name):
    """Layout of a generated map (training-only route shaping), or None for any other map."""
    global _LAYOUTS
    if _LAYOUTS is None:
        _LAYOUTS = {}
        for index in GEN_DIR.glob("*_maps.json"):
            for m in json.loads(index.read_text()):
                if "layout" in m:
                    _LAYOUTS[(str(GEN_DIR / m["wad"]), m["map"])] = m["layout"]
    return _LAYOUTS.get((str(wad), map_name))


def _gen(split):
    index = GEN_DIR / f"{split}_maps.json"
    if not index.exists():
        return []
    return [(str(GEN_DIR / m["wad"]), m["map"]) for m in json.loads(index.read_text())]


SUITES = {
    "train_base": FD2_TRAIN + FD1_ALL + DRILLS,   # the default manifest (configs/train_maps.json)
    "train": FD2_TRAIN + FD1_ALL + DRILLS + _gen("train"),
    "gen_train": _gen("train"),
    "gen_dev": _gen("dev") + [TWO_ROOMS],   # held out; two_rooms is the template's (non-evaluation) test map
    "gen_test": _gen("test"),               # reserved for the final pick
    "dev": [(FD2, f"MAP{i:02d}") for i in range(17, 25)],
    "test": [(FD2, f"MAP{i:02d}") for i in range(25, 33)],
}

def resolve_training_maps(manifest):
    if manifest is None:
        manifest = Path(__file__).parents[1] / "configs" / "train_maps.json"
    if isinstance(manifest, str) and manifest.startswith("suite:"):
        manifest = SUITES[manifest.split(":", 1)[1]]
    elif isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    maps = [(str(Path(w) if Path(w).is_absolute() else Path(_V) / w), m) for w, m in manifest]
    if not maps or len(set(maps)) != len(maps) or not set(maps) <= set(SUITES["train"]):
        raise ValueError("training manifest must contain unique train-suite maps only")
    return maps


def sample_map(rng, step=0, maps=None):
    pool = resolve_training_maps(None) if maps is None else maps
    return pool[int(rng.integers(len(pool)))]


def assert_disjoint() -> None:
    tr, dv, te = (set(SUITES[k]) for k in ("train", "dev", "test"))
    assert not (tr & dv) and not (tr & te) and not (dv & te), "suites overlap"
    gd, gt = set(SUITES["gen_dev"]), set(SUITES["gen_test"])
    assert not (tr & gd) and not (tr & gt) and not (gd & gt), "generated suites overlap"
    assert set(resolve_training_maps(None)) <= tr, "manifest uses non-train maps"
