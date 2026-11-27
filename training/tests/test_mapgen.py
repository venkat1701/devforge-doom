"""Generated small maps: deterministic, tiered, disjoint splits, parsed by the official wrapper."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(ROOT / "env")]
from mapgen.generate import generate
from doomrl import maps


def test_deterministic_and_tier_structure():
    for tier in (1, 2, 3, 4):
        a, ma = generate(1234 + tier, tier)
        b, mb = generate(1234 + tier, tier)
        assert a == b and ma == mb
        assert ma["locks"] == {1: 0, 2: 0, 3: 1, 4: min(2, ma["path_len"])}[tier]
        assert 'special = 243' in a and 'namespace = "zdoom"' in a


def test_suites_disjoint_and_manifest_suite_reference():
    maps.assert_disjoint()
    if not maps.SUITES["gen_train"]:
        pytest.skip("generated maps not built")
    got = maps.resolve_training_maps("suite:train")
    assert set(maps.SUITES["gen_train"]) <= set(got) and not set(maps.SUITES["gen_dev"]) & set(got)
    assert maps.TWO_ROOMS in maps.SUITES["gen_dev"]


def test_wrapper_parses_generated_doors_and_locks():
    if not maps.SUITES["gen_dev"]:
        pytest.skip("generated maps not built")
    from doom_env import make_env
    metas = json.loads((maps.GEN_DIR / "dev_maps.json").read_text())
    for m in [x for x in metas if x["tier"] in (3, 4)][:3]:
        env = make_env(str(maps.GEN_DIR / m["wad"]), m["map"], episode_ticks=2100)
        try:
            env.reset(seed=1)
            lock = env.privileged_state()["door_lock"]
            assert int(np.sum(lock >= 0)) == m["doors"] and int(np.sum(lock > 0)) == m["locks"]
        finally:
            env.close()
