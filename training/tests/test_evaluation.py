import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]


def episodes(delta=0.):
    return [dict(map=m, seed=s, ticks=t, exit=0., keys=delta, kills=1., dead=0., steps=20.,
                 timeout=1., health=100., native_return=delta, ms_mean=1., ms_max=2., cells=5., doors_opened=0., locked_opened=0.,
                 damage=10., distance=100., revisit_ratio=.5, proxy_score=delta,
                 fire_rate=.4, blind_fire=.5, ammo_spent=20., ended_dry=0., on_target_fire=.3)
            for m in ("a", "b") for s in (1, 2) for t in (2100, 4200)]


def test_cap_stratification_and_zero_success_caution():
    from doomrl.evaluate import summarize_episodes
    result = summarize_episodes(episodes())
    assert set(result["by_ticks"]) == {"2100", "4200"}
    assert result["summary"]["exit"]["ci95"] is None
    assert result["summary"]["exit"]["observed_successes"] == 0
    assert result["summary"]["exit"]["independent_maps"] == 2


def test_paired_comparison_matches_jobs_not_order():
    from doomrl.evaluate import compare_paired
    result = compare_paired(episodes(), list(reversed(episodes(.5))))
    assert result["2100"]["keys"]["mean_difference"] == .5
    assert result["4200"]["keys"]["ci95"] == [.5, .5]
    with pytest.raises(ValueError, match="paired"):
        compare_paired(episodes(), episodes(.5)[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        compare_paired(episodes(), episodes(.5) + episodes(.5)[:1])


def test_empty_evaluation_rejected():
    from doomrl.evaluate import summarize_episodes
    with pytest.raises(ValueError):
        summarize_episodes([])
