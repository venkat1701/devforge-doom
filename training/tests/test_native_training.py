import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(Path(__file__).parent)]
from test_learned_agent import observation
from doomrl import envs


class FakeEnv:
    """A contract-shaped engine substitute; no process or simulator side effects."""
    def reset(self, seed=None):
        self.t = 0
        return observation(), {}

    def privileged_state(self):
        return dict(tic=self.t * 4, episode_ticks=8, health=100., kills=self.t, items=0,
                    damage_taken=0., keys=np.zeros(6), dead=False, timeout=self.t == 2,
                    exit_reached=False)

    def step(self, buttons):
        self.t += 1
        return observation(self.t), .125, False, self.t == 2, {"exit_reached": False}

    def close(self):
        pass


def test_native_reward_no_geometric_targets(monkeypatch):
    monkeypatch.setattr(envs, "_make_env", lambda *a, **kw: FakeEnv())
    runner = envs.TrainEpisodeRunner(0, 8, False, False)
    out = runner.reset()
    assert "aux" not in out and "pose" not in out["f"]
    first, final = runner.step(0), runner.step(0)
    assert first["r"] == final["r"] == .125
    assert final["done"] and not final["term"] and final["trunc"]
    assert final["episode"]["return"] == .25
    assert final["f"] is None  # engine has no genuine final observation


def test_custom_rewards_refused():
    with pytest.raises(ValueError):
        envs.TrainEpisodeRunner(0, 8, False, False, reward={"cell": .02})


def test_uniform_manifest_rejects_dev_maps():
    from doomrl.maps import resolve_training_maps, SUITES
    assert resolve_training_maps(None) == SUITES["train_base"]
    with pytest.raises(ValueError):
        resolve_training_maps([SUITES["dev"][0]])
    with pytest.raises(ValueError):
        resolve_training_maps([])


def test_semantic_fingerprint_changes_and_ignores_learner_parameters():
    from doomrl.remote_envs import environment_spec
    cfg = {"seed": 1, "episode_ticks": [2100, 4200], "fast_env": False}
    base = environment_spec(cfg, 2, False, True)
    assert base == environment_spec(dict(cfg, ppo={"lr": 9.}), 2, False, True)
    for update in ({"seed": 2}, {"episode_ticks": [2100]}, {"fast_env": True}):
        assert base != environment_spec(dict(cfg, **update), 2, False, True)
    assert base != environment_spec(cfg, 2, False, False)


@pytest.mark.parametrize("remaining,expected", [(0, 0), (1, 0), (15, 0), (16, 1), (33, 2)])
def test_remaining_budget_never_overshoots(remaining, expected):
    from doomrl.ppo import validate_budget
    cfg = {"num_envs": 2, "total_steps": 100 + remaining, "start_step": 100,
           "ppo": {"num_steps": 8, "num_minibatches": 2, "update_epochs": 1,
                   "gamma": .99, "gae_lambda": .95}}
    assert validate_budget(cfg) == expected
    cfg["total_steps"] = 99
    with pytest.raises(ValueError):
        validate_budget(cfg)


def test_terminal_gae_never_uses_reset_episode_value():
    from doomrl.targets import gae
    rews = torch.tensor([[1.], [2.]])
    vals = torch.tensor([[3.], [1000.]])
    done = torch.ones(2, 1)
    returns = gae(rews, vals, done, torch.tensor([9999.]), .99, .95) + vals
    torch.testing.assert_close(returns, rews)
