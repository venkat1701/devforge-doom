"""Training-only curiosity must not change baseline policies or reward resets."""
import copy
import dataclasses
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(Path(__file__).parent)]

from doom_model.artifacts import load_policy
from doomrl.rnd import RND, RNDConfig, RunningMoments
from test_pixels import small_config
from test_resume_integrity import run


def rnd_config():
    return RNDConfig(coefficient=.01, gamma=.99, gae_lambda=.95, learning_rate=1e-3,
                     feature_dim=16, batch_size=8, epochs=2, observation_clip=5,
                     variance_floor=1e-4, max_grad_norm=.5, value_coef=.5)


def test_running_moments_matches_population():
    x = torch.randn(17, 4, dtype=torch.float64)
    moments = RunningMoments((4,))
    moments.update(x[:7])
    moments.update(x[7:])
    torch.testing.assert_close(moments.mean, x.mean(0))
    torch.testing.assert_close(moments.variance, x.var(0, unbiased=False))
    assert moments.count == 17


def test_target_frozen_and_terminal_reset_reward_zero():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rnd = RND(rnd_config(), 32, "cpu")
    pixels = torch.randint(256, (4, 4, 60, 80), dtype=torch.uint8)
    rnd.observe(pixels)
    frozen = copy.deepcopy(rnd.target.state_dict())
    rewards = rnd.reward(pixels, torch.tensor([0., 1., 0., 1.]))
    assert (rewards[[0, 2]] > 0).all()
    assert torch.equal(rewards[[1, 3]], torch.zeros(2))
    loss = rnd.fit(pixels[None], torch.zeros(1, 4))
    assert loss > 0
    assert all(torch.equal(v, frozen[k]) for k, v in rnd.target.state_dict().items())
    assert all(p.grad is None for p in rnd.target.parameters())
    assert any(p.grad is not None for p in rnd.predictor.parameters())


def test_all_terminal_batch_does_not_train_on_reset_images():
    rnd = RND(rnd_config(), 32, "cpu")
    pixels = torch.randint(256, (1, 2, 4, 60, 80), dtype=torch.uint8)
    count = rnd.observations.count.clone()
    assert rnd.fit(pixels, torch.ones(1, 2)) == 0
    assert rnd.observations.count == count


def test_intrinsic_return_accumulator_resets_per_environment():
    rnd = RND(rnd_config(), 32, "cpu")
    rewards = torch.tensor([[1., 2.], [3., 0.], [0., 4.]])
    dones = torch.tensor([[0., 0.], [0., 1.], [1., 0.]])
    result = rnd.normalize_rewards(rewards, dones)
    assert torch.equal(rnd.discounted, torch.tensor([0., 4.]))
    assert torch.isfinite(result).all()
    assert result[1, 1] == 0 and result[2, 0] == 0
    expected_returns = torch.tensor([1., 2., .99 + 3., .99 * 2.,
                                     (.99 + 3.) * .99, 4.], dtype=torch.float64)
    torch.testing.assert_close(rnd.returns.variance, expected_returns.var(unbiased=False))


def test_rnd_state_restores_stats_predictor_and_resets_episode_returns():
    rnd = RND(rnd_config(), 32, "cpu")
    pixels = torch.randint(256, (2, 4, 60, 80), dtype=torch.uint8)
    rnd.observe(pixels)
    rnd.fit(pixels[None], torch.zeros(1, 2))
    rnd.normalize_rewards(torch.ones(2, 2), torch.tensor([[0., 1.], [0., 0.]]))
    restored = RND(rnd_config(), 32, "cpu")
    restored.restore(copy.deepcopy(rnd.save()))
    assert restored.discounted is None
    for key, value in rnd.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    assert restored.optimizer.state_dict()["state"]
    torch.testing.assert_close(rnd.reward(pixels, torch.zeros(2)),
                               restored.reward(pixels, torch.zeros(2)))


@pytest.mark.parametrize("field,value", [("coefficient", float("nan")), ("gamma", 2),
                                         ("feature_dim", True), ("variance_floor", 0)])
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(rnd_config(), **{field: value})


def test_real_engine_rnd_ppo_export_and_resume(tmp_path):
    cfg = small_config()
    cfg.update(rnd=dataclasses.asdict(rnd_config()), save_every_updates=1)
    run(cfg, tmp_path / "original")
    checkpoint = torch.load(tmp_path / "original/training_16.pt", weights_only=False)
    assert checkpoint["version"] == 4
    assert checkpoint["rnd"]["model"]["observations.count"] > 0
    net, _, _ = load_policy(tmp_path / "original/candidate_32")
    assert not any("rnd" in key for key in net.state_dict())
    resumed = dict(cfg, resume=str(tmp_path / "original/training_16.pt"), start_step=16)
    run(resumed, tmp_path / "resumed")
    final = torch.load(tmp_path / "resumed/training_32.pt", weights_only=False)
    for key in checkpoint["rnd"]["model"]:
        if key.startswith("target."):
            assert torch.equal(final["rnd"]["model"][key], checkpoint["rnd"]["model"][key])
    metrics = [json.loads(x) for x in (tmp_path / "resumed/updates.jsonl").read_text().splitlines()]
    assert len(metrics) == 1 and metrics[0]["rnd_predictor_loss"] >= 0
    assert metrics[0]["lr"] == pytest.approx(cfg["ppo"]["lr"] * .5)
    with pytest.raises(ValueError, match="RND"):
        run(dict(resumed, rnd=None), tmp_path / "mismatch")


def test_legacy_native_checkpoint_config_defaults_remain_resumable(tmp_path):
    cfg = small_config()
    cfg.update(save_every_updates=1)
    run(cfg, tmp_path / "native")
    path = tmp_path / "native/training_16.pt"
    saved = torch.load(path, weights_only=False)
    saved["model_config"].pop("memory_slots", None)
    saved["model_config"].pop("memory_dim", None)
    legacy = tmp_path / "legacy.pt"
    torch.save(saved, legacy)
    run(dict(cfg, resume=str(legacy), start_step=16), tmp_path / "resumed")
    load_policy(tmp_path / "resumed/candidate_32")


def test_rnd_warmup_preserves_bc_policy_and_target_then_unfreezes(tmp_path):
    from doom_model import AgentNet, ModelConfig
    from doom_model.artifacts import export_policy
    cfg = small_config()
    cfg["model"].update(memory_slots=3, memory_dim=8)
    model = ModelConfig(**cfg["model"])
    source = export_policy(AgentNet(model), model, tmp_path / "source", {"purpose": "warmup-test"})
    cfg.update(rnd=dataclasses.asdict(rnd_config()), save_every_updates=1, total_steps=48,
               init_from=str(source / "model.pt"), init_config=str(source / "model_config.json"))
    cfg["ppo"]["critic_warmup_updates"] = 2
    run(cfg, tmp_path / "run")
    policy = torch.load(source / "model.pt", weights_only=True)
    for step in (16, 32):
        frozen = torch.load(tmp_path / f"run/candidate_{step}/model.pt", weights_only=True)
        assert all(torch.equal(v, frozen[k]) for k, v in policy.items())
    final = torch.load(tmp_path / "run/candidate_48/model.pt", weights_only=True)
    assert any(not torch.equal(v, final[k]) for k, v in policy.items())
    first = torch.load(tmp_path / "run/training_16.pt", weights_only=False)["rnd"]["model"]
    second = torch.load(tmp_path / "run/training_32.pt", weights_only=False)["rnd"]["model"]
    assert all(torch.equal(v, second[k]) for k, v in first.items() if k.startswith("target."))
    for prefix in ("value.", "predictor."):
        assert any(not torch.equal(v, second[k]) for k, v in first.items() if k.startswith(prefix))
