"""Real-engine pixels and PPO freeze/unfreeze integration tests."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]
from doom_model import AgentNet, ModelConfig
from doom_model.features import pixels
from doomrl.maps import FD2
from doomrl.ppo import train
from doomrl.train import Logger


def test_real_pixels():
    from doom_env import make_env
    env = make_env(FD2, "MAP01", episode_ticks=20)
    try:
        obs, _ = env.reset(seed=1)
        image = pixels(obs)
        assert image.shape == (4, 60, 80) and image.dtype == np.uint8
        assert image[:3].max() > 0
    finally:
        env.close()


def small_config():
    cfg = json.loads((ROOT / "training/configs/ppo_learned.json").read_text())
    cfg.update(num_envs=2, total_steps=32, device="cpu", episode_ticks=[16, 24],
               training_maps=[(FD2, "MAP01")], purpose="integration-test")
    cfg["model"].update(hidden=32, embed=32)
    cfg["ppo"].update(num_steps=8, num_minibatches=2, update_epochs=1, critic_warmup_updates=1)
    return cfg


def test_critic_warmup_then_actor_unfreezes(tmp_path):
    source_cfg = ModelConfig(hidden=32, embed=32, use_pixels=False)
    source = AgentNet(source_cfg)
    state = {k: v.clone() for k, v in source.state_dict().items() if not k.startswith("v.")}
    torch.save(state, tmp_path / "model.pt")
    source_cfg.save(tmp_path / "model_config.json")
    cfg = small_config()
    cfg.update(init_from=str(tmp_path / "model.pt"), save_every_updates=1, total_steps=48)
    log = Logger(tmp_path / "run")
    try:
        train(cfg, tmp_path / "run", log)
    finally:
        log.close()
    first = torch.load(tmp_path / "run/candidate_16/model.pt", weights_only=True)
    final = torch.load(tmp_path / "run/candidate_32/model.pt", weights_only=True)
    assert all(torch.equal(first[k], v) for k, v in state.items())
    assert torch.count_nonzero(first["enc.pix_proj.weight"]) == 0
    assert any(not torch.equal(final[k], v) for k, v in state.items())
    assert torch.count_nonzero(final["enc.pix_proj.weight"]) > 0
    resume = dict(cfg, resume=str(tmp_path / "run/training_32.pt"), start_step=32, total_steps=48)
    del resume["init_from"]
    log = Logger(tmp_path / "resume")
    try:
        train(resume, tmp_path / "resume", log)
    finally:
        log.close()
    assert (tmp_path / "resume/candidate_48/manifest.json").is_file()
