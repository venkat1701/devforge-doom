"""Inference temperature: default unchanged, 0 = argmax, config file and env override, invalid values refused."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(ROOT / "env"), str(Path(__file__).parent)]
import agent as agent_mod
from doom_env import observation_space, action_space
from doom_model import AgentNet, ModelConfig
from doom_model.artifacts import export_policy
from test_learned_agent import observation


@pytest.fixture
def weights(tmp_path):
    cfg = ModelConfig(hidden=32, embed=32, use_map=False, use_pixels=True, head="actor_critic", privileged_dim=7)
    torch.manual_seed(0)
    out = tmp_path / "w"
    export_policy(AgentNet(cfg), cfg, out, provenance={"test": True})
    return out


def make(weights, monkeypatch, env=None):
    monkeypatch.setattr(agent_mod, "WEIGHTS_DIR", weights)
    if env is None:
        monkeypatch.delenv("DOOM_AGENT_TEMPERATURE", raising=False)
    else:
        monkeypatch.setenv("DOOM_AGENT_TEMPERATURE", env)
    return agent_mod.DoomAgent(observation_space(), action_space(), "cpu")


def test_default_is_sampling_at_temperature_one(weights, monkeypatch):
    assert make(weights, monkeypatch).temperature == 1.0


def test_zero_temperature_is_argmax_and_deterministic(weights, monkeypatch):
    ag = make(weights, monkeypatch, "0")
    ag.reset(); a1 = [ag.act(observation(i)).tolist() for i in range(5)]
    ag.reset(); a2 = [ag.act(observation(i)).tolist() for i in range(5)]
    assert a1 == a2


def test_config_file_and_invalid_values(weights, monkeypatch):
    (weights / "inference.json").write_text(json.dumps({"temperature": 0.5}))
    assert make(weights, monkeypatch).temperature == 0.5
    with pytest.raises(ValueError):
        make(weights, monkeypatch, "-1")


def test_button_order_matches_official_contract():
    from doom_env.spaces import BUTTONS
    assert tuple(agent_mod.BUTTONS) == tuple(BUTTONS)


def test_mask_never_selects_masked_buttons(weights, monkeypatch):
    monkeypatch.setenv("DOOM_AGENT_MASK", "SELECT_NEXT_WEAPON")
    ag = make(weights, monkeypatch, "1.0")
    monkeypatch.setenv("DOOM_AGENT_MASK", "SELECT_NEXT_WEAPON")
    ag = agent_mod.DoomAgent(observation_space(), action_space(), "cpu")
    ag.reset()
    for i in range(200):
        assert ag.act(observation(i % 7))[8] == 0
    (weights / "inference.json").write_text(json.dumps({"temperature": 0, "masked_buttons": ["SELECT_NEXT_WEAPON"]}))
    monkeypatch.delenv("DOOM_AGENT_MASK")
    ag = make(weights, monkeypatch)
    assert bool(ag.action_mask.sum() == 1)
    with pytest.raises(ValueError):
        monkeypatch.setenv("DOOM_AGENT_MASK", "JUMP")
        agent_mod.DoomAgent(observation_space(), action_space(), "cpu")
