"""Submission entry point parity, reset isolation and finite observation extremes."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(Path(__file__).parent)]
import agent as agent_mod
from doom_model import AgentNet, ModelConfig
from doom_model.artifacts import export_policy
from test_learned_agent import observation


def test_agent_fails_closed_and_resets_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "WEIGHTS_DIR", tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        agent_mod.DoomAgent(None, None, "cpu")
    cfg = ModelConfig(hidden=32, embed=32)
    export_policy(AgentNet(cfg), cfg, tmp_path / "test", {"purpose": "test"})
    monkeypatch.setattr(agent_mod, "WEIGHTS_DIR", tmp_path / "test")
    agent = agent_mod.DoomAgent(None, None, "cpu")
    sequences = []
    for _ in range(2):
        agent.reset()
        sequences.append(np.stack([agent.act(observation(i)) for i in range(25)]))
    assert np.array_equal(*sequences)
    assert sequences[0].shape == (25, 9)
    assert set(np.unique(sequences[0])) <= {0, 1}
    assert not hasattr(agent, "bank_pose")


def test_full_length_learned_memory_is_bounded(tmp_path, monkeypatch):
    cfg = ModelConfig(hidden=32, embed=32)
    export_policy(AgentNet(cfg), cfg, tmp_path / "test", {"purpose": "test"})
    monkeypatch.setattr(agent_mod, "WEIGHTS_DIR", tmp_path / "test")
    agent = agent_mod.DoomAgent(None, None, "cpu")
    obs = observation()
    for _ in range(1050):
        agent.act(obs)
    assert agent.h.shape == (1, 1, 32)
    assert torch.isfinite(agent.h).all()
    assert not hasattr(agent.feats, "_last")


def test_observation_extremes_and_sequence_equivalence():
    from doom_env import observation_space
    from doom_model.features import AgentSideFeatures
    cfg = ModelConfig(hidden=32, embed=32)
    net = AgentNet(cfg).eval()
    space = observation_space()
    space.seed(8)
    adapter = AgentSideFeatures()
    sequence = [adapter.reset(space.sample()) for _ in range(12)]
    inputs = {k: torch.as_tensor(np.stack([f[k] for f in sequence]))[:, None] for k in sequence[0]}
    starts = torch.zeros(12, 1)
    starts[[0, 5, 11]] = 1
    full = net.policy(inputs, net.init_state(1), starts)
    h = net.init_state(1)
    logits = []
    for t in range(12):
        out = net.policy({k: v[t:t+1] for k, v in inputs.items()}, h, starts[t:t+1])
        h = out["h"]
        logits.append(out["logits"])
    torch.testing.assert_close(torch.cat(logits), full["logits"], atol=1e-6, rtol=1e-6)
    assert torch.isfinite(full["logits"]).all()
