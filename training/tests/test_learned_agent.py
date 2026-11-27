"""Behavioral regressions for the strictly learned observation/memory path."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]
from doom_model import AgentNet, ModelConfig
from doom_model.features import AgentSideFeatures

torch.set_num_threads(1)


def observation(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "rgb": rng.integers(256, size=(120, 160, 3), dtype=np.uint8),
        "depth": rng.integers(256, size=(120, 160), dtype=np.uint8),
        "rays": {"distance": rng.random(64, dtype=np.float32), "hit_type": rng.integers(6, size=64)},
        "visible_entities": {"features": rng.normal(size=(32, 5)).astype(np.float32),
                             "entity_type": rng.integers(12, size=32), "mask": np.zeros(32, np.float32)},
        "vitals": np.zeros(17, np.float32),
    }


def tensors(obs):
    f = AgentSideFeatures(use_map=False, use_pixels=True).reset(obs)
    return {k: torch.as_tensor(v)[None, None] for k, v in f.items()}


def test_no_geometric_state_even_when_map_disabled():
    features = AgentSideFeatures(use_map=False, use_pixels=True)
    f = features.reset(observation())
    assert set(f) == {"dist", "hit", "ent", "etype", "emask", "vit", "prev", "pix"}
    assert not hasattr(features, "odo") and not hasattr(features, "map")
    for a in range(56):
        f = features.step(a, observation(a))
        assert int(f["prev"]) == a
        assert f["pix"].shape == (4, 60, 80)


@pytest.mark.parametrize("kwargs", [{"use_map": True}, {"use_episodic": True}, {"aux": True},
                                    {"hidden": 0}, {"embed": -1}, {"privileged_dim": -1}])
def test_invalid_or_retired_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_policy_only_matches_training_and_reset_isolates_history():
    net = AgentNet(ModelConfig(hidden=32, embed=32, privileged_dim=3))
    o = tensors(observation())
    starts = torch.ones(1, 1)
    h = net.init_state(1)
    full = net(o, h, starts, privileged=torch.randn(1, 1, 3))
    policy = net.policy(o, h, starts)
    assert set(policy) == {"logits", "h", "core"}
    torch.testing.assert_close(full["logits"], policy["logits"], rtol=0, atol=0)
    changed = net.policy(o, torch.randn_like(h), starts)
    torch.testing.assert_close(changed["logits"], policy["logits"], rtol=0, atol=0)
    assert torch.isfinite(policy["logits"]).all()


def test_entity_permutation_and_empty_mask():
    net = AgentNet(ModelConfig(hidden=32, embed=32))
    o = tensors(observation())
    start = torch.ones(1, 1)
    for mask in (0., 1.):
        o["emask"].fill_(mask)
        a = net.policy(o, net.init_state(1), start)["logits"]
        order = torch.randperm(32)
        permuted = dict(o, ent=o["ent"][:, :, order], etype=o["etype"][:, :, order],
                        emask=o["emask"][:, :, order])
        b = net.policy(permuted, net.init_state(1), start)["logits"]
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_pixel_projection_preserves_no_pixel_policy_then_learns():
    old = AgentNet(ModelConfig(hidden=32, embed=32, use_pixels=False))
    new = AgentNet(ModelConfig(hidden=32, embed=32, use_pixels=True))
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("enc.pix") for k in missing)
    o = tensors(observation())
    st = torch.ones(1, 1)
    a = old.policy(o, old.init_state(1), st)["logits"]
    b = new.policy(o, new.init_state(1), st)["logits"]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    optimizer = torch.optim.Adam(new.parameters(), lr=.001)
    b.square().sum().backward()
    assert new.enc.pix_proj.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad()
    new.policy(o, new.init_state(1), st)["logits"].square().sum().backward()
    assert new.enc.pix[0].weight.grad.abs().sum() > 0
