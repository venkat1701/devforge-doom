"""IMPALA pixel encoder: opt-in, default unchanged, and a nature->impala migration keeps every non-pixel weight."""
import dataclasses
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]
from doom_model import AgentNet, ModelConfig
from doom_model.features import PIX_H, PIX_W
from doomrl.migrate import transfer_policy


def obs(T=3, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"dist": torch.rand(T, B, 64, generator=g), "hit": torch.randint(0, 6, (T, B, 64), generator=g),
            "ent": torch.rand(T, B, 32, 5, generator=g), "etype": torch.randint(0, 12, (T, B, 32), generator=g),
            "emask": torch.ones(T, B, 32), "vit": torch.rand(T, B, 17, generator=g),
            "prev": torch.randint(0, 56, (T, B), generator=g),
            "pix": torch.randint(0, 256, (T, B, 4, PIX_H, PIX_W), generator=g, dtype=torch.uint8)}


def test_default_is_nature_and_invalid_refused():
    assert ModelConfig().pixel_encoder == "nature"
    with pytest.raises(ValueError):
        ModelConfig(pixel_encoder="resnet")


def test_impala_forward_shapes():
    cfg = ModelConfig(hidden=32, embed=32, head="actor_critic", pixel_encoder="impala")
    net = AgentNet(cfg)
    out = net(obs(), net.init_state(2, "cpu"), torch.ones(3, 2))
    assert out["logits"].shape[:2] == (3, 2)


def test_migration_keeps_non_pixel_weights_and_equals_source_without_pixels():
    torch.manual_seed(0)
    src_cfg = ModelConfig(hidden=32, embed=32, head="actor_critic")
    src = AgentNet(src_cfg)
    dst = AgentNet(dataclasses.replace(src_cfg, pixel_encoder="impala"))
    info = transfer_policy(dst, src.state_dict(), dataclasses.asdict(src_cfg))
    assert info["requires_retraining"] and not info["function_preserving"]
    assert any(n.startswith("enc.pix") for n in info["removed"]) and any(n.startswith("enc.pix") for n in info["fresh"])
    for name, t in src.state_dict().items():
        if not name.startswith(("enc.pix", "v.")):
            assert torch.equal(dst.state_dict()[name], t)
    with torch.no_grad():
        src.enc.pix_proj.weight.zero_(); src.enc.pix_proj.bias.zero_()    # source minus its pixel contribution
        o = obs()
        a = src(o, src.init_state(2, "cpu"), torch.ones(3, 2))["logits"]
        b = dst(o, dst.init_state(2, "cpu"), torch.ones(3, 2))["logits"]
    assert torch.allclose(a, b, atol=1e-6)
