import dataclasses
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]
from doom_model import AgentNet, ModelConfig


def test_immutable_export_integrity_and_policy_only_loading(tmp_path):
    from doom_model.artifacts import export_policy, load_policy
    cfg = ModelConfig(hidden=32, embed=32)
    net = AgentNet(cfg)
    destination = tmp_path / "candidate"
    export_policy(net, cfg, destination, {"steps": 128, "purpose": "test"})
    loaded, loaded_cfg, manifest = load_policy(destination)
    assert loaded_cfg == cfg and manifest["provenance"]["steps"] == 128
    assert not hasattr(loaded, "v")
    assert torch.equal(net.pi.weight, loaded.pi.weight)
    with pytest.raises(FileExistsError):
        export_policy(net, cfg, destination, {})
    with (destination / "model.pt").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="hash"):
        load_policy(destination)


def test_nonfinite_export_rejected(tmp_path):
    from doom_model.artifacts import export_policy
    cfg = ModelConfig(hidden=32, embed=32)
    net = AgentNet(cfg)
    with torch.no_grad():
        net.pi.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        export_policy(net, cfg, tmp_path / "bad", {})
    assert not (tmp_path / "bad").exists()


def test_explicit_map_fusion_migration(tmp_path):
    from doomrl.migrate import transfer_policy
    cfg = ModelConfig(hidden=32, embed=32, use_pixels=False)
    source = AgentNet(cfg).state_dict()
    target = AgentNet(cfg)
    original = source["enc.out.0.weight"].clone()
    source["enc.out.0.weight"] = torch.cat([original, torch.randn(32, 128)], dim=1)
    source["enc.map.0.weight"] = torch.randn(32, 10, 3, 3)
    source_cfg = dict(dataclasses.asdict(cfg), use_map=True, aux=True)
    report = transfer_policy(target, source, source_cfg, allow_map_removal=True)
    assert report["function_preserving"] is False
    assert torch.equal(target.enc.out[0].weight, original)
    with pytest.raises(ValueError, match="explicit"):
        transfer_policy(target, source, source_cfg)
    source["pi.weight"] = torch.randn(12, 32)
    with pytest.raises(ValueError, match="shape"):
        transfer_policy(target, source, source_cfg, allow_map_removal=True)
