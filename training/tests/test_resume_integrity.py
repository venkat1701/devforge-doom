"""Regressions for immutable publication and learner-phase resumption."""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(Path(__file__).parent)]
from doom_model import AgentNet, ModelConfig
from doom_model.artifacts import file_hash, verify_artifact
from doomrl import ppo
from doomrl.train import Logger
from test_pixels import small_config


@pytest.mark.parametrize("collision", ["pair", "candidate", "checkpoint"])
def test_save_collision_preserves_published_files(tmp_path, collision):
    cfg = ModelConfig(hidden=32, embed=32)
    net = AgentNet(cfg)
    opt = torch.optim.Adam(net.parameters())
    ppo.checkpoint(net, opt, cfg, {}, tmp_path, 16, 1, {})
    candidate = tmp_path / "candidate_16"
    checkpoint = tmp_path / "training_16.pt"
    if collision == "candidate":
        checkpoint.unlink()
    elif collision == "checkpoint":
        import shutil
        shutil.rmtree(candidate)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with torch.no_grad():
        net.pi.bias.add_(1)
    with pytest.raises(FileExistsError):
        ppo.checkpoint(net, opt, cfg, {}, tmp_path, 16, 1, {})
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_failed_export_does_not_publish_checkpoint(tmp_path, monkeypatch):
    cfg = ModelConfig(hidden=32, embed=32)
    net = AgentNet(cfg)
    opt = torch.optim.Adam(net.parameters())
    def fail_export(*args, **kwargs):
        raise OSError("injected export failure")
    monkeypatch.setattr(ppo, "export_policy", fail_export)
    with pytest.raises(OSError, match="injected"):
        ppo.checkpoint(net, opt, cfg, {}, tmp_path, 16, 1, {})
    assert not (tmp_path / "training_16.pt").exists()
    assert not (tmp_path / "progress.json").exists()


def test_concurrent_saves_publish_one_consistent_pair(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    cfg = ModelConfig(hidden=32, embed=32)
    net = AgentNet(cfg)
    opt = torch.optim.Adam(net.parameters())
    def save():
        try:
            ppo.checkpoint(net, opt, cfg, {}, tmp_path, 16, 1, {})
            return True
        except FileExistsError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: save(), range(2)))
    assert sorted(results) == [False, True]
    manifest = verify_artifact(tmp_path / "candidate_16")
    assert manifest["provenance"]["training_sha256"] == file_hash(tmp_path / "training_16.pt")
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert Path(progress["candidate"]) == tmp_path / "candidate_16"


def run(cfg, path):
    log = Logger(path)
    try:
        return ppo.train(cfg, path, log)
    finally:
        log.close()


@pytest.fixture
def warmup_run(tmp_path):
    cfg = small_config()
    source_cfg = ModelConfig(hidden=32, embed=32, use_pixels=False)
    source = AgentNet(source_cfg)
    torch.save(source.state_dict(), tmp_path / "model.pt")
    source_cfg.save(tmp_path / "model_config.json")
    cfg.update(init_from=str(tmp_path / "model.pt"), total_steps=64, save_every_updates=1)
    cfg["ppo"].update(critic_warmup_updates=3, anneal_lr=True)
    run(cfg, tmp_path / "original")
    resume = dict(cfg, resume=str(tmp_path / "original/training_16.pt"), start_step=16)
    del resume["init_from"]
    run(resume, tmp_path / "resumed")
    return cfg, resume, tmp_path


def test_resume_preserves_learning_rate_horizon(warmup_run):
    cfg, _, path = warmup_run
    metrics = [json.loads(line) for line in (path / "resumed/updates.jsonl").read_text().splitlines()]
    assert [m["lr"] for m in metrics] == pytest.approx(
        [cfg["ppo"]["lr"] * .75, cfg["ppo"]["lr"] * .5, cfg["ppo"]["lr"] * .25])


def test_resume_preserves_warmup_and_migration(warmup_run):
    _, _, path = warmup_run
    initial = torch.load(path / "original/candidate_16/model.pt", weights_only=True)
    for step in (32, 48):
        frozen = torch.load(path / f"resumed/candidate_{step}/model.pt", weights_only=True)
        assert all(torch.equal(value, frozen[key]) for key, value in initial.items())
    unfrozen = torch.load(path / "resumed/candidate_64/model.pt", weights_only=True)
    assert any(not torch.equal(value, unfrozen[key]) for key, value in initial.items())
    manifest = verify_artifact(path / "resumed/candidate_64")
    assert manifest["provenance"]["migration"] is not None
    assert manifest["provenance"]["training_sha256"] == file_hash(path / "resumed/training_64.pt")


def test_resume_rejects_changed_schedule_horizon(warmup_run):
    _, resume, path = warmup_run
    with pytest.raises(ValueError, match="horizon"):
        run(dict(resume, total_steps=80), path / "changed")


def test_resume_loads_rng_on_cpu(warmup_run, monkeypatch):
    _, resume, path = warmup_run
    original_load = torch.load
    def load(*args, **kwargs):
        # Check the deserialization boundary even on machines without CUDA.
        if str(args[0]) == resume["resume"]:
            assert torch.device(kwargs["map_location"]).type == "cpu"
        return original_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", load)
    run(resume, path / "cpu-load")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_cuda_resume_rng_and_optimizer(tmp_path):
    cfg = small_config()
    cfg.update(device="cuda", save_every_updates=1)
    run(cfg, tmp_path / "cuda")
    resumed = dict(cfg, resume=str(tmp_path / "cuda/training_16.pt"), start_step=16)
    run(resumed, tmp_path / "resumed")
    verify_artifact(tmp_path / "resumed/candidate_32")
