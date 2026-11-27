"""Real-engine contract: teacher episodes -> recurrent BC -> memory/RND PPO -> export/resume."""
import copy
import dataclasses
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(Path(__file__).parent)]

from doom_model import AgentNet, ModelConfig
from doom_model.artifacts import export_policy, file_hash, load_policy
from doomrl.demonstrations import collect
from doomrl.learned_stack import run_experiment
from doomrl.maps import FD2
from test_pixels import small_config
from test_resume_integrity import run
from test_rnd import rnd_config


def experiment_config():
    cfg = small_config()
    cfg["model"].update(memory_slots=3, memory_dim=8)
    cfg.update(rnd=dataclasses.asdict(rnd_config()), save_every_updates=1)
    cfg["ppo"]["critic_warmup_updates"] = 0
    return cfg


def bc_parameters():
    cfg = json.loads((ROOT / "training/configs/bc_learned_stack.json").read_text())
    cfg.update(epochs=1, chunk_steps=2)
    return cfg


def test_all_three_real_engine_and_resumable_policy(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = experiment_config()
    cfg["model"] = dataclasses.asdict(ModelConfig(**cfg["model"]))
    monitor = tmp_path / "active-v3-frames"
    monitor.mkdir()
    sentinel = monitor / "train.json"
    sentinel.write_text('{"run":"active-v3"}')
    monkeypatch.setenv("DOOM_FRAME_TAP_DIR", str(monitor))
    from doomrl.frame_tap import FrameTap
    def forbidden_tap():
        pytest.fail("isolated experiment must never read inherited frame-tap settings")
    monkeypatch.setattr(FrameTap, "from_env", forbidden_tap)
    teacher_cfg = ModelConfig(hidden=32, embed=32, use_pixels=True)
    teacher = export_policy(AgentNet(teacher_cfg), teacher_cfg, tmp_path / "teacher",
                            {"purpose": "untrained pipeline test; not skill evidence"})
    teacher_identity = file_hash(teacher / "manifest.json")
    demos = collect(teacher, tmp_path / "demos", training_maps=[(FD2, "MAP01"), (FD2, "MAP02")],
                    seeds=[31, 32], episode_ticks=16, temperature=0,
                    policy_seed=5, torch_threads=1)
    summary = run_experiment(cfg, bc_parameters(), demos, teacher, tmp_path / "experiment",
                             device="cpu", ppo_max_seconds=120)
    assert summary["promotion"].startswith("none")
    assert teacher_identity == file_hash(teacher / "manifest.json")
    bc, bc_cfg, _ = load_policy(summary["bc_candidate"])
    final, final_cfg, manifest = load_policy(summary["ppo"]["candidate"])
    assert final_cfg.memory_slots == bc_cfg.memory_slots == 3
    assert final.init_state(1).shape[-1] > final_cfg.hidden
    assert manifest["provenance"]["bc_manifest_sha256"] == file_hash(
        Path(summary["bc_candidate"]) / "manifest.json")
    assert not any(k.startswith(("v.", "rnd.")) for k in final.state_dict())
    assert any(not torch.equal(v, final.state_dict()[k]) for k, v in bc.state_dict().items())
    run_dir = tmp_path / "experiment/ppo"
    resume = json.loads((run_dir / "config.json").read_text())
    resume.pop("init_from")
    resume.pop("init_config")
    resume.update(resume=str(run_dir / "training_16.pt"), start_step=16)
    run(resume, tmp_path / "resumed")
    resumed, resumed_cfg, _ = load_policy(tmp_path / "resumed/candidate_32")
    assert resumed_cfg == final_cfg
    assert all(torch.isfinite(v).all() for v in resumed.state_dict().values())
    saved = torch.load(tmp_path / "resumed/training_32.pt", weights_only=False)
    assert saved["rnd"]["optimizer"]["state"]
    assert saved["schedule"]["total_steps"] == cfg["total_steps"]
    assert saved["config"]["disable_frame_tap"] is True
    assert sentinel.read_text() == '{"run":"active-v3"}'
    assert list(monitor.iterdir()) == [sentinel]
    with pytest.raises(FileExistsError):
        run_experiment(cfg, bc_parameters(), demos, teacher, tmp_path / "experiment",
                       device="cpu", ppo_max_seconds=120)


@pytest.mark.parametrize("change", [
    {"reward": "progress_v3"}, {"remote_envs": "unapproved:1"}, {"resume": "active-v3.pt"},
    {"rnd": None}, {"total_steps": 0},
])
def test_rejects_unsafe_or_incomplete_stack_before_output(tmp_path, change):
    cfg = copy.deepcopy(experiment_config())
    cfg.update(change)
    with pytest.raises(ValueError):
        run_experiment(cfg, bc_parameters(), tmp_path / "missing", tmp_path / "missing",
                       tmp_path / "output", device="cpu", ppo_max_seconds=1)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("fields", [{"head": "q"}, {"privileged_dim": 999}])
def test_model_conflicts_rejected_before_bc(tmp_path, fields):
    cfg = experiment_config()
    cfg["model"].update(fields)
    with pytest.raises(ValueError):
        run_experiment(cfg, bc_parameters(), tmp_path / "missing", tmp_path / "missing",
                       tmp_path / "output", device="cpu", ppo_max_seconds=1)
    assert not (tmp_path / "output").exists()
