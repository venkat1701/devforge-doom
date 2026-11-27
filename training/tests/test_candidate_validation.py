"""Candidate validation must run against the exact immutable policy export."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submission"))
from doom_model import AgentNet, ModelConfig
from doom_model.artifacts import export_policy, file_hash, verify_artifact

spec = importlib.util.spec_from_file_location(
    "validate_candidate", ROOT / "training/tools/validate_candidate.py"
)
candidate_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate_tool)


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    repo = tmp_path / "source"
    weights = repo / "submission/weights"
    weights.mkdir(parents=True)
    (weights / "stale.pt").write_bytes(b"do not validate this")
    (repo / "submission/agent.py").write_text("# working-tree agent\n")
    (repo / "validate.py").write_text("# official validator\n")
    cfg = ModelConfig(hidden=32, embed=32)
    export = export_policy(
        AgentNet(cfg), cfg, tmp_path / "candidate", {"steps": 17, "run": "test"}
    )
    monkeypatch.setattr(candidate_tool, "ROOT", repo)
    monkeypatch.setattr(sys, "argv", ["validate_candidate.py", "--weights", str(export)])
    return repo, export


def test_exact_export_replaces_stale_weights(candidate, monkeypatch, capsys):
    repo, export = candidate
    before = {p.name: p.read_bytes() for p in export.iterdir()}
    called = []

    def validate(cmd):
        copied = Path(cmd[cmd.index("--repo") + 1])
        weights = copied / "submission/weights"
        assert {p.name: p.read_bytes() for p in weights.iterdir()} == before
        assert verify_artifact(weights)["provenance"] == {"steps": 17, "run": "test"}
        assert (copied / "submission/agent.py").is_file()
        assert cmd[:2] == [sys.executable, str(repo / "validate.py")]
        called.append(copied)
        return 7

    monkeypatch.setattr(candidate_tool.subprocess, "call", validate)
    assert candidate_tool.main() == 7
    assert len(called) == 1
    assert not called[0].exists()
    assert (repo / "submission/weights/stale.pt").read_bytes() == b"do not validate this"
    assert {p.name: p.read_bytes() for p in export.iterdir()} == before
    assert file_hash(export / "manifest.json") in capsys.readouterr().out


@pytest.mark.parametrize("name", ["manifest.json", "model.pt", "model_config.json"])
def test_missing_artifact_never_runs_validator(candidate, monkeypatch, name):
    _, export = candidate
    (export / name).unlink()
    monkeypatch.setattr(
        candidate_tool.subprocess, "call", lambda cmd: pytest.fail("invalid export ran")
    )
    with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
        candidate_tool.main()


@pytest.mark.parametrize("name", ["model.pt", "model_config.json"])
def test_checksum_failure_never_runs_validator(candidate, monkeypatch, name):
    _, export = candidate
    with (export / name).open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(
        candidate_tool.subprocess, "call", lambda cmd: pytest.fail("invalid export ran")
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        candidate_tool.main()


def test_manifest_schema_is_verified(candidate, monkeypatch):
    _, export = candidate
    path = export / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["version"] = -1
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        candidate_tool.subprocess, "call", lambda cmd: pytest.fail("invalid export ran")
    )
    with pytest.raises(ValueError, match="schema"):
        candidate_tool.main()


def test_candidate_identity_cannot_change_during_copy(candidate, monkeypatch):
    _, export = candidate
    original_copy = candidate_tool.shutil.copy2

    def copy(src, dst, *args, **kwargs):
        if Path(src) == export / "manifest.json":
            manifest = json.loads(Path(src).read_text())
            manifest["provenance"]["steps"] += 1
            Path(src).write_text(json.dumps(manifest))
        return original_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(candidate_tool.shutil, "copy2", copy)
    monkeypatch.setattr(
        candidate_tool.subprocess, "call", lambda cmd: pytest.fail("changed export ran")
    )
    with pytest.raises(ValueError, match="identity|changed"):
        candidate_tool.main()


def test_copied_artifact_is_verified(candidate, monkeypatch):
    _, export = candidate
    original_copy = candidate_tool.shutil.copy2

    def copy(src, dst, *args, **kwargs):
        result = original_copy(src, dst, *args, **kwargs)
        if Path(src) == export / "model.pt":
            Path(dst).write_bytes(b"corrupted copy")
        return result

    monkeypatch.setattr(candidate_tool.shutil, "copy2", copy)
    monkeypatch.setattr(
        candidate_tool.subprocess, "call", lambda cmd: pytest.fail("corrupted copy ran")
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        candidate_tool.main()


@pytest.mark.parametrize("linked", ["submission/weights", "submission"])
def test_symlinked_directories_do_not_modify_original(candidate, monkeypatch, tmp_path, linked):
    repo, export = candidate
    original = repo / linked
    external = tmp_path / "external"
    original.rename(external)
    original.symlink_to(external, target_is_directory=True)
    before = {p.relative_to(external): p.read_bytes() for p in external.rglob("*") if p.is_file()}

    def validate(cmd):
        copied = Path(cmd[cmd.index("--repo") + 1]) / "submission/weights"
        assert not copied.is_symlink()
        assert verify_artifact(copied) == verify_artifact(export)
        assert not (copied / "stale.pt").exists()
        return 0

    monkeypatch.setattr(candidate_tool.subprocess, "call", validate)
    assert candidate_tool.main() == 0
    assert {p.relative_to(external): p.read_bytes() for p in external.rglob("*") if p.is_file()} == before


def test_forwarded_flags_cannot_override_candidate_repo(candidate, monkeypatch):
    _, export = candidate
    monkeypatch.setattr(sys, "argv", [
        "validate_candidate.py", "--weights", str(export),
        "--repo", "/wrong/repo", "--map", "example.wad",
        "--docker", "--image", "test-image", "--ticks", "123",
    ])

    def validate(cmd):
        # argparse uses the last value for repeated scalar options.
        repo_index = max(i for i, arg in enumerate(cmd) if arg == "--repo")
        copied = Path(cmd[repo_index + 1])
        assert copied != Path("/wrong/repo")
        verify_artifact(copied / "submission/weights")
        assert cmd[cmd.index("--map") + 1] == "example.wad"
        assert cmd[cmd.index("--ticks") + 1] == "123"
        assert cmd[cmd.index("--image") + 1] == "test-image"
        assert "--docker" in cmd
        return 0

    monkeypatch.setattr(candidate_tool.subprocess, "call", validate)
    assert candidate_tool.main() == 0


def test_validation_cannot_silently_change_artifact(candidate, monkeypatch):
    def validate(cmd):
        copied = Path(cmd[cmd.index("--repo") + 1])
        (copied / "submission/weights/model.pt").write_bytes(b"changed during validation")
        return 0

    monkeypatch.setattr(candidate_tool.subprocess, "call", validate)
    with pytest.raises(ValueError, match="hash mismatch"):
        candidate_tool.main()
