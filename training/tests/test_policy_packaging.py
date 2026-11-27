import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submission"))
from doom_model import AgentNet, ModelConfig
from doom_model import artifacts


@pytest.fixture
def candidate(tmp_path):
    cfg = ModelConfig(hidden=32, embed=32)
    return artifacts.export_policy(AgentNet(cfg), cfg, tmp_path / "candidate", {"steps": 16})


def test_packaging_preserves_exact_export(candidate, tmp_path):
    destination = tmp_path / "weights"
    destination.mkdir()
    artifacts.copy_policy(candidate, destination)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == {
        p.name: p.read_bytes() for p in candidate.iterdir()}
    artifacts.load_policy(destination)


def test_packaging_refuses_existing_files(candidate, tmp_path):
    destination = tmp_path / "weights"
    destination.mkdir()
    (destination / "keep").write_bytes(b"existing data")
    with pytest.raises(FileExistsError):
        artifacts.copy_policy(candidate, destination)
    assert (destination / "keep").read_bytes() == b"existing data"


def test_packaging_rejects_corrupt_source(candidate, tmp_path):
    (candidate / "model.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        artifacts.copy_policy(candidate, tmp_path / "weights")
    assert not (tmp_path / "weights").exists()
