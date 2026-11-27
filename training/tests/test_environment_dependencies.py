"""Environment fingerprints cover runtime dependencies without installation paths."""
import json
from pathlib import Path
import sys

import pytest
import vizdoom

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]

from doom_env import wrapper
from doom_model import artifacts
from doomrl import maps, remote_envs


@pytest.fixture(params=[False, True], ids=["official", "fast"])
def cfg(request):
    return {
        "seed": 7,
        "fast_env": request.param,
        "training_maps": [("freedoom1.wad", "E1M1")],
    }


def spec(cfg):
    return remote_envs.environment_spec(cfg, 1, False, True)


def test_fast_environment_implementation_changes_fingerprint(cfg, tmp_path, monkeypatch):
    source = ROOT / "training/fastenv/__init__.py"
    replacement = tmp_path / "fastenv.py"
    replacement.write_bytes(source.read_bytes())
    file_hash = artifacts.file_hash

    def dependency_hash(path):
        return file_hash(replacement if Path(path) == source else path)

    monkeypatch.setattr(artifacts, "file_hash", dependency_hash)
    before = spec(cfg)
    replacement.write_bytes(replacement.read_bytes() + b"\n# changed implementation\n")
    after = spec(cfg)

    assert before != after
    assert after["code"]["training/fastenv/__init__.py"] == file_hash(replacement)


def test_base_iwad_changes_fingerprint_without_manifest_entry(cfg, tmp_path, monkeypatch):
    assert all(Path(wad).name != "freedoom2.wad"
               for wad, _ in maps.resolve_training_maps(cfg["training_maps"]))
    iwad = tmp_path / "freedoom2.wad"
    iwad.write_bytes(b"base IWAD revision one")
    monkeypatch.setattr(wrapper, "IWAD", str(iwad))
    before = spec(cfg)
    iwad.write_bytes(b"base IWAD revision two")
    after = spec(cfg)

    assert before != after
    assert before["maps"] == after["maps"]
    assert after["iwad"] == artifacts.file_hash(iwad)


def test_runtime_vizdoom_version_changes_fingerprint(cfg, monkeypatch):
    before = spec(cfg)
    monkeypatch.setattr(vizdoom, "__version__", vizdoom.__version__ + ".test")
    after = spec(cfg)

    assert before != after
    assert after["vizdoom_version"] == vizdoom.__version__


def test_fingerprint_records_installed_runtime_dependencies(cfg):
    fingerprint = spec(cfg)

    assert fingerprint["vizdoom_version"] == vizdoom.__version__
    assert fingerprint["iwad"] == artifacts.file_hash(wrapper.IWAD)
    assert fingerprint["maps"] == [(artifacts.file_hash(maps.FD1), "E1M1")]


def test_installation_relocation_preserves_fingerprint(cfg, tmp_path, monkeypatch):
    fingerprints = []
    for installation in (tmp_path / "first", tmp_path / "second"):
        installation.mkdir()
        iwad = installation / "freedoom2.wad"
        scenario = installation / "freedoom1.wad"
        iwad.write_bytes(b"identical base IWAD")
        scenario.write_bytes(b"identical scenario WAD")
        monkeypatch.setattr(wrapper, "IWAD", str(iwad))
        monkeypatch.setattr(maps, "_V", str(installation))
        monkeypatch.setitem(maps.SUITES, "train", [(str(scenario), "E1M1")])
        fingerprints.append(spec(cfg))

    assert fingerprints[0] == fingerprints[1]
    serialized = json.dumps(fingerprints)
    assert str(tmp_path) not in serialized
    assert str(ROOT) not in serialized
    assert str(Path(vizdoom.__file__).parent) not in serialized
    assert all(not Path(path).is_absolute() for path in fingerprints[0]["code"])
