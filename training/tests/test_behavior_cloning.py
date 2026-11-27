"""Synthetic fixtures test collection/BC contracts; optional smoke uses the actual engine."""
import dataclasses
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]

import doom_env
from doom_model import ACTIONS, AgentNet, ModelConfig
from doom_model.artifacts import action_hash, export_policy, file_hash, load_policy
from doom_model.features import FEATURE_SCHEMA
from doomrl import behavior_cloning as bc
from doomrl import demonstrations as demos
from doomrl.maps import SUITES


class TinyTestEnvironment:
    """Only a test simulator; collection still obtains every action from exported network logits."""
    def __init__(self, wad, map_name, *, episode_ticks):
        self.steps = 0
        self.closed = False

    def reset(self, seed):
        self.steps = 0
        space = doom_env.observation_space()
        space.seed(seed)
        self.obs = space.sample()
        return self.obs, {}

    def step(self, action):
        assert action.shape == ACTIONS.shape[1:]
        self.steps += 1
        return self.obs, float(self.steps), False, self.steps == 5, {}

    def privileged_state(self):
        return {"exit_reached": False, "dead": False, "timeout": True, "health": 100., "kills": 0}

    def close(self):
        self.closed = True


@pytest.fixture
def policies(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = ModelConfig(hidden=16, embed=16, use_pixels=False)
    teacher = AgentNet(model)
    with torch.no_grad():
        teacher.pi.weight.zero_()
        teacher.pi.bias.zero_()
        teacher.pi.bias[3] = 2
    source = export_policy(teacher, model, tmp_path / "teacher", {"purpose": "synthetic test policy"})
    student = export_policy(AgentNet(model), model, tmp_path / "student", {"purpose": "synthetic init"})
    return model, source, student


@pytest.fixture
def dataset(tmp_path, policies, monkeypatch):
    monkeypatch.setattr(doom_env, "make_env", TinyTestEnvironment)
    _, teacher, _ = policies
    return demos.collect(teacher, tmp_path / "dataset", training_maps=SUITES["train"][:2],
                         seeds=[31, 32], episode_ticks=20, temperature=0., policy_seed=7, torch_threads=1)


@pytest.fixture
def local_suites(tmp_path, monkeypatch):
    """Small content fixtures; only TinyTestEnvironment consumes these non-engine bytes."""
    root = tmp_path / "installation-a"
    root.mkdir()
    suites = {}
    for split in ("train", "dev", "test", "gen_dev", "gen_test"):
        wad = root / f"{split}.wad"
        wad.write_bytes(f"synthetic WAD identity fixture {split}".encode())
        suites[split] = [(str(wad), "MAP01")]
    suites["train"].append((suites["train"][0][0], "MAP02"))
    for split, maps in suites.items():
        monkeypatch.setitem(SUITES, split, maps)
    return root


@pytest.fixture
def portable_dataset(local_suites, tmp_path, policies, monkeypatch):
    monkeypatch.setattr(doom_env, "make_env", TinyTestEnvironment)
    return demos.collect(policies[1], tmp_path / "portable", training_maps=SUITES["train"],
                         seeds=[31], episode_ticks=20, temperature=0., policy_seed=7, torch_threads=1)


def test_v2_stores_content_identity_and_resolves_local_paths(portable_dataset):
    loaded = demos.load_dataset(portable_dataset)
    digest = file_hash(SUITES["train"][0][0])
    assert loaded.manifest["version"] == 2
    assert loaded.manifest["map_identity_schema"] == demos.MAP_IDENTITY_SCHEMA
    assert loaded.manifest["training_maps"] == [[digest, "MAP01"], [digest, "MAP02"]]
    assert [e.metadata["map"] for e in loaded.episodes] == loaded.manifest["training_maps"]
    assert loaded.resolved_maps == {(digest, name): (str(Path(wad).resolve()), name)
                                    for wad, name in SUITES["train"]}
    assert all(wad not in (portable_dataset / "manifest.json").read_text()
               for wad, _ in SUITES["train"])


def test_dataset_relocation_across_installations(portable_dataset, local_suites, tmp_path,
                                               policies, monkeypatch):
    before = demos.load_dataset(portable_dataset)
    target_root = tmp_path / "installation-b"
    shutil.copytree(local_suites, target_root)
    relocated = tmp_path / "relocated-dataset"
    shutil.copytree(portable_dataset, relocated)
    for split in ("train", "dev", "test", "gen_dev", "gen_test"):
        monkeypatch.setitem(SUITES, split, [(str(target_root / Path(w).name), m)
                                          for w, m in SUITES[split]])
    shutil.rmtree(local_suites)
    after = demos.load_dataset(relocated)
    assert after.sha256 == before.sha256
    assert after.manifest == before.manifest
    assert all(Path(wad).parent == target_root for wad, _ in after.resolved_maps.values())
    candidate = bc.train_bc(config(policies[0], epochs=1), relocated, policies[2], tmp_path / "relocated-bc")
    assert load_policy(candidate)[2]["provenance"]["dataset_sha256"] == before.sha256


def test_changed_wad_content_rejected(portable_dataset):
    Path(SUITES["train"][0][0]).write_bytes(b"changed content at the same path")
    with pytest.raises(ValueError, match="content identity unavailable or changed"):
        demos.load_dataset(portable_dataset)


@pytest.mark.parametrize("heldout", ["dev", "test", "gen_dev", "gen_test"])
def test_copied_heldout_content_rejected_even_if_path_whitelisted(
        portable_dataset, tmp_path, policies, monkeypatch, heldout):
    copied = tmp_path / "renamed-training.wad"
    shutil.copyfile(SUITES[heldout][0][0], copied)
    alias = (str(copied), SUITES[heldout][0][1])
    monkeypatch.setitem(SUITES, "train", SUITES["train"] + [alias])
    with pytest.raises(ValueError, match="content identity overlap"):
        demos.load_dataset(portable_dataset)
    def forbidden(*args, **kwargs):
        pytest.fail("heldout content must be rejected before creating an environment")
    monkeypatch.setattr(doom_env, "make_env", forbidden)
    with pytest.raises(ValueError, match="content identity overlap"):
        demos.collect(policies[1], tmp_path / "rejected", training_maps=[alias], seeds=[1],
                      episode_ticks=4, temperature=0., policy_seed=1, torch_threads=1)
    assert not (tmp_path / "rejected").exists()


def test_wad_is_hashed_once_per_validation(portable_dataset, monkeypatch):
    calls = {}
    original = demos.file_hash
    wad_paths = {Path(w).resolve() for split in ("train", "dev", "test", "gen_dev", "gen_test")
                 for w, _ in SUITES[split]}
    def counted(path):
        path = Path(path).resolve()
        if path in wad_paths:
            calls[path] = calls.get(path, 0) + 1
        return original(path)
    monkeypatch.setattr(demos, "file_hash", counted)
    demos.load_dataset(portable_dataset)
    assert calls == dict.fromkeys(wad_paths, 1)
    demos.load_dataset(portable_dataset)
    assert calls == dict.fromkeys(wad_paths, 2)  # No cross-validation cache can hide mutations.


def test_duplicate_content_aliases_cannot_create_independent_training_maps(
        portable_dataset, tmp_path, policies, monkeypatch):
    copied = tmp_path / "training-alias.wad"
    shutil.copyfile(SUITES["train"][0][0], copied)
    alias = (str(copied), SUITES["train"][0][1])
    monkeypatch.setitem(SUITES, "train", SUITES["train"] + [alias])
    with pytest.raises(ValueError, match="unique whitelisted content"):
        demos.collect(policies[1], tmp_path / "rejected",
                      training_maps=[SUITES["train"][0], alias], seeds=[1],
                      episode_ticks=4, temperature=0., policy_seed=1, torch_threads=1)


def test_wad_mutation_during_collection_prevents_publication(local_suites, tmp_path, policies, monkeypatch):
    class Mutating(TinyTestEnvironment):
        def close(self):
            super().close()
            Path(SUITES["train"][0][0]).write_bytes(b"changed during episode")
    monkeypatch.setattr(doom_env, "make_env", Mutating)
    with pytest.raises(ValueError, match="content identity unavailable or changed"):
        demos.collect(policies[1], tmp_path / "rejected", training_maps=SUITES["train"][:1],
                      seeds=[1], episode_ticks=4, temperature=0., policy_seed=1, torch_threads=1)
    assert not (tmp_path / "rejected").exists()


def config(model, *, epochs=3, chunk_steps=2):
    return {"seed": 3, "device": "cpu", "torch_threads": 1, "model": dataclasses.asdict(model),
            "bc": {"epochs": epochs, "lr": .02, "adam_betas": [.9, .999], "adam_eps": 1e-8,
                   "weight_decay": 0., "chunk_steps": chunk_steps, "max_grad_norm": 5.,
                   "validation_fraction": .5, "validation_strategy": "heldout_maps"}}


def resign_manifest(path, manifest):
    demos._json(path / "manifest.json", manifest)
    (path / "manifest.sha256").write_text(file_hash(path / "manifest.json") + "\n")


def rewrite_episode(path, transform):
    manifest = json.loads((path / "manifest.json").read_text())
    episode_path = path / manifest["episodes"][0]["file"]
    with np.load(episode_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    transform(arrays)
    np.savez_compressed(episode_path, **arrays)
    manifest["episodes"][0]["sha256"] = file_hash(episode_path)
    resign_manifest(path, manifest)


def test_collection_preserves_order_and_complete_episodes(dataset, policies):
    loaded = demos.load_dataset(dataset)
    assert len(loaded.episodes) == 4
    assert [e.metadata["seed"] for e in loaded.episodes] == [31, 32, 31, 32]
    assert loaded.manifest["source"]["manifest_sha256"] == file_hash(policies[1] / "manifest.json")
    assert loaded.manifest["feature_schema"] == FEATURE_SCHEMA
    assert loaded.manifest["action_sha256"] == action_hash()
    for episode in loaded.episodes:
        np.testing.assert_array_equal(episode.actions, np.full(5, 3, dtype=np.int64))
        np.testing.assert_array_equal(episode.features["prev"], [0, 3, 3, 3, 3])
        assert episode.truncated.tolist() == [False, False, False, False, True]
        assert not episode.actions.flags.writeable
        assert episode.metadata["outcome"]["native_return"] == 15.


def test_human_source_uses_same_validated_schema(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text())
    # This remains a synthetic test fixture, not a claim of actual human demonstrations.
    manifest["source"] = {"kind": "human", "manifest_sha256": manifest["source"]["manifest_sha256"],
                          "identifier": "synthetic schema compatibility fixture"}
    resign_manifest(dataset, manifest)
    assert demos.load_dataset(dataset).manifest["source"]["kind"] == "human"


def test_pixel_observations_collect_train_and_export(tmp_path, monkeypatch):
    monkeypatch.setattr(doom_env, "make_env", TinyTestEnvironment)
    model = ModelConfig(hidden=16, embed=16, use_pixels=True)
    source = export_policy(AgentNet(model), model, tmp_path / "pixels-source", {"purpose": "test"})
    path = demos.collect(source, tmp_path / "pixels-data", training_maps=SUITES["train"][:2],
                         seeds=[31], episode_ticks=20, temperature=0., policy_seed=7, torch_threads=1)
    loaded = demos.load_dataset(path)
    assert loaded.episodes[0].features["pix"].shape == (5, 4, 60, 80)
    assert loaded.episodes[0].features["pix"].dtype == np.uint8
    candidate = bc.train_bc(config(model, epochs=1), path, source, tmp_path / "pixels-run")
    assert load_policy(candidate)[1].use_pixels
    rewrite_episode(path, lambda a: a.update(pix=a["pix"].astype(np.float32)))
    with pytest.raises(ValueError, match="pix"):
        demos.load_dataset(path)


@pytest.mark.parametrize("change", [
    lambda a: a["dist"].__setitem__((0, 0), np.nan),
    lambda a: a["ent"].__setitem__((0, 0, 0), np.inf),
    lambda a: a["dist"].__setitem__((0, 0), 1.1),
    lambda a: a["hit"].__setitem__((0, 0), 6),
    lambda a: a["etype"].__setitem__((0, 0), 12),
    lambda a: a["emask"].__setitem__((0, 0), .5),
    lambda a: a["actions"].__setitem__(0, len(ACTIONS)),
    lambda a: a["actions"].__setitem__(0, -1),
    lambda a: a.update(actions=a["actions"].astype(np.float32)),
    lambda a: a.update(actions=a["actions"][:, None]),
    lambda a: a.update(prev=a["prev"][:, None]),
    lambda a: a["prev"].__setitem__(1, 0),
    lambda a: a["prev"].__setitem__(0, 1),
    lambda a: a["truncated"].__setitem__(-1, False),
    lambda a: a["terminated"].__setitem__(0, True),
    lambda a: a["terminated"].__setitem__(-1, True),
    lambda a: a["rewards"].__setitem__(0, np.nan),
    lambda a: a.update(dist=a["dist"].astype(np.float64)),
    lambda a: a.update(actions=np.asarray([object()] * 5)),
])
def test_semantic_corruption_rejected_even_with_recomputed_checksums(dataset, change):
    rewrite_episode(dataset, change)
    with pytest.raises(ValueError):
        demos.load_dataset(dataset)


def test_checksums_and_schema_reject_tampering(dataset):
    episode = dataset / "episode_00000000.npz"
    episode.write_bytes(b"broken archive")
    with pytest.raises(ValueError, match="checksum"):
        demos.load_dataset(dataset)


def test_corrupt_archive_with_recomputed_checksum(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text())
    path = dataset / manifest["episodes"][0]["file"]
    path.write_bytes(b"PK\x03\x04incomplete zip")
    manifest["episodes"][0]["sha256"] = file_hash(path)
    resign_manifest(dataset, manifest)
    with pytest.raises(ValueError):
        demos.load_dataset(dataset)


def test_nonfinite_metadata(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["episodes"][0]["outcome"]["health"] = float("nan")
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    (dataset / "manifest.sha256").write_text(file_hash(dataset / "manifest.json"))
    with pytest.raises(ValueError):
        demos.load_dataset(dataset)


@pytest.mark.parametrize("field,value", [("version", 100), ("version", 1), ("version", True),
                                         ("map_identity_schema", "absolute-path-v1"), ("feature_schema", "other"),
                                         ("action_sha256", "0" * 64)])
def test_manifest_schema(dataset, field, value):
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest[field] = value
    resign_manifest(dataset, manifest)
    with pytest.raises(ValueError, match="schema"):
        demos.load_dataset(dataset)


def test_manifest_checksum(dataset):
    with (dataset / "manifest.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="manifest checksum"):
        demos.load_dataset(dataset)


def test_episode_map_rejection(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["episodes"][0]["map"] = list(SUITES["dev"][0])
    resign_manifest(dataset, manifest)
    with pytest.raises(ValueError, match="training whitelist"):
        demos.load_dataset(dataset)


def test_train_map_rejection_before_environment_creation(tmp_path, policies, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("held-out environment must not be created")
    monkeypatch.setattr(doom_env, "make_env", forbidden)
    for maps in (SUITES["dev"][:1], SUITES["test"][:1], SUITES["gen_dev"][:1]):
        with pytest.raises(ValueError):
            demos.collect(policies[1], tmp_path / "never", training_maps=maps, seeds=[1],
                          episode_ticks=4, temperature=0., policy_seed=1, torch_threads=1)
    assert not (tmp_path / "never").exists()


def test_suite_overlap_fails_closed(tmp_path, policies, monkeypatch):
    monkeypatch.setitem(SUITES, "dev", SUITES["dev"] + SUITES["train"][:1])
    with pytest.raises(ValueError, match="overlap"):
        demos.collect(policies[1], tmp_path / "never", training_maps=SUITES["train"][:1],
                      seeds=[1], episode_ticks=4, temperature=0., policy_seed=1, torch_threads=1)


@pytest.mark.parametrize("temperature", [-1., float("nan"), float("inf"), True])
def test_invalid_temperature(tmp_path, policies, temperature):
    with pytest.raises(ValueError, match="temperature"):
        demos.collect(policies[1], tmp_path / "never", training_maps=SUITES["train"][:1],
                      seeds=[1], episode_ticks=4, temperature=temperature, policy_seed=1, torch_threads=1)


def test_collection_deterministic_and_refuses_overwrite(tmp_path, dataset, policies):
    before = {p.name: p.read_bytes() for p in dataset.iterdir()}
    with pytest.raises(FileExistsError):
        demos.collect(policies[1], dataset, training_maps=SUITES["train"][:2], seeds=[31, 32],
                      episode_ticks=20, temperature=0., policy_seed=7, torch_threads=1)
    assert before == {p.name: p.read_bytes() for p in dataset.iterdir()}
    paths = [demos.collect(policies[1], tmp_path / f"sample-{i}", training_maps=SUITES["train"][:1],
                          seeds=[31], episode_ticks=20, temperature=1., policy_seed=7, torch_threads=1)
             for i in range(2)]
    a, b = (demos.load_dataset(path) for path in paths)
    np.testing.assert_array_equal(a.episodes[0].actions, b.episodes[0].actions)
    assert a.sha256 == b.sha256


def test_tiny_positive_temperature_is_supported(tmp_path, policies, monkeypatch):
    monkeypatch.setattr(doom_env, "make_env", TinyTestEnvironment)
    path = demos.collect(policies[1], tmp_path / "tiny-temperature", training_maps=SUITES["train"][:1],
                         seeds=[1], episode_ticks=20, temperature=float(np.nextafter(0., 1.)),
                         policy_seed=1, torch_threads=1)
    assert np.all(demos.load_dataset(path).episodes[0].actions == 3)


def test_failed_episode_never_published(tmp_path, policies, monkeypatch):
    class Broken(TinyTestEnvironment):
        def step(self, action):
            raise RuntimeError("engine failed")
    monkeypatch.setattr(doom_env, "make_env", Broken)
    with pytest.raises(RuntimeError, match="engine failed"):
        demos.collect(policies[1], tmp_path / "never", training_maps=SUITES["train"][:1],
                      seeds=[1], episode_ticks=4, temperature=0., policy_seed=1, torch_threads=1)
    assert not (tmp_path / "never").exists()
    assert not list(tmp_path.glob(".never-*"))


def test_corrupt_source_rejected_before_collection(tmp_path, policies, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unverified policy must not collect")
    monkeypatch.setattr(doom_env, "make_env", forbidden)
    (policies[1] / "model.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        demos.collect(policies[1], tmp_path / "never", training_maps=SUITES["train"][:1],
                      seeds=[1], episode_ticks=4, temperature=0., policy_seed=1, torch_threads=1)
    assert not (tmp_path / "never").exists()


def test_publication_does_not_replace_empty_directory(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "data").write_text("owned")
    with pytest.raises(FileExistsError):
        demos._publish_directory(source, target)
    assert not list(target.iterdir())
    assert (source / "data").read_text() == "owned"


def test_concurrent_publication_only_one_winner(tmp_path):
    sources = [tmp_path / f"source-{i}" for i in range(2)]
    for index, source in enumerate(sources):
        source.mkdir()
        (source / "data").write_text(str(index))
    target = tmp_path / "destination"
    def publish(source):
        try:
            demos._publish_directory(source, target)
            return True
        except FileExistsError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, sources))
    assert sum(results) == 1
    assert (target / "data").read_text() == str(results.index(True))


def test_chunk_semantics_reset_once_and_carry_detached_state(dataset, policies, monkeypatch):
    net, _, _ = load_policy(policies[2])
    episode = demos.load_dataset(dataset).episodes[0]
    whole = bc.episode_cross_entropy(net, episode, chunk_steps=100, device="cpu")
    calls = []
    original = net.policy
    def tracked(features, h, starts):
        result = original(features, h, starts)
        calls.append((h.detach().clone(), h.grad_fn, starts.clone(), result["h"].detach().clone()))
        return result
    monkeypatch.setattr(net, "policy", tracked)
    chunked = bc.episode_cross_entropy(net, episode, chunk_steps=2, device="cpu", backward=True)
    assert chunked == pytest.approx(whole, abs=1e-6)
    assert [c[2][:, 0].tolist() for c in calls] == [[1., 0.], [0., 0.], [0.]]
    assert all(c[1] is None for c in calls)
    for previous, current in zip(calls, calls[1:]):
        torch.testing.assert_close(previous[3], current[0], rtol=0, atol=0)
    assert any(parameter.grad is not None for parameter in net.parameters())
    calls.clear()
    bc.episode_cross_entropy(net, episode, chunk_steps=2, device="cpu")
    assert torch.count_nonzero(calls[0][0]) == 0
    assert calls[0][2][0, 0] == 1


def test_split_uses_disjoint_training_maps(dataset):
    loaded = demos.load_dataset(dataset)
    train, validation = bc.split_episodes(loaded, fraction=.5, strategy="heldout_maps", seed=3)
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(range(len(loaded.episodes)))
    maps = lambda ids: {tuple(loaded.episodes[i].metadata["map"]) for i in ids}
    assert not maps(train) & maps(validation)
    assert train == tuple(sorted(train))
    assert validation == tuple(sorted(validation))


def test_train_overfits_exports_and_preserves_baseline(dataset, policies, tmp_path, monkeypatch):
    model, teacher, student = policies
    baseline = {p.name: p.read_bytes() for p in student.iterdir()}
    teacher_before = {p.name: p.read_bytes() for p in teacher.iterdir()}
    steps = []
    original = torch.optim.Adam.step
    def tracked(self, *args, **kwargs):
        steps.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", tracked)
    candidate = bc.train_bc(config(model, epochs=12), dataset, student, tmp_path / "run")
    net, result_model, manifest = load_policy(candidate)
    provenance = manifest["provenance"]
    history = provenance["history"]
    assert history[-1]["train_cross_entropy"] < history[0]["train_cross_entropy"] * .1
    assert history[-1]["validation_cross_entropy"] < history[0]["validation_cross_entropy"]
    assert provenance["dataset_sha256"] == demos.load_dataset(dataset).sha256
    assert len(steps) == 12 * len(provenance["train_episodes"])
    assert provenance["updates"] == len(steps)
    assert not hasattr(net, "v")
    assert result_model == model
    assert baseline == {p.name: p.read_bytes() for p in student.iterdir()}
    assert teacher_before == {p.name: p.read_bytes() for p in teacher.iterdir()}
    # Verifies exported tensors can warm-start the existing PPO migration path.
    target = AgentNet(model)
    bc.transfer_policy(target, net.state_dict(), dataclasses.asdict(result_model))
    with pytest.raises(FileExistsError):
        bc.train_bc(config(model), dataset, student, tmp_path / "run")


def test_no_critic_forward_or_optimizer_parameters(dataset, policies, tmp_path, monkeypatch):
    model = policies[0]
    original = torch.optim.Adam
    critic_parameters = []
    original_init = AgentNet.__init__
    def init(net, cfg):
        original_init(net, cfg)
        # Retain references: inference deletes its critic, so Python may otherwise reuse ids.
        critic_parameters.extend(net.v.parameters())
        def forbidden(*args, **kwargs):
            pytest.fail("critic must never run")
        net.v.forward = forbidden
    def optimizer(parameters, *args, **kwargs):
        parameters = list(parameters)
        assert not {id(p) for p in critic_parameters} & {id(p) for p in parameters}
        return original(parameters, *args, **kwargs)
    monkeypatch.setattr(AgentNet, "__init__", init)
    monkeypatch.setattr(torch.optim, "Adam", optimizer)
    bc.train_bc(config(model, epochs=1), dataset, policies[2], tmp_path / "run")


def test_training_deterministic(dataset, policies, tmp_path):
    candidates = [bc.train_bc(config(policies[0], epochs=1), dataset, policies[2], tmp_path / f"run-{i}")
                  for i in range(2)]
    states = [load_policy(path)[0].state_dict() for path in candidates]
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


@pytest.mark.parametrize("add_memory", [False, True])
def test_learned_slot_memory_chunk_and_export(dataset, policies, tmp_path, add_memory):
    model = dataclasses.replace(policies[0], memory_slots=2, memory_dim=8)
    source = export_policy(AgentNet(model), model, tmp_path / "memory-source", {"purpose": "test"})
    net, _, _ = load_policy(source)
    episode = demos.load_dataset(dataset).episodes[0]
    whole = bc.episode_cross_entropy(net, episode, chunk_steps=100, device="cpu")
    chunked = bc.episode_cross_entropy(net, episode, chunk_steps=2, device="cpu", backward=True)
    assert chunked == pytest.approx(whole, abs=1e-6)
    initial = policies[2] if add_memory else source
    candidate = bc.train_bc(config(model, epochs=1), dataset, initial, tmp_path / "memory-run")
    result, result_config, manifest = load_policy(candidate)
    assert manifest["provenance"]["migration"]["requires_retraining"] == add_memory
    assert result_config.memory_slots == 2
    assert result.init_state(1).shape[-1] == model.hidden + model.memory_slots * model.memory_dim


def test_nonfinite_gradient_prevents_publication(dataset, policies, tmp_path, monkeypatch):
    original = bc.episode_cross_entropy
    def poisoned(net, episode, **kwargs):
        loss = original(net, episode, **kwargs)
        if kwargs.get("backward"):
            next(p for p in net.parameters() if p.grad is not None).grad.fill_(float("nan"))
        return loss
    monkeypatch.setattr(bc, "episode_cross_entropy", poisoned)
    with pytest.raises(ValueError, match="gradient"):
        bc.train_bc(config(policies[0], epochs=1), dataset, policies[2], tmp_path / "invalid-run")
    assert not (tmp_path / "invalid-run").exists()


@pytest.mark.parametrize("key,value", [("epochs", 0), ("chunk_steps", 0), ("epochs", True),
                                       ("lr", float("nan")), ("max_grad_norm", 0),
                                       ("adam_eps", -1), ("weight_decay", -1),
                                       ("adam_betas", [1., .9]), ("validation_fraction", 1.),
                                       ("validation_strategy", "dev")])
def test_config_validation(policies, key, value):
    cfg = config(policies[0])
    cfg["bc"][key] = value
    with pytest.raises(ValueError):
        bc.validate_config(cfg)


def test_explicit_hyperparameters_required(policies):
    cfg = config(policies[0])
    del cfg["bc"]["chunk_steps"]
    with pytest.raises(ValueError):
        bc.validate_config(cfg)


@pytest.mark.skipif(os.environ.get("DOOM_BC_REAL_ENGINE") != "1",
                    reason="opt in with DOOM_BC_REAL_ENGINE=1 for tiny real-engine collection")
def test_real_engine_collection_and_bc(tmp_path, policies):
    model, teacher, student = policies
    path = demos.collect(teacher, tmp_path / "real", training_maps=SUITES["train"][:1],
                         seeds=[10, 11], episode_ticks=8, temperature=1., policy_seed=4, torch_threads=1)
    loaded = demos.load_dataset(path)
    assert len(loaded.episodes) == 2
    cfg = config(model, epochs=1)
    cfg["bc"]["validation_strategy"] = "episodes"
    load_policy(bc.train_bc(cfg, path, student, tmp_path / "real-bc"))
