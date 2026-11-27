"""Versioned official-feature episodes; no expert scripts or outcome-based selection."""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]

from doom_env import observation_space
from doom_model import ACTIONS, AgentSideFeatures
from doom_model.artifacts import action_hash, file_hash, load_policy, verify_artifact
from doom_model.features import FEATURE_SCHEMA, PIX_H, PIX_W, VITALS_SCALE
from doomrl.maps import SUITES, resolve_training_maps

DATASET_VERSION = 2
MAP_IDENTITY_SCHEMA = "wad_sha256+map-v1"


@dataclass(frozen=True)
class Episode:
    features: dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class Dataset:
    directory: Path
    manifest: dict
    sha256: str
    episodes: tuple[Episode, ...]
    resolved_maps: dict[tuple[str, str], tuple[str, str]]


def _json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _digest(value):
    if not isinstance(value, str) or len(value) != hashlib.sha256().digest_size * 2:
        raise ValueError("invalid SHA256 digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("invalid SHA256 digest") from exc
    if len(decoded) != hashlib.sha256().digest_size:
        raise ValueError("invalid SHA256 digest")
    return value


def _positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _seed(name, value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(f"{name} must be an integer in [0, 2**32)")


def _map_registry():
    """Resolve content identities from local suites; hash each canonical WAD once per call."""
    hashes, train_paths, local_identities, splits = {}, {}, {}, []
    for split in ("train", "dev", "test", "gen_dev", "gen_test"):
        identities = set()
        for wad, map_name in SUITES[split]:
            path = Path(wad).resolve(strict=True)
            if path not in hashes:
                hashes[path] = file_hash(path)
            identity = (hashes[path], map_name)
            identities.add(identity)
            local_identities[(str(path), map_name)] = identity
            if split == "train":
                train_paths.setdefault(identity, (str(path), map_name))
        splits.append(identities)
    for i, left in enumerate(splits):
        if any(left & right for right in splits[i + 1:]):
            raise ValueError("unsupported train/dev/test content identity overlap")
    return train_paths, local_identities


def _train_maps(manifest):
    """Resolve a local path whitelist and its portable content identities for collection."""
    maps = resolve_training_maps(manifest)
    train_paths, local_identities = _map_registry()
    identities = [local_identities[(str(Path(w).resolve(strict=True)), m)] for w, m in maps]
    if len(set(identities)) != len(identities) or not set(identities) <= train_paths.keys():
        raise ValueError("training maps must have unique whitelisted content identities")
    return maps, identities


def _resolve_dataset_maps(values):
    """Map [wad_sha256, map_name] identities to this installation's trusted train paths."""
    if not isinstance(values, list) or not values:
        raise ValueError("dataset requires explicit training content identities")
    identities = []
    for value in values:
        if (not isinstance(value, list) or len(value) != 2
                or not isinstance(value[1], str) or not value[1]):
            raise ValueError("invalid training map content identity")
        identities.append((_digest(value[0]), value[1]))
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate training map content identities")
    train_paths, _ = _map_registry()
    if not set(identities) <= train_paths.keys():
        raise ValueError("training map content identity unavailable or changed in local whitelist")
    return {identity: train_paths[identity] for identity in identities}


def feature_spec(use_pixels):
    """Shapes, dtypes and bounds come from the official observation contract."""
    if type(use_pixels) is not bool:
        raise ValueError("use_pixels must be boolean")
    space = observation_space()
    rays, entities = space["rays"], space["visible_entities"]
    spec = {}
    for key, box, scale in (("dist", rays["distance"], 1),
                            ("ent", entities["features"], 1),
                            ("vit", space["vitals"], VITALS_SCALE)):
        spec[key] = (box.shape, np.dtype("float32"), box.low / scale, box.high / scale)
    for key, discrete in (("hit", rays["hit_type"]), ("etype", entities["entity_type"])):
        spec[key] = (discrete.shape, np.dtype("int64"), discrete.start,
                     discrete.start + discrete.nvec - 1)
    spec["emask"] = (entities["mask"].shape, np.dtype("float32"), 0, 1)
    spec["prev"] = ((), np.dtype("int64"), 0, len(ACTIONS) - 1)
    if use_pixels:
        spec["pix"] = ((4, PIX_H, PIX_W), np.dtype("uint8"), 0, 255)
    return spec


def _array(name, value, shape, dtype, low=None, high=None):
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
        raise ValueError(f"{name}: invalid shape/dtype, expected {shape}/{dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: nonfinite values")
    if low is not None and (np.any(value < low) or np.any(value > high)):
        raise ValueError(f"{name}: value outside schema range")


def validate_episode(episode, use_pixels):
    """Require one complete episode, scalar action indices, and exact previous-action history."""
    if episode.actions.ndim != 1 or not len(episode.actions):
        raise ValueError("actions must be a nonempty scalar sequence")
    n = len(episode.actions)
    _array("actions", episode.actions, (n,), np.dtype("int64"), 0, len(ACTIONS) - 1)
    spec = feature_spec(use_pixels)
    if set(episode.features) != set(spec):
        raise ValueError("feature schema mismatch")
    for name, (shape, dtype, low, high) in spec.items():
        _array(name, episode.features[name], (n, *shape), dtype, low, high)
    if not np.isin(episode.features["emask"], (0, 1)).all():
        raise ValueError("emask must be binary")
    prev = episode.features["prev"]
    if prev[0] != 0 or not np.array_equal(prev[1:], episode.actions[:-1]):
        raise ValueError("previous actions break episode continuity")
    _array("rewards", episode.rewards, (n,), np.dtype("float64"))
    for name in ("terminated", "truncated"):
        _array(name, getattr(episode, name), (n,), np.dtype("bool"))
    done = episode.terminated | episode.truncated
    if done[:-1].any() or not done[-1] or np.any(episode.terminated & episode.truncated):
        raise ValueError("episode must have exactly one terminal boundary at the end")
    meta = episode.metadata
    if type(meta.get("steps")) is not int or meta["steps"] != n:
        raise ValueError("episode steps mismatch")
    _seed("episode seed", meta.get("seed"))
    _positive_int("episode_ticks", meta.get("episode_ticks"))
    outcome = meta.get("outcome", {})
    if (not isinstance(outcome, dict)
            or type(outcome.get("terminated")) is not bool or type(outcome.get("truncated")) is not bool
            or outcome["terminated"] != bool(episode.terminated[-1])
            or outcome["truncated"] != bool(episode.truncated[-1])):
        raise ValueError("terminal outcome mismatch")
    native_return = outcome.get("native_return")
    if (type(native_return) not in (int, float) or not np.isfinite(native_return)
            or not np.isclose(native_return, episode.rewards.sum(), rtol=1e-12, atol=1e-12)):
        raise ValueError("terminal reward mismatch")


def _publish_directory(staging, destination):
    """Atomic no-replace rename, using the Linux/macOS filesystem exclusion primitive."""
    libc = ctypes.CDLL(None, use_errno=True)
    source, target = os.fsencode(staging), os.fsencode(destination)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = rename(source, target, 0x00000004)  # RENAME_EXCL, <stdio.h>
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = rename(-100, source, -100, target, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise RuntimeError("exclusive directory publication requires Linux or macOS")
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _new_staging(destination):
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))


def load_dataset(directory) -> Dataset:
    """Load v2 NPZ episodes, validating every file and content-identified map split.

    Human recordings use the same features/arrays and episode metadata as collect(), with
    source.kind='human' and source.manifest_sha256 identifying their recording manifest.
    manifest.sha256 covers manifest.json; each ordered episode row covers its NPZ file.
    training_maps and episode.map contain [wad_sha256, map_name], never source WAD paths.
    Dataset.resolved_maps maps those identities to current local train paths; v1 is unsupported.
    """
    directory = Path(directory)
    identity = file_hash(directory / "manifest.json")
    if identity != _digest((directory / "manifest.sha256").read_text().strip()):
        raise ValueError("dataset manifest checksum mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be an object")
    json.dumps(manifest, allow_nan=False)  # Also reject nonfinite values inside metadata.
    if (type(manifest.get("version")) is not int or manifest["version"] != DATASET_VERSION
            or manifest.get("map_identity_schema") != MAP_IDENTITY_SCHEMA
            or manifest.get("feature_schema") != FEATURE_SCHEMA
            or manifest.get("action_sha256") != action_hash()):
        raise ValueError("dataset schema/action mismatch")
    feature_spec(manifest.get("use_pixels"))
    source = manifest.get("source", {})
    if not isinstance(source, dict) or source.get("kind") not in ("learned_policy", "human"):
        raise ValueError("unsupported demonstration source")
    _digest(source.get("manifest_sha256"))
    maps = _resolve_dataset_maps(manifest.get("training_maps"))
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset requires complete episodes")
    expected_files = {"manifest.json", "manifest.sha256"}
    episodes, jobs = [], set()
    for index, meta in enumerate(rows):
        if not isinstance(meta, dict):
            raise ValueError("episode metadata must be an object")
        name = f"episode_{index:08d}.npz"
        if meta.get("order") != index or type(meta.get("order")) is not int or meta.get("file") != name:
            raise ValueError("episode order/file mismatch")
        if meta.get("feature_schema") != FEATURE_SCHEMA or meta.get("action_sha256") != action_hash():
            raise ValueError("episode schema/action mismatch")
        if meta.get("source_manifest_sha256") != source["manifest_sha256"]:
            raise ValueError("episode source identity mismatch")
        map_id = meta.get("map")
        if (not isinstance(map_id, list) or len(map_id) != 2
                or not all(isinstance(part, str) for part in map_id) or tuple(map_id) not in maps):
            raise ValueError("episode map outside training whitelist")
        _seed("episode seed", meta.get("seed"))
        job = (*map_id, meta["seed"])
        if job in jobs:
            raise ValueError("duplicate map/seed episode")
        jobs.add(job)
        path = directory / name
        if path.is_symlink() or file_hash(path) != _digest(meta.get("sha256")):
            raise ValueError("episode checksum mismatch")
        expected_files.add(name)
        try:
            with np.load(path, allow_pickle=False) as saved:
                feature_names = set(feature_spec(manifest["use_pixels"]))
                required = feature_names | {"actions", "rewards", "terminated", "truncated"}
                if set(saved.files) != required or len(saved.files) != len(required):
                    raise ValueError("episode array schema mismatch")
                arrays = {k: saved[k] for k in saved.files}
        except (OSError, EOFError, zipfile.BadZipFile) as exc:
            raise ValueError("invalid episode archive") from exc
        if file_hash(path) != meta["sha256"]:
            raise ValueError("episode changed while loading")
        episode = Episode({k: arrays[k] for k in feature_names}, arrays["actions"], arrays["rewards"],
                          arrays["terminated"], arrays["truncated"], meta)
        validate_episode(episode, manifest["use_pixels"])
        for value in arrays.values():
            value.flags.writeable = False
        episodes.append(episode)
    if {p.name for p in directory.iterdir()} != expected_files:
        raise ValueError("unmanifested dataset files")
    if file_hash(directory / "manifest.json") != identity:
        raise ValueError("dataset changed while loading")
    return Dataset(directory, manifest, identity, tuple(episodes), maps)


def collect(candidate, directory, *, training_maps, seeds, episode_ticks, temperature,
            policy_seed, torch_threads) -> Path:
    """Collect every map×seed complete episode from a verified learned export, in supplied order."""
    from doom_env import make_env

    maps, map_identities = _train_maps(training_maps)
    _positive_int("episode_ticks", episode_ticks)
    _positive_int("torch_threads", torch_threads)
    _seed("policy_seed", policy_seed)
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("unique nonempty episode seeds required")
    for seed in seeds:
        _seed("seed", seed)
    if type(temperature) not in (int, float) or not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    candidate, directory = Path(candidate), Path(directory)
    identity = file_hash(candidate / "manifest.json")
    net, config, _ = load_policy(candidate)
    if config.head != "actor_critic":
        raise ValueError("demonstration collection requires a categorical policy")
    old_threads = torch.get_num_threads()
    staging = _new_staging(directory)
    try:
        torch.set_num_threads(torch_threads)
        manifest = {
            "version": DATASET_VERSION, "feature_schema": FEATURE_SCHEMA, "action_sha256": action_hash(),
            "map_identity_schema": MAP_IDENTITY_SCHEMA,
            "use_pixels": config.use_pixels, "training_maps": map_identities,
            "source": {"kind": "learned_policy", "manifest_sha256": identity,
                       "candidate": str(candidate.resolve())},
            "collection": {"temperature": temperature, "policy_seed": policy_seed,
                           "episode_ticks": episode_ticks, "seeds": seeds, "torch_threads": torch_threads},
            "episodes": [],
        }
        for map_index, (wad, map_name) in enumerate(maps):
            for seed in seeds:
                env = make_env(wad, map_name, episode_ticks=episode_ticks)
                try:
                    obs, _ = env.reset(seed=seed)
                    adapter = AgentSideFeatures(use_map=config.use_map, use_pixels=config.use_pixels)
                    features = adapter.reset(obs)
                    h = net.init_state(1)
                    action_seed = int(np.random.SeedSequence([policy_seed, map_index, seed])
                                      .generate_state(1, dtype=np.uint64)[0])
                    generator = torch.Generator().manual_seed(action_seed)
                    history, actions, rewards, ends, truncations = [], [], [], [], []
                    terminated = truncated = False
                    while not (terminated or truncated):
                        for key, (shape, dtype, low, high) in feature_spec(config.use_pixels).items():
                            _array(key, features[key], shape, dtype, low, high)
                        inputs = {k: torch.as_tensor(v)[None, None] for k, v in features.items()}
                        with torch.no_grad():
                            out = net.policy(inputs, h, torch.tensor([[float(not actions)]]))
                            logits = out["logits"][0, 0]
                            if not torch.isfinite(logits).all():
                                raise ValueError("nonfinite learned policy logits")
                            if temperature == 0:
                                action = int(logits.argmax())
                            else:
                                # Centering first prevents positive overflow at small temperatures.
                                scores = logits.double()
                                probs = ((scores - scores.max()) / temperature).softmax(-1)
                                action = int(torch.multinomial(probs, 1, generator=generator))
                            h = out["h"]
                        history.append({k: v.copy() for k, v in features.items()})
                        actions.append(action)
                        obs, reward, terminated, truncated, _ = env.step(ACTIONS[action].astype(np.int8).copy())
                        rewards.append(reward)
                        ends.append(terminated)
                        truncations.append(truncated)
                        if not (terminated or truncated):
                            features = adapter.step(action, obs)
                    order = len(manifest["episodes"])
                    meta = {"order": order, "file": f"episode_{order:08d}.npz",
                            "map": list(map_identities[map_index]),
                            "seed": seed, "action_seed": action_seed, "steps": len(actions),
                            "episode_ticks": episode_ticks, "feature_schema": FEATURE_SCHEMA,
                            "action_sha256": action_hash(), "source_manifest_sha256": identity,
                            "outcome": {"terminated": bool(terminated), "truncated": bool(truncated),
                                        "native_return": float(np.sum(rewards, dtype=np.float64))}}
                    final = env.privileged_state()
                    meta["outcome"].update({k: final[k] for k in ("exit_reached", "dead", "timeout", "health", "kills")})
                    episode = Episode({k: np.stack([f[k] for f in history]) for k in history[0]},
                                      np.asarray(actions, np.int64), np.asarray(rewards, np.float64),
                                      np.asarray(ends, bool), np.asarray(truncations, bool), meta)
                    validate_episode(episode, config.use_pixels)
                    np.savez_compressed(staging / meta["file"], **episode.features, actions=episode.actions,
                                        rewards=episode.rewards, terminated=episode.terminated,
                                        truncated=episode.truncated)
                    meta["sha256"] = file_hash(staging / meta["file"])
                    manifest["episodes"].append(meta)
                finally:
                    env.close()
        verify_artifact(candidate)
        if file_hash(candidate / "manifest.json") != identity:
            raise ValueError("source candidate changed during collection")
        _json(staging / "manifest.json", manifest)
        (staging / "manifest.sha256").write_text(file_hash(staging / "manifest.json") + "\n")
        load_dataset(staging)
        _publish_directory(staging, directory)
    finally:
        torch.set_num_threads(old_threads)
        if staging.exists():
            shutil.rmtree(staging)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--training-maps", required=True, help="training manifest path or suite:train")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--episode-ticks", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--torch-threads", type=int, required=True)
    args = parser.parse_args()
    result = collect(args.candidate, args.out, training_maps=args.training_maps, seeds=args.seeds,
                     episode_ticks=args.episode_ticks, temperature=args.temperature,
                     policy_seed=args.policy_seed, torch_threads=args.torch_threads)
    print(result)


if __name__ == "__main__":
    main()
