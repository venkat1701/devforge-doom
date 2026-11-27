"""Local/remote parity, configuration rejection, and disconnect handling."""
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import socket
import sys
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]


def server(bind, keyfile):
    os.environ["DOOM_ENV_KEY_FILE"] = keyfile
    from doomrl.remote_envs import serve
    from doomrl.maps import FD2
    serve(bind, 1, 7, [16, 24], False, False, use_pixels=True, training_maps=[(FD2, "MAP01")])


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


def test_remote_local_parity_and_mismatch(tmp_path, monkeypatch):
    from doomrl.envs import ProcEnvs
    from doomrl.remote_envs import RemoteEnvs, environment_spec
    from doomrl.maps import FD2
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    bind = f"127.0.0.1:{port}"
    key = tmp_path / "key"
    key.write_text(secrets.token_hex(32))
    monkeypatch.setenv("DOOM_ENV_KEY_FILE", str(key))
    srv = mp.get_context("spawn").Process(target=server, args=(bind, str(key)))
    srv.start()
    remote = local = None
    try:
        cfg = dict(seed=7, episode_ticks=[16, 24], fast_env=False, training_maps=[(FD2, "MAP01")])
        spec = environment_spec(cfg, 1, False, True)
        with pytest.raises(ValueError, match="fingerprint"):
            RemoteEnvs(bind, expected=dict(spec, seed=8), timeout=20)
        remote = RemoteEnvs(bind, expected=spec, timeout=20)
        local = ProcEnvs(1, 7, ticks=[16, 24], fast=False, use_pixels=True,
                         training_maps=[(FD2, "MAP01")])
        remote.set_global_step(100)
        local.set_global_step(100)
        same(local.reset()[0], remote.reset()[0])
        for action in range(20):
            same(local.step([action])[0], remote.step([action])[0])
    finally:
        if remote:
            remote.close()
        if local:
            local.close()
        srv.join(timeout=5)
        if srv.is_alive():
            srv.terminate()
            srv.join(timeout=5)
