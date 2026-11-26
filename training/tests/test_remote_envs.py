"""RemoteEnvs must return exactly what a local ProcEnvs returns for the same seed and actions.

Starts an env server on 127.0.0.1 with a throwaway auth key, steps a local pool and the remote pool in
lockstep with identical random actions, and compares every field of every step (features, reward,
done flags, privileged/aux arrays, episode records). Also checks that a wrong auth key is rejected.

    uv run --group train python training/tests/test_remote_envs.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import secrets
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]


def _server(bind, n, seed, keyfile):
    os.environ["DOOM_ENV_KEY_FILE"] = keyfile
    from doomrl.remote_envs import serve
    serve(bind, n, seed, [2100, 4200], True, True)


def same(a, b, path="") -> list[str]:
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(a)} != {sorted(b)}"]
        return [m for k in a for m in same(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, np.ndarray):
        return [] if (a.shape == b.shape and np.array_equal(a, b)) else [f"{path}: array mismatch"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def main() -> int:
    from doomrl.envs import ProcEnvs
    from doomrl.remote_envs import RemoteEnvs
    keyfile = Path(tempfile.mkdtemp()) / "key"
    keyfile.write_text(secrets.token_hex(32))
    os.environ["DOOM_ENV_KEY_FILE"] = str(keyfile)
    n, seed, bind = 3, 7, "127.0.0.1:7391"
    srv = mp.get_context("spawn").Process(target=_server, args=(bind, n, seed, str(keyfile)))  # not daemon: it spawns env workers
    srv.start()
    # 1) wrong key first, while the server waits in accept(): must be refused, server must survive
    bad = Path(tempfile.mkdtemp()) / "bad"
    bad.write_text(secrets.token_hex(32))
    os.environ["DOOM_ENV_KEY_FILE"] = str(bad)
    wrong_key_refused = False
    for _ in range(60):
        try:
            RemoteEnvs(bind, retries=1)
            break                                   # accepted: failure, recorded below
        except ConnectionRefusedError:
            import time; time.sleep(2)              # server still starting
        except Exception:
            wrong_key_refused = True
            break
    os.environ["DOOM_ENV_KEY_FILE"] = str(keyfile)
    # 2) the real learner connection
    remote = RemoteEnvs(bind)
    local = ProcEnvs(n, seed, ticks=[2100, 4200], fast=True, use_map=True)
    fails = []
    rng = np.random.default_rng(0)
    fails += same({i: o for i, o in enumerate(local.reset())}, {i: o for i, o in enumerate(remote.reset())}, "reset")
    episodes = 0
    for t in range(400):
        acts = rng.integers(0, 56, size=n)
        lo, ro = local.step(acts), remote.step(acts)
        episodes += sum("episode" in o for o in lo)
        fails += same({i: o for i, o in enumerate(lo)}, {i: o for i, o in enumerate(ro)}, f"step{t}")
        if fails:
            break
    if not wrong_key_refused:
        fails.append("connection with a wrong auth key was accepted")
    remote.close(); local.close()
    srv.join(timeout=30)
    if srv.is_alive():
        srv.terminate()
    print(f"remote vs local: 400 lockstep steps x {n} envs, {episodes} episodes finished, mismatches: {len(fails)}")
    for f in fails[:5]:
        print("  ", f)
    print("PASS" if not fails else "FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
