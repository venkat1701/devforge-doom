"""Environments on another machine: a CPU VM runs the env workers, the GPU VM runs the learner.

Why: on the T4 VM each run was env-bound (350-575 steps/s) while the T4 learner could process ~3,600
samples/s (measured). The env side needs CPU cores; this moves it to a big CPU VM on the same private
network.

  server (CPU VM):  python -m doomrl.remote_envs --bind 10.0.0.5:7000 --n 96 --seed 0
  learner config:   "remote_envs": "10.0.0.5:7000"   (auth key from $DOOM_ENV_KEY_FILE, never in git)

Transport: multiprocessing.connection (pickle over TCP with HMAC challenge-response on the shared auth
key). The server binds to a private address only. The server wraps a normal ProcEnvs, so the env code
and observations are exactly those of local training; only where the processes live changes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import hashlib
import json
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]


def _authkey() -> bytes:
    path = Path(os.environ.get("DOOM_ENV_KEY_FILE", Path.home() / ".doom_env_key"))
    key = path.read_bytes().strip()
    if len(key) < 32:
        raise SystemExit(f"auth key in {path} is missing or too short")
    return key


def _parse(addr: str):
    host, port = addr.rsplit(":", 1)
    return host, int(port)


def environment_spec(cfg, n, use_map, use_pixels):
    import vizdoom
    from doom_env.wrapper import IWAD
    from doomrl.maps import resolve_training_maps
    from doom_model.features import FEATURE_SCHEMA
    from doom_model.actions import ACTIONS
    maps = resolve_training_maps(cfg.get("training_maps"))
    from doom_model.artifacts import file_hash
    # Content hashes, not machine-specific installation paths.
    wads = {w: file_hash(w) for w, _ in maps}
    sources = [Path(__file__), Path(__file__).with_name("envs.py"), Path(__file__).with_name("maps.py"), Path(__file__).with_name("progress.py"),
               ROOT / "submission/doom_model/features.py", ROOT / "training/fastenv/__init__.py",
               ROOT / "training/fastenv/sensors_fast.py"]
    sources += sorted((ROOT / "env/doom_env").glob("*.py"))
    return {"version": 2, "n": n, "seed": cfg["seed"],
            "ticks": cfg.get("episode_ticks", 2100), "fast": cfg.get("fast_env", False),
            "use_map": use_map, "use_pixels": use_pixels, "reward": cfg.get("reward") or "official-wrapper",
            "vizdoom_version": vizdoom.__version__, "iwad": file_hash(IWAD),
            "feature_schema": FEATURE_SCHEMA,
            "actions": hashlib.sha256(ACTIONS.tobytes()).hexdigest(),
            "maps": [(wads[w], m) for w, m in maps],
            "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sources}}


def serve(bind: str, n: int, seed: int, ticks, fast: bool, use_map: bool, reward=None, use_pixels=True,
          training_maps=None):
    from doomrl.envs import ProcEnvs
    spec = environment_spec(dict(seed=seed, episode_ticks=ticks, fast_env=fast,
                                 training_maps=training_maps), n, use_map, use_pixels)
    envs = ProcEnvs(n, seed, ticks=ticks, fast=fast, use_map=use_map, reward=reward,
                    use_pixels=use_pixels, training_maps=training_maps)
    with Listener(_parse(bind), authkey=_authkey()) as listener:
        print(f"[env-server] {n} envs ready, listening on {bind}", flush=True)
        while True:
            try:
                conn = listener.accept()
            except (AuthenticationError, EOFError, OSError) as e:   # wrong key / dropped handshake: keep serving
                print(f"[env-server] rejected a connection: {type(e).__name__}", flush=True)
                continue
            print(f"[env-server] learner connected from {listener.last_accepted}", flush=True)
            try:
                while True:
                    cmd, data = conn.recv()
                    if cmd == "info":
                        conn.send({"n": n, "spec": spec})
                    elif cmd == "reset":
                        conn.send(envs.reset(data))
                    elif cmd == "step":
                        conn.send(envs.step(data))
                    elif cmd == "set_step":
                        envs.set_global_step(data)
                        conn.send(None)
                    elif cmd == "detach":
                        break
                    elif cmd == "close":
                        conn.close()
                        envs.close()
                        print("[env-server] closed by learner", flush=True)
                        return
            except (EOFError, ConnectionResetError, BrokenPipeError):
                print("[env-server] learner disconnected; waiting for a new connection", flush=True)
            finally:
                conn.close()


class RemoteEnvs:
    """Same interface as envs.ProcEnvs (reset / step / set_global_step / close / n / procs)."""

    procs: list = []          # no local worker processes: ram_stats() then reports the learner only

    def __init__(self, address: str, retries: int = 60, expected=None, timeout=120.):
        self.timeout = timeout
        for i in range(retries):
            try:
                self.conn = Client(_parse(address), authkey=_authkey())
                break
            except AuthenticationError:
                raise
            except (ConnectionRefusedError, OSError):
                if i == retries - 1:
                    raise
                time.sleep(5)
        self.conn.send(("info", None))
        info = self._recv()
        self.n = info["n"]
        if expected is not None and info.get("spec") != expected:
            self.conn.send(("detach", None))
            self.conn.close()
            raise ValueError("remote environment configuration/schema fingerprint mismatch")

    def _recv(self):
        if not self.conn.poll(self.timeout):
            raise TimeoutError("remote environment response deadline exceeded")
        return self.conn.recv()

    def reset(self, seeds=None):
        self.conn.send(("reset", seeds))
        return self._recv()

    def step(self, actions):
        self.conn.send(("step", [int(a) for a in actions]))
        return self._recv()

    def set_global_step(self, step):
        self.conn.send(("set_step", int(step)))
        self._recv()

    def close(self):
        try:
            self.conn.send(("close", None))
            self.conn.close()
        except OSError:
            pass


def make_envs(cfg: dict, n: int, use_map: bool, use_pixels: bool = True):
    """Local process pool, or a remote server when cfg['remote_envs'] is set."""
    if cfg.get("remote_envs"):
        envs = RemoteEnvs(cfg["remote_envs"], expected=environment_spec(cfg, n, use_map, use_pixels))
        return envs
    from doomrl.envs import ProcEnvs
    return ProcEnvs(n, cfg["seed"], ticks=cfg.get("episode_ticks", 2100), fast=cfg.get("fast_env", False),
                    use_map=use_map, reward=cfg.get("reward"), use_pixels=use_pixels,
                    training_maps=cfg.get("training_maps"))


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", required=True, help="private ip:port")
    ap.add_argument("--config", required=True, help="training config (ticks, reward, map flags)")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    serve(a.bind, a.n, a.seed, cfg.get("episode_ticks", 2100), cfg.get("fast_env", False),
          cfg["model"].get("use_map", False), cfg.get("reward"), cfg["model"].get("use_pixels", True),
          cfg.get("training_maps"))


if __name__ == "__main__":
    main()
