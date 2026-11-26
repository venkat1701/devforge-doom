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


def serve(bind: str, n: int, seed: int, ticks, fast: bool, use_map: bool, reward=None, use_pixels=False):
    from doomrl.envs import ProcEnvs
    envs = ProcEnvs(n, seed, ticks=ticks, fast=fast, use_map=use_map, reward=reward, use_pixels=use_pixels)
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
                        conn.send({"n": n})
                    elif cmd == "reset":
                        conn.send(envs.reset(data))
                    elif cmd == "step":
                        conn.send(envs.step(data))
                    elif cmd == "set_step":
                        envs.set_global_step(data)
                    elif cmd == "close":
                        conn.close()
                        envs.close()
                        print("[env-server] closed by learner", flush=True)
                        return
            except (EOFError, ConnectionResetError):
                print("[env-server] learner disconnected; waiting for a new connection", flush=True)


class RemoteEnvs:
    """Same interface as envs.ProcEnvs (reset / step / set_global_step / close / n / procs)."""

    procs: list = []          # no local worker processes: ram_stats() then reports the learner only

    def __init__(self, address: str, retries: int = 60):
        for i in range(retries):
            try:
                self.conn = Client(_parse(address), authkey=_authkey())
                break
            except (ConnectionRefusedError, OSError):
                if i == retries - 1:
                    raise
                time.sleep(5)
        self.conn.send(("info", None))
        self.n = self.conn.recv()["n"]

    def reset(self, seeds=None):
        self.conn.send(("reset", seeds))
        return self.conn.recv()

    def step(self, actions):
        self.conn.send(("step", [int(a) for a in actions]))
        return self.conn.recv()

    def set_global_step(self, step):
        self.conn.send(("set_step", int(step)))

    def close(self):
        try:
            self.conn.send(("close", None))
            self.conn.close()
        except OSError:
            pass


def make_envs(cfg: dict, n: int, use_map: bool, use_pixels: bool = False):
    """Local process pool, or a remote server when cfg['remote_envs'] is set."""
    if cfg.get("remote_envs"):
        envs = RemoteEnvs(cfg["remote_envs"])
        if envs.n != n:
            raise SystemExit(f"remote server has {envs.n} envs but config num_envs={n}")
        return envs
    from doomrl.envs import ProcEnvs
    return ProcEnvs(n, cfg["seed"], ticks=cfg.get("episode_ticks", 2100), fast=cfg.get("fast_env", True),
                    use_map=use_map, reward=cfg.get("reward"), use_pixels=use_pixels)


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", required=True, help="private ip:port")
    ap.add_argument("--config", required=True, help="training config (ticks, reward, map flags)")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    serve(a.bind, a.n, a.seed, cfg.get("episode_ticks", 2100), cfg.get("fast_env", True),
          cfg["model"].get("use_map", True), cfg.get("reward"), cfg["model"].get("use_pixels", False))


if __name__ == "__main__":
    main()
