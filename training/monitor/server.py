"""Private loopback-only, read-only training dashboard (stdlib only).

Run: python -m monitor.server --run-dir RUN --frames-dir FRAMES [--port 8765] [--price-per-hour 1.518]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STATIC = Path(__file__).parent / "static"
STATIC_FILES = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "application/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8")}
METRIC_KEYS = ("global_step", "sps", "lr", "entropy", "approx_kl", "clip_fraction", "vloss", "pg", "grad_norm",
               "explained_variance", "hidden_std", "pss_learner_gb", "pss_workers_gb")
EP_KEYS = ("kills", "exit", "dead", "timeout", "return", "keys", "native_return", "cells", "doors_opened",
           "locked_opened", "keys_picked", "revisit_ratio", "distance", "damage")
SHAPING_KEYS = ("cell", "key", "door", "unlock", "kill", "damage", "exit", "blind_ammo", "route", "blind_press", "fist_with_ammo", "step")
STALE_AFTER_S = 120.0
TRAIN_FRAME_UNAVAILABLE = {"available": False, "reason": "no live training frame: the trainer was started "
                           "without DOOM_FRAME_TAP_DIR, or worker 0 has not written one yet"}
TRAIN_FRAME_STALE_S = 30.0


def _num(v):
    """JSON-safe number or None. Missing/non-finite stays None (never 0); bools count as numbers 0/1."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)) and math.isfinite(v):
        return v
    return None


class JsonlTail:
    """Incremental reader of an append-only JSONL file, keeping a bounded history with sequence numbers."""

    def __init__(self, path: Path, maxlen: int):
        self.path, self.items = Path(path), collections.deque(maxlen=maxlen)
        self.offset, self.seq, self.bad = 0, 0, 0   # seq = total records ever read

    def poll(self):
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.offset:                      # truncated/rotated: start over
            self.offset = 0
        if size == self.offset:
            return
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        end = data.rfind(b"\n") + 1                 # only complete lines
        for line in data[:end].splitlines():
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError
            except ValueError:
                self.bad += 1
                continue
            self.seq += 1
            self.items.append((self.seq, rec))
        self.offset += end


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


class Monitor:
    def __init__(self, run_dir, frames_dir=None, price_per_hour=1.518, horizon=19_996_672, configured_horizon=20_000_000,
                 started_at=None, max_updates=2000, max_episodes=5000, clock=time.time, gpu_probe=None, sps_window=20):
        self.run_dir = Path(run_dir)
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.price, self.horizon, self.configured_horizon = price_per_hour, horizon, configured_horizon
        self.started_at, self.clock, self.sps_window = started_at, clock, sps_window
        self.updates = JsonlTail(self.run_dir / "updates.jsonl", max_updates)
        self.episodes = JsonlTail(self.run_dir / "episodes.jsonl", max_episodes)
        self.lock = threading.Lock()
        self._gpu_probe, self._gpu_cache, self._gpu_t = gpu_probe or _nvidia_smi, None, -1e9
        self.first_seen = clock()
        self.start_step, self.evals_dir = 0, None

    def poll(self):
        with self.lock:
            self.updates.poll()
            self.episodes.poll()

    # ---- metrics
    def metrics(self, after=0):
        with self.lock:
            items = [(s, r) for s, r in self.updates.items if s > after]
            first = self.updates.items[0][0] if self.updates.items else self.updates.seq + 1
            gap = after < first - 1 and self.updates.seq > 0
            rows = [dict({"seq": s}, **{k: _num(r.get(k)) for k in METRIC_KEYS}) for s, r in items]
            return {"gap": gap, "first_seq": first, "last_seq": self.updates.seq, "updates": rows}

    # ---- episodes
    def episode_stats(self):
        with self.lock:
            eps = [r for _, r in self.episodes.items]
        def agg(sub):
            out = {"n": len(sub)}
            for k in EP_KEYS:
                out[k] = _mean([_num(e.get(k)) for e in sub])
            return out
        fd = [e for e in eps if str(e.get("map", "")).startswith("freedoom")]
        res = {"total_seen": self.episodes.seq, "last_100": agg(eps[-100:]), "last_500": agg(eps[-500:]),
               "freedoom_last_100": agg(fd[-100:]), "freedoom_last_500": agg(fd[-500:])}
        # reward component breakdown (training-only shaping), mean per episode over the last 500
        sh = [e["shaping"] for e in eps[-500:] if isinstance(e.get("shaping"), dict)]
        res["shaping_last_500"] = {k: _mean([_num(x.get(k)) for x in sh]) for k in SHAPING_KEYS} if sh else None
        caps = sorted({e["ticks"] for e in eps if _num(e.get("ticks")) is not None})
        res["by_cap"] = {str(int(c)): agg([e for e in eps[-500:] if e.get("ticks") == c]) for c in caps}
        # per-episode series for charts (bounded, nulls preserved)
        res["series"] = [{"global_step": _num(e.get("global_step")), "freedoom": str(e.get("map", "")).startswith("freedoom"),
                          **{k: _num(e.get(k)) for k in EP_KEYS}} for e in eps[-1500:]]
        return res

    # ---- fixed-seed evaluations (doomrl.evaluate output), separate from training metrics
    def evals(self):
        d = self.evals_dir
        if not d or not d.is_dir():
            return []
        out = []
        for f in sorted(d.glob("*.json")):
            try:
                r = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            s = r.get("summary") or (r.get("result") or {}).get("summary") or {}
            row = {"name": f.stem, "weights": r.get("weights"), "suite": r.get("suite"), "episodes": r.get("episodes")}
            for k in ("exit", "keys", "kills", "dead", "cells", "doors_opened", "locked_opened", "revisit_ratio",
                      "damage", "proxy_score", "ms_mean"):
                row[k] = _num((s.get(k) or {}).get("mean")) if isinstance(s.get(k), dict) else None
            out.append(row)
        return out

    # ---- status
    def latest_checkpoint(self):
        best = None
        for d in self.run_dir.glob("candidate_*"):
            m = re.fullmatch(r"candidate_(\d+)", d.name)
            if m and (d / "manifest.json").is_file() and (best is None or int(m[1]) > best[0]):
                best = (int(m[1]), d / "manifest.json")
        return best

    def gpu(self):
        now = self.clock()
        if now - self._gpu_t >= 5:
            self._gpu_t = now
            try:
                self._gpu_cache = self._gpu_probe()
            except Exception:
                self._gpu_cache = None
        return self._gpu_cache

    def status(self):
        now = self.clock()
        with self.lock:
            ups = [r for _, r in self.updates.items]
            n_updates = self.updates.seq
        step = next((int(r["global_step"]) for r in reversed(ups) if _num(r.get("global_step")) is not None), 0)
        recent = [_num(r.get("sps")) for r in ups[-self.sps_window:]]
        sps_recent = _mean(recent)
        start = self.started_at
        if start is None:
            try:
                start = (self.run_dir / "config.json").stat().st_mtime
            except OSError:
                start = self.first_seen
        elapsed = max(0.0, now - start)
        done = step - self.start_step
        sps_wall = done / elapsed if elapsed > 0 and done > 0 else None
        remaining = max(0, self.horizon - step)
        eta = remaining / sps_recent if sps_recent else None
        try:
            hb = now - (self.run_dir / "updates.jsonl").stat().st_mtime
        except OSError:
            hb = None
        ck = self.latest_checkpoint()
        ck_age = max(0.0, now - ck[1].stat().st_mtime) if ck else None
        try:
            free = shutil.disk_usage(self.run_dir).free
        except OSError:
            free = None
        spend = elapsed / 3600 * self.price
        return {
            "steps": step, "updates": n_updates, "horizon": self.horizon, "configured_horizon": self.configured_horizon,
            "percent_done": 100.0 * max(0, step - self.start_step) / (self.horizon - self.start_step)
            if self.horizon > self.start_step else None, "start_step": self.start_step,
            "sps_recent": sps_recent, "sps_recent_window": self.sps_window, "sps_wall": sps_wall,
            "eta_seconds": eta, "elapsed_seconds": elapsed,
            "price_per_hour": self.price, "spend_estimated": spend,
            "spend_projected_total_estimated": (elapsed + eta) / 3600 * self.price if eta is not None else None,
            "spend_note": "estimated from the --price-per-hour flag x elapsed time since config.json was written",
            "last_checkpoint_step": ck[0] if ck else None, "last_checkpoint_age_seconds": ck_age,
            "heartbeat_age_seconds": hb, "stale": hb is None or hb > STALE_AFTER_S,
            "gpu": self.gpu(), "disk_free_bytes": free, "now": now,
        }

    # ---- frames
    def spectator_meta(self):
        if not self.frames_dir:
            return None
        try:
            meta = json.loads((self.frames_dir / "spectator.json").read_text())
        except (OSError, ValueError):
            return None
        meta["label"] = "CHECKPOINT PLAYBACK - not the live training rollout"
        if _num(meta.get("checkpoint_mtime")) is not None:
            meta["checkpoint_age_seconds"] = max(0.0, self.clock() - meta["checkpoint_mtime"])
        if _num(meta.get("frame_time")) is not None:
            meta["frame_age_seconds"] = max(0.0, self.clock() - meta["frame_time"])
        return meta

    def train_meta(self):
        """Live frame written by training worker 0 (only when the trainer runs with DOOM_FRAME_TAP_DIR)."""
        if not self.frames_dir:
            return None
        try:
            meta = json.loads((self.frames_dir / "train.json").read_text())
        except (OSError, ValueError):
            return None
        if _num(meta.get("frame_time")) is not None:
            meta["frame_age_seconds"] = max(0.0, self.clock() - meta["frame_time"])
            meta["stale"] = meta["frame_age_seconds"] > TRAIN_FRAME_STALE_S
        meta["available"] = True
        return meta

    def train_png(self):
        if not self.frames_dir:
            return None
        try:
            return (self.frames_dir / "train.png").read_bytes()
        except OSError:
            return None

    def spectator_png(self):
        if not self.frames_dir:
            return None
        try:
            return (self.frames_dir / "spectator.png").read_bytes()
        except OSError:
            return None


def _nvidia_smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4, check=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        u, mu, mt = (float(x) for x in line.split(","))
        gpus.append({"util_percent": u, "mem_used_mib": mu, "mem_total_mib": mt})
    return gpus or None


def make_handler(mon: Monitor):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DoomMonitor"

        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, allow_nan=False).encode(), "application/json")

        def _host_ok(self):
            port = self.server.server_address[1]
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "forbidden host"})
            url = urlparse(self.path)
            path = url.path
            try:
                if path in STATIC_FILES:
                    name, ctype = STATIC_FILES[path]
                    return self._send(200, (STATIC / name).read_bytes(), ctype)
                if path == "/api/status":
                    return self._json(200, mon.status())
                if path == "/api/metrics":
                    q = parse_qs(url.query)
                    try:
                        after = max(0, int(q.get("after", ["0"])[0]))
                    except ValueError:
                        return self._json(400, {"error": "after must be an integer"})
                    return self._json(200, mon.metrics(after))
                if path == "/api/evals":
                    return self._json(200, {"evals": mon.evals()})
                if path == "/api/episodes":
                    return self._json(200, mon.episode_stats())
                if path == "/api/frame/train/meta":
                    meta = mon.train_meta()
                    return self._json(200, meta) if meta else self._json(503, TRAIN_FRAME_UNAVAILABLE)
                if path == "/api/frame/train":
                    png = mon.train_png()
                    return self._send(200, png, "image/png") if png else self._json(503, TRAIN_FRAME_UNAVAILABLE)
                if path == "/api/frame/spectator/meta":
                    meta = mon.spectator_meta()
                    return self._json(200, meta) if meta else self._json(503, {"available": False, "reason": "no spectator frame yet"})
                if path == "/api/frame/spectator":
                    png = mon.spectator_png()
                    return self._send(200, png, "image/png") if png else self._json(503, {"available": False, "reason": "no spectator frame yet"})
            except Exception as e:  # never crash a handler thread
                return self._json(500, {"error": type(e).__name__})
            return self._json(404, {"error": "not found"})

        def _deny(self):
            self.send_response(405)
            self.send_header("Allow", "GET")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _deny
        do_HEAD = _deny

    return Handler


def make_server(mon: Monitor, host="127.0.0.1", port=8765):
    if host != "127.0.0.1":
        raise ValueError("monitor binds to 127.0.0.1 only")
    srv = ThreadingHTTPServer((host, port), make_handler(mon))
    srv.daemon_threads = True
    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--price-per-hour", type=float, default=1.518, help="estimated $/h (GPU + disk)")
    ap.add_argument("--horizon", type=int, default=19_996_672, help="absolute final global step")
    ap.add_argument("--start-step", type=int, default=0, help="global step the run started from (warm start)")
    ap.add_argument("--evals-dir", default=None, help="directory of doomrl.evaluate JSON results")
    ap.add_argument("--started-at", type=float, default=None, help="unix time of run start (default: config.json mtime)")
    a = ap.parse_args()
    mon = Monitor(a.run_dir, a.frames_dir, a.price_per_hour, horizon=a.horizon, started_at=a.started_at)
    mon.start_step = a.start_step
    mon.evals_dir = Path(a.evals_dir) if a.evals_dir else None
    srv = make_server(mon, a.host, a.port)

    def loop():
        while True:
            try:
                mon.poll()
            except Exception:
                pass
            time.sleep(1.0)
    mon.poll()
    threading.Thread(target=loop, daemon=True).start()
    print(f"monitor on http://127.0.0.1:{a.port}/ (run {a.run_dir})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
