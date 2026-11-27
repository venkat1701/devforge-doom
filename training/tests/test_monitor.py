import http.client
import json
import os
import struct
import sys
import threading
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]
from monitor.server import Monitor, make_server
from monitor import spectator


def write_jsonl(path, rows, mode="w"):
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def upd(i, **kw):
    d = {"global_step": 4096 * i, "sps": 100.0 + i, "lr": 2.5e-4, "entropy": 4.0, "approx_kl": 0.01, "clip_fraction": 0.1,
         "vloss": 0.5, "pg": -0.01, "grad_norm": 0.2, "explained_variance": 0.1, "hidden_std": 0.06}
    d.update(kw)
    return d


@pytest.fixture
def run(tmp_path):
    r = tmp_path / "run"
    r.mkdir()
    (r / "config.json").write_text("{}")
    os.utime(r / "config.json", (1000, 1000))
    write_jsonl(r / "updates.jsonl", [upd(i) for i in range(1, 11)])
    write_jsonl(r / "episodes.jsonl", [
        {"ticks": 2100 if i % 2 else 4200, "return": float(i), "kills": i % 3, "keys": 0.0, "exit": i % 4 == 0,
         "dead": i % 2 == 0, "timeout": False, "global_step": i * 100} for i in range(1, 201)])
    return r


@pytest.fixture
def served(run, tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    mon = Monitor(run, frames, price_per_hour=1.518, horizon=1_000_000, clock=lambda: 1000 + 3600.0)
    mon.poll()
    srv = make_server(mon, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield mon, srv, srv.server_address[1], frames
    srv.shutdown()
    srv.server_close()


def req(port, path, method="GET", host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.putrequest(method, path, skip_host=True)
    c.putheader("Host", host or f"127.0.0.1:{port}")
    c.putheader("Content-Length", "0")
    c.endheaders()
    r = c.getresponse()
    return r.status, dict(r.getheaders()), r.read()


def test_host_validation(served):
    _, _, port, _ = served
    assert req(port, "/api/status")[0] == 200
    assert req(port, "/api/status", host=f"localhost:{port}")[0] == 200
    for bad in ("evil.example.com", f"evil.example.com:{port}", "127.0.0.1", f"127.0.0.1:{port + 1}", "0.0.0.0:%d" % port):
        assert req(port, "/api/status", host=bad)[0] == 403
        assert req(port, "/", host=bad)[0] == 403


def test_no_cors_and_static(served):
    _, _, port, _ = served
    for p in ("/", "/app.js", "/style.css", "/api/status"):
        s, h, _ = req(port, p)
        assert s == 200 and not any(k.lower().startswith("access-control") for k in h)


def test_refuses_non_loopback_bind(run):
    mon = Monitor(run)
    for host in ("0.0.0.0", "", "::", "192.168.1.5", "localhost"):
        with pytest.raises(ValueError):
            make_server(mon, host, 0)


def test_no_state_changing_routes(served):
    _, _, port, _ = served
    for m in ("POST", "PUT", "DELETE", "PATCH"):
        for p in ("/api/status", "/api/stop", "/"):
            assert req(port, p, m)[0] == 405
    assert req(port, "/api/stop")[0] == 404


def test_metrics_sequence_gap_and_nulls(run):
    rows = [upd(i) for i in range(1, 6)]
    rows[2]["entropy"] = None
    del rows[3]["approx_kl"]
    write_jsonl(run / "updates.jsonl", rows)
    mon = Monitor(run, max_updates=3)
    mon.poll()
    m = mon.metrics(0)
    assert m["gap"] is True and [u["seq"] for u in m["updates"]] == [3, 4, 5] and m["last_seq"] == 5
    assert m["updates"][0]["entropy"] is None and m["updates"][1]["approx_kl"] is None  # null stays null, not 0
    assert mon.metrics(2)["gap"] is False
    m = mon.metrics(4)
    assert m["gap"] is False and [u["seq"] for u in m["updates"]] == [5]
    assert mon.metrics(5)["updates"] == []
    # incremental tail: partial line is not consumed, then completed
    with open(run / "updates.jsonl", "a") as f:
        f.write(json.dumps(upd(6))[:20])
    mon.poll()
    assert mon.metrics(0)["last_seq"] == 5
    with open(run / "updates.jsonl", "a") as f:
        f.write(json.dumps(upd(6))[20:] + "\n")
    mon.poll()
    assert mon.metrics(0)["last_seq"] == 6


def test_status_math(run):
    (run / "candidate_16384").mkdir()
    (run / "candidate_16384" / "manifest.json").write_text("{}")
    os.utime(run / "candidate_16384" / "manifest.json", (4000, 4000))
    os.utime(run / "updates.jsonl", (4500, 4500))
    now = 1000 + 3600.0 * 2   # 2h after config.json
    mon = Monitor(run, price_per_hour=1.5, horizon=409_600 + 1_000_000, clock=lambda: now, gpu_probe=lambda: None, sps_window=4)
    mon.poll()
    s = mon.status()
    assert s["steps"] == 40960 and s["updates"] == 10
    assert s["sps_recent"] == pytest.approx((107 + 108 + 109 + 110) / 4)
    assert s["sps_wall"] == pytest.approx(40960 / 7200)
    remaining = 409_600 + 1_000_000 - 40960
    assert s["eta_seconds"] == pytest.approx(remaining / 108.5)
    assert s["elapsed_seconds"] == 7200 and s["spend_estimated"] == pytest.approx(3.0)
    assert s["spend_projected_total_estimated"] == pytest.approx((7200 + remaining / 108.5) / 3600 * 1.5)
    assert s["last_checkpoint_step"] == 16384 and s["last_checkpoint_age_seconds"] == 7200 + 1000 - 4000
    assert s["heartbeat_age_seconds"] == 8200 - 4500 and s["stale"] is True   # >120 s
    assert s["gpu"] is None and s["percent_done"] == pytest.approx(100 * 40960 / 1_409_600)
    os.utime(run / "updates.jsonl", (now - 5, now - 5))
    assert mon.status()["stale"] is False


def test_status_with_no_data(tmp_path):
    r = tmp_path / "empty"
    r.mkdir()
    s = Monitor(r, clock=lambda: 5.0, gpu_probe=lambda: 1 / 0).status()
    assert s["steps"] == 0 and s["eta_seconds"] is None and s["stale"] is True and s["gpu"] is None
    assert s["last_checkpoint_step"] is None


def test_episode_stats_and_cap_breakdown(run):
    mon = Monitor(run)
    mon.poll()
    e = mon.episode_stats()
    assert e["last_100"]["n"] == 100 and e["last_500"]["n"] == 200
    assert e["last_100"]["return"] == pytest.approx(np.mean(range(101, 201)))
    assert set(e["by_cap"]) == {"2100", "4200"} and e["by_cap"]["2100"]["n"] == 100
    assert e["last_100"]["exit"] == pytest.approx(0.25)


def test_train_frame_503(served):
    _, _, port, _ = served
    s, _, body = req(port, "/api/frame/train")
    assert s == 503
    assert json.loads(body)["available"] is False and "DOOM_FRAME_TAP_DIR" in json.loads(body)["reason"]
    assert req(port, "/api/frame/train/meta")[0] == 503


def test_train_frame_served_when_tapped(served):
    mon, _, port, frames = served
    import struct as _s, zlib as _z
    png = b"\x89PNG\r\n\x1a\n" + b"fake"
    (frames / "train.png").write_bytes(png)
    (frames / "train.json").write_text(json.dumps({"worker": 0, "map": "freedoom2.wad:MAP01", "episode_step": 3,
                                                   "episode_cap_tics": 2100, "global_step": 4096,
                                                   "frame_time": mon.clock() - 1}))
    s, _, body = req(port, "/api/frame/train")
    assert s == 200 and body == png
    s, _, body = req(port, "/api/frame/train/meta")
    meta = json.loads(body)
    assert s == 200 and meta["available"] and not meta["stale"] and meta["global_step"] == 4096


def test_spectator_frame_endpoints(served):
    mon, _, port, frames = served
    assert req(port, "/api/frame/spectator")[0] == 503 and req(port, "/api/frame/spectator/meta")[0] == 503
    png = spectator.encode_png(np.zeros((4, 5, 3), np.uint8))
    (frames / "spectator.png").write_bytes(png)
    (frames / "spectator.json").write_text(json.dumps({"checkpoint_steps": 8, "checkpoint_mtime": 1000 + 3000, "frame_time": 4599}))
    s, h, body = req(port, "/api/frame/spectator")
    assert s == 200 and body == png and h["Content-Type"] == "image/png"
    meta = json.loads(req(port, "/api/frame/spectator/meta")[2])
    assert meta["checkpoint_age_seconds"] == 600 and "not the live" in meta["label"]


def test_png_encoder_valid():
    img = np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    png = spectator.encode_png(img)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks, idat = 8, [], b""
    while pos < len(png):
        n, tag = struct.unpack(">I4s", png[pos:pos + 8])
        data = png[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", png[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(tag + data)
        chunks.append(tag)
        idat += data if tag == b"IDAT" else b""
        if tag == b"IHDR":
            assert struct.unpack(">IIBBBBB", data) == (160, 120, 8, 2, 0, 0, 0)
        pos += 12 + n
    assert chunks == [b"IHDR", b"IDAT", b"IEND"]
    raw = zlib.decompress(idat)
    rows = np.frombuffer(raw, np.uint8).reshape(120, 1 + 480)
    assert (rows[:, 0] == 0).all() and np.array_equal(rows[:, 1:].reshape(120, 160, 3), img)


def _export(run, steps, tag="ok"):
    from doom_model import AgentNet, ModelConfig
    from doom_model.artifacts import export_policy
    cfg = ModelConfig(hidden=32, embed=32)
    return export_policy(AgentNet(cfg), cfg, run / f"candidate_{steps}", {"steps": steps, "purpose": "test"})


def test_spectator_selection_skips_corrupt_and_labels_old(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    assert spectator.select_candidate(run) is None
    _export(run, 4096)
    c1 = spectator.select_candidate(run)
    assert c1["steps"] == 4096 and c1["old"] is False and len(c1["manifest_sha256"]) == 64
    bad = _export(run, 8192)
    (bad / "manifest.json").write_text("{not json")
    c2 = spectator.select_candidate(run, c1)
    assert c2["steps"] == 4096 and c2["old"] is True
    # tampered weights (hash mismatch) also rejected
    _export(run, 12288)
    with (run / "candidate_12288" / "model.pt").open("ab") as f:
        f.write(b"x")
    assert spectator.select_candidate(run, c1)["steps"] == 4096
    # a later good one wins and is not old
    _export(run, 16384)
    c3 = spectator.select_candidate(run, c2)
    assert c3["steps"] == 16384 and c3["old"] is False
    # nothing verifies: keep previous, labelled old
    only = tmp_path / "run2"
    only.mkdir()
    (only / "candidate_4").mkdir()
    assert spectator.select_candidate(only, c1)["old"] is True and spectator.select_candidate(only) is None


def test_evals_endpoint_reads_evaluate_output(run, tmp_path):
    from monitor.server import Monitor
    d = tmp_path / "evals"; d.mkdir()
    (d / "native_dev.json").write_text(json.dumps({"weights": "w", "suite": "dev", "episodes": 32,
        "summary": {"exit": {"mean": 0.0}, "cells": {"mean": 10.8}, "proxy_score": {"mean": 0.9}}}))
    (d / "broken.json").write_text("{not json")
    mon = Monitor(run)
    mon.evals_dir = d
    rows = mon.evals()
    assert [r["name"] for r in rows] == ["native_dev"]
    assert rows[0]["episodes"] == 32 and rows[0]["cells"] == 10.8 and rows[0]["keys"] is None


def test_start_step_progress_and_freedoom_series(run):
    from monitor.server import Monitor
    mon = Monitor(run, horizon=10**9)
    mon.start_step = 0
    st = mon.status()
    assert st["start_step"] == 0
    e = mon.episode_stats()
    assert all("freedoom" in r for r in e["series"]) and "freedoom_last_100" in e
