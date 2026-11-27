"""Aggregate run logs and evals into one small JSON for the report figures.

usage: agg_report.py OUT.json DEV_MAPS_JSON --runs name=dir ... --evals glob ...
"""
import glob, json, math, os, re, sys
from collections import defaultdict

out, dev_maps = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
runs, evals, mode = {}, [], None
for a in args:
    if a in ("--runs", "--evals"):
        mode = a; continue
    if mode == "--runs":
        k, v = a.split("=", 1); runs[k] = v
    else:
        evals += glob.glob(a, recursive=True)

tier = {}
try:
    for m in json.load(open(dev_maps)):
        tier[f"{m['wad']}:{m['map']}"] = m.get("tier")
except Exception:
    pass

BIN = 1_000_000
res = {"runs": {}, "evals": []}
for name, d in runs.items():
    bins = defaultdict(lambda: defaultdict(list))
    p = os.path.join(d, "episodes.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                e = json.loads(line)
            except Exception:
                continue
            g = e.get("global_step")
            if g is None:
                continue
            b = int(g // BIN)
            kind = "gen" if e.get("map", "").startswith("train_") else "other"
            r = bins[(kind, b)]
            r["exit"].append(float(bool(e.get("exit"))))
            r["dead"].append(float(bool(e.get("dead"))))
            r["native"].append(float(e.get("native_return", 0)))
            r["kills"].append(float(e.get("kills", 0)))
            r["keys"].append(float(e.get("keys_picked", 0) or 0))
            r["locked"].append(float(e.get("locked_opened", 0) or 0))
            a = float(e.get("ammo_spent", 0) or 0); bl = float(e.get("blind_ammo", 0) or 0)
            r["ammo"].append(a); r["blind"].append(bl)
    series = []
    for (kind, b), r in sorted(bins.items(), key=lambda x: (x[0][0], x[0][1])):
        n = len(r["exit"])
        if n < 20:
            continue
        s = {"kind": kind, "step": (b + 0.5) * BIN, "n": n}
        for k, v in r.items():
            s[k] = sum(v) / len(v)
        sa = sum(r["ammo"]); s["blind_frac"] = sum(r["blind"]) / sa if sa else None
        series.append(s)
    sps = []
    p = os.path.join(d, "updates.jsonl")
    if os.path.exists(p):
        vals = []
        for line in open(p):
            try:
                u = json.loads(line)
            except Exception:
                continue
            if "sps" in u:
                vals.append(u["sps"])
        if vals:
            vals.sort(); sps = {"median": vals[len(vals) // 2], "n": len(vals)}
    res["runs"][name] = {"series": series, "sps": sps}

for f in sorted(set(evals)):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    s = r.get("summary", {})
    m = re.search(r"_(\d{6,})_", os.path.basename(f))
    rec = {"file": os.path.relpath(f, os.path.dirname(os.path.dirname(f))), "name": os.path.basename(f),
           "suite": r.get("suite"), "episodes": r.get("episodes"), "step": int(m.group(1)) if m else None,
           "mean": {k: v.get("mean") for k, v in s.items() if isinstance(v, dict) and "mean" in v}}
    if r.get("suite") in ("gen_dev", "gen_test"):
        bt = defaultdict(list)
        for e in r.get("episodes_raw", []):
            t = tier.get(e.get("map"), "template" if "two_rooms" in e.get("map", "") else None)
            bt[str(t)].append(float(e.get("exit", 0) > 0))
        rec["exit_by_tier"] = {k: sum(v) / len(v) for k, v in bt.items()}
    res["evals"].append(rec)

json.dump(res, open(out, "w"))
print(out, len(res["evals"]), "evals;", {k: len(v["series"]) for k, v in res["runs"].items()})
