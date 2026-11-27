"""Compare immutable policy exports on dev jobs, with the first input as baseline.

Select the earliest provenance step >= --step from each --runs directory, or
provide exact --candidates directories. Missing exports fail immediately.
Training runs remain separate; paired intervals resample maps, not episodes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]

from doom_model.artifacts import file_hash, verify_artifact  # noqa: E402
from doomrl.evaluate import compare_paired, evaluate  # noqa: E402


def candidate_identity(path: Path):
    path = Path(path).resolve(strict=True)
    identity = file_hash(path / "manifest.json")
    manifest = verify_artifact(path)
    if file_hash(path / "manifest.json") != identity:
        raise ValueError(f"candidate identity changed during verification: {path}")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"candidate requires provenance with steps: {path}")
    steps = provenance.get("steps")
    if type(steps) is not int or steps < 0:
        raise ValueError(f"candidate provenance steps must be a nonnegative integer: {path}")
    return {"path": str(path), "manifest_sha256": identity, "provenance": provenance}


def select_candidate(run: Path, step: int):
    """Use manifest steps, never filename-derived steps or inferred run families."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    run = Path(run).resolve()
    candidates = [candidate_identity(path) for path in run.glob("candidate_*")]
    eligible = [c for c in candidates if c["provenance"]["steps"] >= step]
    if not eligible:
        raise FileNotFoundError(f"no immutable candidate at or past {step} steps in {run}")
    first_step = min(c["provenance"]["steps"] for c in eligible)
    selected = [c for c in eligible if c["provenance"]["steps"] == first_step]
    if len(selected) != 1:
        raise ValueError(f"ambiguous candidates at step {first_step} in {run}; use --candidates")
    return dict(selected[0], run=str(run))


def verify_selected(candidate):
    current = candidate_identity(Path(candidate["path"]))
    if any(current[key] != candidate[key] for key in current):
        raise ValueError(f"candidate identity changed after selection: {candidate['path']}")


def comparison_text(report):
    lines = [
        "Dev comparison; first input is the baseline.",
        "Differences are candidate minus baseline, stratified by tick cap.",
        "95% paired bootstrap intervals resample maps; training runs are not pooled.",
        "No automatic winner or aggregate competition score is inferred.",
    ]
    for item in report["evaluations"]:
        candidate = item["candidate"]
        lines.append(f"\nCandidate: {candidate['path']}")
        lines.append(f"Manifest SHA256: {candidate['manifest_sha256']}")
        lines.append(f"Provenance: {json.dumps(candidate['provenance'], sort_keys=True)}")
        lines.append(f"Native evaluation by tick cap: {json.dumps(item['result']['by_ticks'], sort_keys=True)}")
    for pair in report["comparisons"]:
        lines.append(f"\n{pair['candidate']['path']} minus {pair['baseline']['path']}")
        for ticks, metrics in pair["paired"].items():
            for metric, values in metrics.items():
                lines.append(
                    f"ticks={ticks} {metric}: difference={values['mean_difference']:.6g}, "
                    f"ci95={values['ci95']}, independent_maps={values['independent_maps']}"
                )
    lines.extend(["", *report["limitations"]])
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs", nargs="+", help="explicit run directories, baseline first")
    source.add_argument("--candidates", nargs="+", help="exact immutable export directories, baseline first")
    ap.add_argument("--step", type=int, help="minimum provenance step; required with --runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ticks", type=int, nargs="+", default=[2100, 4200])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=1800., help="finite evaluation timeout per candidate, seconds")
    a = ap.parse_args(argv)
    if a.runs and a.step is None:
        ap.error("--runs requires --step")
    if a.step is not None and a.step < 0:
        ap.error("--step must be nonnegative")
    if (a.seeds <= 0 or a.workers <= 0 or any(t <= 0 for t in a.ticks)
            or len(set(a.ticks)) != len(a.ticks) or not math.isfinite(a.timeout) or a.timeout <= 0):
        ap.error("positive seeds, workers, unique tick caps and finite positive timeout required")
    paths = [Path(p).resolve() for p in (a.runs or a.candidates)]
    if len(paths) < 2:
        ap.error("provide a baseline and at least one comparison")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate input paths")
    candidates = ([select_candidate(p, a.step) for p in paths] if a.runs
                  else [dict(candidate_identity(p), run=str(p.parent)) for p in paths])
    if len({c["path"] for c in candidates}) != len(candidates):
        raise ValueError("duplicate selected candidate paths")
    if a.step is not None and any(c["provenance"]["steps"] < a.step for c in candidates):
        raise ValueError("explicit candidate is below requested step")
    out = Path(a.out).resolve()
    if any(out.is_relative_to(Path(c["path"])) for c in candidates):
        raise ValueError("comparison output must be outside immutable candidates")
    # Exclusive output directory prevents accidentally replacing an earlier comparison.
    out.mkdir(parents=True, exist_ok=False)
    evaluations = []
    for candidate in candidates:
        verify_selected(candidate)
        result = evaluate(Path(candidate["path"]), "dev", a.seeds, a.ticks, a.workers, timeout=a.timeout)
        verify_selected(candidate)
        if (result["manifest_sha256"] != candidate["manifest_sha256"]
                or result["provenance"] != candidate["provenance"]
                or Path(result["weights"]).resolve() != Path(candidate["path"])):
            raise ValueError("evaluation identity/provenance differs from selected candidate")
        evaluations.append({"candidate": candidate, "result": result})
    baseline = evaluations[0]
    comparisons = [
        {"baseline": baseline["candidate"], "candidate": item["candidate"],
         "paired": compare_paired(baseline["result"]["episodes_raw"], item["result"]["episodes_raw"])}
        for item in evaluations[1:]
    ]
    report = {
        "baseline": baseline["candidate"], "evaluations": evaluations, "comparisons": comparisons,
        "selection": {"minimum_step": a.step, "source": "runs" if a.runs else "candidates"},
        "evaluation": {"suite": "dev", "seeds_per_map": a.seeds, "ticks": a.ticks,
                       "workers": a.workers, "timeout_seconds": a.timeout},
        "limitations": [
            "Intervals describe map uncertainty for these fixed policies, not training-seed variability.",
            "Repeated dev selection and multiple comparisons are not corrected by these intervals.",
            "No winner is selected from marginal CI overlap or paired differences.",
        ],
    }
    # Recheck every pinned artifact before publishing any report.
    for candidate in candidates:
        verify_selected(candidate)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    text = comparison_text(report)
    (out / "comparison.json").write_text(serialized + "\n")
    (out / "DECISION.txt").write_text(text)
    print(text, end="", flush=True)
    return report


if __name__ == "__main__":
    main()
