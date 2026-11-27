"""Selection uses immutable export provenance; comparisons never pool training runs."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]
from doom_model import AgentNet, ModelConfig
from doom_model.artifacts import export_policy, file_hash
from doomrl.evaluate import METRICS, compare_paired, summarize_episodes

spec = importlib.util.spec_from_file_location(
    "decide_at_step", ROOT / "training/tools/decide_at_step.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


@pytest.fixture
def export():
    cfg = ModelConfig(hidden=32, embed=32)
    net = AgentNet(cfg)

    def make(path, steps, **provenance):
        return export_policy(net, cfg, path, {"steps": steps, **provenance})

    return make


def episodes(delta=0.):
    return [
        dict(map=m, seed=s, ticks=t, exit=0., keys=delta, kills=1., dead=0.,
             timeout=1., health=100., native_return=delta, steps=20.,
             ms_mean=1., ms_max=2., cells=5., doors_opened=0., locked_opened=0.,
                 damage=10., distance=100., revisit_ratio=.5, proxy_score=delta,
                 fire_rate=.4, blind_fire=.5, ammo_spent=20., ended_dry=0., on_target_fire=.3)
        for m in ("a", "b") for s in (10000, 10001) for t in (2100, 4200)
    ]


def test_selection_uses_verified_provenance_not_filename(tmp_path, export):
    run = tmp_path / "run"
    export(run / "candidate_900", 10)
    chosen = export(run / "candidate_1", 30, training_sha256="training-identity")
    export(run / "candidate_20", 40)
    (run / "model_25.pt").write_bytes(b"legacy checkpoint")
    selected = tool.select_candidate(run, 20)
    assert selected["path"] == str(chosen.resolve())
    assert selected["manifest_sha256"] == file_hash(chosen / "manifest.json")
    assert selected["provenance"]["steps"] == 30
    assert selected["provenance"]["training_sha256"] == "training-identity"
    assert selected["run"] == str(run.resolve())


def test_legacy_checkpoints_are_not_candidates(tmp_path):
    (tmp_path / "model_200.pt").write_bytes(b"legacy")
    with pytest.raises(FileNotFoundError, match="candidate"):
        tool.select_candidate(tmp_path, 100)


@pytest.mark.parametrize("name", ["model.pt", "model_config.json", "manifest.json"])
def test_incomplete_or_corrupt_export_rejected(tmp_path, export, name):
    candidate = export(tmp_path / "candidate_20", 20)
    (candidate / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        tool.select_candidate(tmp_path, 10)


@pytest.mark.parametrize("steps", [True, -1, 1.5, "20", None])
def test_invalid_step_provenance_rejected(tmp_path, export, steps):
    export(tmp_path / "candidate_20", steps)
    with pytest.raises(ValueError, match="steps"):
        tool.select_candidate(tmp_path, 10)


def test_checksum_mismatch_rejected(tmp_path, export):
    candidate = export(tmp_path / "candidate_20", 20)
    with (candidate / "model.pt").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        tool.select_candidate(tmp_path, 10)


def test_ambiguous_step_requires_explicit_candidate(tmp_path, export):
    export(tmp_path / "candidate_a", 20)
    export(tmp_path / "candidate_b", 20)
    with pytest.raises(ValueError, match="ambiguous"):
        tool.select_candidate(tmp_path, 10)


@pytest.fixture
def comparison(tmp_path, export, monkeypatch):
    candidates = [
        export(tmp_path / parent / "same_s0" / "candidate_20", 20, seed=seed)
        for seed, parent in enumerate(("first", "second", "third"))
    ]
    calls = []

    def evaluate(path, suite, seeds, ticks, workers, timeout):
        path = Path(path)
        calls.append(path)
        manifest = json.loads((path / "manifest.json").read_text())
        eps = episodes(candidates.index(path))
        return dict(weights=str(path), suite=suite, episodes=len(eps),
                    manifest_sha256=file_hash(path / "manifest.json"),
                    provenance=manifest["provenance"], episodes_raw=eps,
                    **summarize_episodes(eps))

    monkeypatch.setattr(tool, "evaluate", evaluate)
    out = tmp_path / "results"
    args = ["--runs", *[str(p.parent) for p in candidates],
            "--step", "10", "--out", str(out)]
    return candidates, calls, out, args


def test_baseline_first_separate_runs_and_paired_native_metrics(comparison, monkeypatch):
    candidates, calls, out, args = comparison
    before = {p: {f.name: f.read_bytes() for f in p.iterdir()} for p in candidates}
    pairs = []

    def paired(a, b):
        pairs.append((a, b))
        return compare_paired(a, b)

    monkeypatch.setattr(tool, "compare_paired", paired)
    tool.main(args)
    report = json.loads((out / "comparison.json").read_text())
    assert calls == candidates
    assert len(report["evaluations"]) == 3
    assert len(report["comparisons"]) == len(pairs) == 2
    assert report["baseline"] == report["evaluations"][0]["candidate"]
    for i, comparison_result in enumerate(report["comparisons"], 1):
        assert pairs[i - 1] == (episodes(), episodes(i))
        assert comparison_result["paired"] == compare_paired(episodes(), episodes(i))
        assert set(comparison_result["paired"]["2100"]) == set(METRICS)
        assert "native_return" in comparison_result["paired"]["2100"]
        assert comparison_result["paired"]["2100"]["native_return"]["independent_maps"] == 2
        assert "winner" not in comparison_result
    assert {p: {f.name: f.read_bytes() for f in p.iterdir()} for p in candidates} == before


def test_explicit_candidates_can_resolve_selection(comparison):
    candidates, calls, out, _ = comparison
    tool.main(["--candidates", *map(str, candidates), "--out", str(out)])
    assert calls == candidates


def test_missing_candidate_fails_without_waiting(tmp_path, monkeypatch):
    runs = [tmp_path / "a", tmp_path / "b"]
    for run in runs:
        run.mkdir()
    monkeypatch.setattr(tool, "evaluate", lambda *a, **k: pytest.fail("evaluated missing export"))
    with pytest.raises(FileNotFoundError, match="candidate"):
        tool.main(["--runs", *map(str, runs), "--step", "10", "--out", str(tmp_path / "out")])


def test_unmatched_jobs_are_rejected(comparison, monkeypatch):
    _, _, out, args = comparison
    original = tool.evaluate

    def mismatched(*a, **kw):
        result = original(*a, **kw)
        if result["provenance"]["seed"] == 1:
            result["episodes_raw"] = result["episodes_raw"][:-1]
        return result

    monkeypatch.setattr(tool, "evaluate", mismatched)
    with pytest.raises(ValueError, match="identical"):
        tool.main(args)
    assert not (out / "comparison.json").exists()


@pytest.mark.parametrize("change", ["manifest", "result_identity", "result_provenance", "weights"])
def test_identity_pinned_across_evaluation(comparison, monkeypatch, change):
    _, _, out, args = comparison
    original = tool.evaluate

    def changed(*a, **kw):
        result = original(*a, **kw)
        path = Path(result["weights"])
        if change == "manifest":
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["provenance"]["steps"] += 1
            (path / "manifest.json").write_text(json.dumps(manifest))
        elif change == "result_identity":
            result["manifest_sha256"] = "different"
        elif change == "result_provenance":
            result["provenance"]["steps"] += 1
        else:
            (path / "model.pt").write_bytes(b"corruption")
        return result

    monkeypatch.setattr(tool, "evaluate", changed)
    with pytest.raises(ValueError, match="identity|changed|hash mismatch"):
        tool.main(args)
    assert not (out / "comparison.json").exists()


def test_duplicate_runs_and_outputs_inside_exports_rejected(comparison):
    candidates, calls, _, _ = comparison
    with pytest.raises(ValueError, match="duplicate"):
        tool.main(["--candidates", str(candidates[0]), str(candidates[0]), "--out", "/unused"])
    with pytest.raises(ValueError, match="outside"):
        tool.main(["--candidates", *map(str, candidates), "--out", str(candidates[0] / "results")])
    assert calls == []


@pytest.mark.parametrize("options", [
    ["--seeds", "0"], ["--workers", "0"], ["--ticks", "0"],
    ["--ticks", "2100", "2100"], ["--timeout", "0"],
    ["--timeout", "inf"], ["--timeout", "nan"], ["--step", "-1"],
])
def test_invalid_evaluation_options_fail_before_evaluation(comparison, options):
    _, calls, _, args = comparison
    with pytest.raises(SystemExit):
        tool.main([*args, *options])
    assert calls == []


def test_explicit_candidate_minimum_step_is_enforced(comparison):
    candidates, calls, out, _ = comparison
    with pytest.raises(ValueError, match="below"):
        tool.main(["--candidates", *map(str, candidates), "--step", "21", "--out", str(out)])
    assert calls == []


def test_run_selection_requires_step(comparison):
    candidates, calls, out, _ = comparison
    with pytest.raises(SystemExit):
        tool.main(["--runs", *[str(p.parent) for p in candidates], "--out", str(out)])
    assert calls == []


def test_evaluation_options_forwarded_and_recorded(comparison, monkeypatch):
    _, _, _, args = comparison
    original = tool.evaluate
    options = []

    def evaluate(path, suite, seeds, ticks, workers, timeout):
        options.append((suite, seeds, ticks, workers, timeout))
        return original(path, suite, seeds, ticks, workers, timeout)

    monkeypatch.setattr(tool, "evaluate", evaluate)
    report = tool.main([*args, "--seeds", "2", "--workers", "1", "--timeout", "30"])
    assert options == [("dev", 2, [2100, 4200], 1, 30.)] * 3
    assert report["evaluation"]["timeout_seconds"] == 30.


def test_unselected_candidate_mutation_is_detected_before_its_evaluation(comparison, monkeypatch):
    candidates, calls, out, args = comparison
    original = tool.evaluate

    def evaluate(*a, **kw):
        result = original(*a, **kw)
        manifest = candidates[1] / "manifest.json"
        data = json.loads(manifest.read_text())
        data["provenance"]["steps"] += 1
        manifest.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(tool, "evaluate", evaluate)
    with pytest.raises(ValueError, match="identity changed"):
        tool.main(args)
    assert calls == candidates[:1]
    assert not (out / "comparison.json").exists()


def test_prior_candidate_mutation_prevents_report_publication(comparison, monkeypatch):
    candidates, _, out, args = comparison
    original = tool.evaluate

    def evaluate(*a, **kw):
        result = original(*a, **kw)
        if result["provenance"]["seed"] == 2:
            (candidates[0] / "model.pt").write_bytes(b"changed after evaluation")
        return result

    monkeypatch.setattr(tool, "evaluate", evaluate)
    with pytest.raises(ValueError, match="hash mismatch"):
        tool.main(args)
    assert not (out / "comparison.json").exists()


def test_existing_output_is_never_overwritten(comparison):
    _, calls, out, args = comparison
    out.mkdir()
    previous = out / "comparison.json"
    previous.write_text("previous comparison")
    with pytest.raises(FileExistsError):
        tool.main(args)
    assert calls == []
    assert previous.read_text() == "previous comparison"
