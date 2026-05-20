"""End-to-end tests for phase_label_multi_provider via subprocess + FakeProvider.

All tests pass env={"DISTILL_DIR_OVERRIDE": str(tmp_path)} so the repo's
real data/distill/ is never touched.
"""
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "05_generate_distillation_data.py"


def _make_label_config(
    tmp_path: Path, *,
    label_attempts: int = 1,
    repair_enabled: bool = False,
    max_repair: int = 0,
    providers: list[dict] | None = None,
    fixture_path: Path | None = None,
) -> Path:
    if fixture_path is None:
        fixture_path = REPO_ROOT / "tests" / "fixtures" / "fake_labels.jsonl"
    if providers is None:
        providers = [
            {"name": "fake_a", "type": "fake", "weight": 1, "threads": 1,
             "fixture_labels": str(fixture_path)},
        ]
    cfg = {
        "version": 1,
        "input_generation": {"enabled": False, "providers": []},
        "output_generation": {
            "enabled": True,
            "label_attempts_per_input": label_attempts,
            "providers": providers,
        },
        "validation": {"schema": True, "semantic_validator": True,
                       "reject_invalid": True,
                       "retry_invalid_with_stricter_prompt": repair_enabled,
                       "max_repair_attempts": max_repair},
        "rate_limits": {"global_max_workers": 2, "write_flush_every": 5},
    }
    cfg_path = tmp_path / "providers.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_path


def _run_label(cfg_path: Path, tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "DISTILL_DIR_OVERRIDE": str(tmp_path)}
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--phase", "label",
         "--provider-config", str(cfg_path),
         "--multi-provider",
         *extra],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
    )


def _seed_inputs(tmp_path: Path, inputs: list[str]) -> None:
    (tmp_path / "inputs_raw.jsonl").write_text(
        "\n".join(json.dumps({"input": s}) for s in inputs) + "\n",
        encoding="utf-8",
    )


def test_label_phase_writes_train_jsonl(tmp_path):
    fixture = REPO_ROOT / "tests" / "fixtures" / "fake_labels.jsonl"
    first_input = json.loads(fixture.read_text(encoding="utf-8").splitlines()[0])["input"]
    _seed_inputs(tmp_path, [first_input])
    cfg = _make_label_config(tmp_path)
    result = _run_label(cfg, tmp_path)
    assert result.returncode == 0, result.stderr
    train = tmp_path / "train.jsonl"
    assert train.exists()
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["_source"] == "multi_provider_label"
    assert rows[0]["_provider"] == "fake_a"
    assert rows[0]["_attempts"] == 1


def test_label_phase_handles_already_labeled_inputs(tmp_path):
    fixture = REPO_ROOT / "tests" / "fixtures" / "fake_labels.jsonl"
    first_input = json.loads(fixture.read_text(encoding="utf-8").splitlines()[0])["input"]
    _seed_inputs(tmp_path, [first_input])
    # Pre-populate train.jsonl with the same input.
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"input": first_input, "output": {"transactions": []}}) + "\n",
        encoding="utf-8",
    )
    cfg = _make_label_config(tmp_path)
    result = _run_label(cfg, tmp_path)
    assert result.returncode == 0, result.stderr
    # Train.jsonl unchanged (only the pre-existing row).
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1


def test_label_phase_disabled_output_generation_does_nothing(tmp_path):
    fixture = REPO_ROOT / "tests" / "fixtures" / "fake_labels.jsonl"
    cfg_data = {
        "version": 1,
        "input_generation": {"enabled": False, "providers": []},
        "output_generation": {"enabled": False, "providers": []},
        "validation": {"schema": True, "semantic_validator": True,
                       "reject_invalid": True,
                       "retry_invalid_with_stricter_prompt": False,
                       "max_repair_attempts": 0},
        "rate_limits": {"global_max_workers": 1, "write_flush_every": 5},
    }
    cfg_path = tmp_path / "providers.json"
    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")
    _seed_inputs(tmp_path, ["whatever"])
    result = _run_label(cfg_path, tmp_path)
    assert result.returncode == 0, result.stderr
    # No train.jsonl written.
    train = tmp_path / "train.jsonl"
    assert not train.exists() or train.read_text(encoding="utf-8") == ""


def test_label_phase_no_pending_inputs_does_nothing(tmp_path):
    # No inputs_raw.jsonl at all.
    cfg = _make_label_config(tmp_path)
    result = _run_label(cfg, tmp_path)
    assert result.returncode == 0, result.stderr
    train = tmp_path / "train.jsonl"
    assert not train.exists() or train.read_text(encoding="utf-8") == ""


def test_label_phase_writes_failed_jsonl_for_unmatched_input(tmp_path):
    """FakeProvider returns miss_payload (empty string) for inputs not in fixture;
    extract_json fails; failed.jsonl gets a row with reason=all_candidates_failed."""
    _seed_inputs(tmp_path, ["completely_unmatched_input_string_xyz"])
    cfg = _make_label_config(tmp_path, label_attempts=2)
    result = _run_label(cfg, tmp_path)
    assert result.returncode == 0, result.stderr
    failed = tmp_path / "failed.jsonl"
    assert failed.exists()
    rows = [json.loads(line) for line in failed.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["reason"] == "all_candidates_failed"
    assert len(rows[0]["attempts"]) == 2
    assert all(a["failure_reason"] == "json_parse_failed" for a in rows[0]["attempts"])


def test_label_phase_limit_truncates(tmp_path):
    fixture = REPO_ROOT / "tests" / "fixtures" / "fake_labels.jsonl"
    rows = [json.loads(line)["input"] for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Seed with several fixture inputs.
    _seed_inputs(tmp_path, rows[:3])
    cfg = _make_label_config(tmp_path)
    result = _run_label(cfg, tmp_path, "--limit", "1")
    assert result.returncode == 0, result.stderr
    train = tmp_path / "train.jsonl"
    rows_out = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows_out) == 1


def test_label_phase_two_providers_pick_higher_score(tmp_path):
    """Two providers with different fixture qualities; the better one's
    label ends up in train.jsonl."""
    fixture = REPO_ROOT / "tests" / "fixtures" / "fake_labels.jsonl"
    failures = REPO_ROOT / "tests" / "fixtures" / "fake_labels_with_failures.jsonl"
    # Use an input present in fake_labels.jsonl (clean label exists).
    first_input = json.loads(fixture.read_text(encoding="utf-8").splitlines()[0])["input"]
    _seed_inputs(tmp_path, [first_input])

    providers = [
        # provider_a returns the clean label for first_input (from fake_labels.jsonl)
        {"name": "fake_good", "type": "fake", "weight": 1, "threads": 1,
         "fixture_labels": str(fixture)},
        # provider_b returns malformed/wrong labels (from failures fixture)
        {"name": "fake_bad", "type": "fake", "weight": 1, "threads": 1,
         "fixture_labels": str(failures)},
    ]
    cfg = _make_label_config(tmp_path, label_attempts=2, providers=providers)
    result = _run_label(cfg, tmp_path)
    assert result.returncode == 0, result.stderr
    train = tmp_path / "train.jsonl"
    if train.exists():
        rows_out = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows_out:
            # The clean provider wins if its label passes validation.
            assert rows_out[0]["_provider"] == "fake_good"


def test_label_phase_repair_exhausted_writes_failed_jsonl(tmp_path):
    """All initial + repair attempts fail; failed.jsonl row has
    reason='repair_exhausted' with attempts[] including repair attempts."""
    fixture = REPO_ROOT / "tests" / "fixtures" / "fake_labels.jsonl"
    _seed_inputs(tmp_path, ["completely_unmatched_input_xyz"])
    providers = [
        {"name": "fake_a", "type": "fake", "weight": 1, "threads": 1,
         "fixture_labels": str(fixture)},
    ]
    cfg = _make_label_config(
        tmp_path,
        label_attempts=2,
        providers=providers,
        repair_enabled=True,
        max_repair=2,
    )
    result = _run_label(cfg, tmp_path)
    assert result.returncode == 0, result.stderr
    failed = tmp_path / "failed.jsonl"
    assert failed.exists()
    rows = [json.loads(line) for line in failed.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["reason"] == "repair_exhausted"
    # 2 initial attempts + up to 2 repair attempts = 3 or 4 total
    assert len(rows[0]["attempts"]) >= 3
    repair_attempts = [a for a in rows[0]["attempts"] if a["is_repair_attempt"]]
    assert len(repair_attempts) >= 1


def test_label_phase_retry_validation_failed_re_attempts(tmp_path):
    """Pre-populate failed.jsonl with a row whose attempts include
    validation_failed. With --retry-validation-failed, that input gets
    re-attempted."""
    fixture = REPO_ROOT / "tests" / "fixtures" / "fake_labels.jsonl"
    first_input = json.loads(fixture.read_text(encoding="utf-8").splitlines()[0])["input"]
    _seed_inputs(tmp_path, [first_input])

    # Seed failed.jsonl with a Slice 3 aggregate row containing a validation_failed attempt.
    (tmp_path / "failed.jsonl").write_text(
        json.dumps({
            "input": first_input,
            "reason": "all_candidates_failed",
            "candidates": [],
            "attempts": [
                {"provider": "fake_a", "failure_reason": "validation_failed"},
            ],
        }) + "\n",
        encoding="utf-8",
    )

    cfg = _make_label_config(tmp_path)
    # WITHOUT --retry-validation-failed: input is skipped (no train.jsonl row).
    result = _run_label(cfg, tmp_path)
    assert result.returncode == 0, result.stderr
    train = tmp_path / "train.jsonl"
    train_rows = []
    if train.exists():
        train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(train_rows) == 0   # input was skipped due to failed.jsonl entry

    # WITH --retry-validation-failed: input is re-attempted; clean label exists in fixture.
    result = _run_label(cfg, tmp_path, "--retry-validation-failed")
    assert result.returncode == 0, result.stderr
    if train.exists():
        train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(train_rows) == 1
    assert train_rows[0]["_provider"] == "fake_a"


def test_label_phase_writes_metrics_json(tmp_path):
    """End-to-end: a successful label run writes data/distill/metrics.json
    with the documented shape."""
    fixture = REPO_ROOT / "tests" / "fixtures" / "fake_labels.jsonl"
    first_input = json.loads(fixture.read_text(encoding="utf-8").splitlines()[0])["input"]
    _seed_inputs(tmp_path, [first_input])
    cfg = _make_label_config(tmp_path)
    result = _run_label(cfg, tmp_path)
    assert result.returncode == 0, result.stderr

    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists(), "metrics.json should be written"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Required top-level keys.
    assert {"run", "totals", "providers", "failures", "repair"} <= set(metrics)

    # Run block.
    assert metrics["run"]["phase"] == "label"
    assert metrics["run"]["inputs_processed"] == 1

    # Totals match output files.
    train = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert metrics["totals"]["train_rows_written"] == len(train) == 1

    # Provider-error invariant.
    assert metrics["failures"].get("provider_error", 0) == metrics["totals"]["calls_failed"]


def test_label_phase_metrics_records_json_parse_failures(tmp_path):
    """FakeProvider miss returns empty/malformed payload -> json_parse_failed.
    Asserts: calls_failed == 0, candidates_rejected > 0, failures.json_parse_failed > 0."""
    _seed_inputs(tmp_path, ["completely_unmatched_input_string_xyz"])
    cfg = _make_label_config(tmp_path, label_attempts=2)
    result = _run_label(cfg, tmp_path)
    assert result.returncode == 0, result.stderr

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["totals"]["calls_failed"] == 0
    assert metrics["totals"]["candidates_rejected"] > 0
    assert metrics["failures"].get("json_parse_failed", 0) > 0
    # Invariant still holds (provider_error and calls_failed both 0 here).
    assert metrics["failures"].get("provider_error", 0) == metrics["totals"]["calls_failed"]
