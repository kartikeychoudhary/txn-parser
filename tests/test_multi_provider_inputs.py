"""End-to-end tests for phase_inputs_multi_provider via subprocess + FakeProvider.

All tests pass `env={"DISTILL_DIR_OVERRIDE": str(tmp_path)}` so the repo's
real data/distill/ is never touched.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "05_generate_distillation_data.py"


def _make_test_config(tmp_path: Path, target: int = 10) -> Path:
    fixtures_src = REPO_ROOT / "tests" / "fixtures" / "fake_inputs.jsonl"
    cfg = {
        "version": 1,
        "input_generation": {
            "enabled": True, "target_inputs": target, "batch_size": 5,
            "providers": [
                {"name": "fake_a", "type": "fake", "weight": 60, "threads": 2,
                 "fixture_inputs": str(fixtures_src)},
                {"name": "fake_b", "type": "fake", "weight": 40, "threads": 2,
                 "fixture_inputs": str(fixtures_src)},
            ],
        },
        "output_generation": {
            "enabled": False,
            "providers": [],
        },
        "validation": {"schema": True, "semantic_validator": True,
                       "reject_invalid": True,
                       "retry_invalid_with_stricter_prompt": False,
                       "max_repair_attempts": 0},
        "rate_limits": {"global_max_workers": 4, "write_flush_every": 5},
    }
    cfg_path = tmp_path / "providers.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_path


def _run_multi_provider(cfg_path: Path, tmp_path: Path, *extra_args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "DISTILL_DIR_OVERRIDE": str(tmp_path)}
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--phase", "inputs",
         "--provider-config", str(cfg_path),
         "--multi-provider",
         *extra_args],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
    )


def test_multi_provider_inputs_produces_target_count(tmp_path):
    cfg = _make_test_config(tmp_path, target=10)
    result = _run_multi_provider(cfg, tmp_path)
    assert result.returncode == 0, result.stderr
    out_file = tmp_path / "inputs_raw.jsonl"
    assert out_file.exists()
    rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    inputs = {r["input"] for r in rows}
    assert len(inputs) == 10


def test_multi_provider_rows_include_metadata(tmp_path):
    cfg = _make_test_config(tmp_path, target=5)
    result = _run_multi_provider(cfg, tmp_path)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (tmp_path / "inputs_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        assert r["_source"] == "synthetic_input"
        assert r["_provider"] in {"fake_a", "fake_b"}
        assert r["_model"] is None    # FakeProvider has no model
        assert "_batch_id" in r
        assert r["_batch_id"]   # non-empty string


def test_multi_provider_resume_appends_only_new(tmp_path):
    """Pre-populate inputs_raw.jsonl with 3 inputs; run with target=10;
    only 7 new rows should be added."""
    out_file = tmp_path / "inputs_raw.jsonl"
    out_file.write_text(
        '{"input":"existing one"}\n'
        '{"input":"existing two"}\n'
        '{"input":"existing three"}\n',
        encoding="utf-8",
    )
    cfg = _make_test_config(tmp_path, target=10)
    result = _run_multi_provider(cfg, tmp_path)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 10
    inputs = {r["input"] for r in rows}
    assert "existing one" in inputs
    assert "existing two" in inputs
    assert "existing three" in inputs


def test_multi_provider_already_at_target_does_nothing(tmp_path):
    out_file = tmp_path / "inputs_raw.jsonl"
    rows_before = [{"input": f"row_{i}"} for i in range(15)]
    out_file.write_text("\n".join(json.dumps(r) for r in rows_before) + "\n",
                        encoding="utf-8")
    cfg = _make_test_config(tmp_path, target=10)   # already past 10
    result = _run_multi_provider(cfg, tmp_path)
    assert result.returncode == 0, result.stderr
    rows_after = out_file.read_text(encoding="utf-8").splitlines()
    assert len(rows_after) == 15   # unchanged


def test_multi_provider_n_inputs_is_ignored_in_multi_mode(tmp_path):
    """target_inputs from config wins over --n-inputs."""
    cfg = _make_test_config(tmp_path, target=5)
    result = _run_multi_provider(cfg, tmp_path, "--n-inputs", "999")
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (tmp_path / "inputs_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    # target_inputs from config (=5) is the source of truth, not --n-inputs (=999).
    assert len({r["input"] for r in rows}) == 5


def test_multi_provider_stops_when_fixture_unique_supply_exhausted(tmp_path):
    """When target_inputs > unique fixture supply, the duplicate-stall guard
    must trigger and exit cleanly with partial progress — must not hang."""
    cfg = _make_test_config(tmp_path, target=999)   # fixture has 20 unique rows
    result = _run_multi_provider(cfg, tmp_path)
    assert result.returncode == 0, result.stderr
    out_file = tmp_path / "inputs_raw.jsonl"
    rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    inputs = {r["input"] for r in rows}
    # Fixture has 20 unique rows; we should get all of them, not the requested 999.
    assert len(inputs) <= 20
    # The global duplicate-stall guard should have logged a warning in stdout.
    output_lower = result.stdout.lower()
    assert (
        "empty batches" in output_lower
        or "duplicate" in output_lower
        or "providers exhausted" in output_lower
    )


def test_multi_provider_disabled_input_generation_does_nothing(tmp_path):
    """enabled=false short-circuits without writing anything."""
    fixtures_src = REPO_ROOT / "tests" / "fixtures" / "fake_inputs.jsonl"
    cfg_data = {
        "version": 1,
        "input_generation": {
            "enabled": False, "target_inputs": 10, "batch_size": 5,
            "providers": [
                {"name": "fake_a", "type": "fake", "weight": 1, "threads": 1,
                 "fixture_inputs": str(fixtures_src)},
            ],
        },
        "output_generation": {"enabled": False, "providers": []},
        "validation": {"schema": True, "semantic_validator": True,
                       "reject_invalid": True,
                       "retry_invalid_with_stricter_prompt": False,
                       "max_repair_attempts": 0},
        "rate_limits": {"global_max_workers": 4, "write_flush_every": 5},
    }
    cfg_path = tmp_path / "providers.json"
    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")
    result = _run_multi_provider(cfg_path, tmp_path)
    assert result.returncode == 0, result.stderr
    out_file = tmp_path / "inputs_raw.jsonl"
    assert not out_file.exists() or out_file.read_text(encoding="utf-8") == ""


def test_inputs_phase_writes_metrics_json(tmp_path):
    """End-to-end: inputs phase writes metrics.json with phase='inputs'."""
    fixture = REPO_ROOT / "tests" / "fixtures" / "fake_inputs.jsonl"
    target = 5
    cfg_data = {
        "version": 1,
        "input_generation": {
            "enabled": True,
            "target_inputs": target,
            "batch_size": 5,
            "dedupe": True,
            "providers": [
                {"name": "fake_a", "type": "fake", "weight": 1, "threads": 1,
                 "fixture_inputs": str(fixture)},
            ],
        },
        "output_generation": {"enabled": False, "providers": []},
        "validation": {"schema": True, "semantic_validator": True,
                       "reject_invalid": True,
                       "retry_invalid_with_stricter_prompt": False,
                       "max_repair_attempts": 0},
        "rate_limits": {"global_max_workers": 1, "write_flush_every": 5},
    }
    cfg_path = tmp_path / "providers.json"
    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")
    env = {**os.environ, "DISTILL_DIR_OVERRIDE": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--phase", "inputs",
         "--provider-config", str(cfg_path),
         "--multi-provider"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
    )
    assert result.returncode == 0, result.stderr

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run"]["phase"] == "inputs"
    assert metrics["totals"]["input_rows_written"] == target
    assert "repair" not in metrics
    # acceptance_rate must NOT appear in any provider block.
    for pname, pblock in metrics["providers"].items():
        assert "acceptance_rate" not in pblock, \
            f"acceptance_rate must be absent in input phase, found in {pname}"
