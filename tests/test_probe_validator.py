import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_SCRIPT = REPO_ROOT / "scripts" / "probe_validator.py"
PROBE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "probe_examples.jsonl"


def run_probe(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROBE_SCRIPT), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )


def test_probe_runs_on_probe_examples_fixture():
    result = run_probe("--input", str(PROBE_FIXTURE))
    assert result.returncode == 0, result.stderr
    assert "Validator probe" in result.stdout
    assert "Rows scanned:" in result.stdout
    assert "Failures by code:" in result.stdout


def test_probe_reports_expected_error_codes():
    """probe_examples.jsonl is designed to hit every validator error code."""
    result = run_probe("--input", str(PROBE_FIXTURE))
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for code in (
        "SCHEMA_INVALID",
        "AMOUNT_NOT_IN_INPUT",
        "SUSPICIOUS_DUPLICATE",
        "SUPERSEDED_AMOUNT_USED",
        "CURRENCY_MISMATCH",
        "TXN_COUNT_EXCEEDS_CANDIDATES",
        "NO_AMOUNT_IN_INPUT",
    ):
        assert code in out, f"missing {code} in probe output:\n{out}"


def test_probe_writes_per_row_report(tmp_path):
    report_path = tmp_path / "probe_report.jsonl"
    result = run_probe(
        "--input", str(PROBE_FIXTURE),
        "--output", str(report_path),
    )
    assert result.returncode == 0, result.stderr
    assert report_path.exists()
    rows = [json.loads(line) for line in report_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 11
    for row in rows:
        assert "line" in row
        assert "validator_ok" in row


def test_probe_creates_output_parent_directory(tmp_path):
    report_path = tmp_path / "nested" / "dir" / "report.jsonl"
    result = run_probe(
        "--input", str(PROBE_FIXTURE),
        "--output", str(report_path),
    )
    assert result.returncode == 0, result.stderr
    assert report_path.exists()


def test_probe_handles_malformed_row(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"input":"ok","output":{"transactions":[{"amount":500,"currency":"INR","item":"beer","category":"Drinks","type":"expense"}]}}\n'
        '{not json\n',
        encoding="utf-8",
    )
    result = run_probe("--input", str(bad))
    # Default: malformed rows do not cause non-zero exit.
    assert result.returncode == 0, result.stderr
    assert "Malformed:" in result.stdout
    assert "1" in result.stdout.split("Malformed:")[1].splitlines()[0]


def test_probe_fail_on_malformed_flips_exit_to_3(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{not json\n', encoding="utf-8")
    result = run_probe("--input", str(bad), "--fail-on-malformed")
    assert result.returncode == 3


def test_probe_missing_input_file_exits_2():
    result = run_probe("--input", "definitely_not_a_real_path.jsonl")
    assert result.returncode == 2


def test_probe_limit_truncates(tmp_path):
    result = run_probe("--input", str(PROBE_FIXTURE), "--limit", "3")
    assert result.returncode == 0, result.stderr
    # Rows scanned should be 3.
    for line in result.stdout.splitlines():
        if line.startswith("Rows scanned:"):
            assert "3" in line
            return
    raise AssertionError("Rows scanned line not found")


def test_probe_mode_warn_does_not_hide_failures():
    """--mode warn flips result.ok to True in serialized report but summary
    classification still uses error severity, so Failed count is unchanged."""
    strict = run_probe("--input", str(PROBE_FIXTURE))
    warn = run_probe("--input", str(PROBE_FIXTURE), "--mode", "warn")
    assert strict.returncode == 0
    assert warn.returncode == 0
    # Extract Failed: line from each.
    def failed_count(out):
        for line in out.splitlines():
            if line.startswith("Failed:"):
                return line
        return None
    assert failed_count(strict.stdout) == failed_count(warn.stdout)
