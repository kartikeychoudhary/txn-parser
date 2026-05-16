"""Subprocess-based tests for the Stage 5 CLI flag matrix.

Uses subprocess.run rather than calling main() directly so the full
argparse path is exercised end-to-end and parser.error semantics
(exit code 2, message to stderr) are verified verbatim.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "05_generate_distillation_data.py"
CONFIG = REPO_ROOT / "configs" / "test_providers.json"


def run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = {**os.environ}
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env=full_env,
    )


def test_help_confirms_cli_imports_cleanly():
    """Sanity: --help still works and exposes the new flags. This proves the
    CLI imports cleanly; it does NOT prove legacy runtime behavior, which is
    covered transitively by the existing legacy tests in the prior slice."""
    result = run_cli("--help")
    assert result.returncode == 0
    assert "--provider-config" in result.stdout
    assert "--dry-run-quota" in result.stdout
    assert "--multi-provider" in result.stdout


def test_provider_config_with_dry_run_quota_succeeds():
    result = run_cli("--provider-config", str(CONFIG), "--dry-run-quota")
    assert result.returncode == 0, result.stderr
    assert "Multi-provider dry run" in result.stdout
    assert "Input generation: enabled" in result.stdout


def test_provider_config_alone_exits_2():
    result = run_cli("--provider-config", str(CONFIG))
    assert result.returncode == 2
    assert "--dry-run-quota" in result.stderr


def test_dry_run_quota_alone_exits_2():
    result = run_cli("--dry-run-quota")
    assert result.returncode == 2
    assert "--provider-config" in result.stderr


def test_multi_provider_alone_exits_2():
    result = run_cli("--multi-provider")
    assert result.returncode == 2
    assert "--provider-config" in result.stderr




def test_invalid_config_path_exits_2():
    result = run_cli("--provider-config", "definitely_not_real.json", "--dry-run-quota")
    assert result.returncode == 2
    assert "Invalid provider config" in result.stderr or "not found" in result.stderr.lower()


def test_provider_config_with_multi_provider_and_phase_label_exits_2():
    result = run_cli("--provider-config", str(CONFIG),
                     "--multi-provider", "--phase", "label")
    assert result.returncode == 2
    assert "Slice 3" in result.stderr


def test_provider_config_with_multi_provider_and_phase_all_exits_2():
    result = run_cli("--provider-config", str(CONFIG),
                     "--multi-provider", "--phase", "all")
    assert result.returncode == 2
    assert "all" in result.stderr.lower()


def test_provider_config_with_multi_provider_and_phase_eval_exits_2():
    result = run_cli("--provider-config", str(CONFIG),
                     "--multi-provider", "--phase", "eval")
    assert result.returncode == 2


def test_provider_config_with_multi_provider_and_dry_run_succeeds_phase_agnostic(tmp_path):
    """Dry-run wins regardless of --phase value."""
    for phase in ("inputs", "label", "all", "eval"):
        result = run_cli("--provider-config", str(CONFIG),
                         "--multi-provider", "--dry-run-quota",
                         "--phase", phase,
                         env={"DISTILL_DIR_OVERRIDE": str(tmp_path)})
        assert result.returncode == 0, f"phase={phase}: {result.stderr}"
        assert "Multi-provider dry run" in result.stdout


def test_provider_config_with_multi_provider_and_phase_inputs_succeeds(tmp_path):
    """The headline Slice 2 capability: --phase inputs --multi-provider runs."""
    result = run_cli(
        "--provider-config", str(CONFIG),
        "--multi-provider", "--phase", "inputs",
        env={"DISTILL_DIR_OVERRIDE": str(tmp_path)},
    )
    # test_providers.json has target_inputs=10 and fake providers, so this should
    # complete successfully and produce a file under tmp_path.
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "inputs_raw.jsonl").exists()
