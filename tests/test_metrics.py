"""Unit tests for scripts/metrics.py.

Covers pure helpers (load_prices, lookup_price, estimate_tokens, percentiles)
and MetricsRecorder (added in Task 2). This file is metrics.py only — the
provider-error orchestrator instrumentation path is tested in
tests/test_label_orchestrator.py.
"""
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# load_prices
# ---------------------------------------------------------------------------


def _write_prices(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "prices.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_load_prices_valid_file_loads_all_entries(tmp_path):
    from metrics import load_prices
    path = _write_prices(tmp_path, {
        "deepseek:deepseek-chat": {"input_per_million": 0.27, "output_per_million": 1.10},
        "gemini:gemini-2.5-flash": {"input_per_million": 0.30, "output_per_million": 2.50},
        "default": {"input_per_million": 0.0, "output_per_million": 0.0},
    })
    prices = load_prices(path)
    assert prices["deepseek:deepseek-chat"]["input_per_million"] == 0.27
    assert prices["gemini:gemini-2.5-flash"]["output_per_million"] == 2.50
    assert prices["default"]["input_per_million"] == 0.0


def test_load_prices_missing_file_returns_default(tmp_path, caplog):
    from metrics import load_prices
    with caplog.at_level(logging.WARNING):
        prices = load_prices(tmp_path / "nope.json")
    assert prices == {"default": {"input_per_million": 0.0, "output_per_million": 0.0}}
    assert any("not found" in rec.message.lower() for rec in caplog.records)


def test_load_prices_malformed_json_raises(tmp_path):
    from metrics import load_prices, MetricsConfigError
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(MetricsConfigError):
        load_prices(path)


def test_load_prices_negative_input_price_raises(tmp_path):
    from metrics import load_prices, MetricsConfigError
    path = _write_prices(tmp_path, {
        "deepseek:deepseek-chat": {"input_per_million": -0.27, "output_per_million": 1.10},
    })
    with pytest.raises(MetricsConfigError):
        load_prices(path)


def test_load_prices_non_numeric_output_price_raises(tmp_path):
    from metrics import load_prices, MetricsConfigError
    path = _write_prices(tmp_path, {
        "deepseek:deepseek-chat": {"input_per_million": 0.27, "output_per_million": "free"},
    })
    with pytest.raises(MetricsConfigError):
        load_prices(path)


def test_load_prices_unknown_field_warns_and_loads(tmp_path, caplog):
    from metrics import load_prices
    path = _write_prices(tmp_path, {
        "gemini:gemini-2.5-flash": {
            "input_per_million": 0.30,
            "output_per_million": 2.50,
            "foo": 123,
        },
    })
    with caplog.at_level(logging.WARNING):
        prices = load_prices(path)
    assert "gemini:gemini-2.5-flash" in prices
    assert any("foo" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# lookup_price
# ---------------------------------------------------------------------------


def test_lookup_price_exact_provider_type_model_wins():
    from metrics import lookup_price
    prices = {
        "deepseek:deepseek-chat": {"input_per_million": 0.27, "output_per_million": 1.10},
        "deepseek-chat": {"input_per_million": 99.0, "output_per_million": 99.0},
        "deepseek": {"input_per_million": 88.0, "output_per_million": 88.0},
        "default": {"input_per_million": 0.0, "output_per_million": 0.0},
    }
    assert lookup_price(prices, "deepseek", "deepseek-chat") == (0.27, 1.10)


def test_lookup_price_falls_back_to_model():
    from metrics import lookup_price
    prices = {
        "deepseek-chat": {"input_per_million": 0.27, "output_per_million": 1.10},
        "default": {"input_per_million": 0.0, "output_per_million": 0.0},
    }
    assert lookup_price(prices, "deepseek", "deepseek-chat") == (0.27, 1.10)


def test_lookup_price_falls_back_to_provider_type():
    from metrics import lookup_price
    prices = {
        "gemini": {"input_per_million": 0.30, "output_per_million": 2.50},
        "default": {"input_per_million": 0.0, "output_per_million": 0.0},
    }
    assert lookup_price(prices, "gemini", "gemini-2.5-pro") == (0.30, 2.50)


def test_lookup_price_falls_back_to_default():
    from metrics import lookup_price
    prices = {
        "default": {"input_per_million": 0.5, "output_per_million": 1.5},
    }
    assert lookup_price(prices, "unknown_type", "unknown-model") == (0.5, 1.5)


def test_lookup_price_no_match_returns_zero():
    from metrics import lookup_price
    prices = {}
    assert lookup_price(prices, "x", "y") == (0.0, 0.0)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_empty_is_zero():
    from metrics import estimate_tokens
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_tokens_one_char_is_one():
    from metrics import estimate_tokens
    assert estimate_tokens("a") == 1


def test_estimate_tokens_ceils_length_over_four():
    from metrics import estimate_tokens
    # 17 chars -> ceil(17/4) = 5
    assert estimate_tokens("a" * 17) == 5
    # 4 chars -> 1
    assert estimate_tokens("abcd") == 1
    # 5 chars -> 2
    assert estimate_tokens("abcde") == 2


# ---------------------------------------------------------------------------
# percentiles
# ---------------------------------------------------------------------------


def test_percentiles_empty_returns_zero():
    from metrics import percentiles
    assert percentiles([]) == {"p50": 0.0, "p95": 0.0}


def test_percentiles_single_value_both_equal():
    from metrics import percentiles
    assert percentiles([42.0]) == {"p50": 42.0, "p95": 42.0}


def test_percentiles_small_sample_p95_is_max():
    from metrics import percentiles
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = percentiles(values)
    assert result["p95"] == 50.0
    # p50 should be roughly the median
    assert 20.0 <= result["p50"] <= 40.0


def test_percentiles_large_sample_uses_quantiles():
    from metrics import percentiles
    values = list(range(1, 101))   # 1..100, n=100 >= 20
    result = percentiles(values)
    # statistics.quantiles(n=20) splits into 20 equal-frequency buckets;
    # index 9 is the 50th percentile (median), index 18 is the 95th.
    assert 45.0 <= result["p50"] <= 55.0
    assert 90.0 <= result["p95"] <= 100.0


# ---------------------------------------------------------------------------
# MetricsRecorder
# ---------------------------------------------------------------------------


def _recorder(prices=None, started_at=None, source="configs/prices.json"):
    from metrics import MetricsRecorder
    return MetricsRecorder(
        prices=prices or {"default": {"input_per_million": 0.0, "output_per_million": 0.0}},
        started_at=started_at or datetime(2026, 5, 17, 14, 33, 1, tzinfo=timezone.utc),
        prices_source=source,
    )


def test_recorder_record_call_accumulates_per_provider():
    r = _recorder()
    for _ in range(3):
        r.record_call(
            provider="gemini_flash", provider_type="gemini", model="gemini-2.5-flash",
            phase="label", attempt_type="initial",
            latency_ms=100.0,
            prompt_tokens=10, completion_tokens=20,
            estimated_tokens=False,
            success=True, failure_reason=None,
        )
    summary = r.finalize(phase="label", inputs_processed=3,
                         finished_at=datetime(2026, 5, 17, 14, 35, 42, tzinfo=timezone.utc))
    assert summary["providers"]["gemini_flash"]["calls"] == 3
    assert summary["providers"]["gemini_flash"]["success_calls"] == 3
    assert summary["providers"]["gemini_flash"]["failed_calls"] == 0
    assert summary["providers"]["gemini_flash"]["prompt_tokens"] == 30
    assert summary["providers"]["gemini_flash"]["completion_tokens"] == 60


def test_recorder_record_call_thread_safety():
    import threading
    r = _recorder()

    def worker():
        for _ in range(100):
            r.record_call(
                provider="p", provider_type="fake", model=None,
                phase="label", attempt_type="initial",
                latency_ms=1.0, prompt_tokens=1, completion_tokens=1,
                estimated_tokens=True, success=True, failure_reason=None,
            )

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    summary = r.finalize(phase="label", inputs_processed=10,
                         finished_at=datetime(2026, 5, 17, 14, 35, 42, tzinfo=timezone.utc))
    assert summary["totals"]["calls"] == 1000


def test_recorder_provider_error_invariant():
    """failures.provider_error == totals.calls_failed exactly."""
    r = _recorder()
    for _ in range(5):
        r.record_call(
            provider="p", provider_type="fake", model=None,
            phase="label", attempt_type="initial",
            latency_ms=10.0, prompt_tokens=0, completion_tokens=0,
            estimated_tokens=True, success=False, failure_reason="provider_error",
        )
    summary = r.finalize(phase="label", inputs_processed=0,
                         finished_at=datetime(2026, 5, 17, 14, 35, 42, tzinfo=timezone.utc))
    assert summary["totals"]["calls_failed"] == 5
    assert summary["failures"]["provider_error"] == 5


def test_recorder_record_output_row_counts():
    r = _recorder()
    r.record_output_row("train")
    r.record_output_row("train")
    r.record_output_row("failed")
    r.record_output_row("input")
    summary = r.finalize(phase="inputs", inputs_processed=1,
                         finished_at=datetime(2026, 5, 17, 14, 35, 42, tzinfo=timezone.utc))
    assert summary["totals"]["train_rows_written"] == 2
    assert summary["totals"]["failed_rows_written"] == 1
    assert summary["totals"]["input_rows_written"] == 1


def test_recorder_empty_finalize_valid_shape():
    r = _recorder()
    summary = r.finalize(phase="label", inputs_processed=0,
                         finished_at=datetime(2026, 5, 17, 14, 35, 42, tzinfo=timezone.utc))
    assert summary["run"]["phase"] == "label"
    assert summary["totals"]["calls"] == 0
    assert summary["totals"]["estimated_cost_usd"] == 0.0
    assert summary["totals"]["has_estimated_tokens"] is False
    assert summary["providers"] == {}
    assert summary["failures"] == {}
    assert summary["repair"] == {"attempts": 0, "accepted": 0, "exhausted": 0}


def test_recorder_finalize_inputs_phase_omits_repair_and_acceptance_rate():
    r = _recorder()
    r.record_call(
        provider="p", provider_type="fake", model=None,
        phase="inputs", attempt_type="input_batch",
        latency_ms=10.0, prompt_tokens=10, completion_tokens=20,
        estimated_tokens=True, success=True, failure_reason=None,
    )
    summary = r.finalize(phase="inputs", inputs_processed=1,
                         finished_at=datetime(2026, 5, 17, 14, 35, 42, tzinfo=timezone.utc))
    assert "repair" not in summary
    assert "acceptance_rate" not in summary["providers"]["p"]


def test_recorder_cost_math():
    prices = {"default": {"input_per_million": 0.30, "output_per_million": 1.50}}
    r = _recorder(prices=prices)
    r.record_call(
        provider="p", provider_type="fake", model=None,
        phase="label", attempt_type="initial",
        latency_ms=1.0,
        prompt_tokens=1_000_000, completion_tokens=2_000_000,
        estimated_tokens=False, success=True, failure_reason=None,
    )
    summary = r.finalize(phase="label", inputs_processed=1,
                         finished_at=datetime(2026, 5, 17, 14, 35, 42, tzinfo=timezone.utc))
    # 1M * $0.30 + 2M * $1.50 = $0.30 + $3.00 = $3.30
    assert summary["providers"]["p"]["estimated_cost_usd"] == 3.30
    assert summary["totals"]["estimated_cost_usd"] == 3.30


def test_recorder_has_estimated_tokens_flag():
    r = _recorder()
    r.record_call(
        provider="p", provider_type="fake", model=None,
        phase="label", attempt_type="initial",
        latency_ms=1.0, prompt_tokens=1, completion_tokens=1,
        estimated_tokens=True, success=True, failure_reason=None,
    )
    summary = r.finalize(phase="label", inputs_processed=1,
                         finished_at=datetime(2026, 5, 17, 14, 35, 42, tzinfo=timezone.utc))
    assert summary["providers"]["p"]["has_estimated_tokens"] is True
    assert summary["totals"]["has_estimated_tokens"] is True


def test_recorder_record_label_candidate_acceptance_rate():
    r = _recorder()
    # 4 accepted, 1 rejected with failure_reason, 1 lost-to-sibling (None).
    for _ in range(4):
        r.record_call(
            provider="p", provider_type="fake", model=None,
            phase="label", attempt_type="initial",
            latency_ms=1.0, prompt_tokens=1, completion_tokens=1,
            estimated_tokens=True, success=True, failure_reason=None,
        )
        r.record_label_candidate(
            provider="p", model=None, score=90,
            accepted=True, failure_reason=None, is_repair_attempt=False,
        )
    r.record_call(
        provider="p", provider_type="fake", model=None,
        phase="label", attempt_type="initial",
        latency_ms=1.0, prompt_tokens=1, completion_tokens=1,
        estimated_tokens=True, success=True, failure_reason=None,
    )
    r.record_label_candidate(
        provider="p", model=None, score=0,
        accepted=False, failure_reason="json_parse_failed", is_repair_attempt=False,
    )
    r.record_call(
        provider="p", provider_type="fake", model=None,
        phase="label", attempt_type="initial",
        latency_ms=1.0, prompt_tokens=1, completion_tokens=1,
        estimated_tokens=True, success=True, failure_reason=None,
    )
    r.record_label_candidate(
        provider="p", model=None, score=85,
        accepted=False, failure_reason=None, is_repair_attempt=False,
    )
    summary = r.finalize(phase="label", inputs_processed=4,
                         finished_at=datetime(2026, 5, 17, 14, 35, 42, tzinfo=timezone.utc))
    pr = summary["providers"]["p"]
    assert pr["candidates_accepted"] == 4
    assert pr["candidates_rejected"] == 2
    assert pr["acceptance_rate"] == pytest.approx(4 / 6, rel=1e-3)
    # failures counts only non-null failure_reason
    assert summary["failures"] == {"json_parse_failed": 1}


def test_recorder_render_stdout_label_phase_shape():
    r = _recorder()
    r.record_call(
        provider="gemini_flash", provider_type="gemini", model="gemini-2.5-flash",
        phase="label", attempt_type="initial",
        latency_ms=800.0, prompt_tokens=100, completion_tokens=50,
        estimated_tokens=False, success=True, failure_reason=None,
    )
    r.record_label_candidate(
        provider="gemini_flash", model="gemini-2.5-flash", score=95,
        accepted=True, failure_reason=None, is_repair_attempt=False,
    )
    summary = r.finalize(phase="label", inputs_processed=1,
                         finished_at=datetime(2026, 5, 17, 14, 35, 42, tzinfo=timezone.utc))
    text = r.render_stdout(summary)
    assert "gemini_flash" in text
    assert "$" in text
    assert "p50" in text or "p95" in text or "812" in text or "800" in text  # latency shown
    assert "Acc" in text


def test_recorder_render_stdout_inputs_phase_omits_acc():
    r = _recorder()
    r.record_call(
        provider="p", provider_type="fake", model=None,
        phase="inputs", attempt_type="input_batch",
        latency_ms=10.0, prompt_tokens=1, completion_tokens=1,
        estimated_tokens=True, success=True, failure_reason=None,
    )
    summary = r.finalize(phase="inputs", inputs_processed=1,
                         finished_at=datetime(2026, 5, 17, 14, 35, 42, tzinfo=timezone.utc))
    text = r.render_stdout(summary)
    # No accuracy column in input phase output.
    assert "Acc" not in text
    assert "Top failure" not in text


def test_recorder_render_stdout_prices_source_none_says_default_zero():
    r = _recorder(source=None)
    summary = r.finalize(phase="label", inputs_processed=0,
                         finished_at=datetime(2026, 5, 17, 14, 35, 42, tzinfo=timezone.utc))
    text = r.render_stdout(summary)
    assert "default zero pricing" in text


def test_recorder_repair_exhausted_not_inflated_by_unrelated_failed_rows():
    """repair.exhausted must reflect explicit calls to record_repair_exhausted,
    not approximate from failed_rows. Two failed rows; only one entered repair."""
    r = _recorder()
    # Two failed inputs total.
    r.record_output_row("failed")
    r.record_output_row("failed")
    # Only one of them entered the repair loop and exhausted it.
    r.record_call(
        provider="p", provider_type="fake", model=None,
        phase="label", attempt_type="repair",
        latency_ms=10.0, prompt_tokens=1, completion_tokens=1,
        estimated_tokens=True, success=True, failure_reason=None,
    )
    r.record_repair_exhausted()
    summary = r.finalize(phase="label", inputs_processed=2,
                         finished_at=datetime(2026, 5, 17, 14, 35, 42, tzinfo=timezone.utc))
    assert summary["totals"]["failed_rows_written"] == 2
    assert summary["repair"]["attempts"] == 1
    assert summary["repair"]["exhausted"] == 1   # NOT 2
