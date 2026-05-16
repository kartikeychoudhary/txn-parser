# Multi-Provider Generation — Slice 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-run metrics report (cost, latency p50/p95, tokens, acceptance, failures) to multi-provider Stage 5 runs. Default behavior is unchanged for every legacy path.

**Architecture:** A pure-stdlib `scripts/metrics.py` module owns `MetricsRecorder` plus `load_prices`/`lookup_price`/`estimate_tokens`/`percentiles` helpers. `DeepSeek`/`Gemini` providers stash SDK usage per-thread and expose `pop_last_usage()`. The orchestrator wraps every provider call in a `try/finally` that times and records the call, then records terminal candidate outcomes (label phase only). At end of `main()`, the recorder writes `data/distill/metrics.json` and logs a compact stdout summary. Recorder is created only for `--multi-provider --phase {inputs,label}` runs (not `--dry-run-quota`, not legacy paths).

**Tech Stack:** Python 3.11+, stdlib only (`threading`, `statistics`, `subprocess`, `datetime`, `math`). Existing pytest harness. No new top-level deps.

**Spec:** `docs/superpowers/specs/2026-05-17-multi-provider-generation-slice4-design.md` (commits `375e269` + `f3cb4d9`).

**Branch:** `feat/multi-provider-slice4` (already exists, branched from `feat/multi-provider-slice3` HEAD `0b79433`; spec commits `375e269` and `f3cb4d9` are on this branch).

**Test baseline at start:** 294 passing on `feat/multi-provider-slice3`. Target at end of plan: **~333 passing** (294 baseline + 18 helper tests + 12 recorder tests + 5 provider tests + 1 orchestrator test + 2 label E2E + 1 inputs E2E + 1 repair-exhausted unit test = 334; exact count may drift up by 1–2 if the Task 12 coverage check adds a small gap-filler test. Treat counts as smoke signals, not hard blockers — `335 passed` is not a failure as long as nothing regressed).

**Single commit at landing.** Same convention as Slices 1–3: spec + amendments live on the branch as their own commits; the implementation lands as one additive commit at Task 13. Data files in `data/distill/`, `.coverage`, and `reports/` are NEVER staged.

---

### Task 0: Preflight

**Files:** none modified.

- [ ] **Step 1: Confirm branch and HEAD**

```bash
cd "C:/work/llm training" && git status --short && git log --oneline -3
```
Expected output:
```
 M data/distill/failed.jsonl
 M data/distill/inputs_raw.jsonl
 M data/distill/train.jsonl
?? .coverage
?? reports/
f3cb4d9 Spec amendments after review
375e269 Spec: multi-provider generation Slice 4 (per-run metrics + cost)
0b79433 Slice 3: validator-gated multi-provider output labeling
```

If `data/distill/*` lines are absent that's fine — they may have been overwritten by smoke runs. The key check: HEAD is `f3cb4d9` on branch `feat/multi-provider-slice4`.

- [ ] **Step 2: Confirm test baseline**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: `294 passed in ~20s`.

- [ ] **Step 3: Sanity-check that `import llm_providers` is SDK-free**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
import sys, llm_providers
loaded = [m for m in sys.modules if m.startswith(('openai','google','torch','unsloth','transformers'))]
assert not loaded, f'leaked: {loaded}'
print('import llm_providers: clean')
"
```
Expected: `import llm_providers: clean`.

If this fails, STOP — Slice 2/3 isolation is broken; do not proceed.

- [ ] **Step 4: Verify the spec file is present**

```bash
cd "C:/work/llm training" && ls docs/superpowers/specs/2026-05-17-multi-provider-generation-slice4-design.md
```
Expected: the file path printed.

---

### Task 1: `metrics.py` exceptions + dataclasses + pure helpers

**Files:**
- Create: `scripts/metrics.py`
- Create: `tests/test_metrics.py`

Spec reference: §3.1, §3.2, §3.3, §3.4.

- [ ] **Step 1: Write the failing unit tests for helpers**

Create `tests/test_metrics.py` with this content:

```python
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
```

- [ ] **Step 2: Run the helper tests — expect import error**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_metrics.py -v 2>&1 | tail -10
```
Expected: every test ERRORS with `ModuleNotFoundError: No module named 'metrics'` (because `scripts/metrics.py` does not exist yet).

- [ ] **Step 3: Create `scripts/metrics.py` with exceptions, dataclasses, and helpers**

Create `scripts/metrics.py` with this exact content:

```python
"""Per-run metrics for multi-provider Stage 5 runs.

Pure stdlib. Importing this module triggers no SDK imports, no I/O.

Public surface:
    - MetricsConfigError
    - ProviderCallMetric, CandidateMetric (dataclasses)
    - load_prices(path) -> dict
    - lookup_price(prices, provider_type, model) -> (input_per_million, output_per_million)
    - estimate_tokens(text) -> int
    - percentiles(values) -> {"p50": ..., "p95": ...}
    - MetricsRecorder (added in Task 2)

Spec: docs/superpowers/specs/2026-05-17-multi-provider-generation-slice4-design.md
"""
from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


logger = logging.getLogger("metrics")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MetricsConfigError(ValueError):
    """Raised when prices.json fails schema or numeric checks."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProviderCallMetric:
    provider: str
    provider_type: str
    model: str | None
    phase: Literal["inputs", "label"]
    attempt_type: Literal["input_batch", "initial", "repair"]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_tokens: bool
    success: bool
    failure_reason: str | None


@dataclass
class CandidateMetric:
    provider: str
    model: str | None
    phase: Literal["label"]
    score: int
    accepted: bool
    failure_reason: str | None
    is_repair_attempt: bool


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


_REQUIRED_PRICE_FIELDS = {"input_per_million", "output_per_million"}


def load_prices(path: Path) -> dict:
    """Load configs/prices.json. See spec §6 for full behavior.

    Returns a dict mapping lookup-key -> {"input_per_million": float,
                                          "output_per_million": float}.
    """
    if not path.exists():
        logger.warning(
            "Prices file not found at %s; cost reports will read $0.00", path
        )
        return {"default": {"input_per_million": 0.0, "output_per_million": 0.0}}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise MetricsConfigError(f"prices.json is not valid JSON: {e}") from e

    if not isinstance(raw, dict):
        raise MetricsConfigError("prices.json top level must be an object")

    out: dict = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            raise MetricsConfigError(
                f"prices.json[{key!r}] must be an object, got {type(entry).__name__}"
            )
        # Validate required numeric fields.
        cleaned = {}
        for field in _REQUIRED_PRICE_FIELDS:
            if field not in entry:
                raise MetricsConfigError(
                    f"prices.json[{key!r}] missing required field {field!r}"
                )
            val = entry[field]
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise MetricsConfigError(
                    f"prices.json[{key!r}].{field} must be a number, got {val!r}"
                )
            if val < 0:
                raise MetricsConfigError(
                    f"prices.json[{key!r}].{field} must be >= 0, got {val}"
                )
            cleaned[field] = float(val)
        # Unknown extra fields are dropped with a warning.
        unknown = set(entry) - _REQUIRED_PRICE_FIELDS
        for u in sorted(unknown):
            logger.warning(
                "prices.json[%r] has unknown field %r; ignoring.", key, u
            )
        out[key] = cleaned
    return out


def lookup_price(
    prices: dict, provider_type: str, model: str | None
) -> tuple[float, float]:
    """Lookup order: 'provider_type:model' -> model -> provider_type -> 'default'.

    Returns (input_per_million, output_per_million); (0.0, 0.0) if nothing matches.
    """
    keys = []
    if model:
        keys.append(f"{provider_type}:{model}")
        keys.append(model)
    keys.append(provider_type)
    keys.append("default")
    for k in keys:
        if k in prices:
            entry = prices[k]
            return (
                float(entry["input_per_million"]),
                float(entry["output_per_million"]),
            )
    return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Token estimation (fallback when SDK usage is unavailable)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str | None) -> int:
    """Cheap char/4 estimate. Returns 0 for None or empty string.

    Used only when a provider does not report SDK usage metadata.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


# ---------------------------------------------------------------------------
# Percentile math
# ---------------------------------------------------------------------------


def percentiles(values: list[float]) -> dict[str, float]:
    """Returns {"p50": ..., "p95": ...}.

    Empty list   -> {"p50": 0.0, "p95": 0.0}.
    n < 20       -> p95 = max(values); p50 = median.
    n >= 20      -> statistics.quantiles(n=20); take index 9 (p50) and 18 (p95).
    """
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    if len(values) == 1:
        return {"p50": float(values[0]), "p95": float(values[0])}
    if len(values) < 20:
        return {
            "p50": float(statistics.median(values)),
            "p95": float(max(values)),
        }
    qs = statistics.quantiles(values, n=20)
    # qs has 19 cut points: index 0 = 5th pct, index 9 = 50th, index 18 = 95th.
    return {"p50": float(qs[9]), "p95": float(qs[18])}
```

- [ ] **Step 4: Run the helper tests — expect all pass**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_metrics.py -v 2>&1 | tail -25
```
Expected: 18 passed (load_prices: 6, lookup_price: 5, estimate_tokens: 3, percentiles: 4).

If you see fewer than 18, scroll up — likely a typo in a test name. The file contains 18 cases exactly.

- [ ] **Step 5: Verify `import metrics` is dependency-free**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
import sys, metrics
loaded = [m for m in sys.modules if m.startswith(('openai','google','torch','unsloth','transformers','numpy','pandas'))]
assert not loaded, f'leaked: {loaded}'
print('import metrics: clean')
"
```
Expected: `import metrics: clean`.

- [ ] **Step 6: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: `312 passed` (294 prior + 18 new).

- [ ] **Step 7: Do NOT commit.** All commits land at Task 13.

---

### Task 2: `MetricsRecorder` class + finalize + render_stdout

**Files:**
- Modify: `scripts/metrics.py` — append `MetricsRecorder` and renderer
- Modify: `tests/test_metrics.py` — append recorder tests

Spec reference: §3.5, §4.1, §4.2.

- [ ] **Step 1: Write the failing recorder tests**

Append to `tests/test_metrics.py`:

```python
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
    assert pr["acceptance_rate"] == pytest.approx(4 / 6)
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
```

- [ ] **Step 2: Run the recorder tests — expect AttributeError**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_metrics.py -v 2>&1 | tail -20
```
Expected: the recorder tests ERROR with `ImportError: cannot import name 'MetricsRecorder' from 'metrics'`.

- [ ] **Step 3: Append `MetricsRecorder` to `scripts/metrics.py`**

Append to `scripts/metrics.py`:

```python
# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

import subprocess
import threading
from collections import Counter
from datetime import datetime, timezone


def _git_short_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:    # noqa: BLE001 — git unavailable; non-fatal
        pass
    return None


def _round4(x: float) -> float:
    return round(float(x), 4)


class MetricsRecorder:
    """Per-run accumulator. Methods are thread-safe (single internal Lock).

    finalize() must run single-threaded after all workers have joined.
    """

    def __init__(
        self,
        *,
        prices: dict,
        started_at: datetime,
        prices_source: str | None,
    ) -> None:
        self._calls: list[ProviderCallMetric] = []
        self._candidates: list[CandidateMetric] = []
        self._input_rows = 0
        self._train_rows = 0
        self._failed_rows = 0
        self._repair_exhausted = 0
        self._lock = threading.Lock()
        self._prices = prices
        self._prices_source = prices_source
        self._started_at = started_at

    # -- recording --

    def record_call(
        self,
        *,
        provider: str,
        provider_type: str,
        model: str | None,
        phase: str,
        attempt_type: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_tokens: bool,
        success: bool,
        failure_reason: str | None,
    ) -> None:
        m = ProviderCallMetric(
            provider=provider,
            provider_type=provider_type,
            model=model,
            phase=phase,
            attempt_type=attempt_type,
            latency_ms=float(latency_ms),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(prompt_tokens) + int(completion_tokens),
            estimated_tokens=bool(estimated_tokens),
            success=bool(success),
            failure_reason=failure_reason,
        )
        with self._lock:
            self._calls.append(m)

    def record_label_candidate(
        self,
        *,
        provider: str,
        model: str | None,
        score: int,
        accepted: bool,
        failure_reason: str | None,
        is_repair_attempt: bool,
    ) -> None:
        m = CandidateMetric(
            provider=provider,
            model=model,
            phase="label",
            score=int(score),
            accepted=bool(accepted),
            failure_reason=failure_reason,
            is_repair_attempt=bool(is_repair_attempt),
        )
        with self._lock:
            self._candidates.append(m)

    def record_output_row(self, kind: Literal["input", "train", "failed"]) -> None:
        with self._lock:
            if kind == "input":
                self._input_rows += 1
            elif kind == "train":
                self._train_rows += 1
            elif kind == "failed":
                self._failed_rows += 1
            else:
                raise ValueError(f"unknown output row kind: {kind!r}")

    def record_repair_exhausted(self) -> None:
        """Called once per input whose repair loop finished without accepting
        any candidate. Distinct from record_output_row('failed') because a
        failed row may or may not have entered repair."""
        with self._lock:
            self._repair_exhausted += 1

    # -- finalize --

    def finalize(
        self,
        *,
        phase: str,
        inputs_processed: int,
        finished_at: datetime | None = None,
    ) -> dict:
        finished_at = finished_at or datetime.now(timezone.utc)
        duration_ms = (finished_at - self._started_at).total_seconds() * 1000.0
        duration_ms = max(0.0, duration_ms)

        # Per-provider aggregation.
        provider_keys: list[str] = []
        seen_provider = set()
        for c in self._calls:
            if c.provider not in seen_provider:
                seen_provider.add(c.provider)
                provider_keys.append(c.provider)

        providers_out: dict = {}
        for pname in provider_keys:
            pcalls = [c for c in self._calls if c.provider == pname]
            # provider_type/model from the first call (all calls for one provider share these)
            ptype = pcalls[0].provider_type
            pmodel = pcalls[0].model
            success_calls = sum(1 for c in pcalls if c.success)
            failed_calls = sum(1 for c in pcalls if not c.success)
            prompt_tokens = sum(c.prompt_tokens for c in pcalls)
            completion_tokens = sum(c.completion_tokens for c in pcalls)
            has_est = any(c.estimated_tokens for c in pcalls)
            input_pm, output_pm = lookup_price(self._prices, ptype, pmodel)
            cost = (prompt_tokens / 1_000_000.0) * input_pm + \
                   (completion_tokens / 1_000_000.0) * output_pm
            lat = percentiles([c.latency_ms for c in pcalls])
            call_failure_rate = (failed_calls / len(pcalls)) if pcalls else 0.0

            block: dict = {
                "provider_type": ptype,
                "model": pmodel,
                "calls": len(pcalls),
                "success_calls": success_calls,
                "failed_calls": failed_calls,
                "call_failure_rate": _round4(call_failure_rate),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "has_estimated_tokens": has_est,
                "estimated_cost_usd": _round4(cost),
                "latency_ms": {
                    "p50": _round4(lat["p50"]),
                    "p95": _round4(lat["p95"]),
                },
            }

            if phase == "label":
                pcands = [c for c in self._candidates if c.provider == pname]
                accepted = sum(1 for c in pcands if c.accepted)
                rejected = sum(1 for c in pcands if not c.accepted)
                total_cands = accepted + rejected
                block["candidates_accepted"] = accepted
                block["candidates_rejected"] = rejected
                block["acceptance_rate"] = _round4(accepted / total_cands) \
                    if total_cands else 0.0

            providers_out[pname] = block

        # Totals.
        total_calls = len(self._calls)
        calls_succeeded = sum(1 for c in self._calls if c.success)
        calls_failed = total_calls - calls_succeeded
        total_cost = sum(p["estimated_cost_usd"] for p in providers_out.values())
        totals: dict = {
            "calls": total_calls,
            "calls_succeeded": calls_succeeded,
            "calls_failed": calls_failed,
            "candidates_accepted": sum(1 for c in self._candidates if c.accepted),
            "candidates_rejected": sum(1 for c in self._candidates if not c.accepted),
            "estimated_cost_usd": _round4(total_cost),
            "input_rows_written": self._input_rows,
            "train_rows_written": self._train_rows,
            "failed_rows_written": self._failed_rows,
            "has_estimated_tokens": any(c.estimated_tokens for c in self._calls),
        }

        # Failures: terminal failure_reason from calls + candidates; only non-null.
        failures: Counter = Counter()
        for c in self._calls:
            if c.failure_reason:
                failures[c.failure_reason] += 1
        for c in self._candidates:
            if c.failure_reason:
                failures[c.failure_reason] += 1

        # Throughput.
        duration_s = duration_ms / 1000.0
        throughput = {
            "inputs_per_sec": _round4(inputs_processed / duration_s) if duration_s > 0 else 0.0,
            "calls_per_sec": _round4(total_calls / duration_s) if duration_s > 0 else 0.0,
        }

        out: dict = {
            "run": {
                "started_at": self._started_at.isoformat().replace("+00:00", "Z"),
                "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
                "duration_ms": int(duration_ms),
                "phase": phase,
                "inputs_processed": int(inputs_processed),
                "git_sha": _git_short_sha(),
                "prices_source": self._prices_source,
                "throughput": throughput,
            },
            "totals": totals,
            "providers": providers_out,
            "failures": dict(failures),
        }

        # Repair block only for label phase.
        if phase == "label":
            repair_attempts = sum(
                1 for c in self._calls if c.attempt_type == "repair"
            )
            repair_accepted = sum(
                1 for c in self._candidates
                if c.is_repair_attempt and c.accepted
            )
            out["repair"] = {
                "attempts": repair_attempts,
                "accepted": repair_accepted,
                "exhausted": self._repair_exhausted,
            }

        return out

    # -- stdout summary --

    def render_stdout(self, summary: dict) -> str:
        run = summary["run"]
        totals = summary["totals"]
        providers = summary["providers"]
        phase = run["phase"]
        duration_ms = run["duration_ms"]
        mins, secs = divmod(duration_ms // 1000, 60)
        dur = f"{mins}m{secs:02d}s" if mins else f"{secs}s"

        lines = [
            f"=== Stage 5 metrics (phase={phase}, "
            f"{run['inputs_processed']} inputs, {dur}) ===",
            f"Calls: {totals['calls']} ({totals['calls_succeeded']} ok, "
            f"{totals['calls_failed']} failed)"
            + (f"  Accepted: {totals['candidates_accepted']}  "
               f"Failed rows: {totals['failed_rows_written']}"
               if phase == "label" else
               f"  Input rows: {totals['input_rows_written']}"),
        ]
        prices_source = run.get("prices_source")
        cost_label = (
            f"rates from {prices_source}" if prices_source
            else "rates from default zero pricing"
        )
        lines.append(
            f"Estimated cost: ${totals['estimated_cost_usd']:.4f} ({cost_label})"
        )
        lines.append("")

        # Per-provider table.
        if phase == "label":
            header = ["Provider", "Calls", "Ok", "Fail%", "Acc", "Cost", "p50", "p95", "Top failure"]
        else:
            header = ["Provider", "Calls", "Ok", "Fail%", "Cost", "p50", "p95"]

        rows = [header]
        # Compute per-provider top failure from summary['failures'] is global —
        # we have to recompute per-provider by walking _calls/_candidates.
        for pname, pblock in providers.items():
            if phase == "label":
                top_fail = self._top_failure_for(pname)
                rows.append([
                    pname,
                    str(pblock["calls"]),
                    str(pblock["success_calls"]),
                    f"{pblock['call_failure_rate']*100:.2f}%",
                    str(pblock.get("candidates_accepted", 0)),
                    f"${pblock['estimated_cost_usd']:.4f}",
                    f"{int(pblock['latency_ms']['p50'])}",
                    f"{int(pblock['latency_ms']['p95'])}",
                    top_fail,
                ])
            else:
                rows.append([
                    pname,
                    str(pblock["calls"]),
                    str(pblock["success_calls"]),
                    f"{pblock['call_failure_rate']*100:.2f}%",
                    f"${pblock['estimated_cost_usd']:.4f}",
                    f"{int(pblock['latency_ms']['p50'])}",
                    f"{int(pblock['latency_ms']['p95'])}",
                ])

        widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
        for row in rows:
            lines.append("  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))

        if phase == "label" and "repair" in summary:
            rep = summary["repair"]
            lines.append("")
            lines.append(
                f"Repair: {rep['attempts']} attempted, "
                f"{rep['accepted']} accepted, {rep['exhausted']} exhausted"
            )

        return "\n".join(lines)

    def _top_failure_for(self, provider: str) -> str:
        c: Counter = Counter()
        for call in self._calls:
            if call.provider == provider and call.failure_reason:
                c[call.failure_reason] += 1
        for cand in self._candidates:
            if cand.provider == provider and cand.failure_reason:
                c[cand.failure_reason] += 1
        if not c:
            return ""
        reason, n = c.most_common(1)[0]
        return f"{reason} ({n})"
```

- [ ] **Step 4: Run the recorder tests — expect all pass**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_metrics.py -v 2>&1 | tail -30
```
Expected: all metrics tests pass (18 helpers + 13 recorder = 31 in `test_metrics.py`).

- [ ] **Step 5: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: `325 passed` (294 + 31).

- [ ] **Step 6: Do NOT commit.**

---

### Task 3: Seed `configs/prices.json` + `.gitignore`

**Files:**
- Create: `configs/prices.json`
- Modify: `.gitignore`

Spec reference: §6, §7.4.

- [ ] **Step 1: Create `configs/prices.json`**

Create `configs/prices.json` with this exact content:

```json
{
  "deepseek:deepseek-chat": {"input_per_million": 0.27, "output_per_million": 1.10},
  "gemini:gemini-2.5-flash": {"input_per_million": 0.30, "output_per_million": 2.50},
  "local_teacher": {"input_per_million": 0.0, "output_per_million": 0.0},
  "fake": {"input_per_million": 0.0, "output_per_million": 0.0},
  "default": {"input_per_million": 0.0, "output_per_million": 0.0}
}
```

- [ ] **Step 2: Verify the seed config loads under the loader**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
from pathlib import Path
from metrics import load_prices, lookup_price
prices = load_prices(Path('configs/prices.json'))
print('deepseek:deepseek-chat ->', lookup_price(prices, 'deepseek', 'deepseek-chat'))
print('gemini:gemini-2.5-flash ->', lookup_price(prices, 'gemini', 'gemini-2.5-flash'))
print('unknown ->', lookup_price(prices, 'unknown', 'mystery'))
"
```
Expected:
```
deepseek:deepseek-chat -> (0.27, 1.1)
gemini:gemini-2.5-flash -> (0.3, 2.5)
unknown -> (0.0, 0.0)
```

- [ ] **Step 3: Check current `.gitignore`**

```bash
cd "C:/work/llm training" && grep -n "data/distill" .gitignore || echo "no data/distill entry"
```

If `data/distill/*` is already covered by a glob (e.g. `data/distill/*.jsonl`), `metrics.json` would not match that glob. The spec requires `data/distill/metrics.json` to be explicitly ignored. Add it regardless.

- [ ] **Step 4: Append the metrics.json ignore line**

Append `data/distill/metrics.json` to `.gitignore` (one new line):

```bash
cd "C:/work/llm training" && echo "data/distill/metrics.json" >> .gitignore
```

Verify:
```bash
cd "C:/work/llm training" && tail -3 .gitignore
```
Expected: the new line appears at the end.

- [ ] **Step 5: Verify a fake metrics.json is ignored**

```bash
cd "C:/work/llm training" && touch data/distill/metrics.json && git status --short data/distill/metrics.json
```
Expected: no output (file is ignored). Then clean up:
```bash
cd "C:/work/llm training" && rm data/distill/metrics.json
```

- [ ] **Step 6: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: still `325 passed` (no test changes).

- [ ] **Step 7: Do NOT commit.**

---

### Task 4: DeepSeek + Gemini stash SDK usage per-thread

**Files:**
- Modify: `scripts/llm_providers.py`
- Modify: `tests/test_real_providers.py` (or add small test file if cleaner — see Step 1)

Spec reference: §5.1.

**Semantics note:** usage is stashed AFTER a successful SDK response object is returned, just before `return resp.choices[0].message.content` (DeepSeek) / `return resp.text` (Gemini). If the SDK raises BEFORE returning a response object, `pop_last_usage` returns whatever was stashed by the previous successful call on the same thread — which is why each `pop_last_usage` immediately follows the `try/finally` in the orchestrator and resets the slot to `None`. Acceptable Slice 4 limitation: if our own response processing raises AFTER the SDK returns but before `_stash_usage` runs, metrics fall back to estimated tokens. We do not introduce a deeper try/finally inside `_call_api_*` for this — the cost is one mis-attributed call, the benefit isn't worth it.

- [ ] **Step 1: Write failing tests for `pop_last_usage`**

Add these tests to `tests/test_real_providers.py` (append at end):

```python
def test_deepseek_pop_last_usage_returns_sdk_usage(monkeypatch, patch_sdk_clients):
    """DeepSeek stashes usage from resp.usage and pop_last_usage returns it."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")

    class _FakeUsage:
        prompt_tokens = 123
        completion_tokens = 45

    class _FakeChoice:
        class message:
            content = '{"transactions": []}'

    class _FakeResp:
        usage = _FakeUsage()
        choices = [_FakeChoice()]

    from llm_providers import DeepSeekProvider
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    p._client.chat.completions.create = lambda **kw: _FakeResp()  # type: ignore[attr-defined]
    _ = p.generate_label("foo")
    assert p.pop_last_usage() == {"prompt_tokens": 123, "completion_tokens": 45}
    # Second pop returns None.
    assert p.pop_last_usage() is None


def test_deepseek_pop_last_usage_handles_missing_usage(monkeypatch, patch_sdk_clients):
    """If SDK response lacks usage, pop_last_usage returns None."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")

    class _FakeChoice:
        class message:
            content = '{"transactions": []}'

    class _FakeResp:
        choices = [_FakeChoice()]
        # No `usage` attribute.

    from llm_providers import DeepSeekProvider
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    p._client.chat.completions.create = lambda **kw: _FakeResp()  # type: ignore[attr-defined]
    _ = p.generate_label("foo")
    assert p.pop_last_usage() is None


def test_gemini_pop_last_usage_returns_sdk_usage(monkeypatch, patch_sdk_clients):
    """Gemini stashes usage from resp.usage_metadata."""
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    class _FakeMeta:
        prompt_token_count = 80
        candidates_token_count = 20

    class _FakeResp:
        usage_metadata = _FakeMeta()
        text = '{"transactions": []}'

    from llm_providers import GeminiProvider
    p = GeminiProvider(name="g", model="gemini-2.5-flash")
    p._client.models.generate_content = lambda **kw: _FakeResp()  # type: ignore[attr-defined]
    _ = p.generate_label("foo")
    assert p.pop_last_usage() == {"prompt_tokens": 80, "completion_tokens": 20}


def test_gemini_pop_last_usage_handles_missing_usage(monkeypatch, patch_sdk_clients):
    """If Gemini SDK response lacks usage_metadata, pop_last_usage returns None."""
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    class _FakeResp:
        text = '{"transactions": []}'
        # No usage_metadata attribute.

    from llm_providers import GeminiProvider
    p = GeminiProvider(name="g", model="gemini-2.5-flash")
    p._client.models.generate_content = lambda **kw: _FakeResp()  # type: ignore[attr-defined]
    _ = p.generate_label("foo")
    assert p.pop_last_usage() is None


def test_fake_provider_instance_has_no_pop_last_usage(tmp_path):
    """FakeProvider deliberately does not expose pop_last_usage; orchestrator
    uses getattr fallback. Check on an INSTANCE so dynamic attrs are caught."""
    from llm_providers import FakeProvider
    # Construct with a minimal fixture (one line is enough for instantiation).
    fixture = tmp_path / "fake_inputs.jsonl"
    fixture.write_text('{"input":"hello"}\n', encoding="utf-8")
    # FakeProvider's constructor signature varies between input/output use —
    # use whichever form your test_fake_provider.py already exercises. The
    # simplest hermetic construction:
    try:
        p = FakeProvider(name="fake", fixture_inputs=fixture)
    except TypeError:
        # If FakeProvider uses fixture_labels in this codebase variant,
        # fall back to that.
        fixture2 = tmp_path / "fake_labels.jsonl"
        fixture2.write_text(
            '{"input":"hello","output":{"transactions":[]}}\n', encoding="utf-8",
        )
        p = FakeProvider(name="fake", fixture_labels=fixture2)
    assert not hasattr(p, "pop_last_usage")
```

- [ ] **Step 2: Run the new tests — expect AttributeError**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_real_providers.py -v -k "pop_last_usage or no_pop_last_usage" 2>&1 | tail -10
```
Expected: 4 of 5 tests fail with `AttributeError: 'DeepSeekProvider' object has no attribute 'pop_last_usage'` (or similar). The `test_fake_provider_has_no_pop_last_usage_attribute` should pass right away.

- [ ] **Step 3: Add `pop_last_usage` + `_stash_usage` to DeepSeekProvider**

In `scripts/llm_providers.py`, in `DeepSeekProvider.__init__`, AFTER the line `self._sleep = time.sleep` (around line 347), add this block:

```python
        self._tls = __import__("threading").local()
```

(We use `__import__` inline to avoid moving the import to module top; threading is stdlib and cheap, but we want minimal diff.)

Actually, simpler: add `import threading` to the file's top-level imports (it's stdlib and likely already imported elsewhere in this file from Slice 3). Check first:

```bash
cd "C:/work/llm training" && grep -n "^import threading" scripts/llm_providers.py
```

If present, just use `self._tls = threading.local()`. If absent, add `import threading` near the other top imports.

The init change (using `threading.local()` cleanly):

```python
        self._tls = threading.local()
```

Then add these methods inside `DeepSeekProvider` (anywhere after `__init__`, before `_call_api_messages`):

```python
    def pop_last_usage(self) -> dict | None:
        u = getattr(self._tls, "usage", None)
        self._tls.usage = None
        return u

    def _stash_usage(self, prompt_tokens=None, completion_tokens=None) -> None:
        if prompt_tokens is None and completion_tokens is None:
            self._tls.usage = None
        else:
            self._tls.usage = {
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
            }
```

In `_call_api_messages` (line ~386), at the END of the method just before `return`, add:

```python
        usage = getattr(resp, "usage", None)
        self._stash_usage(
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
        return resp.choices[0].message.content or ""
```

In `_call_api` (line ~409), at the END of the method just before `return`, add the same block:

```python
        usage = getattr(resp, "usage", None)
        self._stash_usage(
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
        return resp.choices[0].message.content or ""
```

- [ ] **Step 4: Add `pop_last_usage` + `_stash_usage` to GeminiProvider**

In `GeminiProvider.__init__` (line ~469), AFTER `self._sleep = time.sleep`:

```python
        self._tls = threading.local()
```

Add the same `pop_last_usage` and `_stash_usage` methods inside `GeminiProvider`:

```python
    def pop_last_usage(self) -> dict | None:
        u = getattr(self._tls, "usage", None)
        self._tls.usage = None
        return u

    def _stash_usage(self, prompt_tokens=None, completion_tokens=None) -> None:
        if prompt_tokens is None and completion_tokens is None:
            self._tls.usage = None
        else:
            self._tls.usage = {
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
            }
```

In `_call_api_label` (line ~523), at the END just before `return resp.text or ""`, add:

```python
        usage = getattr(resp, "usage_metadata", None)
        self._stash_usage(
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )
        return resp.text or ""
```

In `_call_api` (line ~553), at the END just before `return resp.text or ""`, add the same block:

```python
        usage = getattr(resp, "usage_metadata", None)
        self._stash_usage(
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )
        return resp.text or ""
```

- [ ] **Step 5: Run the pop_last_usage tests — expect all pass**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_real_providers.py -v -k "pop_last_usage or no_pop_last_usage" 2>&1 | tail -15
```
Expected: 5 passed.

- [ ] **Step 6: Verify `import llm_providers` is still SDK-free**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
import sys, llm_providers
loaded = [m for m in sys.modules if m.startswith(('openai','google','torch','unsloth','transformers'))]
assert not loaded, f'leaked: {loaded}'
print('import llm_providers: still clean')
"
```
Expected: `import llm_providers: still clean`.

- [ ] **Step 7: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: `330 passed` (325 + 5).

- [ ] **Step 8: Do NOT commit.**

---

### Task 5: Orchestrator instrumentation — label phase

**Files:**
- Modify: `scripts/05_generate_distillation_data.py`

Spec reference: §5.2, §5.3, §7.2.

The label phase touches `_try_one_attempt`, `_process_one_input`, and `phase_label_multi_provider`. Recorder threads in via an optional kwarg.

- [ ] **Step 1: Add recorder parameter to `_try_one_attempt`**

Open `scripts/05_generate_distillation_data.py`. Find `def _try_one_attempt(` (around line 467) and add three new keyword parameters: `recorder`, `provider_type`, and `attempt_type`. The new signature:

```python
def _try_one_attempt(
    input_text: str,
    provider,
    pcfg,
    *,
    is_repair: bool = False,
    failure_summary: str = "",
    parser_candidates: list[dict] | None = None,
    provider_priority_rank: int | None = None,
    recorder=None,                        # NEW
    provider_type: str = "",              # NEW
):
    """Run one provider attempt; validate; score. Never raises."""
    from label_selection import CandidateOutcome, score_candidate, build_repair_prompt
    from metrics import estimate_tokens
    import time
```

(Note: `attempt_type` is derived from `is_repair`; we don't need it as a separate param.)

- [ ] **Step 2: Wrap the provider call with timing + metrics capture**

Replace the existing `try / except Exception as e:` block inside `_try_one_attempt` (currently around lines 480-502) with this wrapped version:

```python
    start = time.perf_counter()
    usage = None
    if is_repair:
        prompt_text = build_repair_prompt(
            input_text,
            candidates=parser_candidates,
            failure_summary=failure_summary,
        )
    else:
        prompt_text = input_text
    try:
        if is_repair:
            raw = provider.generate_label(prompt_text)
        else:
            raw = provider.generate_label(input_text)
    except Exception as e:    # noqa: BLE001 — unify all provider failures
        latency_ms = (time.perf_counter() - start) * 1000.0
        pop = getattr(provider, "pop_last_usage", None)
        usage = pop() if pop else None
        if recorder is not None:
            if usage:
                pt, ct, est = usage["prompt_tokens"], usage["completion_tokens"], False
            else:
                pt = estimate_tokens(prompt_text)
                ct = 0
                est = True
            recorder.record_call(
                provider=pcfg.name,
                provider_type=provider_type,
                model=getattr(provider, "model", None),
                phase="label",
                attempt_type="repair" if is_repair else "initial",
                latency_ms=latency_ms,
                prompt_tokens=pt,
                completion_tokens=ct,
                estimated_tokens=est,
                success=False,
                failure_reason="provider_error",
            )
        return CandidateOutcome(
            provider=pcfg.name,
            model=getattr(provider, "model", None),
            raw_output="",
            parsed_output=None,
            validation=None,
            failure_reason="provider_error",
            score=0,
            is_repair_attempt=is_repair,
            provider_priority_rank=provider_priority_rank,
            error=repr(e),
        )
    latency_ms = (time.perf_counter() - start) * 1000.0
    pop = getattr(provider, "pop_last_usage", None)
    usage = pop() if pop else None
    if recorder is not None:
        if usage:
            pt, ct, est = usage["prompt_tokens"], usage["completion_tokens"], False
        else:
            pt = estimate_tokens(prompt_text)
            ct = estimate_tokens(raw)
            est = True
        recorder.record_call(
            provider=pcfg.name,
            provider_type=provider_type,
            model=getattr(provider, "model", None),
            phase="label",
            attempt_type="repair" if is_repair else "initial",
            latency_ms=latency_ms,
            prompt_tokens=pt,
            completion_tokens=ct,
            estimated_tokens=est,
            success=True,
            failure_reason=None,
        )
```

(This replaces the original 22-line `try / except` block with a ~50-line version that adds metrics capture. The rest of `_try_one_attempt` — the post-processing block from `try: parsed = extract_json(raw)` onwards — is unchanged.)

- [ ] **Step 3: Add recorder parameter to `_process_one_input` and pass it through**

Find `def _process_one_input(` (around line 596). Add `recorder=None` as a new keyword parameter and pass it into all `_try_one_attempt(...)` calls.

New signature:

```python
def _process_one_input(
    input_text: str,
    *,
    providers_by_name: dict,
    pcfgs_by_name: dict,
    scheduler,
    priority_map: dict,
    provider_limits: dict,
    cfg,
    parser_candidates: list[dict],
    recorder=None,                        # NEW
):
```

Inside the loop, modify the two `_try_one_attempt` call sites to pass `recorder=recorder, provider_type=pcfgs_by_name[provider_name].provider_type`:

```python
    for _ in range(cfg.output_generation.label_attempts_per_input):
        provider_name = scheduler.next_provider()
        with provider_limits[provider_name]:
            outcomes.append(_try_one_attempt(
                input_text,
                provider=providers_by_name[provider_name],
                pcfg=pcfgs_by_name[provider_name],
                provider_priority_rank=priority_map.get(provider_name),
                recorder=recorder,
                provider_type=pcfgs_by_name[provider_name].provider_type,
            ))
```

And the repair call:

```python
    if repair_enabled:
        repair_provider_name = _highest_priority_provider(cfg, providers_by_name)
        for _ in range(cfg.validation.max_repair_attempts):
            with provider_limits[repair_provider_name]:
                repair_outcome = _try_one_attempt(
                    input_text,
                    provider=providers_by_name[repair_provider_name],
                    pcfg=pcfgs_by_name[repair_provider_name],
                    is_repair=True,
                    failure_summary=_summarize_failures(outcomes),
                    parser_candidates=parser_candidates,
                    provider_priority_rank=priority_map.get(repair_provider_name),
                    recorder=recorder,
                    provider_type=pcfgs_by_name[repair_provider_name].provider_type,
                )
            outcomes.append(repair_outcome)
            if repair_outcome.failure_reason is None:
                return (input_text, repair_outcome, outcomes)

    return (input_text, None, outcomes)
```

**Note on recording semantics:** worker threads return `CandidateOutcome` objects via `_process_one_input`. The main thread (inside `phase_label_multi_provider`) records all candidate metrics after best selection, so exactly one outcome can be marked `accepted=True` and the valid-but-not-selected outcomes get `accepted=False, failure_reason=None`. This is the place `record_label_candidate` is called from — never from inside `_try_one_attempt` itself, which only records the provider call.

- [ ] **Step 4: Add recorder parameter to `phase_label_multi_provider`, thread through, return int**

Find `def phase_label_multi_provider(args: argparse.Namespace) -> None:` (around line 740). Change to:

```python
def phase_label_multi_provider(args: argparse.Namespace, recorder=None) -> int:
```

At every early-`return` inside the function, change `return` to `return 0`. Specifically:

- `return` after the `output_generation.enabled=False` log: → `return 0`
- `return` after the `if not pending:` log: → `return 0`

In the body, pass `recorder=recorder` into `pool.submit(_process_one_input, ...)`:

```python
        futures = {
            pool.submit(
                _process_one_input,
                inp,
                providers_by_name=providers_by_name,
                pcfgs_by_name=pcfgs_by_name,
                scheduler=scheduler,
                priority_map=priority_map,
                provider_limits=provider_limits,
                cfg=cfg,
                parser_candidates=candidates_by_input[inp],
                recorder=recorder,
            ): inp
            for inp in pending
        }
```

After the `train_out.write(...)` line (acceptance case), add a `record_label_candidate(accepted=True, ...)` call followed by `record_output_row("train")`. After the `fail_out.write(...)` line (failure case), add `record_label_candidate` calls for each candidate in `all_outcomes` (except provider_error ones), then `record_output_row("failed")`.

Locate the block starting at `for fut in concurrent.futures.as_completed(futures):` (around line 824). Replace its body with:

```python
        for fut in concurrent.futures.as_completed(futures):
            input_text, best, all_outcomes = fut.result()
            for o in all_outcomes:
                if o.failure_reason:
                    attempt_failure_reasons[o.failure_reason] += 1

            # Emit label-candidate metrics for every outcome EXCEPT provider_error
            # (provider_error is already accounted for via record_call).
            if recorder is not None:
                for o in all_outcomes:
                    if o.failure_reason == "provider_error":
                        continue
                    recorder.record_label_candidate(
                        provider=o.provider,
                        model=o.model,
                        score=o.score,
                        accepted=(best is not None and o is best),
                        failure_reason=o.failure_reason,
                        is_repair_attempt=o.is_repair_attempt,
                    )

            if best is not None:
                row = {
                    "input": input_text,
                    "output": best.parsed_output,
                    "_source": "multi_provider_label",
                    "_provider": best.provider,
                    "_model": best.model,
                    "_validation_score": best.score,
                    "_attempts": len(all_outcomes),
                }
                train_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_kept += 1
                if best.is_repair_attempt:
                    n_repaired += 1
                if recorder is not None:
                    recorder.record_output_row("train")
            else:
                repair_enabled = (
                    cfg.validation.retry_invalid_with_stricter_prompt
                    and cfg.validation.max_repair_attempts > 0
                )
                if all(o.failure_reason == "provider_error" for o in all_outcomes):
                    top_reason = "provider_error"
                else:
                    top_reason = "repair_exhausted" if repair_enabled else "all_candidates_failed"
                failed_row = {
                    "input": input_text,
                    "reason": top_reason,
                    "candidates": candidates_by_input[input_text],
                    "attempts": [_serialize_outcome(o) for o in all_outcomes],
                }
                fail_out.write(json.dumps(failed_row, ensure_ascii=False) + "\n")
                n_failed += 1
                failure_reasons[top_reason] += 1
                if recorder is not None:
                    recorder.record_output_row("failed")
                    # Explicit repair-exhausted accounting: only count inputs
                    # that actually entered the repair loop.
                    if any(o.is_repair_attempt for o in all_outcomes):
                        recorder.record_repair_exhausted()
            if (n_kept + n_failed) % cfg.rate_limits.write_flush_every == 0:
                train_out.flush()
                fail_out.flush()
            bar.update(1)

        train_out.flush()
        fail_out.flush()
```

At the very end of `phase_label_multi_provider` (after the `logging.info` summary block), add `return n_kept + n_failed`:

```python
    keep_rate = (n_kept / max(1, n_kept + n_failed)) * 100
    logging.info(
        "Phase label (multi-provider) done. kept=%d (repair=%d) failed=%d keep_rate=%.1f%%",
        n_kept, n_repaired, n_failed, keep_rate,
    )
    if failure_reasons:
        logging.info("Top-level failure reasons: %s", dict(failure_reasons))
    if attempt_failure_reasons:
        logging.info("Attempt-level failure reasons: %s", dict(attempt_failure_reasons))
    return n_kept + n_failed
```

- [ ] **Step 5: Run all existing label-phase tests — expect all pass (recorder=None is the default)**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_multi_provider_labels.py tests/test_label_orchestrator.py -q 2>&1 | tail -5
```
Expected: same pass count as before (19 tests = 9 + 10).

- [ ] **Step 6: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: `330 passed`.

- [ ] **Step 7: Do NOT commit.**

---

### Task 6: Orchestrator instrumentation — input phase

**Files:**
- Modify: `scripts/05_generate_distillation_data.py`

Spec reference: §5.2, §7.2.

- [ ] **Step 1: Add recorder parameter to `_provider_worker`**

Find `def _provider_worker(` (around line 936). Add `recorder=None` and `provider_type: str = ""` as new keyword parameters.

New signature:

```python
def _provider_worker(
    *,
    provider,
    pcfg,
    batch_size: int,
    results_queue: "queue.Queue",
    stop_event: threading.Event,
    recorder=None,                       # NEW
    provider_type: str = "",             # NEW
) -> None:
```

Inside the worker, wrap the `provider.generate_inputs(prompt, batch_size)` call with timing + recording:

Replace the block that contains the existing call:
```python
        try:
            lines = provider.generate_inputs(prompt, batch_size)
        except Exception as e:
            logging.error("Provider %s batch failed: %s", pcfg.name, e)
            error_streak += 1
            continue
```

With:

```python
        import time
        from metrics import estimate_tokens as _est
        start = time.perf_counter()
        try:
            lines = provider.generate_inputs(prompt, batch_size)
        except Exception as e:    # noqa: BLE001 — surface any provider failure as an error streak
            latency_ms = (time.perf_counter() - start) * 1000.0
            pop = getattr(provider, "pop_last_usage", None)
            usage = pop() if pop else None
            if recorder is not None:
                pt = (usage or {}).get("prompt_tokens", _est(prompt))
                ct = (usage or {}).get("completion_tokens", 0)
                recorder.record_call(
                    provider=pcfg.name,
                    provider_type=provider_type,
                    model=getattr(provider, "model", None),
                    phase="inputs",
                    attempt_type="input_batch",
                    latency_ms=latency_ms,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    estimated_tokens=usage is None,
                    success=False,
                    failure_reason="provider_error",
                )
            logging.error("Provider %s batch failed: %s", pcfg.name, e)
            error_streak += 1
            continue
        latency_ms = (time.perf_counter() - start) * 1000.0
        pop = getattr(provider, "pop_last_usage", None)
        usage = pop() if pop else None
        if recorder is not None:
            if usage:
                pt, ct, est = usage["prompt_tokens"], usage["completion_tokens"], False
            else:
                pt = _est(prompt)
                # Estimate completion by total chars across returned lines.
                ct = _est("\n".join(lines))
                est = True
            recorder.record_call(
                provider=pcfg.name,
                provider_type=provider_type,
                model=getattr(provider, "model", None),
                phase="inputs",
                attempt_type="input_batch",
                latency_ms=latency_ms,
                prompt_tokens=pt,
                completion_tokens=ct,
                estimated_tokens=est,
                success=True,
                failure_reason=None,
            )
```

(The `import time` and `from metrics import estimate_tokens as _est` lines should go at the top of the function on first entry; you can hoist them to the function head if cleaner, but inline-imports inside the loop body work fine and isolate the metrics dependency.)

Cleanup recommendation: hoist `import time` and `from metrics import estimate_tokens as _est` to the very top of `_provider_worker` so they execute once per worker, not once per iteration.

- [ ] **Step 2: Add recorder parameter to `phase_inputs_multi_provider`, thread through, return int**

Find `def phase_inputs_multi_provider(args: argparse.Namespace) -> None:` (around line 991). Change to:

```python
def phase_inputs_multi_provider(args: argparse.Namespace, recorder=None) -> int:
```

Change each `return` (the early-exits when `not ig.enabled` and `pending == 0`) to `return 0`.

In the worker-spawn loop, pass `recorder` and `provider_type` into `_provider_worker`:

```python
        for pcfg in ig.providers:
            q = quota[pcfg.name]
            if q <= 0:
                continue
            n_workers = min(pcfg.threads, max(1, math.ceil(q / ig.batch_size)))
            for _ in range(n_workers):
                futures.append(pool.submit(
                    _provider_worker,
                    provider=providers[pcfg.name],
                    pcfg=pcfg,
                    batch_size=ig.batch_size,
                    results_queue=results_queue,
                    stop_event=stop_event,
                    recorder=recorder,
                    provider_type=pcfg.provider_type,
                ))
```

When a row is accepted in the main loop, add `recorder.record_output_row("input")`. Find:

```python
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            accepted_total += 1
```

And add after:

```python
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            accepted_total += 1
            if recorder is not None:
                recorder.record_output_row("input")
```

At the end of `phase_inputs_multi_provider`, after the closing `logging.info(...)`, add:

```python
    return accepted_total
```

**Critical:** `accepted_total` is the count of unique rows written THIS RUN — not `pending`, not `ig.target_inputs`, not `len(existing_inputs)`. The E2E test in Task 10 asserts `input_rows_written == target_inputs` for a fresh run; if `accepted_total` drifts, the E2E will fail with a misleading off-by-one. Also confirm every `fout.write(...)` is followed by an `if recorder is not None: recorder.record_output_row("input")` call so the two counters (return value vs recorder counter) stay in lockstep.

- [ ] **Step 3: Run all existing input-phase tests — expect all pass**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_multi_provider_inputs.py -q 2>&1 | tail -3
```
Expected: same count as before (whatever the current count is; recorder defaults to `None`, so behavior is identical).

- [ ] **Step 4: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: `330 passed`.

- [ ] **Step 5: Do NOT commit.**

---

### Task 7: Wire recorder lifecycle in `main()`

**Files:**
- Modify: `scripts/05_generate_distillation_data.py`

Spec reference: §7.1, §7.3.

- [ ] **Step 1: Add imports needed for the recorder lifecycle**

At the top of `scripts/05_generate_distillation_data.py`, ensure these imports are present (they may already be):

```python
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
```

`datetime` and `timezone` are likely already imported; check:

```bash
cd "C:/work/llm training" && grep -n "from datetime" scripts/05_generate_distillation_data.py
```

If `timezone` is missing from the existing `from datetime import` line, add it.

- [ ] **Step 2: Update `main()` to create + thread + finalize the recorder**

Find `def main() -> int:` (around line 1261). Locate the multi-provider branch:

```python
    if args.multi_provider:
        assert args.phase in {"inputs", "label"}
        assert args.provider_config is not None
        if args.phase == "inputs":
            phase_inputs_multi_provider(args)
        else:
            phase_label_multi_provider(args)
        logging.info("Stage 5 done.")
        return 0
```

Replace it with:

```python
    if args.multi_provider:
        assert args.phase in {"inputs", "label"}
        assert args.provider_config is not None
        from metrics import MetricsRecorder, load_prices
        prices_path = REPO_ROOT / "configs" / "prices.json"
        prices = load_prices(prices_path)
        recorder = MetricsRecorder(
            prices=prices,
            started_at=datetime.now(timezone.utc),
            prices_source=str(prices_path.relative_to(REPO_ROOT))
                          if prices_path.exists() else None,
        )
        if args.phase == "inputs":
            processed = phase_inputs_multi_provider(args, recorder=recorder)
        else:
            processed = phase_label_multi_provider(args, recorder=recorder)
        summary = recorder.finalize(
            phase=args.phase,
            inputs_processed=processed,
            finished_at=datetime.now(timezone.utc),
        )
        metrics_path = DISTILL_DIR / "metrics.json"
        metrics_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for line in recorder.render_stdout(summary).splitlines():
            logging.info(line)
        logging.info("Stage 5 done.")
        return 0
```

Note: the `--dry-run-quota` path is handled earlier in `main()` (`if args.dry_run_quota:` returns before reaching the `if args.multi_provider:` branch). No change needed there — dry-run still gets no recorder.

- [ ] **Step 3: Sanity-check the change**

```bash
cd "C:/work/llm training" && python -c "
import ast, pathlib
src = pathlib.Path('scripts/05_generate_distillation_data.py').read_text(encoding='utf-8')
tree = ast.parse(src)
print('parses cleanly')
"
```
Expected: `parses cleanly`.

- [ ] **Step 4: Quick smoke: run --phase label with seeded inputs**

The label phase needs `inputs_raw.jsonl` to have something to process. Seed it from a fixture row so FakeProvider has a matching label.

```bash
cd "C:/work/llm training" && rm -rf /tmp/s4_smoke && mkdir -p /tmp/s4_smoke && \
  python -c "
import json, pathlib
fixture = pathlib.Path('tests/fixtures/fake_labels.jsonl').read_text(encoding='utf-8')
row = json.loads(fixture.splitlines()[0])
pathlib.Path('/tmp/s4_smoke/inputs_raw.jsonl').write_text(
    json.dumps({'input': row['input']}) + '\n', encoding='utf-8',
)
" && \
  DISTILL_DIR_OVERRIDE=/tmp/s4_smoke python scripts/05_generate_distillation_data.py --phase label --provider-config configs/test_providers.json --multi-provider --limit 2 2>&1 | tail -20
```
Expected: completes with exit 0; logs show `=== Stage 5 metrics (phase=label, ...)`; `/tmp/s4_smoke/metrics.json` exists.

```bash
ls /tmp/s4_smoke/ && head -40 /tmp/s4_smoke/metrics.json
```
Expected: `metrics.json` present; content includes `"run"`, `"totals"`, `"providers"`, `"failures"`, `"repair"` keys; at least one provider has `"calls" > 0`.

- [ ] **Step 5: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: `330 passed`.

- [ ] **Step 6: Do NOT commit.**

---

### Task 8: Orchestrator unit test — provider-error invariant

**Files:**
- Modify: `tests/test_label_orchestrator.py`

Spec reference: §8.2.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_label_orchestrator.py`:

```python
def test_provider_error_records_call_failure_only():
    """The provider-error path records record_call(success=False) and does
    NOT call record_label_candidate. Keystone of the no-double-count invariant.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

    from datetime import datetime, timezone
    from metrics import MetricsRecorder
    from llm_providers import ProviderError

    # Spying recorder.
    class _SpyRecorder(MetricsRecorder):
        def __init__(self):
            super().__init__(
                prices={"default": {"input_per_million": 0.0, "output_per_million": 0.0}},
                started_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
                prices_source=None,
            )
            self.call_log = []
            self.candidate_log = []
        def record_call(self, **kw):
            self.call_log.append(kw)
            super().record_call(**kw)
        def record_label_candidate(self, **kw):
            self.candidate_log.append(kw)
            super().record_label_candidate(**kw)

    class _RaisingProvider:
        name = "boom"
        model = "m"
        def generate_label(self, _):
            raise ProviderError("simulated failure")

    class _Pcfg:
        name = "boom"
        provider_type = "fake"

    from importlib import import_module
    stage5 = import_module("05_generate_distillation_data")

    rec = _SpyRecorder()
    outcome = stage5._try_one_attempt(
        "input text",
        provider=_RaisingProvider(),
        pcfg=_Pcfg(),
        recorder=rec,
        provider_type="fake",
    )
    assert outcome.failure_reason == "provider_error"
    assert len(rec.call_log) == 1
    assert rec.call_log[0]["success"] is False
    assert rec.call_log[0]["failure_reason"] == "provider_error"
    assert len(rec.candidate_log) == 0
```

- [ ] **Step 2: Run the test — expect pass**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_label_orchestrator.py::test_provider_error_records_call_failure_only -v 2>&1 | tail -10
```
Expected: PASS.

If FAIL with "module name starts with digit" — `import_module("05_generate_distillation_data")` is necessary because the module name starts with `05`; this is what the test already uses inside this file (check existing tests in `test_label_orchestrator.py` for the pattern they use).

- [ ] **Step 3: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: `331 passed` (330 + 1).

- [ ] **Step 4: Do NOT commit.**

---

### Task 9: E2E tests — label phase metrics

**Files:**
- Modify: `tests/test_multi_provider_labels.py`

Spec reference: §8.3.

- [ ] **Step 1: Append two E2E tests**

Append to `tests/test_multi_provider_labels.py`:

```python
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
```

- [ ] **Step 2: Run the new tests**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_multi_provider_labels.py -v -k "metrics" 2>&1 | tail -10
```
Expected: 2 passed.

- [ ] **Step 3: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: `333 passed` (331 + 2).

- [ ] **Step 4: Do NOT commit.**

---

### Task 10: E2E test — inputs phase metrics

**Files:**
- Modify: `tests/test_multi_provider_inputs.py`

Spec reference: §8.3.

- [ ] **Step 1: Check current structure**

```bash
cd "C:/work/llm training" && head -50 tests/test_multi_provider_inputs.py
```
Note the imports + helper function names (e.g., `_make_inputs_config`, `_run_inputs`). The new test will follow the same convention.

- [ ] **Step 2: Append the new test**

Append to `tests/test_multi_provider_inputs.py`. If the file uses `_run_inputs(...)` / `_make_inputs_config(...)`, reuse those. Otherwise adapt the pattern below.

```python
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
```

(If `test_multi_provider_inputs.py` already imports `os`, `subprocess`, `sys`, `json`, `REPO_ROOT`, `SCRIPT` — it should, as it's structured like `test_multi_provider_labels.py` — you don't need to re-import. If anything is missing, add it to the imports at the top of the file.)

- [ ] **Step 3: Run the new test**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_multi_provider_inputs.py::test_inputs_phase_writes_metrics_json -v 2>&1 | tail -15
```
Expected: PASS.

If it FAILS because the fixture path is wrong or helper names differ, inspect `tests/test_multi_provider_inputs.py` more carefully and adapt the test to its conventions.

- [ ] **Step 4: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: `334 passed` (333 + 1 from inputs E2E). Exact count may drift if Task 12 adds a coverage gap-filler test — treat as a smoke signal, not a hard blocker.

- [ ] **Step 5: Do NOT commit.**

---

### Task 11: README + provider_config.md docs

**Files:**
- Modify: `README.md`
- Modify: `docs/provider_config.md`

Spec reference: §9.

- [ ] **Step 1: Append the Slice 4 README subsection**

Open `README.md`. Find the heading `### Multi-provider output labeling (Slice 3)`. After that subsection (look for the next `## ` or `### ` heading), APPEND:

```markdown
### Per-run metrics (Slice 4)

Every multi-provider Stage 5 run emits `data/distill/metrics.json` with per-
provider call counts, accepted/rejected candidates, latency (p50/p95), token
counts, and estimated USD cost. A compact summary is logged at the end of the
run:

```
=== Stage 5 metrics (phase=label, 100 inputs, 2m41s) ===
Calls: 213 (198 ok, 15 failed)  Accepted: 87  Failed rows: 13
Estimated cost: $0.0342 (rates from configs/prices.json)

Provider          Calls  Ok    Fail%   Acc   Cost      p50    p95    Top failure
gemini_flash      108    105   2.78%   52    $0.0283   812    1421   validation_failed (7)
deepseek_v4_pro   105    93    11.43%  35    $0.0059   1104   2210   provider_error (12)

Repair: 12 attempted, 5 accepted, 7 exhausted
```

Costs are estimates based on `configs/prices.json` at run time. The script
does not fetch live pricing — update `configs/prices.json` to match current
provider rates if you care about USD accuracy.

Metrics fire only for `--multi-provider --phase {inputs,label}`. Legacy
single-provider phases, `--phase eval`, `--phase all`, and `--dry-run-quota`
produce no metrics file.
```

- [ ] **Step 2: Append a `## Metrics` section to `docs/provider_config.md`**

Open `docs/provider_config.md`. At the very end of the file, append:

```markdown

## Metrics (Slice 4)

Multi-provider runs (`--multi-provider --phase {inputs,label}`) emit
`data/distill/metrics.json` and log a compact stdout summary.

Per-model USD prices come from `configs/prices.json`. The lookup order is:

1. `<provider_type>:<model>` (e.g., `gemini:gemini-2.5-flash`)
2. `<model>` (e.g., `gemini-2.5-flash`)
3. `<provider_type>` (e.g., `gemini`)
4. `default`

Each entry has the shape:

```json
{"input_per_million": 0.30, "output_per_million": 2.50}
```

If `configs/prices.json` is missing, the run continues with $0.00 cost and a
warning log line. Costs are estimates — the script never contacts a live
pricing API.
```

- [ ] **Step 3: Verify both files are valid Markdown (cursory check)**

```bash
cd "C:/work/llm training" && grep -c "^### Per-run metrics (Slice 4)" README.md
cd "C:/work/llm training" && grep -c "^## Metrics (Slice 4)" docs/provider_config.md
```
Expected: both print `1`.

- [ ] **Step 4: Verify prior README sections are intact**

```bash
cd "C:/work/llm training" && python -c "
content = open('README.md', encoding='utf-8').read()
for s in [
    'Multi-provider scaffolding (Slice 1)',
    'Multi-provider input generation (Slice 2)',
    'Multi-provider output labeling (Slice 3)',
    'Per-run metrics (Slice 4)',
]:
    print(s, '->', s in content)
"
```
Expected: all four print `True`.

- [ ] **Step 5: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: `334 passed` (333 + 1 from inputs E2E). Exact count may drift if Task 12 adds a coverage gap-filler test — treat as a smoke signal, not a hard blocker.

- [ ] **Step 6: Do NOT commit.**

---

### Task 12: Manual smokes + verify clean staging area

**Files:** none modified.

Spec reference: §10.

- [ ] **Step 1: Smoke 1 — Slice 2 input phase regression**

```bash
cd "C:/work/llm training" && rm -rf /tmp/s4_smoke && mkdir -p /tmp/s4_smoke && DISTILL_DIR_OVERRIDE=/tmp/s4_smoke python scripts/05_generate_distillation_data.py --phase inputs --provider-config configs/test_providers.json --multi-provider 2>&1 | tail -10
```
Expected: completes with exit 0; logs include `=== Stage 5 metrics (phase=inputs, ...)`. Verify outputs:
```bash
ls /tmp/s4_smoke/ && wc -l /tmp/s4_smoke/inputs_raw.jsonl && head -30 /tmp/s4_smoke/metrics.json
```
Expected: `inputs_raw.jsonl` has 10 lines; `metrics.json` has `"phase": "inputs"`, `"input_rows_written": 10`.

- [ ] **Step 2: Smoke 2 — Slice 3 label phase against fake providers**

```bash
cd "C:/work/llm training" && DISTILL_DIR_OVERRIDE=/tmp/s4_smoke python scripts/05_generate_distillation_data.py --phase label --provider-config configs/test_providers.json --multi-provider --limit 5 2>&1 | tail -15
```
Expected: completes with exit 0; logs show `=== Stage 5 metrics (phase=label, 5 inputs, ...)` and a provider table. Verify:
```bash
head -40 /tmp/s4_smoke/metrics.json
```
Expected: `"phase": "label"`, `"providers"` block with `acceptance_rate` per provider, `"repair"` block present.

- [ ] **Step 3: Smoke 3 — Dry-run produces NO metrics.json**

```bash
cd "C:/work/llm training" && rm -rf /tmp/s4_dry && mkdir -p /tmp/s4_dry && DISTILL_DIR_OVERRIDE=/tmp/s4_dry python scripts/05_generate_distillation_data.py --provider-config configs/smoke_real_providers.json --dry-run-quota 2>&1 | tail -5
ls /tmp/s4_dry/
```
Expected: exit 0; the directory listing does NOT include `metrics.json`. (The dry-run prints to stdout but writes no files.)

- [ ] **Step 4: Smoke 4 — Legacy phase produces NO metrics.json**

```bash
cd "C:/work/llm training" && rm -rf /tmp/s4_legacy && mkdir -p /tmp/s4_legacy && DISTILL_DIR_OVERRIDE=/tmp/s4_legacy python scripts/05_generate_distillation_data.py --phase eval --force-eval-copy 2>&1 | tail -5
ls /tmp/s4_legacy/
```
Expected: exit 0; directory listing shows `eval.jsonl` (or similar) but NOT `metrics.json`. (Eval is a legacy path; no recorder lifecycle.)

If the eval path fails (e.g., source eval missing), the test is still informative: even with a SystemExit, no `metrics.json` should be created.

- [ ] **Step 5: Smoke 5 — Missing prices.json fallback**

```bash
cd "C:/work/llm training" && mv configs/prices.json configs/prices.json.bak && rm -rf /tmp/s4_noprice && mkdir -p /tmp/s4_noprice && DISTILL_DIR_OVERRIDE=/tmp/s4_noprice python scripts/05_generate_distillation_data.py --phase label --provider-config configs/test_providers.json --multi-provider --limit 2 2>&1 | tail -10
mv configs/prices.json.bak configs/prices.json
```
Expected: completes with exit 0; logs include `Prices file not found` warning AND `Estimated cost: $0.0000 (rates from default zero pricing)`.

Verify:
```bash
python -c "import json; m=json.load(open('/tmp/s4_noprice/metrics.json')); print('cost:', m['totals']['estimated_cost_usd']); print('prices_source:', m['run']['prices_source'])"
```
Expected: `cost: 0.0`, `prices_source: None`.

- [ ] **Step 6: Verify clean staging area**

```bash
cd "C:/work/llm training" && git status --short
```
Look at the list. These MUST NOT be staged:
- `data/distill/*.jsonl`
- `data/distill/metrics.json` (should be gitignored)
- `reports/`, `.coverage`, `htmlcov/`

(They may appear as unstaged working-tree changes or untracked — that's fine; the rule is they NEVER end up in the commit at Task 13.)

- [ ] **Step 7: Final full suite**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: `334 passed` (333 + 1 from inputs E2E). Exact count may drift if Task 12 adds a coverage gap-filler test — treat as a smoke signal, not a hard blocker.

- [ ] **Step 8: Coverage final check**

```bash
cd "C:/work/llm training" && python -m pytest tests/ --cov=metrics --cov=llm_providers --cov-report=term 2>&1 | tail -10
```
Expected:
- `metrics.py`: ≥ 95%
- `llm_providers.py`: ≥ 85%

If `metrics.py` coverage is below 95%, identify the gap (likely an error branch in `load_prices` or `record_output_row`) and add one small unit test in `test_metrics.py` to cover it before proceeding.

- [ ] **Step 9: Do NOT commit yet. Final commit is Task 13.**

---

### Task 13: Final commit

**Files:** none modified — this is the commit step.

- [ ] **Step 1: Confirm the staged set**

CRITICAL: do NOT stage `data/distill/*`, `reports/*`, `.coverage`, `htmlcov/*`. Stage only source + tests + configs + docs.

```bash
cd "C:/work/llm training" && git add \
    scripts/metrics.py \
    scripts/llm_providers.py \
    scripts/05_generate_distillation_data.py \
    configs/prices.json \
    tests/test_metrics.py \
    tests/test_real_providers.py \
    tests/test_label_orchestrator.py \
    tests/test_multi_provider_labels.py \
    tests/test_multi_provider_inputs.py \
    README.md \
    docs/provider_config.md \
    .gitignore

git status --short
```

Verify the staged section shows ONLY the listed files. Pre-existing `data/distill/*` modifications must remain unstaged.

If a file in the list shows no changes (e.g., you didn't actually need to modify `tests/test_real_providers.py`), drop it from the add command.

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
Slice 4: per-run metrics + cost tracking

Adds a metrics report emitted at the end of every multi-provider Stage 5 run
covering cost (USD estimate from configs/prices.json), latency (p50/p95),
token counts (SDK-reported with char/4 fallback), and acceptance/failure
breakdown by reason.

Adds:
- scripts/metrics.py — MetricsRecorder + load_prices/lookup_price/
  estimate_tokens/percentiles + dataclasses (ProviderCallMetric,
  CandidateMetric). Pure stdlib; threading.Lock around list appends; thread-
  safety verified with a 10x100 concurrent stress test.
- DeepSeek/Gemini providers: per-thread threading.local() stash of SDK
  usage metadata; pop_last_usage() called by orchestrator. getattr fallback
  everywhere so missing/changed SDK shape never breaks the run.
- Orchestrator instrumentation in 05_generate_distillation_data.py:
  - _try_one_attempt wraps each provider call in try/finally with timing +
    pop_last_usage + record_call; record_label_candidate at terminal
    outcomes (never on provider-error path — keystone of the no-double-count
    invariant).
  - _provider_worker (input phase) wraps each generate_inputs call the same
    way with attempt_type="input_batch".
  - Both phase functions now return int (rows processed).
  - main() creates recorder only for --multi-provider --phase {inputs,label}
    (not --dry-run-quota, not legacy paths), writes data/distill/metrics.json
    + logs compact stdout summary at end of run.
- configs/prices.json — seed price table; user-maintained estimate.

Invariants pinned:
- record_label_candidate NEVER called on provider-error path; ensures
  failures.provider_error == totals.calls_failed exactly.
- Legacy phases (eval/all/single-provider) and --dry-run-quota produce no
  metrics.json.
- prices.json missing → run continues with $0 cost + warning log.
- Provider-side usage extraction uses getattr() throughout; missing fields
  fall back to estimate_tokens() with has_estimated_tokens=True surfaced.

Single additive commit on feat/multi-provider-slice4 from
feat/multi-provider-slice3 HEAD (0b79433).

Spec: docs/superpowers/specs/2026-05-17-multi-provider-generation-slice4-design.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify the commit**

```bash
cd "C:/work/llm training" && git log --oneline -5 && git status --short
```
Expected: new commit on top of `f3cb4d9` (spec amendments) and `375e269` (spec). Working tree shows only pre-existing `data/distill/*` modifications and untracked artifacts.

- [ ] **Step 4: Report**

Surface to the user:
- The new commit SHA.
- Final test pass count (~334 passed; a couple over is fine, regressions are not).
- Coverage percentages for `metrics.py` and `llm_providers.py`.
- Confirmation that all five smokes succeeded (input phase, label phase, dry-run, legacy phase, missing prices fallback).
- Suggested next step: optional real-API smoke with `configs/smoke_real_providers.json` to measure actual DeepSeek + Gemini cost/latency (requires `DEEPSEEK_API_KEY` + `GOOGLE_API_KEY`), or move on to merging Slice 4 into `feat/multi-provider-generation`.
