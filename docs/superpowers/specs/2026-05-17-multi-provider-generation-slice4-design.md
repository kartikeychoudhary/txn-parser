# Multi-Provider Generation — Slice 4 Design Spec

**Date:** 2026-05-17
**Status:** Approved, ready for implementation plan
**Scope:** Slice 4 of 4 in the multi-provider synthetic-data roadmap.
**Prerequisite:** Slice 3 merged (commit `0b79433` on `feat/multi-provider-slice3`, parent `feat/multi-provider-generation`).

---

## 1. Problem

Slices 1–3 wired multi-provider input generation and validator-gated output labeling, but every Stage 5 run is opaque to cost, latency, and per-provider reliability. We don't know which provider is producing accepted labels, what we spent in USD, or which provider's calls are slowest. Slice 4 instruments the multi-provider phases to emit a per-run metrics report covering cost, latency (p50/p95), token counts, and acceptance/failure breakdown.

The original Slice 1 roadmap reserved Slice 4 for "metrics, cost tracking, per-provider analysis." This spec implements exactly that — no ensemble voting, no per-input attribution in `train.jsonl`, no run history.

## 2. Solution overview

Three new/modified scripts and one config:

```
scripts/
  metrics.py                       (NEW)   MetricsRecorder + price lookup + render
  llm_providers.py                 (MODIFY) DeepSeek/Gemini stash SDK usage per-thread
  05_generate_distillation_data.py (MODIFY) Instrument multi-provider phases; emit metrics at end

configs/
  prices.json                      (NEW)   model → {input_per_million, output_per_million}

tests/
  test_metrics.py                  (NEW)   Recorder unit tests + price lookup + percentile math
  test_multi_provider_labels.py    (MODIFY) +2 E2E (metrics.json shape, json_parse_failed counting)
  test_multi_provider_inputs.py    (MODIFY) +1 E2E (input-phase metrics.json shape)

README.md, docs/provider_config.md  (MODIFY)
.gitignore                         (MODIFY) Add data/distill/metrics.json
```

**Default behavior unchanged.** Metrics only fire when `args.multi_provider and args.phase in {"inputs","label"}`. Legacy single-provider phases (`eval`, plain `inputs`/`label`, `all`), and `--dry-run-quota`, produce no `metrics.json` and no metrics output.

**Out of scope (deferred):**

- Per-input attribution annotations in `train.jsonl` (rows already carry `_provider`/`_model` from Slice 3, but no token/cost columns).
- Run-history append to a `runs.jsonl` (each run overwrites `metrics.json`).
- Latency p99, per-model breakdown, or histogram output.
- Cost predictions surfaced via `--dry-run-quota`.
- Ensemble amount-voting across candidates (was an earlier Slice 4 candidate, declined).
- Eval-set expansion or back-cleaning of existing `train.jsonl`.

## 3. Module: `scripts/metrics.py`

Pure stdlib. `import metrics` triggers no SDK imports, no I/O.

### 3.1 Exceptions

```python
class MetricsConfigError(ValueError):
    """Raised when prices.json fails schema or numeric checks."""
```

Kept distinct from `ConfigError` (Slice 1 provider-config loader) so callers can catch either independently.

### 3.2 Dataclasses

```python
from dataclasses import dataclass
from typing import Literal

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
    total_tokens: int           # prompt + completion (precomputed for ergonomics)
    estimated_tokens: bool      # True iff we used estimate_tokens() fallback
    success: bool
    failure_reason: str | None  # "provider_error" or None

@dataclass
class CandidateMetric:
    provider: str
    model: str | None
    phase: Literal["label"]     # candidates only exist in the label phase
    score: int                  # 0 when rejected; 0-100 when scored
    accepted: bool
    failure_reason: str | None  # "json_parse_failed" / "schema_invalid" / "validation_failed" / etc.
    is_repair_attempt: bool
```

### 3.3 `load_prices` and `lookup_price`

```python
def load_prices(path: Path) -> dict:
    """Load configs/prices.json. Returns a dict mapping lookup-key to
    {"input_per_million": float, "output_per_million": float}.

    File missing  → returns {"default": {"input_per_million": 0.0,
                                          "output_per_million": 0.0}}
                    and logs a warning.
    File present  → parsed; negative price → raises MetricsConfigError;
                    non-numeric → raises MetricsConfigError;
                    unknown fields inside an entry → logging.warning, dropped.
    Top-level keys are arbitrary lookup keys (provider_type:model, model,
    provider_type, "default"). They are NOT validated against a fixed list.
    """

def lookup_price(prices: dict, provider_type: str, model: str | None) -> tuple[float, float]:
    """Return (input_per_million, output_per_million).

    Tries in order:
        f"{provider_type}:{model}"
        model
        provider_type
        "default"
    Returns (0.0, 0.0) if none match.
    """
```

### 3.4 Helpers

```python
def estimate_tokens(text: str) -> int:
    """Cheap fallback for providers that don't return SDK usage.
    Returns max(1, ceil(len(text) / 4)) for non-empty text; 0 for empty/None."""

def percentiles(values: list[float]) -> dict[str, float]:
    """Returns {'p50': ..., 'p95': ...}.
    Empty list   → {'p50': 0, 'p95': 0}.
    n < 20       → p95 = max(values) (avoid spurious quantile interpolation).
    n >= 20      → use statistics.quantiles with n=20, take 9 (p50) and 18 (p95).
    """
```

### 3.5 `MetricsRecorder`

```python
class MetricsRecorder:
    def __init__(self, prices: dict, started_at: datetime, prices_source: str | None):
        self._calls: list[ProviderCallMetric] = []
        self._candidates: list[CandidateMetric] = []
        self._input_rows = 0
        self._train_rows = 0
        self._failed_rows = 0
        self._lock = threading.Lock()
        self._prices = prices
        self._prices_source = prices_source
        self._started_at = started_at

    def record_call(self, **kw) -> None: ...            # appends ProviderCallMetric under lock
    def record_label_candidate(self, **kw) -> None: ... # appends CandidateMetric under lock
    def record_output_row(self, kind: Literal["input","train","failed"]) -> None: ...

    def finalize(self, *, phase: str, inputs_processed: int,
                 finished_at: datetime | None = None) -> dict:
        """Compute aggregates and return the metrics.json shape (Section 4)."""

    def render_stdout(self, summary: dict) -> str:
        """Plain-text summary for end-of-run logging (Section 4)."""
```

**Concurrency:** `record_*` methods are called from worker threads (per-input executor in label phase, per-provider workers in input phase). The internal `_lock` serializes list appends and counter increments. `finalize()` runs single-threaded after all workers join.

**Provider-error invariant:** when a provider call raises, the orchestrator calls `record_call(success=False, failure_reason="provider_error")` and does NOT call `record_label_candidate`. This ensures `failures["provider_error"] == totals["calls_failed"]` at finalize time. The candidate failure_reason space is reserved for outcomes after a provider returned text.

## 4. Report shapes

### 4.1 `metrics.json`

Written to `<distill_dir>/metrics.json` (honors `DISTILL_DIR_OVERRIDE`). Overwrites any prior run's file. Single artifact per run, never committed (see `.gitignore` change in §7).

```json
{
  "run": {
    "started_at": "2026-05-17T14:33:01Z",
    "finished_at": "2026-05-17T14:35:42Z",
    "duration_ms": 161000,
    "phase": "label",
    "inputs_processed": 100,
    "git_sha": "0b79433",
    "prices_source": "configs/prices.json",
    "throughput": {"inputs_per_sec": 0.62, "calls_per_sec": 1.32}
  },
  "totals": {
    "calls": 213,
    "calls_succeeded": 198,
    "calls_failed": 15,
    "candidates_accepted": 87,
    "candidates_rejected": 26,
    "estimated_cost_usd": 0.0342,
    "input_rows_written": 0,
    "train_rows_written": 87,
    "failed_rows_written": 13,
    "has_estimated_tokens": false
  },
  "providers": {
    "gemini_flash": {
      "provider_type": "gemini",
      "model": "gemini-2.5-flash",
      "calls": 108,
      "success_calls": 105,
      "failed_calls": 3,
      "call_failure_rate": 0.0278,
      "prompt_tokens": 24310,
      "completion_tokens": 8420,
      "has_estimated_tokens": false,
      "estimated_cost_usd": 0.0283,
      "latency_ms": {"p50": 812, "p95": 1421},
      "candidates_accepted": 52,
      "candidates_rejected": 11,
      "acceptance_rate": 0.825
    }
  },
  "failures": {
    "json_parse_failed": 4,
    "schema_invalid": 1,
    "validation_failed": 8,
    "suspicious_duplicate": 2,
    "provider_error": 15
  },
  "repair": {
    "attempts": 12,
    "accepted": 5,
    "exhausted": 7
  }
}
```

**Field semantics:**

| Field | Definition |
|---|---|
| `run.git_sha` | `git rev-parse --short HEAD` via subprocess. `null` if not a repo / call fails. |
| `run.prices_source` | Relative path to the prices file actually loaded. `null` if file was missing. |
| `run.throughput.inputs_per_sec` | `inputs_processed / (duration_ms / 1000)`. 0 if duration is 0. |
| `run.throughput.calls_per_sec` | `totals.calls / (duration_ms / 1000)`. 0 if duration is 0. |
| `totals.calls` | Total provider invocations across all phases this run. |
| `totals.calls_failed` | Provider invocations that raised. |
| `totals.candidates_accepted` | Label candidates selected into `train.jsonl`. Always 0 for input phase. |
| `totals.candidates_rejected` | Label candidates not selected into `train.jsonl`. Includes both validation/parse failures (`failure_reason != None`) and valid lower-scoring candidates that lost to a better sibling (`failure_reason == None, score > 0`). Always 0 for input phase. |
| `totals.failed_rows_written` | Rows that ended in `failed.jsonl`. Distinct from `candidates_rejected` (one input may have many candidates). |
| `totals.has_estimated_tokens` | True iff any `ProviderCallMetric.estimated_tokens` was True. |
| `providers.<name>.call_failure_rate` | `failed_calls / calls`; 0 if calls is 0. |
| `providers.<name>.acceptance_rate` | `candidates_accepted / (candidates_accepted + candidates_rejected)`. Present only for label phase. |
| `providers.<name>.estimated_cost_usd` | `(prompt_tokens/1e6 * input_price) + (completion_tokens/1e6 * output_price)`, rounded to 4 decimals. |
| `providers.<name>.has_estimated_tokens` | True iff any call for that provider used token estimation. |
| `failures` | Counts of terminal `failure_reason` across both `record_call` (provider_error) and `record_label_candidate` (everything else). **Only non-null `failure_reason` values are counted** — valid lower-scoring candidates (`failure_reason == None`) do not appear here. No double-count: a provider exception never produces a candidate metric. |
| `repair.attempts` | Number of repair provider calls (i.e., `record_call(attempt_type="repair", ...)` events). |
| `repair.accepted` | Number of inputs whose accepted candidate came from a repair attempt. |
| `repair.exhausted` | Number of inputs whose repair loop finished without accepting any candidate. |

**Always present:** `run`, `totals`, `providers`, `failures`.
**Conditionally present:** `repair` block omitted for input phase. `acceptance_rate` field omitted from each `providers.<name>` for input phase.

**Cost rounding:** all `estimated_cost_usd` rounded to 4 decimal places at finalize time (cents and sub-cents matter at distillation volumes).

### 4.2 Stdout summary

Emitted via `logging.info` (multi-line) at the end of each multi-provider run. Same content as `metrics.json` in compact form:

```
=== Stage 5 metrics (phase=label, 100 inputs, 2m41s) ===
Calls: 213 (198 ok, 15 failed)  Accepted: 87  Failed rows: 13
Estimated cost: $0.0342 (rates from configs/prices.json)
# When configs/prices.json is missing, the cost line reads:
#   Estimated cost: $0.0000 (rates from default zero pricing)

Provider          Calls  Ok    Fail%   Acc   Cost      p50    p95    Top failure
gemini_flash      108    105   2.78%   52    $0.0283   812    1421   validation_failed (7)
deepseek_v4_pro   105    93    11.43%  35    $0.0059   1104   2210   provider_error (12)

Repair: 12 attempted, 5 accepted, 7 exhausted
```

- Columns padded with `ljust`. Cost formatted as `$<4-decimal>`.
- `Top failure` = mode of failure reasons attributed to that provider; format `reason (count)`; empty string if no failures.
- For `phase=="inputs"`: drop `Acc` and `Top failure` columns. `record_output_row("input")` is recorded at the totals level only (no per-provider attribution in v1; if you need per-provider input row counts later, expand `record_output_row` to take a provider arg). `success_calls` already conveys API reliability per provider.

## 5. Capture mechanism (provider + orchestrator)

### 5.1 Provider-side: stash SDK usage per-thread

`DeepSeekProvider.__init__` and `GeminiProvider.__init__` gain:

```python
self._tls = threading.local()

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

Inside each `_call_api_*` method after a successful SDK response:

```python
# DeepSeek
usage = getattr(resp, "usage", None)
self._stash_usage(
    getattr(usage, "prompt_tokens", None),
    getattr(usage, "completion_tokens", None),
)
```

```python
# Gemini
usage = getattr(resp, "usage_metadata", None)
self._stash_usage(
    getattr(usage, "prompt_token_count", None),
    getattr(usage, "candidates_token_count", None),
)
```

`FakeProvider` and `LocalTeacherProvider` do NOT gain `pop_last_usage`. The orchestrator uses `getattr(provider, "pop_last_usage", lambda: None)()` so callers don't need to type-switch. Missing or partial SDK metadata → `usage=None` → orchestrator falls back to `estimate_tokens()`.

### 5.2 Orchestrator-side: wrap each call

The capture pattern used everywhere:

```python
start = time.perf_counter()
usage = None
try:
    raw = provider.generate_label(...)        # or generate_inputs(...)
    success = True
    failure_reason = None
except Exception:                              # noqa: BLE001 — matches existing _try_one_attempt
    raw = ""
    success = False
    failure_reason = "provider_error"
finally:
    pop = getattr(provider, "pop_last_usage", None)
    usage = pop() if pop else None
    latency_ms = (time.perf_counter() - start) * 1000.0

if usage:
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    estimated = False
else:
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(raw)
    estimated = True

recorder.record_call(
    provider=cfg.name, provider_type=cfg.provider_type, model=cfg.model,
    phase=phase, attempt_type=attempt_type,
    latency_ms=latency_ms,
    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    total_tokens=prompt_tokens + completion_tokens,
    estimated_tokens=estimated,
    success=success, failure_reason=failure_reason,
)
```

**`prompt_text` per phase/attempt_type:**

| Phase | attempt_type | `prompt_text` for estimate fallback |
|---|---|---|
| inputs | `input_batch` | The full batch prompt sent to the provider |
| label | `initial` | `input_text` (the transaction string being labeled) |
| label | `repair` | The repair prompt body |

This is approximate (the system message and chat formatting tokens aren't included), but it's consistent across providers and only fires when SDK didn't return usage. Real DeepSeek and Gemini calls report usage directly, so the estimate is only used by FakeProvider and LocalTeacherProvider.

### 5.3 Candidate recording (label phase only)

After a successful provider call, downstream parsing/validation may still reject the candidate. The orchestrator emits `record_label_candidate(...)` at the terminal outcome point for each candidate:

- accepted by the validator and selected → `accepted=True, failure_reason=None`
- accepted by parser but validator-rejected → `accepted=False, failure_reason="validation_failed"` (or specific code)
- parser-rejected → `accepted=False, failure_reason="json_parse_failed"` / `"schema_invalid"`
- scored but lost to a higher-scoring sibling → `accepted=False, failure_reason=None, score=<score>`

The last category (valid lower-scoring candidates) increments `candidates_rejected` at finalize time but does NOT increment any bucket in `failures` — `failures` aggregates only non-null `failure_reason` values.

`record_label_candidate` is NEVER called on the provider-exception path. That's the cornerstone of the no-double-count invariant.

## 6. Configuration: `configs/prices.json`

Seed file committed at this shape (representative as of 2026-05-17):

```json
{
  "deepseek:deepseek-chat": {"input_per_million": 0.27, "output_per_million": 1.10},
  "gemini:gemini-2.5-flash": {"input_per_million": 0.30, "output_per_million": 2.50},
  "local_teacher": {"input_per_million": 0.0, "output_per_million": 0.0},
  "fake": {"input_per_million": 0.0, "output_per_million": 0.0},
  "default": {"input_per_million": 0.0, "output_per_million": 0.0}
}
```

**Documented as user-maintained.** Both README and `docs/provider_config.md` carry the disclaimer: *"Costs are estimates based on configs/prices.json at run time. The script does not fetch live pricing."*

**Loader behavior:**

- File missing → log a warning ("prices file not found; cost reports will read $0.00"), use `{"default": {"input_per_million": 0.0, "output_per_million": 0.0}}`, continue.
- File present but malformed JSON → `MetricsConfigError`.
- Negative price → `MetricsConfigError("input_per_million must be >= 0; got -0.27 at key 'deepseek:deepseek-chat'")`.
- Non-numeric `input_per_million`/`output_per_million` → `MetricsConfigError`.
- Unknown field inside an entry (e.g., `"foo": 123`) → `logging.warning`; entry still parsed.
- Top-level keys are arbitrary lookup keys; no validation against a fixed list.

**No CLI flag** to override the path. `configs/prices.json` is the single location. (YAGNI for Slice 4; can be added in a future slice.)

## 7. Wiring into `05_generate_distillation_data.py`

### 7.1 Guard at the top of `main()`

```python
recorder: MetricsRecorder | None = None
if args.multi_provider and args.phase in {"inputs", "label"} and not args.dry_run_quota:
    prices_path = REPO_ROOT / "configs" / "prices.json"
    prices = load_prices(prices_path)
    recorder = MetricsRecorder(
        prices=prices,
        started_at=datetime.now(timezone.utc),
        prices_source=str(prices_path.relative_to(REPO_ROOT))
                       if prices_path.exists() else None,
    )
```

Single-provider legacy paths, `--phase eval`, `--phase all`, and `--dry-run-quota` never instantiate a recorder.

### 7.2 Pass into phase functions

Both phase functions gain an optional `recorder: MetricsRecorder | None = None` parameter and return an `int` (rows processed):

```python
def phase_inputs_multi_provider(args, recorder: MetricsRecorder | None = None) -> int:
    ...
    return n_input_rows_written_this_run

def phase_label_multi_provider(args, recorder: MetricsRecorder | None = None) -> int:
    ...
    return n_kept + n_failed
```

Inside each, every provider call uses the §5.2 wrap pattern. Every `record_output_row` call accompanies a JSONL write.

If `recorder is None`, all `record_*` calls become no-ops — guarded with `if recorder: recorder.record_call(...)` everywhere, or via a `_NullRecorder` shim. Recommended: explicit `if recorder:` guards. Cheap and visible.

### 7.3 Finalize at end of `main()`

```python
if recorder is not None:
    summary = recorder.finalize(
        phase=args.phase,
        inputs_processed=processed,
        finished_at=datetime.now(timezone.utc),
    )
    distill_dir = _get_distill_dir()
    (distill_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    for line in recorder.render_stdout(summary).splitlines():
        logging.info(line)
```

The metrics.json write is unconditional on log level; the stdout-style summary goes through `logging.info` so it respects log handlers but still appears in default runs.

### 7.4 `.gitignore` addition

```
data/distill/metrics.json
```

Single line. Other `data/distill/*.jsonl` artifacts remain in the existing ignore pattern (or are already gitignored — to be checked at implementation time).

## 8. Testing

### 8.1 `tests/test_metrics.py` (~25 cases)

**`load_prices`:**
- Valid file loads with all four entries.
- Missing file → returns default-only dict; emits warning (caplog asserts text).
- Malformed JSON → `MetricsConfigError`.
- Negative `input_per_million` → `MetricsConfigError`.
- Non-numeric `output_per_million` → `MetricsConfigError`.
- Unknown field inside an entry → `logging.warning` (caplog), entry still loads.

**`lookup_price`:**
- `provider_type:model` exact hit wins over weaker keys.
- Fallback to `model` when `provider_type:model` absent.
- Fallback to `provider_type` when both `provider_type:model` and `model` absent.
- Fallback to `default` when nothing else matches.
- Returns `(0.0, 0.0)` when no key (not even `default`) matches.

**`estimate_tokens`:**
- Empty / None → 0.
- 1-char → 1.
- 17 chars → 5 (ceil 17/4).

**`percentiles`:**
- Empty list → `{p50: 0, p95: 0}`.
- Single value → both equal to that value.
- 5 values → p95 == max.
- 20+ values → uses `statistics.quantiles`.

**`MetricsRecorder`:**
- `record_call` accumulates per-provider; multi-thread (10×100) yields exact totals.
- `record_label_candidate` accumulates per-provider; never called on provider-error path (asserted by a direct test: simulate exception path, finalize, assert `failures["provider_error"] == totals["calls_failed"]`).
- `record_output_row("input"/"train"/"failed")` increments the right totals field.
- `finalize()` empty-state → all counts 0, p50/p95 0, has_estimated_tokens False, valid JSON shape.
- `finalize()` with mixed providers and failures → expected counts; `failures["json_parse_failed"]` sums candidate failures, `failures["provider_error"]` equals `totals["calls_failed"]`.
- `finalize()` with `estimated_tokens=True` on at least one call → `totals.has_estimated_tokens` and that provider's `has_estimated_tokens` both True.
- `finalize()` for `phase="inputs"` omits `repair` block and per-provider `acceptance_rate`.
- `finalize()` cost calc: prompt 1_000_000 tokens at $0.30/M = $0.30 exactly (rounded 4dp).
- `render_stdout` contains every provider name, contains `$` with 4-decimal cost, contains the top-failure reason for each provider.
- `render_stdout` for `phase="inputs"` omits `Acc` and `Top failure` columns.

`test_metrics.py` covers `metrics.py` only. The provider-error instrumentation path is covered in `tests/test_label_orchestrator.py` (see §8.2 below).

### 8.2 Orchestrator additions

**`tests/test_label_orchestrator.py` (+1):**
- `test_provider_error_records_call_failure_only`: a `RaisingFakeProvider` that always raises is passed through `_try_one_attempt` (or the §5.2 wrap pattern); recorder receives exactly one `record_call(success=False, failure_reason="provider_error")` and zero `record_label_candidate` calls. This is the keystone test of the no-double-count invariant.

### 8.3 E2E additions

**`tests/test_multi_provider_labels.py` (+2):**
- `test_label_phase_writes_metrics_json`: after a successful run with FakeProvider hitting `fake_labels.jsonl`, `metrics.json` exists, contains `run`/`totals`/`providers`/`failures`/`repair` keys, `train_rows_written` matches `train.jsonl` line count, `phase == "label"`.
- `test_label_phase_metrics_records_json_parse_failures`: FakeProvider configured to miss → metrics.json shows `calls_failed == 0`, `candidates_rejected > 0`, `failures.json_parse_failed > 0`. Also asserts `failures["provider_error"] == totals["calls_failed"]` (both 0 here, but invariant proven).

**`tests/test_multi_provider_inputs.py` (+1):**
- `test_inputs_phase_writes_metrics_json`: after a successful input-phase run, `metrics.json` exists with `phase == "inputs"`, `input_rows_written == target_inputs`, no `repair` key, no `acceptance_rate` in any `providers.<name>`.

### 8.4 Suite size check

Pre-Slice 4: 294 tests. Estimate: +~25 unit + 3 E2E ≈ +28. Target: ~322 passing at end of Slice 4. Coverage targets: `metrics.py` ≥ 95%, `llm_providers.py` ≥ 85% (currently 89%).

## 9. Documentation

**`README.md`:** new subsection under Stage 5, after "Multi-provider output labeling (Slice 3)":

```markdown
### Per-run metrics (Slice 4)

Every multi-provider Stage 5 run emits `data/distill/metrics.json` with per-
provider call counts, accepted/rejected candidates, latency (p50/p95), token
counts, and estimated USD cost. A compact summary is logged at the end of the
run.

Costs are estimates based on `configs/prices.json` at run time. The script
does not fetch live pricing — update `configs/prices.json` to match current
provider rates if you care about USD accuracy.

Metrics fire only for `--multi-provider --phase {inputs,label}`. Legacy
single-provider phases and `--dry-run-quota` produce no metrics file.
```

**`docs/provider_config.md`:** append `## Metrics` section pointing at `configs/prices.json` schema and the lookup-order rule.

## 10. Scope guardrails

- No new top-level deps. `statistics`, `subprocess`, `threading`, `datetime` are stdlib.
- No behavior change to generation or selection. Removing `metrics.py` and the `record_*` calls leaves Slice 1/2/3 tests passing.
- No metrics for `--phase eval`, `--phase all`, single-provider phases, or `--dry-run-quota`.
- No metrics surfaced from `--dry-run-quota`. (Cost predictions could be a future slice.)
- `metrics.json` is single-file, overwritten each run. No timestamping, no run history.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| SDK usage shape changes upstream and we crash mid-run | All extraction via `getattr(..., None)`. Missing fields → estimate fallback, `has_estimated_tokens=True`. |
| Concurrent `record_*` calls drop events under contention | `threading.Lock` around list appends; tested under 10×100 concurrent stress in unit tests. |
| `prices.json` drifts from reality | Documented as user-maintained estimate; `prices_source` field in `metrics.json` makes audit trivial. |
| Adding the recorder slows the hot path | Per-call overhead is one lock-protected list append (~microseconds). Compared to seconds of API latency, negligible. |
| Token estimation is wrong for non-English / verbose JSON | Acknowledged; `has_estimated_tokens` flag surfaces this in reports. Real DeepSeek/Gemini return actual usage and won't trigger estimation. |
| Slice 4 introduces a regression in Slice 1/2/3 flows | Guard at `main()` top means recorder is `None` for legacy paths; all existing Slice 1/2/3 tests are run unchanged. |

## 12. Single-commit landing

Plan will target a single additive commit on a new branch `feat/multi-provider-slice4` from `feat/multi-provider-slice3` HEAD (`0b79433`). Same convention as Slice 3.

Files staged: `scripts/metrics.py`, `scripts/llm_providers.py`, `scripts/05_generate_distillation_data.py`, `configs/prices.json`, `tests/test_metrics.py`, `tests/test_multi_provider_labels.py`, `tests/test_multi_provider_inputs.py`, `README.md`, `docs/provider_config.md`, `.gitignore`.

Files NEVER staged: `data/distill/*.jsonl`, `data/distill/metrics.json`, `.coverage`, `reports/`, `htmlcov/`.
