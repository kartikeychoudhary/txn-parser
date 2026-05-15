# Validator + Amount Parser — Design Spec

**Date:** 2026-05-16
**Status:** Approved, ready for implementation plan
**Scope:** Slice 1 of 6 in the transaction-parser quality roadmap (see "Out of scope" for the other five)

---

## 1. Problem

The current pipeline accepts any teacher label that passes JSON-schema validation and ships a Q5 GGUF that fails semantic correctness ~80% of the time on the smoke eval. The two dominant failure modes are:

- **Duplicate transaction rows** the input does not justify.
- **Amount drift** — e.g. `50000` predicted as `500`, or a corrected amount (`500 wait no 600`) still emitted.

Schema validation does not catch any of this. We need a deterministic, model-independent check that an output is consistent with the literal numeric/semantic content of the input. That check needs to run both as a **training-time gate** (Stage 5 distillation labels) and as a **runtime post-check** (eval + future on-device fallback).

## 2. Solution overview

Two new pure-Python modules under `scripts/`, no model/GPU dependencies:

```
scripts/
  amount_parser.py   # text  -> list[AmountCandidate]
  validator.py       # (input_text, output_obj) -> ValidationResult
  _lib.py            # re-exports parse_amounts, validate_example, etc. at file bottom
```

`amount_parser.py` is the only place numeric normalization lives. It turns raw input text into a list of `AmountCandidate` objects with `value`, `raw`, `span`, `status ∈ {active, superseded}`, `source`, and optional `currency_hint`. It does not know about the JSON schema, categories, or model output.

`validator.py` consumes the candidate list plus the model's JSON output and returns a structured `ValidationResult` describing input-vs-output consistency. It does not raise on normal failures and does not call the model.

Two integration points in this slice:

- **Stage 5** (`scripts/05_generate_distillation_data.py`) — strict validation gate replaces the schema-only check; rejected rows go to `failed.jsonl` with structured error info.
- **Stage 4** (`scripts/04_eval.py`) — new per-example fields and aggregate rows derived from `ValidationResult`, plus eval-only `amount_exact` / `txn_count_exact` computed against expected output.

Deliberately out of scope: grammar-constrained decoding, schema changes, training-loop changes, dataset expansion, eval-set expansion, Qwen3-0.6B benchmark, back-cleaning the existing `data/distill/train.jsonl` (separate one-shot, see §10).

---

## 3. Module: `scripts/amount_parser.py`

### 3.1 Data shape

```python
@dataclass(frozen=True)
class AmountCandidate:
    value: float
    raw: str                      # exact substring from input
    span: tuple[int, int]         # (start, end) char offsets
    status: Literal["active", "superseded"]
    source: Literal[
        "decimal_k", "k_suffix",
        "digits_lakh", "digits_hazaar", "digits_sau",
        "currency_symbol", "slash_dash",
        "digits",
        "number_word_en", "number_word_hi",
    ]
    currency_hint: Literal["INR", "USD"] | None = None

def parse_amounts(text: str) -> list[AmountCandidate]: ...
```

- List is returned in **input order**.
- `superseded` candidates are kept (not dropped) so the validator can detect "model emitted a superseded amount."
- Spans never overlap. Overlap is resolved by priority order (§3.2); the higher-priority matcher wins and blocks the lower-priority matcher from claiming any overlapping characters.
- `value` is `float`; downstream comparisons use `math.isclose(a, b, rel_tol=0, abs_tol=0.001)`. Never compare floats directly.

### 3.2 Matcher priority

Higher priority runs first; a matched span blocks later matchers from claiming overlapping characters.

| Priority | Rule | Examples | Notes |
|---:|---|---|---|
| 1 | Decimal-k | `1.5k`, `2.5K` | × 1000 |
| 2 | Integer-k | `50k`, `5K` | × 1000 |
| 3 | Digits + lakh | `1 lakh`, `1.5 lakh` | × 100000, source `digits_lakh` |
| 4 | Digits + hazaar/thousand | `5 hazaar`, `2 thousand` | × 1000, source `digits_hazaar` |
| 5 | Digits + sau/hundred | `2 sau`, `5 hundred` | × 100, source `digits_sau` |
| 6 | Currency-prefixed/suffixed digits | `₹300`, `$50`, `rs 500`, `Rs.500`, `500/-` | Sets `currency_hint`; source `currency_symbol` or `slash_dash` |
| 7 | Plain digit groups | `5000`, `50,000`, `12500` | Strips commas; source `digits`. Subject to rejection filters (§3.3) |
| 8 | English number-words + unit | `five hundred`, `two thousand`, `one lakh`, `twenty five hundred` | Restricted vocabulary; source `number_word_en` |
| 9 | Hindi number-words + unit | `paanch sau`, `do hazaar`, `char lakh` | Only when followed by a unit word; source `number_word_hi` |

**Unit-bearing matchers (3, 4, 5, 8, 9) must beat plain digit matcher (7).** This is the rule that prevents `"1 lakh"` from being captured as just `1`.

### 3.3 Plain-digit rejection rules

A digit group at priority 7 is **rejected** (not emitted as a candidate) when any of:

- Length ≥ 7 consecutive digits.
- Preceded by `phone`, `mobile`, `otp`, `pin`, `account`, `upi id`, `order id` within ~5 tokens.
- The digit's span falls inside a wider time match `\b\d{1,2}:\d{2}\b` in the input. (Not just "matches the pattern" — the rejection check runs against the full surrounding substring so that `12` and `30` in `"12:30"` are both rejected.)
- The digit's span falls inside a wider date match `\b\d{1,2}/\d{1,2}/\d{2,4}\b` in the input. So `12`, `05`, and `2026` in `"12/05/2026"` are all rejected.

Implementation note: scan for time/date patterns **before** plain-digit emission; build a "blocked spans" set; reject any plain-digit candidate whose span overlaps a blocked span.

Large amounts with explicit cues (e.g., `"paid 125000 rent deposit"`, `"₹125000 deposit"`, `"1.25 lakh deposit"`) remain parseable because the cue or unit beats the rejection.

### 3.4 English number-word vocabulary (v1)

Bounded vocabulary to avoid ambiguity:

- `<one..twenty> hundred` (e.g., `five hundred`, `twenty hundred`)
- `<one..twenty> thousand` (e.g., `two thousand`)
- `<one..ten> lakh` (e.g., `one lakh`, `five lakh`)
- Compound `<one..twenty> <one..nine> hundred` (e.g., `twenty five hundred` → 2500)

Out of v1 vocabulary (`twelve fifty`, casual compounds, word-form decimals, ranges, Devanagari numerals, mixed-script). Documented as known gaps; surfaced via `NO_AMOUNT_IN_INPUT` or `AMOUNT_NOT_IN_INPUT` rather than silent passes.

### 3.5 Hindi number-words rule

Hindi number words are parsed **only when followed by a unit word**. Bare Hindi number words are NOT parsed in v1 — this prevents `"do coffee"`, `"char people"`, `"paanch minute"` from being misread as amounts.

v1 Hindi numerals: `ek` (1), `do` (2), `teen` (3), `char` / `chaar` (4), `paanch` (5), `chhe` / `che` (6), `saat` (7), `aath` (8), `nau` (9), `das` (10).
v1 Hindi units: `sau` (×100), `hazaar` (×1000), `lakh` (×100000).

Composite forms beyond `<numeral> <unit>` (e.g., `do hazaar paanch sau` for 2500) are **out of v1 scope**. Spec covers the simple `numeral × unit` form only; multi-unit compounds will surface as `AMOUNT_NOT_IN_INPUT` and are addressed in a follow-up parser slice.

### 3.6 Correction / supersession

After collecting raw candidates, a second pass scans for correction markers. **Bare `wait` is NOT a marker** — only stronger compound forms.

- `wait no`, `wait actually`, `wait sorry`, `wait correction`, `wait,`(comma)
- `i mean`, `actually`, `scratch that`, `nope`
- `umm ... no`, `uhh no`
- `nahi`, `nahin`, `galat`
- `correction:`

`wait` followed by a non-marker word (e.g., `"500 beer wait and 600 candy"`) is treated as a stop-word, NOT a correction.

Rule (exact wording):

> A correction marker supersedes the nearest previous active candidate only when **all** of:
> 1. The marker is after that candidate.
> 2. Another amount candidate appears after the marker.
> 3. The previous candidate is within `N=6` tokens before the marker.
> 4. The next candidate is within `M=6` tokens after the marker.

This correctly handles:

- `"500 beer wait no 600 beer"` → 500 superseded, 600 active.
- `"500 no 600 no 700"` → 500 superseded, 600 superseded, 700 active.
- `"500 beer no extra charges"` → 500 stays active (no replacement candidate after the marker).
- `"500 beer and 50 candy"` → both active (no marker between them).

---

## 4. Module: `scripts/validator.py`

### 4.1 Public API

```python
@dataclass(frozen=True)
class ValidationError:
    code: str
    path: str                  # JSON pointer, e.g. "/transactions/0/amount"
    message: str
    severity: Literal["error", "warning"]

@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[ValidationError]
    amount_values_match_active_candidates: bool
    txn_count_matches_active_candidates: bool
    duplicate_transactions_found: bool
    superseded_amount_used: bool
    candidates: list[AmountCandidate]

def validate_example(
    input_text: str,
    output_obj: dict,
    *,
    mode: Literal["strict", "warn"] = "strict",
) -> ValidationResult: ...

def serialize_validation_result(result: ValidationResult) -> dict: ...
```

- `mode="strict"` (default): `ok = not any(e.severity == "error" for e in errors)`.
- `mode="warn"`: `ok = True` regardless, errors list still populated.
- `serialize_validation_result` returns `{ok, amount_values_match_active_candidates, txn_count_matches_active_candidates, duplicate_transactions_found, superseded_amount_used, errors: [...]}`. It **excludes** `candidates` — `candidates` are serialized separately by callers (Stage 5 writes them at the top level of the failed.jsonl row per §6.3). This keeps the validation block stable when `candidates` schema evolves.
- `AmountCandidate.span` JSON-serializes as a 2-element array.

### 4.2 Imports

The project's existing scripts use the flat-script import pattern (`sys.path.insert(0, scripts_dir)` then `from _lib import ...`). The new modules and tests use the same style:

- `validator.py` does `from amount_parser import parse_amounts` at module load.
- `validator.py` does `from _lib import is_schema_valid, schema_errors` **lazily** inside `validate_example` to break the circular load (`_lib` re-exports validator symbols at its file bottom).
- Tests import via `sys.path.insert(0, scripts_dir)` then flat imports.
- Smoke check at end of step 4 of the rollout uses `PYTHONPATH`:

  ```bash
  PYTHONPATH=scripts python -c "from _lib import validate_example, parse_amounts; print('ok')"
  ```

If a future refactor turns `scripts/` into a package, the imports must be migrated everywhere together — out of scope for this slice.

### 4.3 Pipeline

All semantic checks run regardless of earlier semantic failures so `failed.jsonl` rows carry a complete picture. Schema failure returns early because downstream checks assume valid shape.

1. **Schema preflight** — call `is_schema_valid`. On failure: emit `SCHEMA_INVALID` errors and **return** (skip steps 2–7).
2. **Parse candidates** — `candidates = parse_amounts(input_text)`. Let `active = [c for c in candidates if c.status == "active"]`.
3. **No-amount precedence** — if `len(active) == 0` and `len(output["transactions"]) ≥ 1`: emit a single `NO_AMOUNT_IN_INPUT` at `/transactions` and **skip step 4** (per-txn amount checks). Continue with steps 5–7.
4. **Per-transaction amount check** — for each txn, find `matches = [c for c in candidates if math.isclose(c.value, txn.amount, abs_tol=0.001)]`:
   - `matches == []` → `AMOUNT_NOT_IN_INPUT`.
   - Every match has `status == "superseded"` → `SUPERSEDED_AMOUNT_USED`.
   - Otherwise → pass.
5. **Per-transaction currency check** — find matched **active** candidates with a non-null `currency_hint`. If any exist, `txn.currency` must equal one of their hints; mismatch → `CURRENCY_MISMATCH`. If none of the matches has a `currency_hint`, txn currency is unconstrained (default `INR` is fine). **Per matched candidate, not global** — this allows mixed-currency inputs like `"$50 coffee and ₹300 lunch"`.
6. **Transaction count check**:
   - `len(transactions) > len(active)` → `TXN_COUNT_EXCEEDS_CANDIDATES` (error).
   - `len(transactions) < len(active)` → `TXN_COUNT_BELOW_CANDIDATES` (warning). `TXN_COUNT_BELOW_CANDIDATES` is warning severity, so a result containing only this warning has `ok=True` even in strict mode.
7. **Duplicate detection** — see §4.5.

### 4.4 Error codes

| Code | Severity | Trigger | Path example |
|---|---|---|---|
| `SCHEMA_INVALID` | error | jsonschema violation | `/transactions/0/amount` |
| `AMOUNT_NOT_IN_INPUT` | error | txn amount matches no candidate | `/transactions/0/amount` |
| `SUPERSEDED_AMOUNT_USED` | error | txn amount matches only `superseded` candidates | `/transactions/0/amount` |
| `CURRENCY_MISMATCH` | error | matched active candidate has explicit `currency_hint` differing from txn | `/transactions/0/currency` |
| `TXN_COUNT_EXCEEDS_CANDIDATES` | error | more transactions than active candidates | `/transactions` |
| `TXN_COUNT_BELOW_CANDIDATES` | warning | fewer transactions than active candidates | `/transactions` |
| `SUSPICIOUS_DUPLICATE` | error | duplicate not justified by input (§4.5) | `/transactions/1` |
| `NO_AMOUNT_IN_INPUT` | error | no active candidates but ≥1 transaction emitted | `/transactions` |

Note: `JSON_PARSE_FAILED` is a Stage 4 / Stage 5 wrapper error code, **not** emitted by `validate_example`. The validator only sees parsed objects.

### 4.5 Duplicate-detection rule

A transaction at index `i` is **suspiciously duplicated** when it is an exact normalized duplicate of an earlier transaction `j < i` AND the input does not justify the repetition.

Normalized duplicate key: `(amount-bucket-via-isclose, item.lower().strip(), category, type, currency)`. "amount-bucket-via-isclose" means two amounts are the same key iff `math.isclose(a, b, abs_tol=0.001)`.

Let `dup_count(key)` be the number of transactions in the output sharing that normalized key (so `dup_count ≥ 2` is the duplicate trigger).

A duplicate is **justified** only when:

1. The same amount AND the same item phrase each appear in the input at least `dup_count(key)` times, OR
2. An explicit repetition cue appears near the amount/item phrase: `twice`, `2x`, `two times`, `do baar`, `again`, `another`, `aur ek`, `phir se`.

Worked cases:

- `"500 beer"` → two beer txns → `SUSPICIOUS_DUPLICATE`.
- `"500 beer and 500 beer"` → two beer txns → OK.
- `"500 beer twice"` → two beer txns → OK.
- `"500 beer and 500 candy"` → two beer txns → `SUSPICIOUS_DUPLICATE` (input does not contain `beer` twice).
- `"500 beer and 500 candy"` → beer + candy txns → OK.

### 4.6 What the validator does NOT check (v1)

- Item-text correctness (too subjective; eval-side concern).
- Category correctness (eval-side confusion matrix).
- Type expense/income correctness (eval-side).
- Transaction ordering.

These belong to Stage 4's expected-vs-predicted comparison, not the input-vs-predicted validator.

---

## 5. `scripts/_lib.py` changes

A single block appended at the **bottom** of `_lib.py`, after all existing schema/constants/functions are defined:

```python
# Re-exports — placed at the bottom to avoid circular imports.
from amount_parser import AmountCandidate, parse_amounts
from validator import (
    ValidationError, ValidationResult,
    validate_example, serialize_validation_result,
)
```

Existing imports of `_lib` symbols (e.g., `from _lib import is_schema_valid`) are untouched. Quick smoke after the change:

```bash
python -c "from scripts._lib import validate_example, parse_amounts; print('ok')"
```

---

## 6. Stage 5 integration — `scripts/05_generate_distillation_data.py`

### 6.1 Gate replacement

In the labeling-phase loop, the existing schema-only gate is replaced. The validator already runs the schema preflight internally, so Stage 5 does **not** call `is_schema_valid` separately.

```python
from _lib import extract_json, validate_example

for inp, raw_output in teacher_outputs:
    obj = extract_json(raw_output)
    if obj is None:
        append_failed(inp, raw_output=raw_output, reason="json_parse_failed")
        continue

    result = validate_example(inp, obj, mode="strict")
    if result.ok:
        append_train(inp, obj)
    else:
        append_failed(inp, raw_output=raw_output, parsed_output=obj,
                      result=result, reason="validation_failed")
```

### 6.2 New flag: `--retry-validation-failed`

Definition: re-attempts rows in `failed.jsonl` where `reason == "validation_failed"`. Analogous to the existing `--retry-failed`. Error-code filtering (`--error-code AMOUNT_NOT_IN_INPUT`) deferred to a later slice.

### 6.3 `failed.jsonl` shape

Backwards-incompatible. The old shape is not read by the new tooling — delete or archive `data/distill/failed.jsonl` before running the new Stage 5 flow.

Field naming standardized: `raw_output` everywhere (not `raw`).

```json
// reason == "validation_failed"
{
  "input": "fifty thousand for laptop",
  "raw_output": "{\"transactions\":[{\"amount\":500,...",
  "parsed_output": {"transactions": [...]},
  "candidates": [
    {"value": 50000.0, "raw": "fifty thousand", "span": [0,14],
     "status": "active", "source": "number_word_en", "currency_hint": null}
  ],
  "validation": {
    "ok": false,
    "amount_values_match_active_candidates": false,
    "txn_count_matches_active_candidates": true,
    "duplicate_transactions_found": false,
    "superseded_amount_used": false,
    "errors": [
      {"code": "AMOUNT_NOT_IN_INPUT", "path": "/transactions/0/amount",
       "message": "Predicted amount 500 not in candidates [50000]", "severity": "error"}
    ]
  },
  "reason": "validation_failed"
}

// reason == "json_parse_failed"
{
  "input": "...",
  "raw_output": "<raw model text>",
  "candidates": [...],
  "reason": "json_parse_failed"
}

// reason == "teacher_error"
{
  "input": "...",
  "raw_output": null,
  "candidates": [...],
  "reason": "teacher_error",
  "error": "<exception or HTTP error text>"
}
```

`candidates` is included for **all** failure types. `validation` and `parsed_output` only for `validation_failed`. `error` only for `teacher_error`.

**`teacher_error` granularity.** The current labeling loop batches teacher inference. When a batch raises, the implementation writes one `teacher_error` row per input in that batch (so retry tooling can target individual inputs); the same `error` string is repeated across all rows of the failing batch. Batch-level error context (batch size, batch index) is logged to the per-run logfile, not duplicated in `failed.jsonl`.

---

## 7. Stage 4 integration — `scripts/04_eval.py`

### 7.1 New per-example fields

```python
# from validator (input vs predicted output)
amount_values_match_active_candidates: bool
txn_count_matches_active_candidates: bool
duplicate_transactions_found: bool
superseded_amount_used: bool
validation_errors: list[dict]   # serialized ValidationError

# eval-only (predicted vs expected)
amount_exact: bool              # multiset compare via math.isclose(abs_tol=0.001)
txn_count_exact: bool           # len(predicted.transactions) == len(expected.transactions)
```

### 7.2 Invalid-JSON row shape (pinned)

When `extract_json(raw)` returns `None`:

```json
{
  "amount_values_match_active_candidates": false,
  "txn_count_matches_active_candidates": false,
  "duplicate_transactions_found": false,
  "superseded_amount_used": false,
  "validation_errors": [
    {"code": "JSON_PARSE_FAILED", "path": "",
     "message": "Model output was not valid JSON", "severity": "error"}
  ],
  "amount_exact": false,
  "txn_count_exact": false
}
```

`JSON_PARSE_FAILED` is emitted by Stage 4 wrapper code, not by `validate_example`.

### 7.3 New summary aggregates

Added to the end-of-eval summary table:

```
amount_exact                  84.2%
txn_count_exact               91.5%
duplicate_rate                 2.7%   (= examples_with_duplicate_transactions_found / total_eval_examples)
superseded_amount_used_rate    0.4%   (= examples_with_superseded_amount_used / total_eval_examples)
```

`duplicate_rate` denominator is total eval examples, not total transactions.

### 7.4 Stage 4 metric helper

A small testable helper isolates the wrapper logic so the invalid-JSON path can be unit-tested without spinning a model:

```python
def score_validation_fields(
    input_text: str,
    expected: dict,
    predicted_raw: str,
    predicted: dict | None = None,
) -> dict:
    """Returns the per-example fields described in §7.1 / §7.2.

    If `predicted` is provided, it is used as-is and `predicted_raw` is
    only stored for diagnostics. If `predicted is None`, the helper
    calls `extract_json(predicted_raw)` itself; `None` from that call
    triggers the §7.2 invalid-JSON row shape.
    """
```

Stage 4 callers that already have `predicted` from `extract_json` should pass it in so JSON is not parsed twice.

---

## 8. Testing strategy

All tests pure-Python, no GPU, < 5s total.

### 8.1 `tests/test_amount_parser.py` — target ≥95% line coverage

Fixture groups:

1. **One-rule-per-row positives** (~80 cases): each `source` × ≥ 3 phrasings.
   - `("1.5k", [Cand(value=1500.0, source="decimal_k", status="active")])`
   - `("5 hazaar", [Cand(value=5000.0, source="digits_hazaar", status="active")])`
   - `("paanch hazaar", [Cand(value=5000.0, source="number_word_hi", status="active")])`
   - `("five thousand", [Cand(value=5000.0, source="number_word_en", status="active")])`
2. **Priority regression**:
   - `"1 lakh rent"` → `[Cand(value=100000.0, source="digits_lakh")]`
   - `"1.5 lakh deposit"` → `[Cand(value=150000.0, source="digits_lakh")]`
   - `"5 hazaar rent"` → `[Cand(value=5000.0, source="digits_hazaar")]`
   - `"2 sau chai"` → `[Cand(value=200.0, source="digits_sau")]`
3. **Currency hint** (pinned):
   - `"$50 coffee"` → `value=50, currency_hint="USD"`
   - `"₹300 lunch"` → `value=300, currency_hint="INR"`
   - `"rs 500 dinner"` → `value=500, currency_hint="INR"`
   - `"Rs.500 petrol"` → `value=500, currency_hint="INR"`
   - `"500/- chai"` → `value=500, currency_hint="INR"`
   - `"500 chai"` → `value=500, currency_hint=None`
4. **Non-amount rejection**:
   - `"phone number 9876543210"` → `[]`
   - `"otp 123456"` → `[]`
   - `"order id 12345678"` → `[]`
   - `"two minutes"` → `[]`
   - `"second item"` → `[]`
   - `"12:30 train"` → `[]`
5. **Supersession**:
   - `"500 beer wait no 600 beer"` → `[500 superseded, 600 active]`
   - `"500 no 600 no 700"` → `[500 superseded, 600 superseded, 700 active]`
   - `"500 beer no extra charges"` → `[500 active]`
   - `"500 beer and 500 beer"` → both active
   - `"paanch sau nahi chhe sau"` → `[500 superseded, 600 active]`
6. **Window edge** — marker beyond N=6 tokens from prior candidate; no supersession.

### 8.2 `tests/test_validator.py` — target ≥90% line coverage

Every error code exercised:

- `SCHEMA_INVALID` × 3 (wrong type, missing field, bad enum).
- `AMOUNT_NOT_IN_INPUT` × 2.
- `SUPERSEDED_AMOUNT_USED` × 2.
- `CURRENCY_MISMATCH` × 2, including the mixed-currency happy path `"$50 coffee and ₹300 lunch"` (must pass).
- `TXN_COUNT_EXCEEDS_CANDIDATES` × 2.
- `TXN_COUNT_BELOW_CANDIDATES` × 2 — warning only; assert `result.ok == True` in strict mode.
- `SUSPICIOUS_DUPLICATE` × 4 — the four worked cases in §4.5.
- `NO_AMOUNT_IN_INPUT` × 1 — assert per-txn `AMOUNT_NOT_IN_INPUT` is NOT also emitted.
- `mode="warn"` × 1 — same failing input passes with `ok=True` and non-empty errors.
- Serialization round-trip: `serialize_validation_result(result)` is JSON-encodable and re-decodable.

### 8.3 `tests/test_stage4_metrics.py` — wrapper smoke

One small test for `score_validation_fields`:

- `predicted_raw = "not json"` →
  - `validation_errors[0]["code"] == "JSON_PARSE_FAILED"`,
  - `amount_exact == False`, `txn_count_exact == False`,
  - all validator-derived flags `False`.

Stage 5 wiring stays manual smoke.

### 8.4 Integration smoke (no test runner)

- `python scripts/05_generate_distillation_data.py --phase label --limit 50` against the current teacher adapter. Inspect `failed.jsonl` for error-code distribution.
- `python scripts/04_eval.py --model models/student/gguf --limit 50`. Verify new metric rows appear and invalid-JSON rows have the §7.2 shape.

---

## 9. Dependencies

No new third-party packages in the runtime path. New dev file:

```
requirements-dev.txt
  pytest
  pytest-cov
```

README dev-setup snippet:

```bash
pip install -r requirements-dev.txt
pytest --cov=scripts.amount_parser --cov=scripts.validator
```

---

## 10. Rollout sequence

1. Add `requirements-dev.txt` (`pytest`, `pytest-cov`).
2. Implement `scripts/amount_parser.py` + `tests/test_amount_parser.py` → green.
3. Implement `scripts/validator.py` + `tests/test_validator.py` → green.
4. Append re-exports to bottom of `scripts/_lib.py`. Import smoke check.
5. Implement Stage 4 helper `score_validation_fields` + `tests/test_stage4_metrics.py` → green.
6. Integrate Stage 4 per-example fields + aggregate rows.
7. Integrate Stage 5 strict validation gate; rename failed-row writer; add `--retry-validation-failed`.
8. Manual Stage 5 smoke: `--phase label --limit 50`. Inspect distribution.
9. Manual Stage 4 smoke: `--model models/student/gguf --limit 50`.
10. README delta — paragraphs in Stage 4 and Stage 5 sections documenting new fields, new flag, new `failed.jsonl` shape, dev-test command.

Commit split (two commits, for review/rollback):

- Commit A: parser + validator + tests + `_lib` re-exports.
- Commit B: Stage 4/Stage 5 integration + README delta.

---

## 11. Decision points during rollout

- **After step 8**: of rows that were previously JSON-valid AND schema-valid (denominator pinned), if more than ~25% now fail validation, the parser has gaps. Triage the top failure codes, expand `amount_parser` fixtures, iterate before declaring the slice done.
- **After step 9**: if `amount_exact` on the shipped Q5 student is materially below the existing `exact_match` (~20%), that's the headline number to put in the eval-expansion slice (slice 5).

---

## 12. Out of scope (next slices)

- **Slice 2**: Grammar-constrained decoding (GBNF + first-object stop) in the llama.cpp inference path.
- **Slice 3**: Expanded eval set (3k examples across the slice mix in the analysis) with per-slice metrics.
- **Slice 4**: Re-generate distillation data using the validator as a label gate; targeted dataset mix per the analysis (100k+).
- **Slice 5**: One-shot `scripts/07_filter_distill_train.py` to back-clean the existing `data/distill/train.jsonl` against the new validator.
- **Slice 6**: Qwen3-0.6B benchmark track (parallel student candidate).
- **Stretch (not blocking)**: Viewer route `GET /api/validate?file=...&index=...` — requires a Python CLI bridge (`scripts/validate_one.py`); kept out of this slice's critical path.

---

## 13. Public-surface dependency graph

```
amount_parser.py
  └─ no project deps, no schema deps, no model deps

validator.py
  ├─ imports amount_parser (at load)
  └─ imports _lib (lazy, inside validate_example)

_lib.py
  ├─ owns schema, system prompt, helpers (unchanged)
  └─ re-exports amount_parser + validator symbols at file bottom

scripts/05_generate_distillation_data.py
  └─ imports validate_example, extract_json from _lib (teacher-label gate)

scripts/04_eval.py
  └─ imports validate_example, extract_json from _lib (per-example fields + aggregates)
```
