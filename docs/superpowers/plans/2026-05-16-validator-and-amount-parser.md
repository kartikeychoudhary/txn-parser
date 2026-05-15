# Validator + Amount Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic amount parser and a semantic validator that gate Stage 5 distillation labels and surface new per-example metrics in Stage 4, without changing the training loop, schema, or runtime decoding.

**Architecture:** Two new pure-Python modules under `scripts/` (`amount_parser.py`, `validator.py`) wired together through re-exports added to the bottom of `scripts/_lib.py`. Stage 5's labeling loop swaps its schema-only check for `validate_example(...)`; Stage 4 adds new per-example fields and aggregate rows. The spec at `docs/superpowers/specs/2026-05-16-validator-and-amount-parser-design.md` is the source of truth — when in doubt, refer back to it.

**Tech Stack:** Python 3.11, stdlib only for the new modules (`re`, `dataclasses`, `math`), `jsonschema` (already in `requirements.txt`), `pytest` + `pytest-cov` for tests. No model dependencies. Project uses flat-script imports (`sys.path.insert(0, scripts_dir)` then `from _lib import ...`); new modules and tests must follow that pattern.

**File map:**

| File | Purpose |
|---|---|
| `requirements-dev.txt` (new) | Dev deps: `pytest`, `pytest-cov` |
| `scripts/amount_parser.py` (new) | Text → `list[AmountCandidate]`; no schema/model deps |
| `scripts/validator.py` (new) | `(input, output_dict) → ValidationResult`; lazy import of `_lib` |
| `scripts/_lib.py` (modify, append-only) | Add re-export block at bottom |
| `scripts/04_eval.py` (modify) | Add `score_validation_fields` helper + per-example/aggregate metrics |
| `scripts/05_generate_distillation_data.py` (modify) | Replace schema-only gate; new `failed.jsonl` shape; `--retry-validation-failed` flag |
| `tests/test_amount_parser.py` (new) | ≥95% line coverage on parser |
| `tests/test_validator.py` (new) | ≥90% line coverage on validator |
| `tests/test_stage4_metrics.py` (new) | One smoke test for `score_validation_fields` |
| `README.md` (modify) | Short deltas in Stage 4 / Stage 5 sections + dev-test command |

**Commit split:**
- **Commit A** (after Task 11): parser + validator + tests + `_lib` re-exports.
- **Commit B** (after Task 15): Stage 4/Stage 5 integration + README delta.

---

### Task 1: Add `requirements-dev.txt` and verify pytest works

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/__init__.py` (empty, so pytest picks the dir up)
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `requirements-dev.txt`**

```text
pytest>=8.0
pytest-cov>=5.0
```

- [ ] **Step 2: Install dev deps**

Run (PowerShell or bash, depending on shell):
```bash
pip install -r requirements-dev.txt
```
Expected: pytest and pytest-cov install cleanly. Verify with `pytest --version`.

- [ ] **Step 3: Create empty `tests/__init__.py`**

Just an empty file so pytest's default rootdir discovery is unambiguous.

- [ ] **Step 4: Create `tests/conftest.py` to put `scripts/` on `sys.path`**

```python
"""Pytest config — put scripts/ on sys.path so tests can use the same
flat-import style as the scripts themselves (`from amount_parser import ...`)."""
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
```

- [ ] **Step 5: Sanity check — pytest collects no tests but exits 5 (not 1)**

Run: `pytest tests/`
Expected: exit code 5 ("no tests ran"), no errors. This proves discovery works and `conftest.py` doesn't crash on import.

- [ ] **Step 6: Do not commit yet.** First substantive commit happens at Task 11.

---

### Task 2: AmountCandidate data shape + parser scaffolding

**Files:**
- Create: `scripts/amount_parser.py`
- Create: `tests/test_amount_parser.py`

Spec reference: §3.1 (data shape), §3.4–3.5 (vocabulary), §3.6 (supersession).

- [ ] **Step 1: Write the first failing test for the data shape and empty-input behavior**

`tests/test_amount_parser.py`:
```python
import math
from amount_parser import AmountCandidate, parse_amounts


def test_amountcandidate_is_frozen_dataclass():
    c = AmountCandidate(
        value=500.0, raw="500", span=(0, 3),
        status="active", source="digits", currency_hint=None,
    )
    assert c.value == 500.0
    assert c.raw == "500"
    assert c.span == (0, 3)
    assert c.status == "active"
    assert c.source == "digits"
    assert c.currency_hint is None


def test_parse_amounts_empty_string_returns_empty_list():
    assert parse_amounts("") == []


def test_parse_amounts_text_without_amounts_returns_empty_list():
    assert parse_amounts("just some words here") == []
```

- [ ] **Step 2: Run, watch it fail (module missing)**

Run: `pytest tests/test_amount_parser.py -v`
Expected: `ModuleNotFoundError: No module named 'amount_parser'`.

- [ ] **Step 3: Implement the scaffolding**

`scripts/amount_parser.py`:
```python
"""Deterministic amount parser. Text in -> list of AmountCandidate.

This module knows nothing about the JSON schema, transaction categories,
or model outputs. It is pure stdlib (re, dataclasses, math) so it can be
imported by training-time tooling AND embedded in runtime validators.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Status = Literal["active", "superseded"]
Source = Literal[
    "decimal_k", "k_suffix",
    "digits_lakh", "digits_hazaar", "digits_sau",
    "currency_symbol", "slash_dash",
    "digits",
    "number_word_en", "number_word_hi",
]
CurrencyHint = Literal["INR", "USD"]


@dataclass(frozen=True)
class AmountCandidate:
    value: float
    raw: str
    span: tuple[int, int]
    status: Status
    source: Source
    currency_hint: CurrencyHint | None = None


def parse_amounts(text: str) -> list[AmountCandidate]:
    """Return amount candidates in input order. Spans never overlap.

    Implemented incrementally — see subsequent tasks for each priority
    band of matchers.
    """
    if not text:
        return []
    return []  # stub — filled in by later tasks
```

- [ ] **Step 4: Run tests, all three pass**

Run: `pytest tests/test_amount_parser.py -v`
Expected: 3 passed.

---

### Task 3: Priority 1–2 matchers (`decimal_k`, `k_suffix`)

**Files:**
- Modify: `scripts/amount_parser.py`
- Modify: `tests/test_amount_parser.py`

Spec reference: §3.2 priorities 1–2.

- [ ] **Step 1: Add failing tests for k/K suffix amounts**

Append to `tests/test_amount_parser.py`:
```python
import pytest


@pytest.mark.parametrize("text,expected_value,expected_source", [
    ("1.5k",        1500.0, "decimal_k"),
    ("2.5K",        2500.0, "decimal_k"),
    ("0.5k",         500.0, "decimal_k"),
    ("50k",        50000.0, "k_suffix"),
    ("5K",          5000.0, "k_suffix"),
    ("12k",        12000.0, "k_suffix"),
])
def test_k_suffix_amounts(text, expected_value, expected_source):
    cands = parse_amounts(text)
    assert len(cands) == 1
    c = cands[0]
    assert math.isclose(c.value, expected_value, abs_tol=0.001)
    assert c.source == expected_source
    assert c.status == "active"
```

- [ ] **Step 2: Run, watch them fail**

Run: `pytest tests/test_amount_parser.py -v -k k_suffix`
Expected: 6 failures (all return empty list).

- [ ] **Step 3: Implement priority 1–2**

In `scripts/amount_parser.py`, replace the stub body of `parse_amounts` and add module-level constants/helpers:

```python
import re


# Each matcher returns list of (start, end, AmountCandidate) tuples for
# this priority band. parse_amounts runs them in priority order and
# discards any later candidate whose span overlaps an already-claimed
# range.

_DECIMAL_K_RE = re.compile(r"(\d+\.\d+)\s*[kK]\b")
_INT_K_RE     = re.compile(r"(?<!\.)\b(\d+)\s*[kK]\b")


def _match_decimal_k(text: str) -> list[AmountCandidate]:
    out = []
    for m in _DECIMAL_K_RE.finditer(text):
        out.append(AmountCandidate(
            value=float(m.group(1)) * 1000.0,
            raw=m.group(0),
            span=(m.start(), m.end()),
            status="active",
            source="decimal_k",
        ))
    return out


def _match_int_k(text: str) -> list[AmountCandidate]:
    out = []
    for m in _INT_K_RE.finditer(text):
        out.append(AmountCandidate(
            value=float(m.group(1)) * 1000.0,
            raw=m.group(0),
            span=(m.start(), m.end()),
            status="active",
            source="k_suffix",
        ))
    return out


def _overlaps(span: tuple[int, int], claimed: list[tuple[int, int]]) -> bool:
    s, e = span
    return any(not (e <= cs or s >= ce) for cs, ce in claimed)


def parse_amounts(text: str) -> list[AmountCandidate]:
    if not text:
        return []
    claimed: list[tuple[int, int]] = []
    out: list[AmountCandidate] = []

    # Priority order: each band runs and claims spans; later bands skip
    # any candidate whose span overlaps an already-claimed range.
    for band in (_match_decimal_k, _match_int_k):
        for cand in band(text):
            if _overlaps(cand.span, claimed):
                continue
            out.append(cand)
            claimed.append(cand.span)

    out.sort(key=lambda c: c.span[0])
    return out
```

- [ ] **Step 4: Run all parser tests, all pass**

Run: `pytest tests/test_amount_parser.py -v`
Expected: all tests pass (3 from Task 2 + 6 from Task 3 = 9 passed).

---

### Task 4: Priority 3–5 matchers (digits + lakh/hazaar/sau)

**Files:**
- Modify: `scripts/amount_parser.py`
- Modify: `tests/test_amount_parser.py`

Spec reference: §3.2 priorities 3–5. These matchers MUST beat the plain-digit matcher (added in Task 6) so `"1 lakh"` is captured as 100000 rather than as `1`.

- [ ] **Step 1: Add failing tests for digit + Hindi/English unit**

Append to `tests/test_amount_parser.py`:
```python
@pytest.mark.parametrize("text,expected_value,expected_source", [
    # lakh
    ("1 lakh",          100000.0, "digits_lakh"),
    ("1.5 lakh",        150000.0, "digits_lakh"),
    ("2 lakhs",         200000.0, "digits_lakh"),
    # hazaar / thousand
    ("5 hazaar",          5000.0, "digits_hazaar"),
    ("2 thousand",        2000.0, "digits_hazaar"),
    ("10 hazaar",        10000.0, "digits_hazaar"),
    # sau / hundred
    ("2 sau",              200.0, "digits_sau"),
    ("5 hundred",          500.0, "digits_sau"),
])
def test_digit_with_unit(text, expected_value, expected_source):
    cands = parse_amounts(text)
    assert len(cands) == 1, f"expected one candidate, got {cands}"
    c = cands[0]
    assert math.isclose(c.value, expected_value, abs_tol=0.001)
    assert c.source == expected_source
    assert c.status == "active"


def test_unit_beats_plain_digit_priority_regression():
    """1 lakh must produce 100000, not '1' as a plain digit."""
    cands = parse_amounts("1 lakh rent")
    assert len(cands) == 1
    assert math.isclose(cands[0].value, 100000.0)
```

- [ ] **Step 2: Run, watch them fail**

Run: `pytest tests/test_amount_parser.py -v -k "digit_with_unit or priority_regression"`
Expected: 9 failures.

- [ ] **Step 3: Implement priority 3–5**

In `scripts/amount_parser.py`, add three matchers and wire them into the priority loop AHEAD of `_match_decimal_k` would be wrong (decimal_k beats `5.5 lakh` since k is more specific). Actual order matches spec §3.2: decimal_k, int_k, then lakh/hazaar/sau.

```python
_DIGITS_LAKH_RE   = re.compile(r"\b(\d+(?:\.\d+)?)\s*lakhs?\b", re.IGNORECASE)
_DIGITS_HAZAAR_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:hazaar|hazar|thousand)\b", re.IGNORECASE)
_DIGITS_SAU_RE    = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:sau|hundred)\b", re.IGNORECASE)


def _match_digits_lakh(text: str) -> list[AmountCandidate]:
    return [
        AmountCandidate(
            value=float(m.group(1)) * 100000.0,
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="digits_lakh",
        )
        for m in _DIGITS_LAKH_RE.finditer(text)
    ]


def _match_digits_hazaar(text: str) -> list[AmountCandidate]:
    return [
        AmountCandidate(
            value=float(m.group(1)) * 1000.0,
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="digits_hazaar",
        )
        for m in _DIGITS_HAZAAR_RE.finditer(text)
    ]


def _match_digits_sau(text: str) -> list[AmountCandidate]:
    return [
        AmountCandidate(
            value=float(m.group(1)) * 100.0,
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="digits_sau",
        )
        for m in _DIGITS_SAU_RE.finditer(text)
    ]
```

Update the priority loop in `parse_amounts`:
```python
    for band in (
        _match_decimal_k, _match_int_k,
        _match_digits_lakh, _match_digits_hazaar, _match_digits_sau,
    ):
        for cand in band(text):
            if _overlaps(cand.span, claimed):
                continue
            out.append(cand)
            claimed.append(cand.span)
```

- [ ] **Step 4: Run all parser tests, all pass**

Run: `pytest tests/test_amount_parser.py -v`
Expected: all pass.

---

### Task 5: Priority 6 matcher (currency-prefixed digits + currency_hint)

**Files:**
- Modify: `scripts/amount_parser.py`
- Modify: `tests/test_amount_parser.py`

Spec reference: §3.2 priority 6.

- [ ] **Step 1: Add failing tests for currency-marked amounts**

Append to `tests/test_amount_parser.py`:
```python
@pytest.mark.parametrize("text,value,currency,source", [
    ("$50 coffee",      50.0, "USD", "currency_symbol"),
    ("₹300 lunch",     300.0, "INR", "currency_symbol"),
    ("rs 500 dinner",  500.0, "INR", "currency_symbol"),
    ("Rs.500 petrol",  500.0, "INR", "currency_symbol"),
    ("500/- chai",     500.0, "INR", "slash_dash"),
])
def test_currency_hint(text, value, currency, source):
    cands = parse_amounts(text)
    assert len(cands) == 1, f"expected one candidate, got {cands}"
    c = cands[0]
    assert math.isclose(c.value, value, abs_tol=0.001)
    assert c.currency_hint == currency
    assert c.source == source


def test_plain_digit_has_no_currency_hint():
    cands = parse_amounts("500 chai")
    assert len(cands) == 1
    assert cands[0].currency_hint is None
```

- [ ] **Step 2: Run, watch them fail**

Run: `pytest tests/test_amount_parser.py -v -k currency_hint`
Expected: 5 failures (the no-hint test will fail too because plain-digit matcher doesn't exist yet, see Task 6).

- [ ] **Step 3: Implement priority 6**

Add to `scripts/amount_parser.py`:
```python
_CURRENCY_PREFIX_RE = re.compile(
    r"""
    (?P<sym>\$|₹|Rs\.?|rs\.?)      # currency symbol
    \s*
    (?P<num>\d+(?:[.,]\d+)*)       # number, allow commas/decimals
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SLASH_DASH_RE = re.compile(r"\b(\d+(?:[.,]\d+)*)\s*/-")


def _to_float(numstr: str) -> float:
    return float(numstr.replace(",", ""))


def _match_currency_prefix(text: str) -> list[AmountCandidate]:
    out = []
    for m in _CURRENCY_PREFIX_RE.finditer(text):
        sym = m.group("sym").lower()
        currency: CurrencyHint = "USD" if sym == "$" else "INR"
        out.append(AmountCandidate(
            value=_to_float(m.group("num")),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="currency_symbol",
            currency_hint=currency,
        ))
    return out


def _match_slash_dash(text: str) -> list[AmountCandidate]:
    return [
        AmountCandidate(
            value=_to_float(m.group(1)),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="slash_dash",
            currency_hint="INR",
        )
        for m in _SLASH_DASH_RE.finditer(text)
    ]
```

Update priority loop to include `_match_currency_prefix, _match_slash_dash` after `_match_digits_sau`.

The `test_plain_digit_has_no_currency_hint` test still fails until Task 6 (plain digits not implemented). That's acceptable — it'll pass once Task 6 is in.

- [ ] **Step 4: Run currency_hint tests, 5 of 6 pass**

Run: `pytest tests/test_amount_parser.py -v -k currency_hint`
Expected: the 5 parameterized cases pass; `test_plain_digit_has_no_currency_hint` still fails (no plain-digit matcher yet — Task 6).

---

### Task 6: Priority 7 matcher (plain digits) with rejection rules

**Files:**
- Modify: `scripts/amount_parser.py`
- Modify: `tests/test_amount_parser.py`

Spec reference: §3.2 priority 7, §3.3 rejection rules.

- [ ] **Step 1: Add failing tests for plain digits and non-amount rejections**

Append to `tests/test_amount_parser.py`:
```python
@pytest.mark.parametrize("text,expected_value", [
    ("5000",       5000.0),
    ("50,000",    50000.0),
    ("12500",     12500.0),
    ("paid 125000 rent",  125000.0),
])
def test_plain_digits(text, expected_value):
    cands = parse_amounts(text)
    assert len(cands) == 1
    assert math.isclose(cands[0].value, expected_value, abs_tol=0.001)
    assert cands[0].source == "digits"


@pytest.mark.parametrize("text", [
    "phone number 9876543210",
    "otp 123456",
    "mobile 9876543210",
    "order id 12345678",
    "pin 1234",                    # preceded-by-keyword
    "account 1122334455",
    "upi id 1234",
])
def test_plain_digits_rejected_by_keyword_or_length(text):
    assert parse_amounts(text) == []


@pytest.mark.parametrize("text", [
    "12:30 train",
    "at 9:45 meeting",
    "12/05/2026 due",
    "due 01/01/2027",
])
def test_plain_digits_rejected_by_time_date_pattern(text):
    assert parse_amounts(text) == []
```

- [ ] **Step 2: Run, watch them fail**

Run: `pytest tests/test_amount_parser.py -v -k plain_digits`
Expected: failures on the positive parametrize (no candidate produced) AND the rejection tests would produce candidates if implemented naively.

- [ ] **Step 3: Implement priority 7 with rejection**

Add to `scripts/amount_parser.py`:
```python
_PLAIN_DIGITS_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d+)\b")

_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")

_REJECT_KEYWORDS = (
    "phone", "mobile", "otp", "pin",
    "account", "upi id", "order id",
)
_KEYWORD_WINDOW_CHARS = 30  # roughly ~5 tokens


def _blocked_spans(text: str) -> list[tuple[int, int]]:
    """Pre-compute time/date pattern spans so any digit overlapping them is rejected."""
    spans: list[tuple[int, int]] = []
    for rx in (_TIME_RE, _DATE_RE):
        for m in rx.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def _preceded_by_keyword(text: str, start: int) -> bool:
    lookback_start = max(0, start - _KEYWORD_WINDOW_CHARS)
    snippet = text[lookback_start:start].lower()
    return any(kw in snippet for kw in _REJECT_KEYWORDS)


def _match_plain_digits(
    text: str, blocked: list[tuple[int, int]],
) -> list[AmountCandidate]:
    out = []
    for m in _PLAIN_DIGITS_RE.finditer(text):
        raw = m.group(0)
        digits_only = raw.replace(",", "")
        if len(digits_only) >= 7:
            continue
        if _overlaps((m.start(), m.end()), blocked):
            continue
        if _preceded_by_keyword(text, m.start()):
            continue
        out.append(AmountCandidate(
            value=_to_float(raw),
            raw=raw, span=(m.start(), m.end()),
            status="active", source="digits",
        ))
    return out
```

Wire it into `parse_amounts`:
```python
def parse_amounts(text: str) -> list[AmountCandidate]:
    if not text:
        return []
    claimed: list[tuple[int, int]] = []
    blocked = _blocked_spans(text)
    out: list[AmountCandidate] = []

    higher_priority_bands = (
        _match_decimal_k, _match_int_k,
        _match_digits_lakh, _match_digits_hazaar, _match_digits_sau,
        _match_currency_prefix, _match_slash_dash,
    )
    for band in higher_priority_bands:
        for cand in band(text):
            if _overlaps(cand.span, claimed):
                continue
            out.append(cand)
            claimed.append(cand.span)

    for cand in _match_plain_digits(text, blocked):
        if _overlaps(cand.span, claimed):
            continue
        out.append(cand)
        claimed.append(cand.span)

    out.sort(key=lambda c: c.span[0])
    return out
```

- [ ] **Step 4: Run parser tests, all pass**

Run: `pytest tests/test_amount_parser.py -v`
Expected: all parser tests pass, including the Task 5 `test_plain_digit_has_no_currency_hint` that depended on the plain-digit matcher.

---

### Task 7: Priority 8–9 matchers (English and Hindi number-word + unit)

**Files:**
- Modify: `scripts/amount_parser.py`
- Modify: `tests/test_amount_parser.py`

Spec reference: §3.2 priorities 8–9, §3.4–3.5 (vocabularies).

- [ ] **Step 1: Add failing tests for number-word + unit**

Append to `tests/test_amount_parser.py`:
```python
@pytest.mark.parametrize("text,expected_value,expected_source", [
    # English
    ("five hundred",       500.0, "number_word_en"),
    ("two thousand",      2000.0, "number_word_en"),
    ("one lakh",        100000.0, "number_word_en"),
    ("five lakh",       500000.0, "number_word_en"),
    ("twenty five hundred", 2500.0, "number_word_en"),
    # Hindi
    ("paanch sau",         500.0, "number_word_hi"),
    ("do hazaar",         2000.0, "number_word_hi"),
    ("char lakh",       400000.0, "number_word_hi"),
    ("chhe sau",           600.0, "number_word_hi"),
    ("ek hazaar",         1000.0, "number_word_hi"),
])
def test_number_word_with_unit(text, expected_value, expected_source):
    cands = parse_amounts(text)
    assert len(cands) == 1, f"expected one candidate, got {cands}"
    c = cands[0]
    assert math.isclose(c.value, expected_value, abs_tol=0.001)
    assert c.source == expected_source


@pytest.mark.parametrize("text", [
    "do coffee",      # bare Hindi numeral, no unit
    "char people",
    "paanch minute",
    "two minutes",    # English numeral, no amount unit
    "second item",    # ordinal, not a numeral
])
def test_bare_number_words_not_parsed(text):
    assert parse_amounts(text) == []
```

- [ ] **Step 2: Run, watch them fail**

Run: `pytest tests/test_amount_parser.py -v -k number_word`
Expected: 10 failures from the positive cases; the bare-word tests pass coincidentally (parser returns `[]` because none of the existing matchers fires).

- [ ] **Step 3: Implement priority 8–9**

Add to `scripts/amount_parser.py`:
```python
_EN_NUMERAL = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_HI_NUMERAL = {
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4,
    "paanch": 5, "chhe": 6, "che": 6,
    "saat": 7, "aath": 8, "nau": 9, "das": 10,
}

# English numeral word + (hundred|thousand|lakh)
_EN_UNIT_WORD = {"hundred": 100, "thousand": 1000, "lakh": 100000}
_HI_UNIT_WORD = {"sau": 100, "hazaar": 1000, "hazar": 1000, "lakh": 100000}

_EN_SIMPLE_RE = re.compile(
    r"\b(" + "|".join(_EN_NUMERAL) + r")\s+(" + "|".join(_EN_UNIT_WORD) + r")\b",
    re.IGNORECASE,
)
# Compound English: <one..twenty> <one..nine> hundred -> N*100 + M*100? No —
# spec §3.4: "twenty five hundred" = 25 * 100 = 2500. So the leading word is
# tens (twenty), the second is ones (five), they combine before * 100.
_EN_COMPOUND_RE = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+"
    r"(one|two|three|four|five|six|seven|eight|nine)\s+(hundred)\b",
    re.IGNORECASE,
)
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_HI_UNIT_RE = re.compile(
    r"\b(" + "|".join(sorted(_HI_NUMERAL, key=len, reverse=True))
    + r")\s+(" + "|".join(sorted(_HI_UNIT_WORD, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _match_number_word_en(text: str) -> list[AmountCandidate]:
    out: list[AmountCandidate] = []
    # Compound first so "twenty five hundred" beats "five hundred"
    for m in _EN_COMPOUND_RE.finditer(text):
        tens = _TENS[m.group(1).lower()]
        ones = _EN_NUMERAL[m.group(2).lower()]
        out.append(AmountCandidate(
            value=float((tens + ones) * 100),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="number_word_en",
        ))
    for m in _EN_SIMPLE_RE.finditer(text):
        num = _EN_NUMERAL[m.group(1).lower()]
        unit = _EN_UNIT_WORD[m.group(2).lower()]
        out.append(AmountCandidate(
            value=float(num * unit),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="number_word_en",
        ))
    return out


def _match_number_word_hi(text: str) -> list[AmountCandidate]:
    out: list[AmountCandidate] = []
    for m in _HI_UNIT_RE.finditer(text):
        num = _HI_NUMERAL[m.group(1).lower()]
        unit = _HI_UNIT_WORD[m.group(2).lower()]
        out.append(AmountCandidate(
            value=float(num * unit),
            raw=m.group(0), span=(m.start(), m.end()),
            status="active", source="number_word_hi",
        ))
    return out
```

Wire them in as the **last** priority bands (after plain digits) so unit-bearing numeric forms still win when they overlap:

```python
    # English / Hindi number-word + unit — lowest priority because they're
    # the most fuzzy; plain digits still win head-to-head via overlap.
    for band in (_match_number_word_en, _match_number_word_hi):
        for cand in band(text):
            if _overlaps(cand.span, claimed):
                continue
            out.append(cand)
            claimed.append(cand.span)
```

- [ ] **Step 4: Run parser tests, all pass**

Run: `pytest tests/test_amount_parser.py -v`
Expected: all pass.

---

### Task 8: Supersession pass (correction markers)

**Files:**
- Modify: `scripts/amount_parser.py`
- Modify: `tests/test_amount_parser.py`

Spec reference: §3.6.

- [ ] **Step 1: Add failing tests for supersession**

Append to `tests/test_amount_parser.py`:
```python
def _statuses(text):
    return [(c.value, c.status) for c in parse_amounts(text)]


def test_supersession_simple_wait_no():
    assert _statuses("500 beer wait no 600 beer") == [
        (500.0, "superseded"), (600.0, "active"),
    ]


def test_supersession_rapid_fire():
    assert _statuses("500 no 600 no 700") == [
        (500.0, "superseded"), (600.0, "superseded"), (700.0, "active"),
    ]


def test_no_supersession_when_no_replacement():
    """'no' after a candidate but no later candidate -> stays active."""
    assert _statuses("500 beer no extra charges") == [(500.0, "active")]


def test_no_supersession_for_real_two_transactions():
    """No marker between candidates -> both active."""
    assert _statuses("500 beer and 500 candy") == [(500.0, "active"), (500.0, "active")]


def test_no_supersession_for_repeated_amount_no_marker():
    assert _statuses("500 beer and 500 beer") == [(500.0, "active"), (500.0, "active")]


def test_hindi_correction_marker_nahi():
    assert _statuses("paanch sau nahi chhe sau") == [
        (500.0, "superseded"), (600.0, "active"),
    ]


def test_bare_wait_is_not_a_marker():
    """'wait and' / 'wait for' / bare 'wait' should NOT supersede."""
    assert _statuses("500 beer wait and 600 candy") == [
        (500.0, "active"), (600.0, "active"),
    ]


def test_window_too_far_no_supersession():
    """A marker far beyond 6 tokens from the previous candidate must not supersede."""
    text = "500 beer was nice and tasty and refreshing on a hot day no actually 600"
    statuses = _statuses(text)
    # 500 stays active because the marker is too far from it.
    assert (500.0, "active") in statuses
```

- [ ] **Step 2: Run, watch them fail**

Run: `pytest tests/test_amount_parser.py -v -k supersession or test_hindi_correction_marker_nahi or test_bare_wait_is_not_a_marker or test_window`
Expected: failures — supersession not yet implemented.

- [ ] **Step 3: Implement supersession**

Add to `scripts/amount_parser.py`:
```python
import dataclasses


# Multi-token compound markers MUST be matched before bare-token forms
# to avoid spurious bare-'no' matches inside 'wait no'.
_COMPOUND_MARKER_PHRASES = (
    "wait no", "wait actually", "wait sorry", "wait correction",
    "wait,", "i mean", "scratch that", "umm no", "uhh no",
)
_BARE_MARKER_WORDS = (
    "actually", "nope", "nahi", "nahin", "galat", "correction:",
)
_BARE_NO = re.compile(r"\bno\b", re.IGNORECASE)


def _find_correction_marker_spans(text: str) -> list[tuple[int, int]]:
    """Return spans where a correction marker appears, in input order."""
    spans: list[tuple[int, int]] = []
    lower = text.lower()
    # Compound phrases first
    for phrase in _COMPOUND_MARKER_PHRASES:
        start = 0
        while True:
            i = lower.find(phrase, start)
            if i == -1:
                break
            spans.append((i, i + len(phrase)))
            start = i + len(phrase)
    # Bare markers — \b boundary
    for word in _BARE_MARKER_WORDS:
        for m in re.finditer(rf"\b{re.escape(word)}\b", lower):
            spans.append((m.start(), m.end()))
    # Bare 'no' — but only if NOT already inside a compound marker span.
    for m in _BARE_NO.finditer(lower):
        s, e = m.start(), m.end()
        if any(cs <= s and e <= ce for cs, ce in spans):
            continue
        spans.append((s, e))

    spans.sort()
    return spans


_TOKEN_RE = re.compile(r"\S+")
_WINDOW_TOKENS = 6


def _token_index_at_char(text: str, char_pos: int) -> int:
    """Return the index of the first token whose start >= char_pos.

    Used to measure 'distance in tokens' between a candidate and a marker.
    """
    idx = 0
    for i, m in enumerate(_TOKEN_RE.finditer(text)):
        if m.start() >= char_pos:
            return i
        idx = i + 1
    return idx


def _apply_supersession(
    text: str, candidates: list[AmountCandidate],
) -> list[AmountCandidate]:
    if len(candidates) < 2:
        return candidates
    marker_spans = _find_correction_marker_spans(text)
    if not marker_spans:
        return candidates

    # Build (token_index, candidate_index) lookup for candidates.
    cand_token_idx = [_token_index_at_char(text, c.span[0]) for c in candidates]
    marker_token_idx = [_token_index_at_char(text, s) for s, _ in marker_spans]

    superseded: set[int] = set()
    for mtok in marker_token_idx:
        # Find nearest active candidate strictly before this marker, within window.
        prev_i = -1
        for i, ctok in enumerate(cand_token_idx):
            if ctok < mtok and (mtok - ctok) <= _WINDOW_TOKENS and i not in superseded:
                prev_i = i  # keep the latest qualifying one
        if prev_i == -1:
            continue
        # Find a replacement candidate AFTER the marker within window.
        next_i = -1
        for j, ctok in enumerate(cand_token_idx):
            if ctok > mtok and (ctok - mtok) <= _WINDOW_TOKENS and j not in superseded:
                next_i = j
                break
        if next_i == -1:
            continue
        superseded.add(prev_i)

    return [
        dataclasses.replace(c, status="superseded") if i in superseded else c
        for i, c in enumerate(candidates)
    ]
```

Wire it in at the end of `parse_amounts`:
```python
    out = _apply_supersession(text, out)
    return out
```

- [ ] **Step 4: Run all parser tests, all pass**

Run: `pytest tests/test_amount_parser.py -v`
Expected: all pass.

- [ ] **Step 5: Check coverage target (≥95% lines on `amount_parser.py`)**

Run: `pytest tests/test_amount_parser.py --cov=amount_parser --cov-report=term-missing`
Expected: coverage ≥95%. If under, look at the "missing" line numbers and add a fixture. Common gaps: comma-stripping in `_to_float`, the `lookback_start = 0` edge in `_preceded_by_keyword`. Add one or two more parametrized rows to cover them.

---

### Task 9: Validator data shape + schema preflight

**Files:**
- Create: `scripts/validator.py`
- Create: `tests/test_validator.py`

Spec reference: §4.1 (API), §4.2 (imports), §4.3 step 1 (schema preflight).

- [ ] **Step 1: Write failing tests for the shape and schema preflight**

`tests/test_validator.py`:
```python
import json

from validator import (
    ValidationError, ValidationResult,
    validate_example, serialize_validation_result,
)


def _good_output(amount=500, currency="INR", item="beer", category="Drinks", type_="expense"):
    return {"transactions": [{
        "amount": amount, "currency": currency, "item": item,
        "category": category, "type": type_,
    }]}


def test_validationerror_shape():
    e = ValidationError(code="SCHEMA_INVALID", path="/transactions",
                        message="x", severity="error")
    assert e.code == "SCHEMA_INVALID"
    assert e.severity == "error"


def test_schema_invalid_missing_field_returns_early():
    bad = {"transactions": [{"amount": 500}]}  # missing currency/item/etc
    result = validate_example("500 beer", bad)
    assert not result.ok
    codes = [e.code for e in result.errors]
    # SCHEMA_INVALID present, no downstream codes since validator returns early
    assert "SCHEMA_INVALID" in codes
    assert "AMOUNT_NOT_IN_INPUT" not in codes


def test_schema_invalid_wrong_type():
    bad = {"transactions": "not an array"}
    result = validate_example("500 beer", bad)
    assert not result.ok
    assert any(e.code == "SCHEMA_INVALID" for e in result.errors)


def test_schema_invalid_bad_enum():
    bad = _good_output()
    bad["transactions"][0]["category"] = "NotACategory"
    result = validate_example("500 beer", bad)
    assert not result.ok
    assert any(e.code == "SCHEMA_INVALID" for e in result.errors)


def test_serialize_validation_result_excludes_candidates_and_is_json_safe():
    result = validate_example("500 beer", _good_output())
    payload = serialize_validation_result(result)
    assert "candidates" not in payload
    assert "errors" in payload
    # Round-trip through JSON cleanly.
    json.loads(json.dumps(payload))
```

- [ ] **Step 2: Run, watch them fail**

Run: `pytest tests/test_validator.py -v`
Expected: `ModuleNotFoundError: No module named 'validator'`.

- [ ] **Step 3: Implement validator scaffolding + schema preflight**

`scripts/validator.py`:
```python
"""Semantic validator: (input_text, output_obj) -> ValidationResult.

The validator runs schema preflight (lazy import from _lib) and then
input-vs-output consistency checks. It never calls a model. Errors
returned via ValidationResult.errors carry stable machine-readable codes.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Literal

from amount_parser import AmountCandidate, parse_amounts


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationError:
    code: str
    path: str
    message: str
    severity: Severity


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[ValidationError]
    amount_values_match_active_candidates: bool
    txn_count_matches_active_candidates: bool
    duplicate_transactions_found: bool
    superseded_amount_used: bool
    candidates: list[AmountCandidate]


def _result_from(
    errors: list[ValidationError],
    *,
    mode: str,
    candidates: list[AmountCandidate],
    amount_match_active: bool = False,
    txn_count_match_active: bool = False,
    duplicate_found: bool = False,
    superseded_used: bool = False,
) -> ValidationResult:
    has_error = any(e.severity == "error" for e in errors)
    ok = True if mode == "warn" else not has_error
    return ValidationResult(
        ok=ok, errors=errors, candidates=candidates,
        amount_values_match_active_candidates=amount_match_active,
        txn_count_matches_active_candidates=txn_count_match_active,
        duplicate_transactions_found=duplicate_found,
        superseded_amount_used=superseded_used,
    )


def validate_example(
    input_text: str,
    output_obj: dict,
    *,
    mode: Literal["strict", "warn"] = "strict",
) -> ValidationResult:
    # Lazy import to avoid a circular load with _lib (which re-exports
    # this module's symbols at its file bottom).
    from _lib import is_schema_valid, schema_errors

    candidates = parse_amounts(input_text)

    if not is_schema_valid(output_obj):
        errs = [
            ValidationError(
                code="SCHEMA_INVALID",
                path="/" + err.split(":", 1)[0].strip("<>") if ":" in err else "/transactions",
                message=err,
                severity="error",
            )
            for err in schema_errors(output_obj)
        ]
        return _result_from(errs, mode=mode, candidates=candidates)

    # Downstream checks (amount, currency, count, duplicate) — implemented in later tasks.
    return _result_from([], mode=mode, candidates=candidates,
                        amount_match_active=True, txn_count_match_active=True)


def serialize_validation_result(result: ValidationResult) -> dict:
    """Stable JSON-encodable shape. Excludes `candidates` so callers can
    write them at a different top-level key in failed.jsonl."""
    return {
        "ok": result.ok,
        "amount_values_match_active_candidates": result.amount_values_match_active_candidates,
        "txn_count_matches_active_candidates": result.txn_count_matches_active_candidates,
        "duplicate_transactions_found": result.duplicate_transactions_found,
        "superseded_amount_used": result.superseded_amount_used,
        "errors": [dataclasses.asdict(e) for e in result.errors],
    }
```

- [ ] **Step 4: Run validator tests, all pass**

Run: `pytest tests/test_validator.py -v`
Expected: all 5 tests pass.

---

### Task 10: Validator — amount + currency + count + duplicate + no-amount

**Files:**
- Modify: `scripts/validator.py`
- Modify: `tests/test_validator.py`

Spec reference: §4.3 steps 2–7, §4.4 (error codes), §4.5 (duplicate rule).

This task is larger because the checks share the same `candidates` list and are easier to land together. Step-by-step it's still TDD: test first, run-watch-fail, implement, run-watch-pass.

- [ ] **Step 1: Add failing tests for all remaining error codes**

Append to `tests/test_validator.py`:
```python
def test_amount_not_in_input_when_model_invents_amount():
    out = _good_output(amount=500)
    result = validate_example("fifty thousand for laptop", out)
    assert not result.ok
    assert any(e.code == "AMOUNT_NOT_IN_INPUT" for e in result.errors)


def test_amount_in_input_passes():
    out = _good_output(amount=500)
    result = validate_example("500 beer", out)
    assert result.ok
    assert result.amount_values_match_active_candidates is True


def test_superseded_amount_used_flagged():
    out = _good_output(amount=500)
    result = validate_example("500 beer wait no 600 beer", out)
    assert not result.ok
    assert any(e.code == "SUPERSEDED_AMOUNT_USED" for e in result.errors)
    assert result.superseded_amount_used is True


def test_currency_mismatch_uses_per_candidate_hint():
    out = _good_output(amount=50, currency="INR")
    result = validate_example("$50 coffee", out)
    assert not result.ok
    assert any(e.code == "CURRENCY_MISMATCH" for e in result.errors)


def test_mixed_currency_input_happy_path():
    out = {"transactions": [
        {"amount": 50,  "currency": "USD", "item": "coffee", "category": "Drinks", "type": "expense"},
        {"amount": 300, "currency": "INR", "item": "lunch",  "category": "Food",   "type": "expense"},
    ]}
    result = validate_example("$50 coffee and ₹300 lunch", out)
    assert result.ok, result.errors


def test_txn_count_exceeds_candidates():
    out = {"transactions": [
        _good_output()["transactions"][0],
        _good_output(item="chai", category="Drinks")["transactions"][0],
    ]}
    result = validate_example("500 beer", out)
    assert not result.ok
    assert any(e.code == "TXN_COUNT_EXCEEDS_CANDIDATES" for e in result.errors)


def test_txn_count_below_candidates_is_warning_only():
    out = _good_output(amount=500)
    result = validate_example("500 beer and 50 candy", out)
    codes = [e.code for e in result.errors]
    assert "TXN_COUNT_BELOW_CANDIDATES" in codes
    severities = {e.code: e.severity for e in result.errors}
    assert severities["TXN_COUNT_BELOW_CANDIDATES"] == "warning"
    # Warning-only -> ok still True in strict mode.
    assert result.ok is True


def test_suspicious_duplicate_unjustified():
    out = {"transactions": [
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
    ]}
    result = validate_example("500 beer", out)
    assert not result.ok
    assert any(e.code == "SUSPICIOUS_DUPLICATE" for e in result.errors)
    assert result.duplicate_transactions_found is True


def test_duplicate_justified_by_repeated_amount_and_item():
    out = {"transactions": [
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
    ]}
    result = validate_example("500 beer and 500 beer", out)
    assert result.ok, result.errors


def test_duplicate_justified_by_repetition_cue():
    out = {"transactions": [
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
    ]}
    result = validate_example("500 beer twice", out)
    assert result.ok, result.errors


def test_duplicate_not_justified_by_different_item_with_same_amount():
    """'500 beer and 500 candy' -> two beer txns is still suspicious."""
    out = {"transactions": [
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
        {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"},
    ]}
    result = validate_example("500 beer and 500 candy", out)
    assert not result.ok
    assert any(e.code == "SUSPICIOUS_DUPLICATE" for e in result.errors)


def test_no_amount_in_input_suppresses_per_txn_errors():
    out = _good_output(amount=500)
    result = validate_example("just some words here", out)
    codes = [e.code for e in result.errors]
    assert "NO_AMOUNT_IN_INPUT" in codes
    assert "AMOUNT_NOT_IN_INPUT" not in codes


def test_warn_mode_returns_ok_true_with_errors():
    out = _good_output(amount=500)
    result = validate_example("fifty thousand for laptop", out, mode="warn")
    assert result.ok is True
    assert any(e.code == "AMOUNT_NOT_IN_INPUT" for e in result.errors)
```

- [ ] **Step 2: Run, watch them fail**

Run: `pytest tests/test_validator.py -v`
Expected: failures on all the new tests (validator returns OK for everything that passes schema).

- [ ] **Step 3: Implement the remaining checks**

Replace the body of `validate_example` after the schema preflight return. The full function:

```python
_REPETITION_CUES = (
    "twice", "2x", "two times", "do baar",
    "again", "another", "aur ek", "phir se",
)


def _matches_amount(c: AmountCandidate, amount: float) -> bool:
    return math.isclose(c.value, amount, abs_tol=0.001)


def _input_supports_repeat(input_text: str, amount: float, item: str,
                           candidates: list[AmountCandidate],
                           dup_count: int) -> bool:
    """A duplicate group of size dup_count is justified iff:
      1. same amount appears >= dup_count times AND same item phrase
         appears >= dup_count times in the input, OR
      2. a repetition cue appears in the input.
    """
    lower = input_text.lower()
    if any(cue in lower for cue in _REPETITION_CUES):
        return True
    amount_count = sum(1 for c in candidates if _matches_amount(c, amount))
    item_count = lower.count(item.lower())
    return amount_count >= dup_count and item_count >= dup_count


def validate_example(
    input_text: str,
    output_obj: dict,
    *,
    mode: Literal["strict", "warn"] = "strict",
) -> ValidationResult:
    from _lib import is_schema_valid, schema_errors

    candidates = parse_amounts(input_text)

    if not is_schema_valid(output_obj):
        errs = [
            ValidationError(
                code="SCHEMA_INVALID", path="/transactions",
                message=err, severity="error",
            )
            for err in schema_errors(output_obj)
        ]
        return _result_from(errs, mode=mode, candidates=candidates)

    errors: list[ValidationError] = []
    active = [c for c in candidates if c.status == "active"]
    txns = output_obj["transactions"]

    # Step 3 — no-amount precedence.
    skip_per_txn_amount = False
    if len(active) == 0 and len(txns) >= 1:
        errors.append(ValidationError(
            code="NO_AMOUNT_IN_INPUT", path="/transactions",
            message="No amount candidates parsed from input but transactions emitted",
            severity="error",
        ))
        skip_per_txn_amount = True

    # Step 4 — per-transaction amount check.
    amount_match_active = True
    superseded_used = False
    if not skip_per_txn_amount:
        for i, txn in enumerate(txns):
            matches = [c for c in candidates if _matches_amount(c, txn["amount"])]
            path = f"/transactions/{i}/amount"
            if not matches:
                errors.append(ValidationError(
                    code="AMOUNT_NOT_IN_INPUT", path=path,
                    message=f"Predicted amount {txn['amount']} not in candidates "
                            f"{[c.value for c in candidates]}",
                    severity="error",
                ))
                amount_match_active = False
            elif all(c.status == "superseded" for c in matches):
                errors.append(ValidationError(
                    code="SUPERSEDED_AMOUNT_USED", path=path,
                    message=f"Predicted amount {txn['amount']} matches only a "
                            f"superseded candidate",
                    severity="error",
                ))
                superseded_used = True
                amount_match_active = False
    else:
        amount_match_active = False

    # Step 5 — per-transaction currency check (per matched active candidate).
    for i, txn in enumerate(txns):
        matched_active = [
            c for c in candidates
            if c.status == "active" and _matches_amount(c, txn["amount"])
        ]
        hints = {c.currency_hint for c in matched_active if c.currency_hint}
        if hints and txn["currency"] not in hints:
            errors.append(ValidationError(
                code="CURRENCY_MISMATCH",
                path=f"/transactions/{i}/currency",
                message=f"Txn currency {txn['currency']!r} does not match "
                        f"matched-candidate hints {sorted(hints)}",
                severity="error",
            ))

    # Step 6 — transaction count check.
    txn_count_match_active = (len(txns) == len(active))
    if len(txns) > len(active):
        errors.append(ValidationError(
            code="TXN_COUNT_EXCEEDS_CANDIDATES", path="/transactions",
            message=f"{len(txns)} transactions but only {len(active)} active candidates",
            severity="error",
        ))
    elif len(txns) < len(active):
        errors.append(ValidationError(
            code="TXN_COUNT_BELOW_CANDIDATES", path="/transactions",
            message=f"{len(txns)} transactions but {len(active)} active candidates",
            severity="warning",
        ))

    # Step 7 — duplicate detection.
    keys: list[tuple] = []
    for t in txns:
        keys.append((
            round(float(t["amount"]) * 1000) / 1000.0,
            t["item"].lower().strip(),
            t["category"], t["type"], t["currency"],
        ))
    dup_groups: dict[tuple, int] = {}
    for k in keys:
        dup_groups[k] = dup_groups.get(k, 0) + 1

    duplicate_found = False
    for i, k in enumerate(keys):
        if dup_groups[k] < 2:
            continue
        # Only report once per repeated occurrence after the first.
        first_occurrence = keys.index(k)
        if i == first_occurrence:
            continue
        amount = k[0]; item = k[1]
        if _input_supports_repeat(input_text, amount, item, candidates, dup_groups[k]):
            continue
        duplicate_found = True
        errors.append(ValidationError(
            code="SUSPICIOUS_DUPLICATE", path=f"/transactions/{i}",
            message=f"Transaction is an unjustified duplicate of /transactions/{first_occurrence}",
            severity="error",
        ))

    return _result_from(
        errors, mode=mode, candidates=candidates,
        amount_match_active=amount_match_active,
        txn_count_match_active=txn_count_match_active,
        duplicate_found=duplicate_found,
        superseded_used=superseded_used,
    )
```

- [ ] **Step 4: Run validator tests, all pass**

Run: `pytest tests/test_validator.py -v`
Expected: all pass.

- [ ] **Step 5: Check coverage target (≥90% lines on `validator.py`)**

Run: `pytest tests/test_validator.py --cov=validator --cov-report=term-missing`
Expected: coverage ≥ 90%. If under, look at missing line numbers and add fixtures for the uncovered branches. Common gaps: the `else: amount_match_active = False` branch of skip-per-txn-amount, the `first_occurrence == i` skip in dup detection — add one or two more parametrized rows.

---

### Task 11: `_lib.py` re-exports + import smoke + Commit A

**Files:**
- Modify: `scripts/_lib.py` (append-only, at file bottom)

Spec reference: §5.

- [ ] **Step 1: Append re-export block to bottom of `scripts/_lib.py`**

Open `scripts/_lib.py` and add at the **very end of the file** (after every existing definition):

```python


# ---------------------------------------------------------------------------
# Re-exports — placed at the bottom of _lib.py so amount_parser/validator can
# `from _lib import is_schema_valid, schema_errors` lazily without a circular
# load. Existing imports of _lib symbols are untouched.
# ---------------------------------------------------------------------------

from amount_parser import AmountCandidate, parse_amounts  # noqa: E402
from validator import (  # noqa: E402
    ValidationError, ValidationResult,
    validate_example, serialize_validation_result,
)
```

- [ ] **Step 2: Smoke check the import path**

Run (bash):
```bash
PYTHONPATH=scripts python -c "from _lib import validate_example, parse_amounts, is_schema_valid; print('ok')"
```

Or PowerShell:
```powershell
$env:PYTHONPATH = "scripts"; python -c "from _lib import validate_example, parse_amounts, is_schema_valid; print('ok')"; Remove-Item Env:PYTHONPATH
```

Expected: prints `ok`.

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: every test from Tasks 1–10 passes.

- [ ] **Step 4: Run coverage one more time**

Run: `pytest tests/ --cov=amount_parser --cov=validator --cov-report=term`
Expected: parser ≥95%, validator ≥90%.

- [ ] **Step 5: Commit A**

```bash
git add scripts/amount_parser.py scripts/validator.py scripts/_lib.py \
        tests/__init__.py tests/conftest.py tests/test_amount_parser.py \
        tests/test_validator.py requirements-dev.txt
git commit -m "$(cat <<'EOF'
Add amount_parser + validator + tests + _lib re-exports

Pure-Python modules with no model/GPU deps. Parser turns raw input text
into ranked AmountCandidate list with active/superseded status, currency
hints, and source tags. Validator runs schema preflight then input-vs-
output consistency checks (amount, currency, count, duplicate). Stage 5
gate / Stage 4 metrics wired up in the next commit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Stage 4 metric helper `score_validation_fields`

**Files:**
- Modify: `scripts/04_eval.py`
- Create: `tests/test_stage4_metrics.py`

Spec reference: §7.1, §7.2, §7.4.

- [ ] **Step 1: Write the failing tests for the helper**

`tests/test_stage4_metrics.py`:
```python
import importlib


eval_mod = importlib.import_module("04_eval")
score_validation_fields = eval_mod.score_validation_fields


def _expected():
    return {"transactions": [{
        "amount": 500, "currency": "INR", "item": "beer",
        "category": "Drinks", "type": "expense",
    }]}


def test_invalid_json_produces_json_parse_failed():
    fields = score_validation_fields("500 beer", _expected(), "not json")
    codes = [e["code"] for e in fields["validation_errors"]]
    assert codes == ["JSON_PARSE_FAILED"]
    assert fields["amount_exact"] is False
    assert fields["txn_count_exact"] is False
    assert fields["amount_values_match_active_candidates"] is False
    assert fields["txn_count_matches_active_candidates"] is False
    assert fields["duplicate_transactions_found"] is False
    assert fields["superseded_amount_used"] is False


def test_helper_accepts_pre_parsed_predicted_to_avoid_double_parse():
    predicted = {"transactions": [{
        "amount": 500, "currency": "INR", "item": "beer",
        "category": "Drinks", "type": "expense",
    }]}
    fields = score_validation_fields("500 beer", _expected(),
                                     predicted_raw="<unused>",
                                     predicted=predicted)
    assert fields["amount_exact"] is True
    assert fields["txn_count_exact"] is True
    assert fields["amount_values_match_active_candidates"] is True


def test_amount_exact_uses_isclose_so_int_vs_float_equal():
    predicted = {"transactions": [{
        "amount": 500.0, "currency": "INR", "item": "beer",
        "category": "Drinks", "type": "expense",
    }]}
    fields = score_validation_fields("500 beer", _expected(),
                                     predicted_raw="<unused>",
                                     predicted=predicted)
    assert fields["amount_exact"] is True
```

(Importing `04_eval` by string works because Python file names beginning with digits aren't valid identifiers; `importlib.import_module("04_eval")` sidesteps the syntax issue. The test depends on `tests/conftest.py` having put `scripts/` on `sys.path`.)

- [ ] **Step 2: Run, watch them fail**

Run: `pytest tests/test_stage4_metrics.py -v`
Expected: `AttributeError: module '04_eval' has no attribute 'score_validation_fields'`.

- [ ] **Step 3: Add `score_validation_fields` to `scripts/04_eval.py`**

After the existing `score_example` function (around line 240), add:

```python
def _multiset_close(a: list[float], b: list[float]) -> bool:
    """Compare two multisets of floats with abs_tol=0.001."""
    if len(a) != len(b):
        return False
    remaining = list(b)
    for x in a:
        for i, y in enumerate(remaining):
            if math.isclose(x, y, abs_tol=0.001):
                del remaining[i]
                break
        else:
            return False
    return True


def score_validation_fields(
    input_text: str,
    expected: dict,
    predicted_raw: str,
    predicted: dict | None = None,
) -> dict:
    """Returns the per-example fields described in the spec §7.1 / §7.2.

    If ``predicted`` is provided, it is used as-is. Otherwise the helper
    calls ``extract_json(predicted_raw)`` itself; ``None`` from that call
    triggers the invalid-JSON row shape with a JSON_PARSE_FAILED wrapper
    error (NOT produced by the validator).
    """
    from _lib import validate_example, serialize_validation_result  # local: avoids cycle at import time

    if predicted is None:
        predicted = extract_json(predicted_raw)

    if predicted is None:
        return {
            "amount_values_match_active_candidates": False,
            "txn_count_matches_active_candidates": False,
            "duplicate_transactions_found": False,
            "superseded_amount_used": False,
            "validation_errors": [{
                "code": "JSON_PARSE_FAILED", "path": "",
                "message": "Model output was not valid JSON", "severity": "error",
            }],
            "amount_exact": False,
            "txn_count_exact": False,
        }

    result = validate_example(input_text, predicted, mode="strict")
    serialized = serialize_validation_result(result)

    exp_txns = expected.get("transactions", [])
    pred_txns = predicted.get("transactions", [])
    amount_exact = _multiset_close(
        [float(t["amount"]) for t in exp_txns],
        [float(t.get("amount", float("nan"))) for t in pred_txns],
    )
    txn_count_exact = len(exp_txns) == len(pred_txns)

    return {
        "amount_values_match_active_candidates": serialized["amount_values_match_active_candidates"],
        "txn_count_matches_active_candidates": serialized["txn_count_matches_active_candidates"],
        "duplicate_transactions_found": serialized["duplicate_transactions_found"],
        "superseded_amount_used": serialized["superseded_amount_used"],
        "validation_errors": serialized["errors"],
        "amount_exact": amount_exact,
        "txn_count_exact": txn_count_exact,
    }
```

Add `import math` to the file's existing import block at the top (currently no `math` import).

- [ ] **Step 4: Run the helper tests, all pass**

Run: `pytest tests/test_stage4_metrics.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests pass.

---

### Task 13: Stage 4 integration — wire helper into the eval loop

**Files:**
- Modify: `scripts/04_eval.py`

Spec reference: §7.1, §7.3.

- [ ] **Step 1: Wire `score_validation_fields` into the main loop**

In `scripts/04_eval.py`, locate the per-example write block in `main` (currently around lines 376–407). Replace the `fout.write(json.dumps({...}` block with:

```python
                    val_fields = score_validation_fields(
                        inp, expected, raw, predicted=scored["predicted"],
                    )
                    # Accumulate new aggregates.
                    if val_fields["duplicate_transactions_found"]:
                        n_duplicate += 1
                    if val_fields["superseded_amount_used"]:
                        n_superseded += 1
                    if val_fields["amount_exact"]:
                        n_amount_exact += 1
                    if val_fields["txn_count_exact"]:
                        n_txn_count_exact += 1

                    row = {
                        "input": inp,
                        "expected": expected,
                        "predicted_raw": raw,
                        "predicted": scored["predicted"],
                        "json_valid": scored["json_valid"],
                        "schema_valid": scored["schema_valid"],
                        "exact_match": scored["exact_match"],
                        "schema_errors": scored["schema_errors"],
                        "latency_ms": round(latency_ms, 2),
                        **val_fields,
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
```

Above the `with results_path.open(...)` block, initialize the new counters next to the existing ones:

```python
    n_json = n_schema = n_exact = 0
    n_duplicate = n_superseded = n_amount_exact = n_txn_count_exact = 0
```

- [ ] **Step 2: Add the new aggregate rows to the summary block**

After the existing `Exact match` line in the summary (around line 421), add:

```python
    logging.info("Amount exact:       %d  (%s)", n_amount_exact, pct(n_amount_exact))
    logging.info("Txn count exact:    %d  (%s)", n_txn_count_exact, pct(n_txn_count_exact))
    logging.info("Duplicate rate:     %d  (%s)", n_duplicate, pct(n_duplicate))
    logging.info("Superseded used:    %d  (%s)", n_superseded, pct(n_superseded))
```

- [ ] **Step 3: Sanity-import the modified script (no model load yet)**

Run:
```bash
PYTHONPATH=scripts python -c "import importlib; m = importlib.import_module('04_eval'); print(m.score_validation_fields.__name__)"
```
Expected: prints `score_validation_fields`.

- [ ] **Step 4: Full test suite stays green**

Run: `pytest tests/ -v`
Expected: all pass. The Stage 4 integration is exercised manually after Task 15.

---

### Task 14: Stage 5 integration — strict gate + new `failed.jsonl` shape + flag

**Files:**
- Modify: `scripts/05_generate_distillation_data.py`

Spec reference: §6.

- [ ] **Step 1: Update imports**

Replace the existing `from _lib import (...)` block in `scripts/05_generate_distillation_data.py` (around lines 46–54) with:

```python
from _lib import (  # noqa: E402
    BATCH_FOCUSES,
    build_messages,
    call_deepseek,
    extract_json,
    load_jsonl,
    parse_amounts,
    serialize_validation_result,
    validate_example,
)
```

(We drop `is_schema_valid` and `schema_errors` because the validator handles schema checks internally; we add `validate_example`, `serialize_validation_result`, `parse_amounts`.)

- [ ] **Step 2: Add the new CLI flag**

In `parse_args()` (around line 472, next to the existing `--retry-failed` flag), add:

```python
    p.add_argument("--retry-validation-failed", action="store_true",
                   help="Phase 2: re-attempt rows in failed.jsonl where "
                        "reason == 'validation_failed'. Analogous to --retry-failed "
                        "but scoped to semantic-validation failures.")
```

- [ ] **Step 3: Update the failed-row reader to honor the new flag**

Inside `phase_label`, the current logic builds `failed_set` from `failed.jsonl` when `--retry-failed` is not set. Replace it with:

```python
    failed_set: set[str] = set()
    if FAILED_FILE.exists():
        for r in load_jsonl(FAILED_FILE):
            reason = r.get("reason")
            if args.retry_failed:
                # Old behavior: re-attempt every failed row regardless of reason.
                continue
            if args.retry_validation_failed and reason == "validation_failed":
                continue
            failed_set.add(r["input"])
```

- [ ] **Step 4: Replace the labeling loop body**

Locate the labeling loop inside `phase_label` (lines ~377–406). Replace the per-input handling so the gate uses the validator and writes the new `failed.jsonl` shape:

```python
    with TRAIN_FILE.open("a", encoding="utf-8") as train_out, \
         FAILED_FILE.open("a", encoding="utf-8") as fail_out:
        for start in tqdm(range(0, len(pending), batch_size), desc="labeling", unit="batch"):
            chunk = pending[start : start + batch_size]
            try:
                raws = batch_infer(chunk)
                batch_error: str | None = None
            except Exception as e:  # noqa: BLE001
                logging.error("teacher batch failed at start=%d size=%d: %s",
                              start, len(chunk), e)
                raws = [""] * len(chunk)
                batch_error = repr(e)

            for inp, raw in zip(chunk, raws):
                candidates = parse_amounts(inp)
                cand_payload = [
                    {
                        "value": c.value, "raw": c.raw, "span": list(c.span),
                        "status": c.status, "source": c.source,
                        "currency_hint": c.currency_hint,
                    }
                    for c in candidates
                ]

                if batch_error is not None:
                    fail_out.write(json.dumps({
                        "input": inp,
                        "raw_output": None,
                        "candidates": cand_payload,
                        "reason": "teacher_error",
                        "error": batch_error,
                    }, ensure_ascii=False) + "\n")
                    n_failed += 1
                    failure_reasons["teacher_error"] = failure_reasons.get("teacher_error", 0) + 1
                    continue

                obj = extract_json(raw)
                if obj is None:
                    fail_out.write(json.dumps({
                        "input": inp,
                        "raw_output": raw,
                        "candidates": cand_payload,
                        "reason": "json_parse_failed",
                    }, ensure_ascii=False) + "\n")
                    n_failed += 1
                    failure_reasons["json_parse_failed"] = failure_reasons.get("json_parse_failed", 0) + 1
                    continue

                result = validate_example(inp, obj, mode="strict")
                if result.ok:
                    train_out.write(json.dumps({
                        "input": inp,
                        "output": obj,
                        "_source": "teacher_label",
                    }, ensure_ascii=False) + "\n")
                    n_kept += 1
                else:
                    fail_out.write(json.dumps({
                        "input": inp,
                        "raw_output": raw,
                        "parsed_output": obj,
                        "candidates": cand_payload,
                        "validation": serialize_validation_result(result),
                        "reason": "validation_failed",
                    }, ensure_ascii=False) + "\n")
                    n_failed += 1
                    failure_reasons["validation_failed"] = failure_reasons.get("validation_failed", 0) + 1
            train_out.flush()
            fail_out.flush()
```

- [ ] **Step 5: Confirm the script still imports**

Run:
```bash
PYTHONPATH=scripts python -c "import importlib; importlib.import_module('05_generate_distillation_data'); print('ok')"
```
Expected: prints `ok` with no import errors.

---

### Task 15: README delta, manual smokes, Commit B

**Files:**
- Modify: `README.md`
- Modify: `data/distill/failed.jsonl` (deletion or archive)

Spec reference: §10.

- [ ] **Step 1: Archive the existing `failed.jsonl` (incompatible shape)**

```bash
mv "data/distill/failed.jsonl" "data/distill/failed.jsonl.pre-validator.bak"
```

(Powershell equivalent: `Rename-Item data/distill/failed.jsonl data/distill/failed.jsonl.pre-validator.bak`.)

- [ ] **Step 2: Manual Stage 5 smoke**

```bash
python scripts/05_generate_distillation_data.py --phase label --limit 50 --retry-failed
```

What to check (do NOT proceed if any of these go sideways without an explanation):

1. Script completes without crash.
2. New rows in `data/distill/failed.jsonl` have the §6.3 shape — `raw_output`, `candidates`, `reason` populated; `validation` only on `reason=validation_failed`.
3. Pass-rate decision point: of rows that were **previously JSON-valid AND schema-valid**, fewer than ~25% now fail validation. If higher, inspect the top failure codes in `failed.jsonl` and consider adding parser fixtures before declaring this slice done.

- [ ] **Step 3: Manual Stage 4 smoke**

```bash
python scripts/04_eval.py --model models/student/gguf --limit 50
```

What to check:

1. Summary block shows the four new aggregate rows (`Amount exact`, `Txn count exact`, `Duplicate rate`, `Superseded used`).
2. `eval_results/<name>.jsonl` rows contain `amount_exact`, `txn_count_exact`, `duplicate_transactions_found`, `superseded_amount_used`, `validation_errors`.
3. Invalid-JSON rows have `validation_errors: [{"code": "JSON_PARSE_FAILED", ...}]` and all four new flags `false`.

- [ ] **Step 4: README delta**

In `README.md`, locate the Stage 5 section (around the line `## Stage 5 — Teacher generates distillation data`). Append after the existing "Useful flags (Phase 2)" list:

```markdown
**Validator gate (Phase 2).** Phase 2 uses `scripts/validator.py` to gate teacher
labels: a label must pass the JSON schema AND the semantic checks (amount-in-input,
no-superseded-amount, currency hint, txn count, no unjustified duplicates). Rejected
rows land in `data/distill/failed.jsonl` with `reason ∈ {validation_failed,
json_parse_failed, teacher_error}` and structured `validation.errors[]` carrying
machine-readable codes. The `failed.jsonl` shape changed in this release — delete
or archive the old file before re-running. Re-attempt only semantic-validation
failures with `--retry-validation-failed`.
```

In the Stage 4 section (after the "Reports % JSON-valid..." paragraph, around line 256), append:

```markdown
Stage 4 also reports four validator-derived aggregates: `amount_exact` (predicted
amounts equal expected as multisets), `txn_count_exact`, `duplicate_rate` (fraction
of examples with `duplicate_transactions_found`), and `superseded_amount_used_rate`.
Per-example rows in `eval_results/<name>.jsonl` carry the same fields plus
`validation_errors[]` for failed rows.
```

In the Prerequisites section, after the existing pip-install commands, append:

```markdown
### Dev dependencies

Tests for the amount parser and validator use `pytest`:

```bash
pip install -r requirements-dev.txt
pytest tests/ --cov=amount_parser --cov=validator
```
```

- [ ] **Step 5: Run the full test suite one more time**

Run: `pytest tests/ --cov=amount_parser --cov=validator --cov-report=term`
Expected: all pass; parser ≥95%, validator ≥90%.

- [ ] **Step 6: Commit B**

```bash
git add scripts/04_eval.py scripts/05_generate_distillation_data.py \
        tests/test_stage4_metrics.py README.md data/distill/failed.jsonl.pre-validator.bak
git commit -m "$(cat <<'EOF'
Wire validator into Stage 4 and Stage 5

Stage 5: replace schema-only gate with validate_example(...); rewrite
failed.jsonl shape (reason in {validation_failed, json_parse_failed,
teacher_error}, candidates always populated, structured errors with
machine codes); add --retry-validation-failed.

Stage 4: new per-example fields (amount_values_match_active_candidates,
txn_count_matches_active_candidates, duplicate_transactions_found,
superseded_amount_used, validation_errors) plus eval-only amount_exact
and txn_count_exact; four new aggregate rows in the summary.

The old failed.jsonl is incompatible — archived to .pre-validator.bak.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Final verification

- [ ] **Step 1: Confirm both commits exist on `main`**

Run: `git log --oneline -5`
Expected: the two new commits (parser/validator + Stage 4/5 wiring) on top of the spec commit `9d3733c`.

- [ ] **Step 2: Confirm no uncommitted changes from this slice (other than the pre-existing modified data files)**

Run: `git status`
Expected: only the pre-existing `data/distill/{inputs_raw,train}.jsonl` modifications remain (these are independent of this slice).

- [ ] **Step 3: Mark slice 1 done and surface the next slice**

Re-read spec §12. The natural next slice (based on the rollout-decision-point in §11) is whichever of these the smoke runs justify:
- **Eval expansion (slice 3)** if Stage 4 smoke shows the new aggregates are interpretable but tiny.
- **Re-distill (slice 4)** if Stage 5 rejection rate is moderate (~5–20%) and label quality matters more than runtime structure.
- **Grammar-constrained decoding (slice 2)** if Stage 4 shows lots of `JSON_PARSE_FAILED` in invalid-JSON rows.

Report the smoke numbers and recommended next slice to the user; do not start the next slice without explicit go-ahead.
