# Multi-Provider Generation — Slice 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--phase label --provider-config <path> --multi-provider` actually run with validator-gated, multi-provider output labeling: DeepSeek + Gemini API calls plus a new `LocalTeacherProvider` wrapping the fp16 Unsloth backend; per-input candidate scoring; best-of-N selection; repair-on-failure retry. Legacy `--phase label` (no `--multi-provider`) stays byte-identical.

**Architecture:** New pure-function module `scripts/label_selection.py` (scoring + best-pick + repair-prompt). Real `generate_label` implementations on `DeepSeekProvider` + `GeminiProvider` in `scripts/llm_providers.py`; new `LocalTeacherProvider` with lazy backend load + `threading.Lock`. New `_lib.build_teacher_fp16_backend` (additive, heavy imports inside the function only). New `phase_label_multi_provider` orchestrator in `scripts/05_generate_distillation_data.py` using `ThreadPoolExecutor` + `as_completed` with main-thread JSONL writes and a never-raises `_try_one_attempt` helper. The Slice 2 contract enforcer drops its `phase=="label"` block. Spec at `docs/superpowers/specs/2026-05-16-multi-provider-generation-slice3-design.md` is the source of truth.

**Tech Stack:** Python 3.11+, stdlib (`dataclasses`, `threading`, `concurrent.futures`, `queue`, `argparse`, `json`, `collections.Counter`). `openai>=1.50` + `google-genai>=1.73.1` already installed by Slice 2. `unsloth` + `torch` optional (only needed for real local-teacher smoke; tests monkeypatch the backend). `tests/conftest.py` already puts `scripts/` on `sys.path`.

**File map:**

| File | Purpose |
|---|---|
| `scripts/label_selection.py` (new) | `CandidateOutcome` dataclass + `score_candidate` + `pick_best` + `build_repair_prompt`. No model/API/I/O deps. |
| `scripts/_lib.py` (modify, additive) | +`build_teacher_fp16_backend` + `_FP16Backend`. Heavy imports inside the function bodies. |
| `scripts/llm_providers.py` (modify) | Real `generate_label` on `DeepSeekProvider` + `GeminiProvider`. New `LocalTeacherProvider`. New `_label_schema_for_gemini`. Factory wiring. |
| `scripts/05_generate_distillation_data.py` (modify) | +`phase_label_multi_provider` + helpers; refactor `_build_label_backend` to call `_lib.build_teacher_fp16_backend`; remove `phase=="label"` enforcement block; update `main()` dispatch + help text. |
| `configs/example_providers.json` (unchanged) | Stays safe-by-default. |
| `configs/smoke_real_providers.json` (modify) | Add `output_generation` block (opt-in). |
| `configs/smoke_local_teacher_providers.json` (new) | Opt-in local-teacher + Gemini config. |
| `docs/provider_config.md` (modify) | Update "What loads and runs". |
| `tests/conftest.py` (modify) | Add shared SDK-stub fixture (opt-in, not autouse). |
| `tests/test_candidate_scoring.py` (new) | Pure-function tests for `label_selection.py`. |
| `tests/test_label_providers.py` (new) | DeepSeek/Gemini/LocalTeacher `generate_label`; SDK stubs. |
| `tests/test_label_orchestrator.py` (new) | Direct unit tests for `_try_one_attempt`, `_process_one_input`, fixture-failure mapping. |
| `tests/test_multi_provider_labels.py` (new) | Subprocess end-to-end with FakeProvider. |
| `tests/test_stage5_flag_contract.py` (modify) | Delete the "label exits 2" test; add "label succeeds" test. |
| `tests/fixtures/fake_labels_with_failures.jsonl` (new) | 6 rows mapping inputs → specific attempt-level failure reasons. |
| `README.md` (modify) | "Slice 3: Multi-provider output labeling" subsection. |

**Commit policy:** Single commit at Task 15. Every preceding task says **do NOT commit** explicitly. `data/distill/*`, `reports/`, `.coverage`, `htmlcov/` are NEVER staged.

---

### Task 0: Preflight — verify prerequisites

**Files:** none modified.

Spec reference: §10 Task 0.

- [ ] **Step 1: Confirm branch**

```bash
cd "C:/work/llm training" && git branch --show-current
```
Expected: `feat/multi-provider-slice3`. If not, STOP.

- [ ] **Step 2: Verify Slice 1+2 imports**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "from _lib import validate_example, serialize_validation_result, parse_amounts, normalize_input, clean_input_line, build_messages; print('prior slices ok')"
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "from llm_providers import FakeProvider, UnimplementedProvider, DeepSeekProvider, GeminiProvider, create_provider, ProviderError, _parse_input_lines; print('Slice 2 ok')"
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "from generation_config import load_generation_config, ConfigError, ProviderConfig; print('config ok')"
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "from generation_orchestrator import allocate_quota, RoundRobinScheduler, render_dry_run; print('orchestrator ok')"
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "from _retry import retry_with_backoff; print('retry ok')"
```
Expected: five lines, each printing `<name> ok`.

- [ ] **Step 3: SDK imports resolve**

```bash
python -c "from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError; print('openai ok')"
python -c "from google import genai; from google.genai import types, errors; print('genai ok')"
```
Expected: both print `ok`. (These were installed in Slice 2.)

- [ ] **Step 4: Optional — `unsloth` availability for local-teacher smoke**

```bash
python -c "from unsloth import FastLanguageModel; print('unsloth ok')" || echo "unsloth NOT installed; LocalTeacherProvider tests monkeypatch the backend so this is OK for pytest. GPU smoke will be unavailable."
```
Both outcomes are acceptable. Record which.

- [ ] **Step 5: Prior tests green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: prior baseline (~244 tests) all passing. If anything fails, STOP and report.

- [ ] **Step 6: Do NOT commit.**

---

### Task 1: `scripts/label_selection.py` + tests

**Files:**
- Create: `scripts/label_selection.py`
- Create: `tests/test_candidate_scoring.py`

Spec reference: §4.

- [ ] **Step 1: Write the failing tests**

`tests/test_candidate_scoring.py`:
```python
import pytest

from label_selection import (
    CandidateOutcome,
    build_repair_prompt,
    pick_best,
    score_candidate,
)


def _ok_validation(**overrides) -> dict:
    base = {
        "ok": True,
        "amount_values_match_active_candidates": True,
        "txn_count_matches_active_candidates": True,
        "duplicate_transactions_found": False,
        "superseded_amount_used": False,
        "errors": [],
    }
    base.update(overrides)
    return base


def _validation_with(errors: list[dict], **overrides) -> dict:
    base = _ok_validation()
    base["errors"] = errors
    base["ok"] = not any(e["severity"] == "error" for e in errors)
    base.update(overrides)
    return base


# ---- score_candidate ------------------------------------------------------

def test_score_perfect_candidate():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=_ok_validation(),
        failure_reason=None,
        provider_priority_rank=0,
    )
    assert score == 100


def test_score_hard_reject_returns_zero():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=_ok_validation(),
        failure_reason="json_parse_failed",
    )
    assert score == 0


def test_score_parsed_none_returns_zero():
    score = score_candidate(
        parsed_output=None,
        validation=_ok_validation(),
        failure_reason=None,
    )
    assert score == 0


def test_score_validation_none_returns_zero():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=None,
        failure_reason=None,
    )
    assert score == 0


def test_score_any_error_severity_returns_zero():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=_validation_with([
            {"code": "AMOUNT_NOT_IN_INPUT", "path": "/transactions/0/amount",
             "message": "x", "severity": "error"},
        ]),
        failure_reason=None,
        provider_priority_rank=0,
    )
    assert score == 0


def test_score_clean_pass_no_priority_bonus():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=_ok_validation(),
        failure_reason=None,
        provider_priority_rank=1,
    )
    # 80 base + 10 count + 5 no-warnings + 0 priority = 95
    assert score == 95


def test_score_warning_only_with_count_mismatch():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=_validation_with(
            [{"code": "TXN_COUNT_BELOW_CANDIDATES", "path": "/transactions",
              "message": "x", "severity": "warning"}],
            txn_count_matches_active_candidates=False,
        ),
        failure_reason=None,
        provider_priority_rank=0,
    )
    # 80 base + 0 count + 0 warnings + 5 priority = 85
    assert score == 85


def test_score_no_priority_rank_at_all():
    score = score_candidate(
        parsed_output={"transactions": []},
        validation=_ok_validation(),
        failure_reason=None,
        provider_priority_rank=None,
    )
    # 80 + 10 + 5 + 0 = 95
    assert score == 95


# ---- pick_best ------------------------------------------------------------

def _make_outcome(provider="a", score=80, failure_reason=None, rank=None):
    return CandidateOutcome(
        provider=provider, model=None, raw_output="", parsed_output={} if failure_reason is None else None,
        validation=_ok_validation() if failure_reason is None else None,
        failure_reason=failure_reason, score=score, is_repair_attempt=False,
        provider_priority_rank=rank,
    )


def test_pick_best_returns_highest_score():
    a = _make_outcome("a", score=50)
    b = _make_outcome("b", score=95)
    c = _make_outcome("c", score=80)
    assert pick_best([a, b, c]).provider == "b"


def test_pick_best_returns_none_when_all_failed():
    a = _make_outcome("a", score=0, failure_reason="json_parse_failed")
    assert pick_best([a]) is None


def test_pick_best_ignores_zero_score_outcomes():
    a = _make_outcome("a", score=0, failure_reason="provider_error")
    b = _make_outcome("b", score=70)
    assert pick_best([a, b]).provider == "b"


def test_pick_best_tie_break_by_provider_priority_rank():
    low = _make_outcome("alpha", score=95, rank=2)
    high = _make_outcome("zebra", score=95, rank=0)
    # Tied score; rank 0 beats rank 2 regardless of alphabetical order.
    assert pick_best([low, high]).provider == "zebra"


def test_pick_best_tie_break_by_provider_name_when_priority_equal():
    a = _make_outcome("zebra", score=95, rank=1)
    b = _make_outcome("alpha", score=95, rank=1)
    # Tied score AND tied rank; alphabetical name wins.
    assert pick_best([a, b]).provider == "alpha"


def test_pick_best_treats_none_rank_as_lowest_priority():
    high = _make_outcome("alpha", score=95, rank=0)
    none = _make_outcome("zebra", score=95, rank=None)
    # rank None is treated as +infinity (lowest priority).
    assert pick_best([high, none]).provider == "alpha"


# ---- build_repair_prompt --------------------------------------------------

def test_build_repair_prompt_includes_input_and_failures():
    prompt = build_repair_prompt(
        "500 beer",
        candidates=[{"value": 500.0, "raw": "500", "span": [0, 3],
                     "status": "active", "source": "digits", "currency_hint": None}],
        failure_summary="fake_a: AMOUNT_NOT_IN_INPUT",
    )
    assert "500 beer" in prompt
    assert "AMOUNT_NOT_IN_INPUT" in prompt
    assert "500" in prompt
    assert "Output ONLY the JSON object" in prompt


def test_build_repair_prompt_handles_empty_candidates():
    prompt = build_repair_prompt(
        "no amounts here",
        candidates=[],
        failure_summary="fake_a: NO_AMOUNT_IN_INPUT",
    )
    assert "no amount candidates parsed" in prompt
    assert "NO_AMOUNT_IN_INPUT" in prompt


def test_build_repair_prompt_handles_none_candidates():
    prompt = build_repair_prompt(
        "x",
        candidates=None,
        failure_summary="x: y",
    )
    assert "no amount candidates parsed" in prompt
```

- [ ] **Step 2: Run, expect ModuleNotFoundError**

```bash
cd "C:/work/llm training" && pytest tests/test_candidate_scoring.py -v
```
Expected: `ModuleNotFoundError: No module named 'label_selection'`.

- [ ] **Step 3: Create `scripts/label_selection.py`**

```python
"""Pure scoring + selection logic for multi-provider label candidates.

No I/O, no model deps. All inputs are explicit and all outputs are
derived deterministically. Tests are pure unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateOutcome:
    """A single provider's attempt at labeling one input."""
    provider: str
    model: str | None
    raw_output: str
    parsed_output: dict | None
    validation: dict | None
    failure_reason: str | None
    score: int
    is_repair_attempt: bool
    provider_priority_rank: int | None = None
    error: str | None = None


def score_candidate(
    parsed_output: dict | None,
    validation: dict | None,
    failure_reason: str | None,
    provider_priority_rank: int | None = None,
) -> int:
    """Hard-reject any candidate with a failure_reason or any error-severity
    validator failure. Survivors get a base 80 plus bonuses for count match
    (+10), no warnings (+5), and provider priority rank 0 (+5). Max 100.
    """
    if failure_reason is not None or parsed_output is None or validation is None:
        return 0
    error_codes = {
        e["code"] for e in validation.get("errors", []) if e["severity"] == "error"
    }
    if error_codes:
        return 0
    score = 80
    if validation.get("txn_count_matches_active_candidates"):
        score += 10
    warning_codes = {
        e["code"] for e in validation.get("errors", []) if e["severity"] == "warning"
    }
    if not warning_codes:
        score += 5
    if provider_priority_rank == 0:
        score += 5
    return score


def pick_best(outcomes: list[CandidateOutcome]) -> CandidateOutcome | None:
    """Return the highest-scoring non-failure outcome, or None if all failed.

    Tie-break order:
      1. Higher score wins.
      2. Lower provider_priority_rank wins (None treated as +inf).
      3. Alphabetical provider name (stable, deterministic).
    """
    valid = [o for o in outcomes if o.failure_reason is None and o.score > 0]
    if not valid:
        return None
    valid.sort(key=lambda o: (
        -o.score,
        o.provider_priority_rank if o.provider_priority_rank is not None else float("inf"),
        o.provider,
    ))
    return valid[0]


_REPAIR_PROMPT_TEMPLATE = """The previous label attempt for this input failed validation.

INPUT:
{input_text}

PARSED AMOUNT CANDIDATES (from deterministic parser):
{candidates_summary}

FAILURE SUMMARY:
{failure_summary}

Generate a corrected JSON label that:
1. Uses ONLY amounts from the parsed candidates above (do not invent new amounts).
2. Marks the correct currency per the candidate's currency_hint.
3. Has the correct number of transactions matching the active candidates.
4. Does NOT include any superseded amounts (markers like "wait no", "actually").
5. Does NOT duplicate transactions unless the input explicitly repeats them.

Output ONLY the JSON object."""


def build_repair_prompt(
    input_text: str,
    *,
    candidates: list[dict] | None = None,
    failure_summary: str,
) -> str:
    """Build a stricter prompt for a repair attempt."""
    if candidates:
        lines = [
            f"  - {c['raw']!r} -> {c['value']} "
            f"({c.get('currency_hint') or '(no hint, default INR)'}, {c['status']})"
            for c in candidates
        ]
        candidates_summary = "\n".join(lines)
    else:
        candidates_summary = "  (no amount candidates parsed)"
    return _REPAIR_PROMPT_TEMPLATE.format(
        input_text=input_text,
        candidates_summary=candidates_summary,
        failure_summary=failure_summary,
    )
```

- [ ] **Step 4: Run tests, expect all pass**

```bash
cd "C:/work/llm training" && pytest tests/test_candidate_scoring.py -v
```
Expected: 16 passed.

- [ ] **Step 5: Coverage check**

```bash
cd "C:/work/llm training" && pytest tests/test_candidate_scoring.py --cov=label_selection --cov-report=term
```
Expected: ≥ 95% on `label_selection.py`.

- [ ] **Step 6: Full suite stays green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: prior baseline + 16.

- [ ] **Step 7: Do NOT commit.**

---

### Task 2: `_lib.build_teacher_fp16_backend` + `_FP16Backend`

**Files:**
- Modify: `scripts/_lib.py` — additive helper

Spec reference: §3.8.

These are pure additive helpers. Heavy imports (`unsloth`, `torch`) happen INSIDE function bodies, never at module load. No tests in this task — Task 6 (LocalTeacherProvider tests) exercises it via monkeypatching.

- [ ] **Step 1: Append helpers to `scripts/_lib.py`**

Open `scripts/_lib.py`. Locate the Slice 1 re-export block at the bottom (the comment `# Re-exports — placed at the bottom...`). Insert the new helpers IMMEDIATELY BEFORE that block:

```python


# ---------------------------------------------------------------------------
# Teacher fp16 backend builder (Slice 3). Used by:
#   - scripts/llm_providers.py::LocalTeacherProvider (multi-provider labeling)
#   - scripts/05_generate_distillation_data.py::_build_label_backend
#     (legacy single-teacher labeling)
#
# Heavy imports (unsloth, torch) are INSIDE function bodies so that
# `import _lib` stays light. Importing _lib must NOT pull in CUDA/Unsloth.
# ---------------------------------------------------------------------------


def build_teacher_fp16_backend(
    *,
    adapter_dir,
    max_seq_length: int = 1024,
):
    """Build an fp16 transformers backend wrapping the Unsloth-trained
    teacher LoRA adapter. Returns an _FP16Backend instance.

    Heavy imports happen inside this function (unsloth + torch); do not
    move them to module scope.
    """
    from unsloth import FastLanguageModel    # lazy

    model, processor = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return _FP16Backend(model=model, processor=processor, tokenizer=tokenizer)


class _FP16Backend:
    """Wraps the loaded model + tokenizer. NOT thread-safe;
    LocalTeacherProvider owns the lock."""

    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

    def generate_label(self, messages: list[dict], *, max_new_tokens: int) -> str:
        import torch    # lazy

        templater = (
            self.processor if hasattr(self.processor, "apply_chat_template") else self.tokenizer
        )
        prompt = templater.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        enc = self.tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True,
            max_length=self.tokenizer.model_max_length or 1024,
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        return self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
```

The re-export block at the very bottom of `_lib.py` must stay last. Insert before it.

- [ ] **Step 2: Verify `import _lib` is still light (no torch/unsloth import)**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
import sys
sdk_before = {m for m in sys.modules if m.startswith(('torch', 'unsloth', 'transformers'))}
import _lib
sdk_after = {m for m in sys.modules if m.startswith(('torch', 'unsloth', 'transformers'))}
leaked = sdk_after - sdk_before
print('leaked at import time:', sorted(leaked) if leaked else 'none')
"
```
Expected: `leaked at import time: none`. If anything leaked, STOP and check that the imports are properly inside function bodies.

- [ ] **Step 3: Confirm the new symbols are exposed**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "from _lib import build_teacher_fp16_backend, _FP16Backend; print(build_teacher_fp16_backend.__name__, _FP16Backend.__name__)"
```
Expected: `build_teacher_fp16_backend _FP16Backend`.

- [ ] **Step 4: Full suite stays green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: no regressions.

- [ ] **Step 5: Do NOT commit.**

---

### Task 3: Move SDK-stub fixture to `tests/conftest.py`

**Files:**
- Modify: `tests/conftest.py` — add shared fixture
- Modify: `tests/test_real_providers.py` — use the shared fixture

Spec reference: §8 (shared fixture rule).

The Slice 2 SDK-stub autouse fixture lives in `tests/test_real_providers.py`. Slice 3 needs the same fixture in `tests/test_label_providers.py`. Move it to `conftest.py` so both files share it. Change `autouse=True` to opt-in (callers reference it by parameter name).

- [ ] **Step 1: Look at current Slice 2 fixture location**

```bash
cd "C:/work/llm training" && grep -n "class _Stub\|_patch_sdk_clients\|autouse" tests/test_real_providers.py | head -20
```

Note the fixture name `_patch_sdk_clients` and the stub classes (`_StubChat`, `_StubChatCompletions`, `_StubOpenAIClient`, `_StubGeminiModels`, `_StubGeminiClient`).

- [ ] **Step 2: Add the fixture to `tests/conftest.py`**

Open `tests/conftest.py`. After the existing `sys.path` setup, append:

```python


# ---------------------------------------------------------------------------
# Shared SDK-stub fixture used by both test_real_providers.py (Slice 2) and
# test_label_providers.py (Slice 3). Opt-in via parameter name — NOT autouse
# globally, so tests that want to verify lazy SDK imports can skip it.
# ---------------------------------------------------------------------------


class _StubChatCompletions:
    def create(self, **kwargs):
        raise AssertionError("real OpenAI client should not be called in tests")


class _StubChat:
    def __init__(self):
        self.completions = _StubChatCompletions()


class _StubOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = _StubChat()


class _StubGeminiModels:
    def generate_content(self, **kwargs):
        raise AssertionError("real Gemini client should not be called in tests")


class _StubGeminiClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models = _StubGeminiModels()


import pytest as _pytest    # alias to avoid confusing with test-level pytest imports


@_pytest.fixture
def patch_sdk_clients(monkeypatch):
    """Patch openai.OpenAI and google.genai.Client so provider __init__
    never makes network calls. Opt-in: tests reference `patch_sdk_clients`
    by parameter name when they want SDK construction stubbed.
    """
    import openai
    monkeypatch.setattr(openai, "OpenAI", _StubOpenAIClient)
    from google import genai
    monkeypatch.setattr(genai, "Client", _StubGeminiClient)
```

- [ ] **Step 3: Update `tests/test_real_providers.py` to use the shared fixture**

Open `tests/test_real_providers.py`. Find the existing `_patch_sdk_clients` autouse fixture and the stub classes — DELETE them (they now live in `conftest.py`).

Find every test in the file that depends on the SDK stubs. They are currently autouse, so the test functions don't take the fixture as a parameter. Add `patch_sdk_clients` as a parameter to each test that uses provider construction. Concrete pattern:

Tests that previously looked like:
```python
def test_deepseek_constructs_with_env_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    ...
```

become:
```python
def test_deepseek_constructs_with_env_key(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    ...
```

Every test in `test_real_providers.py` needs the fixture; add `patch_sdk_clients` to each signature.

Quick search to find them all:
```bash
cd "C:/work/llm training" && grep -n "^def test_" tests/test_real_providers.py
```

For each `def test_...(monkeypatch)`, change to `def test_...(monkeypatch, patch_sdk_clients)`. For each `def test_...()` (no params), change to `def test_...(patch_sdk_clients)`.

- [ ] **Step 4: Run Slice 2 tests to confirm still green**

```bash
cd "C:/work/llm training" && pytest tests/test_real_providers.py -v 2>&1 | tail -15
```
Expected: all Slice 2 provider tests still pass.

- [ ] **Step 5: Full suite stays green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: same baseline + 16 from Task 1.

- [ ] **Step 6: Do NOT commit.**

---

### Task 4: `DeepSeekProvider.generate_label` + tests

**Files:**
- Modify: `scripts/llm_providers.py` — add `_call_api_messages`, replace `generate_label` NotImplementedError
- Create: `tests/test_label_providers.py` — DeepSeek half only (Gemini in Task 5, LocalTeacher in Task 6)

Spec reference: §3.3.

- [ ] **Step 1: Create `tests/test_label_providers.py` with DeepSeek tests**

```python
"""Tests for real provider generate_label implementations.

Uses the shared `patch_sdk_clients` fixture from tests/conftest.py so
provider __init__ never makes network calls.
"""
import pytest

from llm_providers import DeepSeekProvider, ProviderError


# ---- DeepSeek generate_label ---------------------------------------------

def test_deepseek_generate_label_returns_raw(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    monkeypatch.setattr(
        p, "_call_api_messages",
        lambda msgs: '{"transactions":[{"amount":500,"currency":"INR","item":"beer","category":"Drinks","type":"expense"}]}',
    )
    raw = p.generate_label("500 beer")
    assert '"transactions"' in raw
    assert '"beer"' in raw


def test_deepseek_label_uses_build_messages(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    captured = {}
    def capture(msgs):
        captured["msgs"] = msgs
        return "{}"
    monkeypatch.setattr(p, "_call_api_messages", capture)
    p.generate_label("500 beer")
    msgs = captured["msgs"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "500 beer"


# ---- DeepSeek omits None kwargs in label path -----------------------------

class _FakeMessage:
    content = "ok line"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResp:
    choices = [_FakeChoice()]


def test_deepseek_label_omits_none_kwargs(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    captured = {}
    p = DeepSeekProvider(name="ds", model="deepseek-chat",
                         temperature=None, max_tokens=None)
    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResp()
    monkeypatch.setattr(p._client.chat.completions, "create", fake_create)
    p.generate_label("500 beer")
    assert "temperature" not in captured
    assert "max_tokens" not in captured
    assert captured["model"] == "deepseek-chat"
    assert captured["messages"][1]["content"] == "500 beer"


# ---- DeepSeek retry on transient (using fake exception class) -------------

class _FakeTransientError(Exception):
    pass


def test_deepseek_label_retries_on_transient(monkeypatch, patch_sdk_clients):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat", max_retries=2)
    p._retryable_excs = (_FakeTransientError,)
    p._sleep = lambda _: None

    state = {"calls": 0}
    def flaky(_msgs):
        state["calls"] += 1
        if state["calls"] < 2:
            raise _FakeTransientError("transient")
        return "ok"
    monkeypatch.setattr(p, "_call_api_messages", flaky)

    result = p.generate_label("500 beer")
    assert state["calls"] == 2
    assert result == "ok"
```

- [ ] **Step 2: Run, expect failures (`generate_label` still raises NotImplementedError)**

```bash
cd "C:/work/llm training" && pytest tests/test_label_providers.py -v
```
Expected: tests fail because `generate_label` raises `NotImplementedError` ("output labeling lands in Slice 3").

- [ ] **Step 3: Replace `DeepSeekProvider.generate_label` body**

Open `scripts/llm_providers.py`. Find `DeepSeekProvider.generate_label`. Current body:

```python
def generate_label(self, input_text: str) -> str:
    raise NotImplementedError(
        f"deepseek provider {self.name!r}: output labeling lands in Slice 3"
    )
```

Replace with:

```python
def generate_label(self, input_text: str) -> str:
    """Single chat-completion call to label this input. Returns raw text;
    orchestrator validates."""
    from _lib import build_messages
    msgs = build_messages(input_text)
    return self._call_with_retry_label(msgs)


def _call_with_retry_label(self, messages: list[dict]) -> str:
    from _retry import retry_with_backoff
    return retry_with_backoff(
        lambda: self._call_api_messages(messages),
        retryable=self._retryable_excs,
        max_retries=self.max_retries,
        logger_name=f"deepseek.{self.name}",
        sleep=self._sleep,
    )


def _call_api_messages(self, messages: list[dict]) -> str:
    kwargs = {
        "model": self.model,
        "messages": messages,
        "timeout": self.timeout,
    }
    if self.temperature is not None:
        kwargs["temperature"] = self.temperature
    if self.max_tokens is not None:
        kwargs["max_tokens"] = self.max_tokens
    resp = self._client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""
```

(The new `_call_with_retry_label` + `_call_api_messages` go inside the `DeepSeekProvider` class, alongside the existing `_call_with_retry` + `_call_api` from Slice 2.)

- [ ] **Step 4: Run tests, expect 4 passed**

```bash
cd "C:/work/llm training" && pytest tests/test_label_providers.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Full suite stays green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: prior baseline + 16 (Task 1) + 4 (Task 4).

- [ ] **Step 6: Do NOT commit.**

---

### Task 5: `GeminiProvider.generate_label` + `_label_schema_for_gemini` + tests

**Files:**
- Modify: `scripts/llm_providers.py` — add `_label_schema_for_gemini`, replace `GeminiProvider.generate_label`
- Modify: `tests/test_label_providers.py` — append Gemini tests

Spec reference: §3.4, §3.5.

- [ ] **Step 1: Append Gemini tests to `tests/test_label_providers.py`**

```python
# ---- Gemini structured output --------------------------------------------

def test_label_schema_for_gemini_uses_lib_enums():
    from llm_providers import _label_schema_for_gemini
    from _lib import CATEGORIES, TYPES, CURRENCIES
    schema = _label_schema_for_gemini()
    txn_props = schema["properties"]["transactions"]["items"]["properties"]
    assert set(txn_props["category"]["enum"]) == set(CATEGORIES)
    assert set(txn_props["type"]["enum"]) == set(TYPES)
    assert set(txn_props["currency"]["enum"]) == set(CURRENCIES)


def test_gemini_label_with_structured_output_passes_schema(monkeypatch, patch_sdk_clients):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash", structured_output=True)
    captured = {}

    class FakeResp:
        text = '{"transactions":[]}'

    def fake_generate(model, contents, config):
        captured["model"] = model
        captured["contents"] = contents
        captured["config"] = config
        return FakeResp()
    monkeypatch.setattr(p._client.models, "generate_content", fake_generate)
    p.generate_label("500 beer")
    assert captured["model"] == "gemini-2.5-flash"
    assert captured["contents"] == "500 beer"
    assert captured["config"] is not None


def test_gemini_label_without_structured_output_still_runs(monkeypatch, patch_sdk_clients):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash", structured_output=False)
    captured = {}

    class FakeResp:
        text = '{"transactions":[]}'

    def fake_generate(model, contents, config):
        captured["config"] = config
        return FakeResp()
    monkeypatch.setattr(p._client.models, "generate_content", fake_generate)
    p.generate_label("500 beer")
    # config exists (because we always pass system_instruction); just runs cleanly.
    assert captured["config"] is not None


def test_gemini_generate_inputs_still_ignores_structured_output(monkeypatch, patch_sdk_clients):
    """structured_output=True must NOT affect generate_inputs (regression check)."""
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash", structured_output=True)
    monkeypatch.setattr(p, "_call_api", lambda prompt: "500 beer\n200 chai")
    result = p.generate_inputs("ignored", n=2)
    assert result == ["500 beer", "200 chai"]


# ---- Gemini label retry behavior ------------------------------------------

class _FakeGeminiRetryable(Exception):
    def __init__(self, status):
        super().__init__()
        self.status_code = status


def test_gemini_label_retries_on_429(monkeypatch, patch_sdk_clients):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash", max_retries=2)
    p._retryable_excs = (_FakeGeminiRetryable,)
    p._sleep = lambda _: None

    state = {"calls": 0}
    def flaky(input_text):
        state["calls"] += 1
        if state["calls"] < 2:
            raise _FakeGeminiRetryable(429)
        return "ok"
    monkeypatch.setattr(p, "_call_api_label", flaky)
    result = p.generate_label("500 beer")
    assert state["calls"] == 2
    assert result == "ok"
```

- [ ] **Step 2: Run, expect ImportError on `_label_schema_for_gemini` + failures on label tests**

```bash
cd "C:/work/llm training" && pytest tests/test_label_providers.py -v -k "gemini or label_schema"
```
Expected: ImportError for `_label_schema_for_gemini` and `NotImplementedError` for `generate_label`.

- [ ] **Step 3: Add `_label_schema_for_gemini` to `scripts/llm_providers.py`**

After the existing `_is_retryable_gemini_error` function, append:

```python
def _label_schema_for_gemini() -> dict:
    """Gemini-shaped schema for transaction labels. Enums sourced from
    _lib so they stay in lockstep with the validator."""
    from _lib import CATEGORIES, TYPES, CURRENCIES
    return {
        "type": "OBJECT",
        "properties": {
            "transactions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "amount":   {"type": "NUMBER"},
                        "currency": {"type": "STRING", "enum": list(CURRENCIES)},
                        "item":     {"type": "STRING"},
                        "category": {"type": "STRING", "enum": list(CATEGORIES)},
                        "type":     {"type": "STRING", "enum": list(TYPES)},
                    },
                    "required": ["amount", "currency", "item", "category", "type"],
                },
            },
        },
        "required": ["transactions"],
    }
```

- [ ] **Step 4: Replace `GeminiProvider.generate_label` body**

Find `GeminiProvider.generate_label` in `scripts/llm_providers.py`. Current body raises `NotImplementedError`. Replace with:

```python
def generate_label(self, input_text: str) -> str:
    """Single content-generation call to label this input. Returns raw text."""
    return self._call_with_retry_label(input_text)


def _call_with_retry_label(self, input_text: str) -> str:
    from _retry import retry_with_backoff
    return retry_with_backoff(
        lambda: self._call_api_label(input_text),
        retryable=self._retryable_excs,
        is_retryable=_is_retryable_gemini_error,
        max_retries=self.max_retries,
        logger_name=f"gemini.{self.name}",
        sleep=self._sleep,
    )


def _call_api_label(self, input_text: str) -> str:
    from google.genai import types
    from _lib import SYSTEM_PROMPT
    cfg_kwargs = {"system_instruction": SYSTEM_PROMPT}
    if self.temperature is not None:
        cfg_kwargs["temperature"] = self.temperature
    if self.max_tokens is not None:
        cfg_kwargs["max_output_tokens"] = self.max_tokens
    if self.structured_output:
        cfg_kwargs["response_mime_type"] = "application/json"
        cfg_kwargs["response_schema"] = _label_schema_for_gemini()
    config = types.GenerateContentConfig(**cfg_kwargs)
    resp = self._client.models.generate_content(
        model=self.model,
        contents=input_text,
        config=config,
    )
    return resp.text or ""
```

(These three methods go inside `GeminiProvider`, alongside the existing input-generation methods.)

- [ ] **Step 5: Run Gemini tests, expect 5 passed**

```bash
cd "C:/work/llm training" && pytest tests/test_label_providers.py -v -k "gemini or label_schema"
```
Expected: 5 passed.

- [ ] **Step 6: Run the whole label_providers file**

```bash
cd "C:/work/llm training" && pytest tests/test_label_providers.py -v
```
Expected: 9 passed (4 DeepSeek + 5 Gemini).

- [ ] **Step 7: Full suite stays green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```

- [ ] **Step 8: Do NOT commit.**

---

### Task 6: `LocalTeacherProvider` + factory update + tests

**Files:**
- Modify: `scripts/llm_providers.py` — add `LocalTeacherProvider`, update `create_provider` for `local_teacher`
- Modify: `tests/test_label_providers.py` — append LocalTeacher tests

Spec reference: §3.6, §3.7.

- [ ] **Step 1: Append LocalTeacher tests to `tests/test_label_providers.py`**

```python
# ---- LocalTeacherProvider -------------------------------------------------

def test_local_teacher_factory_requires_cfg_model(monkeypatch):
    """Factory raises ConfigError when local_teacher has no model (adapter dir)."""
    from llm_providers import create_provider
    from generation_config import ProviderConfig, ConfigError

    cfg = ProviderConfig(name="lt", provider_type="local_teacher", model=None)
    with pytest.raises(ConfigError) as exc:
        create_provider(cfg)
    assert "local_teacher" in str(exc.value)
    assert "model" in str(exc.value).lower() or "adapter" in str(exc.value).lower()


def test_local_teacher_init_does_no_io(tmp_path):
    from llm_providers import LocalTeacherProvider
    p = LocalTeacherProvider(name="lt", adapter_dir=tmp_path / "fake_adapter")
    assert p._backend is None
    assert p.model is None


def test_local_teacher_generate_inputs_raises():
    from llm_providers import LocalTeacherProvider
    p = LocalTeacherProvider(name="lt", adapter_dir=None)
    with pytest.raises(NotImplementedError) as exc:
        p.generate_inputs("prompt", n=5)
    assert "label-only" in str(exc.value)


def test_local_teacher_missing_adapter_dir_raises(monkeypatch, tmp_path):
    """If adapter_dir doesn't exist, first generate_label raises ProviderError."""
    from llm_providers import LocalTeacherProvider, ProviderError
    p = LocalTeacherProvider(name="lt", adapter_dir=tmp_path / "does_not_exist")
    with pytest.raises(ProviderError) as exc:
        p.generate_label("500 beer")
    assert "adapter dir not found" in str(exc.value).lower()


def test_local_teacher_lazy_loads_on_first_call(monkeypatch, tmp_path):
    """First generate_label triggers backend load; second reuses."""
    from llm_providers import LocalTeacherProvider
    fake_adapter = tmp_path / "fake_adapter"
    fake_adapter.mkdir()
    p = LocalTeacherProvider(name="lt", adapter_dir=fake_adapter)

    state = {"build_calls": 0, "generate_calls": 0}

    class FakeBackend:
        def generate_label(self, msgs, *, max_new_tokens):
            state["generate_calls"] += 1
            return '{"transactions":[]}'

    def fake_build(*, adapter_dir, max_seq_length):
        state["build_calls"] += 1
        return FakeBackend()

    import _lib
    monkeypatch.setattr(_lib, "build_teacher_fp16_backend", fake_build)

    p.generate_label("500 beer")
    assert state["build_calls"] == 1
    assert state["generate_calls"] == 1
    assert p._backend is not None
    assert p.model == "fake_adapter"

    p.generate_label("200 chai")
    assert state["build_calls"] == 1   # reused
    assert state["generate_calls"] == 2


def test_local_teacher_serializes_concurrent_calls(monkeypatch, tmp_path):
    """Lock ensures multiple threads don't call generate_label concurrently."""
    import threading
    import time
    from llm_providers import LocalTeacherProvider
    fake_adapter = tmp_path / "fake_adapter"
    fake_adapter.mkdir()
    p = LocalTeacherProvider(name="lt", adapter_dir=fake_adapter)

    in_flight = [0]
    max_concurrent = [0]
    counter_lock = threading.Lock()

    class FakeBackend:
        def generate_label(self, msgs, *, max_new_tokens):
            with counter_lock:
                in_flight[0] += 1
                max_concurrent[0] = max(max_concurrent[0], in_flight[0])
            time.sleep(0.02)
            with counter_lock:
                in_flight[0] -= 1
            return '{"transactions":[]}'

    import _lib
    monkeypatch.setattr(_lib, "build_teacher_fp16_backend",
                        lambda *, adapter_dir, max_seq_length: FakeBackend())

    threads = [threading.Thread(target=lambda: p.generate_label("x")) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert max_concurrent[0] == 1


def test_local_teacher_via_factory_smoke(monkeypatch, tmp_path):
    """create_provider builds a LocalTeacherProvider when cfg.model is set."""
    from llm_providers import create_provider, LocalTeacherProvider
    from generation_config import ProviderConfig

    fake_adapter = tmp_path / "adapter"
    fake_adapter.mkdir()
    cfg = ProviderConfig(name="lt", provider_type="local_teacher", model=str(fake_adapter))
    p = create_provider(cfg)
    assert isinstance(p, LocalTeacherProvider)
    assert p.adapter_dir == fake_adapter
    assert p.max_new_tokens == 384   # default
```

- [ ] **Step 2: Run, expect ImportError on `LocalTeacherProvider`**

```bash
cd "C:/work/llm training" && pytest tests/test_label_providers.py -v -k "local_teacher"
```
Expected: ImportError.

- [ ] **Step 3: Add `LocalTeacherProvider` to `scripts/llm_providers.py`**

Append AFTER `GeminiProvider`:

```python
class LocalTeacherProvider:
    """Real local teacher via Unsloth + transformers fp16 backend.

    Lazy backend load: __init__ stashes config; the first generate_label
    call builds the backend. A threading.Lock protects both init and
    inference (single GPU is not safely concurrent).
    """

    provider_type = "local_teacher"

    def __init__(
        self,
        name: str,
        *,
        adapter_dir,
        max_seq_length: int = 1024,
        max_new_tokens: int = 384,
    ) -> None:
        self.name = name
        self.model = None   # populated lazily after backend load
        self.adapter_dir = adapter_dir
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self._backend = None
        self._lock = threading.Lock()

    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        raise NotImplementedError(
            f"local_teacher provider {self.name!r}: input generation is not "
            f"supported — local_teacher is label-only."
        )

    def generate_label(self, input_text: str) -> str:
        with self._lock:
            if self._backend is None:
                from _lib import build_teacher_fp16_backend
                if self.adapter_dir is None or not self.adapter_dir.exists():
                    raise ProviderError(
                        f"LocalTeacherProvider {self.name!r}: adapter dir not found at {self.adapter_dir}"
                    )
                self._backend = build_teacher_fp16_backend(
                    adapter_dir=self.adapter_dir,
                    max_seq_length=self.max_seq_length,
                )
                self.model = self.adapter_dir.name
            from _lib import build_messages
            msgs = build_messages(input_text)
            return self._backend.generate_label(msgs, max_new_tokens=self.max_new_tokens)
```

- [ ] **Step 4: Update `create_provider` factory's `local_teacher` branch**

In `scripts/llm_providers.py`, find `create_provider`. The `local_teacher` branch currently looks like:
```python
if cfg.provider_type == "local_teacher":
    return UnimplementedProvider(
        name=cfg.name, provider_type="local_teacher", model=cfg.model,
    )
```

Replace with:
```python
if cfg.provider_type == "local_teacher":
    from generation_config import ConfigError
    if not cfg.model:
        raise ConfigError(
            f"local_teacher provider {cfg.name!r} requires `model` "
            f"(path to the adapter directory)."
        )
    return LocalTeacherProvider(
        name=cfg.name,
        adapter_dir=Path(cfg.model),
        max_seq_length=1024,
        max_new_tokens=cfg.max_tokens or 384,
    )
```

(The `Path` import is already present in the file from Slice 1.)

- [ ] **Step 5: Run LocalTeacher tests, expect 7 passed**

```bash
cd "C:/work/llm training" && pytest tests/test_label_providers.py -v -k "local_teacher"
```
Expected: 7 passed.

- [ ] **Step 6: Run the whole label_providers file**

```bash
cd "C:/work/llm training" && pytest tests/test_label_providers.py -v
```
Expected: 16 passed (4 DeepSeek + 5 Gemini + 7 LocalTeacher).

- [ ] **Step 7: Verify Slice 1 SDK-leak guard test still passes**

```bash
cd "C:/work/llm training" && pytest tests/test_fake_provider.py -v 2>&1 | tail -5
```
Expected: all Slice 1 fake_provider tests pass. The leak-guard test on `UnimplementedProvider` stays valid because `local_teacher` no longer routes through `UnimplementedProvider`.

NOTE: The Slice 1 test `test_create_provider_returns_unimplemented_for_real_types` may now fail because `local_teacher` no longer returns `UnimplementedProvider`. Edit that test if so:

```bash
cd "C:/work/llm training" && grep -n "test_create_provider_returns_unimplemented_for_real_types" tests/test_fake_provider.py
```

If the test still references `local_teacher` in the loop, update it to only check types that still route through UnimplementedProvider. Since after this task NO types route through UnimplementedProvider (deepseek/gemini went real in Slice 2, local_teacher goes real now), the test should be DELETED or replaced with a comment that UnimplementedProvider is no longer reached by `create_provider` for any real type — only `fake` and explicit `UnimplementedProvider(...)` construction remain.

Pragmatic edit: delete the test. Or rename and have it assert `create_provider("unknown_type")` raises ValueError, which was already covered by `test_create_provider_raises_on_unknown_type`.

- [ ] **Step 8: Full suite stays green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```

- [ ] **Step 9: Do NOT commit.**

---

### Task 7: `phase_label_multi_provider` orchestrator

**Files:**
- Modify: `scripts/05_generate_distillation_data.py` — refactor legacy `_build_label_backend`; add `phase_label_multi_provider` + helpers

Spec reference: §5.

This task is the largest single block of new code. No unit tests in this task — Task 8 covers helpers directly, Task 10 covers end-to-end via subprocess.

- [ ] **Step 1: Refactor legacy `_build_label_backend` to call the new shared helper**

In `scripts/05_generate_distillation_data.py`, find `_build_label_backend` (~line 254). The current `transformers` branch does:

```python
from unsloth import FastLanguageModel
import torch

logging.info("Loading teacher (fp16) from %s", TEACHER_ADAPTER_DIR)
model, processor = FastLanguageModel.from_pretrained(
    model_name=str(TEACHER_ADAPTER_DIR),
    max_seq_length=args.max_seq_length,
    dtype=None,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

# ... lots more code constructing the closure ...
```

Refactor this transformers branch to delegate to `build_teacher_fp16_backend`:

```python
if args.backend == "gguf":
    # ... existing GGUF branch unchanged ...
    return batch_infer   # unchanged

# Default: transformers / Unsloth fp16
from _lib import build_teacher_fp16_backend

logging.info("Loading teacher (fp16) from %s", TEACHER_ADAPTER_DIR)
backend = build_teacher_fp16_backend(
    adapter_dir=TEACHER_ADAPTER_DIR,
    max_seq_length=args.max_seq_length,
)

# The legacy batch_infer closure wraps build_messages → templater → tokenize →
# model.generate → decode, all of which now live in _FP16Backend.generate_label.
# Adapt the batch closure to call backend.generate_label per item.
def batch_infer(inputs: list[str]) -> list[str]:
    out: list[str] = []
    for s in inputs:
        msgs = build_messages(s)
        out.append(backend.generate_label(msgs, max_new_tokens=args.max_new_tokens))
    return out

return batch_infer
```

The legacy batched-inference behavior becomes per-item iteration; this is acceptable because the legacy `phase_label` path is end-of-life (multi-provider is the recommended path). Pin this as a deliberate simplification: legacy `phase_label` is behaviorally unchanged from the user's POV (it still produces train.jsonl + failed.jsonl with the same row shapes), just slower if batch was previously used. If perf concerns surface later, the `_FP16Backend` class can grow a `generate_labels(list[messages])` method.

(Verify after the refactor: legacy phase_label still imports cleanly and the schema-only gate from the validator slice still operates.)

- [ ] **Step 2: Add module imports and constants**

At the top of `scripts/05_generate_distillation_data.py`, ensure these stdlib imports are present (they were added in Slice 2 for input-phase orchestration):

```python
import concurrent.futures
import math
import queue
import threading
```

Update the `from _lib import (...)` block to include any missing names (validator slice + Slice 2):

```python
from _lib import (  # noqa: E402
    BATCH_FOCUSES,
    build_messages,
    build_teacher_fp16_backend,
    call_deepseek,
    clean_input_line,
    extract_json,
    is_schema_valid,
    load_jsonl,
    normalize_input,
    parse_amounts,
    schema_errors,
    serialize_validation_result,
    validate_example,
)
```

Add this near other module-level helper imports:

```python
from collections import Counter
```

- [ ] **Step 3: Append the new helpers + orchestrator function**

Insert this block AFTER `phase_inputs_multi_provider` (which is around line 594) and BEFORE `phase_eval`:

```python
# ---------------------------------------------------------------------------
# Multi-provider OUTPUT labeling (Slice 3). Concurrency model:
#   - Per-input worker via ThreadPoolExecutor; pool size =
#     min(global_max_workers, len(pending)).
#   - Each worker: round-robin scheduler picks providers, calls generate_label,
#     validates, scores. Workers never write to JSONL.
#   - Main thread: as_completed loop reads worker results and writes rows to
#     train.jsonl or failed.jsonl. Sole writer.
#   - LocalTeacherProvider serializes its own GPU calls via internal lock.
# ---------------------------------------------------------------------------


def _try_one_attempt(
    input_text: str,
    provider,
    pcfg,
    *,
    is_repair: bool = False,
    failure_summary: str = "",
    parser_candidates: list[dict] | None = None,
    provider_priority_rank: int | None = None,
):
    """Run one provider attempt; validate; score. Never raises."""
    from label_selection import CandidateOutcome, score_candidate, build_repair_prompt

    try:
        if is_repair:
            prompt = build_repair_prompt(
                input_text,
                candidates=parser_candidates,
                failure_summary=failure_summary,
            )
            raw = provider.generate_label(prompt)
        else:
            raw = provider.generate_label(input_text)
    except Exception as e:    # noqa: BLE001 — unify all provider failures
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

    parsed = extract_json(raw)
    if parsed is None:
        return CandidateOutcome(
            provider=pcfg.name,
            model=getattr(provider, "model", None),
            raw_output=raw, parsed_output=None, validation=None,
            failure_reason="json_parse_failed",
            score=0, is_repair_attempt=is_repair,
            provider_priority_rank=provider_priority_rank,
        )

    result = validate_example(input_text, parsed, mode="strict")
    validation_dict = serialize_validation_result(result)

    error_codes = {
        e["code"] for e in validation_dict["errors"] if e["severity"] == "error"
    }
    if "SCHEMA_INVALID" in error_codes:
        failure_reason = "schema_invalid"
    elif "SUPERSEDED_AMOUNT_USED" in error_codes:
        failure_reason = "superseded_amount_used"
    elif "CURRENCY_MISMATCH" in error_codes:
        failure_reason = "currency_mismatch"
    elif "SUSPICIOUS_DUPLICATE" in error_codes:
        failure_reason = "suspicious_duplicate"
    elif error_codes:
        failure_reason = "validation_failed"   # AMOUNT_NOT_IN_INPUT, NO_AMOUNT_IN_INPUT, etc.
    else:
        failure_reason = None

    score = score_candidate(
        parsed_output=parsed,
        validation=validation_dict,
        failure_reason=failure_reason,
        provider_priority_rank=provider_priority_rank,
    )
    return CandidateOutcome(
        provider=pcfg.name,
        model=getattr(provider, "model", None),
        raw_output=raw,
        parsed_output=parsed,
        validation=validation_dict,
        failure_reason=failure_reason,
        score=score,
        is_repair_attempt=is_repair,
        provider_priority_rank=provider_priority_rank,
    )


def _highest_priority_provider(cfg, providers_by_name: dict) -> str:
    """First entry of provider_priority, falling back to first declared provider.
    Raises ValueError if the resolved name isn't in providers_by_name."""
    if cfg.output_generation.provider_priority:
        name = cfg.output_generation.provider_priority[0]
    else:
        name = cfg.output_generation.providers[0].name
    if name not in providers_by_name:
        raise ValueError(
            f"_highest_priority_provider: {name!r} not in providers_by_name"
        )
    return name


def _summarize_failures(outcomes) -> str:
    """≤200-char summary of why all attempts failed."""
    parts = []
    for o in outcomes:
        if o.failure_reason == "provider_error":
            parts.append(f"{o.provider}: provider error")
        elif o.validation:
            codes = [e["code"] for e in o.validation["errors"] if e["severity"] == "error"]
            parts.append(f"{o.provider}: " + ", ".join(codes[:3]))
        elif o.failure_reason:
            parts.append(f"{o.provider}: {o.failure_reason}")
    return "; ".join(parts)[:200]


def _process_one_input(
    input_text: str,
    *,
    providers_by_name: dict,
    pcfgs_by_name: dict,
    scheduler,
    priority_map: dict,
    cfg,
    parser_candidates: list[dict],
):
    """Process one input end-to-end. Returns (input_text, best_or_None, all_outcomes)."""
    from label_selection import pick_best

    outcomes = []

    for _ in range(cfg.output_generation.label_attempts_per_input):
        provider_name = scheduler.next_provider()
        outcomes.append(_try_one_attempt(
            input_text,
            provider=providers_by_name[provider_name],
            pcfg=pcfgs_by_name[provider_name],
            provider_priority_rank=priority_map.get(provider_name),
        ))

    best = pick_best(outcomes)
    if best is not None:
        return (input_text, best, outcomes)

    repair_enabled = (
        cfg.validation.retry_invalid_with_stricter_prompt
        and cfg.validation.max_repair_attempts > 0
    )
    if repair_enabled:
        repair_provider_name = _highest_priority_provider(cfg, providers_by_name)
        for _ in range(cfg.validation.max_repair_attempts):
            repair_outcome = _try_one_attempt(
                input_text,
                provider=providers_by_name[repair_provider_name],
                pcfg=pcfgs_by_name[repair_provider_name],
                is_repair=True,
                failure_summary=_summarize_failures(outcomes),
                parser_candidates=parser_candidates,
                provider_priority_rank=priority_map.get(repair_provider_name),
            )
            outcomes.append(repair_outcome)
            if repair_outcome.failure_reason is None:
                return (input_text, repair_outcome, outcomes)

    return (input_text, None, outcomes)


def _read_labeled_inputs(path) -> set:
    """Set of inputs already present in train.jsonl. Tolerant of legacy rows."""
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("input"), str):
                out.add(obj["input"])
    return out


def _row_has_validation_failure_attempt(obj: dict) -> bool:
    validator_reasons = {
        "validation_failed", "schema_invalid", "superseded_amount_used",
        "currency_mismatch", "suspicious_duplicate",
    }
    for attempt in obj.get("attempts", []):
        if attempt.get("failure_reason") in validator_reasons:
            return True
    return False


def _read_failed_skip_set(path, args) -> set:
    """Inputs to SKIP this run based on failed.jsonl + retry flags."""
    if not path.exists():
        return set()
    if args.retry_failed:
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get("input"), str):
                continue
            reason = obj.get("reason")
            if args.retry_validation_failed:
                if reason == "validation_failed":
                    continue
                if reason in {"all_candidates_failed", "repair_exhausted"} \
                        and _row_has_validation_failure_attempt(obj):
                    continue
            out.add(obj["input"])
    return out


def _build_priority_map(og) -> dict:
    if og.provider_priority:
        return {name: i for i, name in enumerate(og.provider_priority)}
    return {p.name: i for i, p in enumerate(og.providers)}


def _serialize_candidate(c) -> dict:
    return {
        "value": c.value, "raw": c.raw, "span": list(c.span),
        "status": c.status, "source": c.source, "currency_hint": c.currency_hint,
    }


def _serialize_outcome(o) -> dict:
    return {
        "provider": o.provider,
        "model": o.model,
        "raw_output": o.raw_output,
        "parsed_output": o.parsed_output,
        "validation": o.validation,
        "score": o.score,
        "failure_reason": o.failure_reason,
        "is_repair_attempt": o.is_repair_attempt,
        "provider_priority_rank": o.provider_priority_rank,
        "error": o.error,
    }


def phase_label_multi_provider(args: argparse.Namespace) -> None:
    """Slice 3 validator-gated multi-provider label generation."""
    from generation_config import load_generation_config
    from generation_orchestrator import RoundRobinScheduler
    from llm_providers import create_provider

    cfg = load_generation_config(args.provider_config)
    og = cfg.output_generation
    if not og.enabled:
        logging.info(
            "Phase label (multi-provider): output_generation.enabled=False — nothing to do."
        )
        return

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    all_inputs = read_inputs_jsonl(INPUTS_FILE)

    # Normalize-dedupe in first-seen order.
    seen_norm = set()
    all_inputs_unique = []
    for inp in all_inputs:
        key = normalize_input(inp)
        if key in seen_norm:
            continue
        seen_norm.add(key)
        all_inputs_unique.append(inp)

    labeled_keys = {normalize_input(s) for s in _read_labeled_inputs(TRAIN_FILE)}
    failed_skip_keys = {normalize_input(s) for s in _read_failed_skip_set(FAILED_FILE, args)}
    pending = [
        inp for inp in all_inputs_unique
        if normalize_input(inp) not in labeled_keys
        and normalize_input(inp) not in failed_skip_keys
    ]
    if args.limit > 0:
        pending = pending[:args.limit]
    logging.info(
        "Phase label (multi-provider): all_unique=%d labeled=%d failed_skip=%d pending=%d",
        len(all_inputs_unique), len(labeled_keys), len(failed_skip_keys), len(pending),
    )
    if not pending:
        return

    providers_by_name = {p.name: create_provider(p) for p in og.providers}
    pcfgs_by_name = {p.name: p for p in og.providers}
    scheduler = RoundRobinScheduler(og.providers)
    priority_map = _build_priority_map(og)
    pool_size = min(cfg.rate_limits.global_max_workers, max(1, len(pending)))

    n_kept = 0
    n_failed = 0
    n_repaired = 0
    failure_reasons = Counter()
    attempt_failure_reasons = Counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as pool, \
         TRAIN_FILE.open("a", encoding="utf-8") as train_out, \
         FAILED_FILE.open("a", encoding="utf-8") as fail_out, \
         tqdm(total=len(pending), desc="multi-provider labels") as bar:

        candidates_by_input = {
            inp: [_serialize_candidate(c) for c in parse_amounts(inp)]
            for inp in pending
        }

        futures = {
            pool.submit(
                _process_one_input,
                inp,
                providers_by_name=providers_by_name,
                pcfgs_by_name=pcfgs_by_name,
                scheduler=scheduler,
                priority_map=priority_map,
                cfg=cfg,
                parser_candidates=candidates_by_input[inp],
            ): inp
            for inp in pending
        }

        for fut in concurrent.futures.as_completed(futures):
            input_text, best, all_outcomes = fut.result()
            for o in all_outcomes:
                if o.failure_reason:
                    attempt_failure_reasons[o.failure_reason] += 1

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
            if (n_kept + n_failed) % cfg.rate_limits.write_flush_every == 0:
                train_out.flush()
                fail_out.flush()
            bar.update(1)

        train_out.flush()
        fail_out.flush()

    keep_rate = (n_kept / max(1, n_kept + n_failed)) * 100
    logging.info(
        "Phase label (multi-provider) done. kept=%d (repair=%d) failed=%d keep_rate=%.1f%%",
        n_kept, n_repaired, n_failed, keep_rate,
    )
    if failure_reasons:
        logging.info("Top-level failure reasons: %s", dict(failure_reasons))
    if attempt_failure_reasons:
        logging.info("Attempt-level failure reasons: %s", dict(attempt_failure_reasons))
```

- [ ] **Step 4: Sanity check the file still imports**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
import importlib
m = importlib.import_module('05_generate_distillation_data')
print(m.phase_label_multi_provider.__name__)
print(m._try_one_attempt.__name__)
print(m._process_one_input.__name__)
print(m._serialize_outcome.__name__)
"
```
Expected: 4 function names printed.

- [ ] **Step 5: Full suite stays green (no new tests yet; Task 8 + 10 cover the new code)**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```

- [ ] **Step 6: Do NOT commit.**

---

### Task 8: Direct unit tests for `_try_one_attempt` + fixture verification

**Files:**
- Create: `tests/test_label_orchestrator.py`
- Create: `tests/fixtures/fake_labels_with_failures.jsonl`

Spec reference: §8.2, §8.4.

These are helper-level tests; faster and more precise than subprocess. Subprocess end-to-end coverage comes in Task 10.

- [ ] **Step 1: Create `tests/fixtures/fake_labels_with_failures.jsonl`**

Exactly 6 rows as pinned in spec §8.2:

```jsonl
{"input":"500 beer","output":{"transactions":[{"amount":500,"currency":"INR","item":"beer","category":"Drinks","type":"expense"}]}}
{"input":"schema fail case","raw_output":"{\"transactions\":[{\"amount\":100}]}"}
{"input":"json fail case","raw_output":"{not json"}
{"input":"no amount input","raw_output":"{\"transactions\":[{\"amount\":999,\"currency\":\"INR\",\"item\":\"thing\",\"category\":\"Other\",\"type\":\"expense\"}]}"}
{"input":"500 beer wait no 600 beer","raw_output":"{\"transactions\":[{\"amount\":500,\"currency\":\"INR\",\"item\":\"beer\",\"category\":\"Drinks\",\"type\":\"expense\"}]}"}
{"input":"500 beer once","raw_output":"{\"transactions\":[{\"amount\":500,\"currency\":\"INR\",\"item\":\"beer\",\"category\":\"Drinks\",\"type\":\"expense\"},{\"amount\":500,\"currency\":\"INR\",\"item\":\"beer\",\"category\":\"Drinks\",\"type\":\"expense\"}]}"}
```

- [ ] **Step 2: Verify file is valid JSONL**

```bash
cd "C:/work/llm training" && python -c "
import json
rows = []
with open('tests/fixtures/fake_labels_with_failures.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
print(f'fake_labels_with_failures.jsonl: {len(rows)} rows OK')
"
```
Expected: `fake_labels_with_failures.jsonl: 6 rows OK`.

- [ ] **Step 3: Write the orchestrator unit tests**

`tests/test_label_orchestrator.py`:
```python
"""Direct helper-level tests for the multi-provider label orchestrator.

Exercises _try_one_attempt and the fixture-failure-reason mapping
without needing subprocess invocation.
"""
import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "fake_labels_with_failures.jsonl"


def _orchestrator():
    """Import the orchestrator module (filename starts with digit)."""
    return importlib.import_module("05_generate_distillation_data")


# ---- _try_one_attempt happy path ------------------------------------------

class _SuccessProvider:
    name = "succ"
    model = "fake-model-1"
    provider_type = "fake"

    def generate_label(self, text):
        return '{"transactions":[{"amount":500,"currency":"INR","item":"beer","category":"Drinks","type":"expense"}]}'


class _RaisingProvider:
    name = "rais"
    model = None
    provider_type = "fake"

    def generate_label(self, text):
        raise RuntimeError("simulated provider failure")


class _BadJsonProvider:
    name = "badj"
    model = "fake-model-2"
    provider_type = "fake"

    def generate_label(self, text):
        return "{not json"


class _ValidatorFailingProvider:
    """Returns a schema-valid but amount-invented label."""
    name = "valid_fail"
    model = "fake-model-3"
    provider_type = "fake"

    def generate_label(self, text):
        # input may be "500 beer" but we emit 999 -> AMOUNT_NOT_IN_INPUT
        return '{"transactions":[{"amount":999,"currency":"INR","item":"beer","category":"Drinks","type":"expense"}]}'


class _SchemaInvalidProvider:
    name = "sch_inv"
    model = "fake-model-4"
    provider_type = "fake"

    def generate_label(self, text):
        return '{"transactions":[{"amount":100}]}'   # missing currency/item/etc


class _PcfgStub:
    def __init__(self, name):
        self.name = name


def test_try_one_attempt_success(monkeypatch):
    orch = _orchestrator()
    p = _SuccessProvider()
    outcome = orch._try_one_attempt(
        "500 beer", provider=p, pcfg=_PcfgStub("succ"),
    )
    assert outcome.failure_reason is None
    assert outcome.score > 0
    assert outcome.parsed_output["transactions"][0]["amount"] == 500
    assert outcome.is_repair_attempt is False


def test_try_one_attempt_provider_error():
    orch = _orchestrator()
    p = _RaisingProvider()
    outcome = orch._try_one_attempt(
        "500 beer", provider=p, pcfg=_PcfgStub("rais"),
    )
    assert outcome.failure_reason == "provider_error"
    assert outcome.score == 0
    assert outcome.error and "simulated provider failure" in outcome.error
    assert outcome.parsed_output is None
    assert outcome.validation is None


def test_try_one_attempt_json_parse_failed():
    orch = _orchestrator()
    p = _BadJsonProvider()
    outcome = orch._try_one_attempt(
        "500 beer", provider=p, pcfg=_PcfgStub("badj"),
    )
    assert outcome.failure_reason == "json_parse_failed"
    assert outcome.score == 0
    assert outcome.parsed_output is None


def test_try_one_attempt_schema_invalid():
    orch = _orchestrator()
    p = _SchemaInvalidProvider()
    outcome = orch._try_one_attempt(
        "500 beer", provider=p, pcfg=_PcfgStub("sch_inv"),
    )
    assert outcome.failure_reason == "schema_invalid"
    assert outcome.score == 0


def test_try_one_attempt_validation_failed():
    orch = _orchestrator()
    p = _ValidatorFailingProvider()
    outcome = orch._try_one_attempt(
        "500 beer", provider=p, pcfg=_PcfgStub("valid_fail"),
    )
    assert outcome.failure_reason == "validation_failed"
    assert outcome.score == 0
    assert outcome.parsed_output is not None
    assert outcome.validation is not None


def test_try_one_attempt_is_repair_attempt_flag():
    orch = _orchestrator()
    p = _SuccessProvider()
    outcome = orch._try_one_attempt(
        "500 beer", provider=p, pcfg=_PcfgStub("succ"),
        is_repair=True,
        failure_summary="prev failure",
        parser_candidates=[],
    )
    assert outcome.is_repair_attempt is True
    assert outcome.failure_reason is None


# ---- Fixture failure-reason mapping ---------------------------------------

def test_fixture_failure_codes_match_expectation():
    """Each row in fake_labels_with_failures.jsonl maps to a specific
    attempt-level failure reason (or None for clean pass). If validator
    code drifts, this test catches it first."""
    import json
    orch = _orchestrator()
    rows = []
    with FIXTURE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Map input string -> expected failure_reason (or None for clean pass)
    expected = {
        "500 beer": None,
        "schema fail case": "schema_invalid",
        "json fail case": "json_parse_failed",
        "no amount input": "validation_failed",
        "500 beer wait no 600 beer": "superseded_amount_used",
        "500 beer once": "suspicious_duplicate",
    }

    for row in rows:
        # Build a provider that returns this row's raw_output (or json-dumped output).
        if "raw_output" in row:
            payload = row["raw_output"]
        else:
            payload = json.dumps(row["output"], separators=(",", ":"))

        class FakeProv:
            name = "fp"
            model = None
            def generate_label(self_inner, text):
                return payload

        outcome = orch._try_one_attempt(
            row["input"], provider=FakeProv(), pcfg=_PcfgStub("fp"),
        )
        assert outcome.failure_reason == expected[row["input"]], (
            f"row {row['input']!r}: got {outcome.failure_reason!r}, "
            f"expected {expected[row['input']]!r}"
        )


# ---- _process_one_input integration ---------------------------------------

class _StubScheduler:
    """Deterministic scheduler that yields names from a pre-set list."""
    def __init__(self, names: list[str]):
        self._names = names
        self._idx = 0

    def next_provider(self) -> str:
        name = self._names[self._idx % len(self._names)]
        self._idx += 1
        return name


class _CfgStub:
    """Minimal cfg duck-type used by _process_one_input."""
    class _Og:
        label_attempts_per_input = 2
        providers = []
        provider_priority = ()
    class _V:
        retry_invalid_with_stricter_prompt = False
        max_repair_attempts = 0
    output_generation = _Og()
    validation = _V()


def test_process_one_input_picks_best_of_two():
    """Two providers; only one returns a valid label; best is the valid one."""
    orch = _orchestrator()
    providers = {
        "succ": _SuccessProvider(),
        "rais": _RaisingProvider(),
    }
    pcfgs = {n: _PcfgStub(n) for n in providers}
    scheduler = _StubScheduler(["succ", "rais"])
    cfg = _CfgStub()

    inp, best, all_outcomes = orch._process_one_input(
        "500 beer",
        providers_by_name=providers,
        pcfgs_by_name=pcfgs,
        scheduler=scheduler,
        priority_map={"succ": 0, "rais": 1},
        cfg=cfg,
        parser_candidates=[],
    )
    assert best is not None
    assert best.provider == "succ"
    assert len(all_outcomes) == 2


def test_process_one_input_returns_none_when_all_fail():
    """All providers fail; best is None; all outcomes have failure_reason."""
    orch = _orchestrator()
    providers = {
        "bad1": _BadJsonProvider(),
        "bad2": _RaisingProvider(),
    }
    pcfgs = {n: _PcfgStub(n) for n in providers}
    scheduler = _StubScheduler(["bad1", "bad2"])
    cfg = _CfgStub()

    inp, best, all_outcomes = orch._process_one_input(
        "500 beer",
        providers_by_name=providers,
        pcfgs_by_name=pcfgs,
        scheduler=scheduler,
        priority_map={"bad1": 0, "bad2": 1},
        cfg=cfg,
        parser_candidates=[],
    )
    assert best is None
    assert len(all_outcomes) == 2
    assert all(o.failure_reason is not None for o in all_outcomes)


def test_process_one_input_repair_succeeds():
    """First attempt fails; repair succeeds."""
    orch = _orchestrator()

    class FlakyProvider:
        """Fails on first call, succeeds on repair (second call)."""
        name = "flaky"
        model = "fake-model-flaky"
        def __init__(self):
            self.calls = 0
        def generate_label(self, text):
            self.calls += 1
            if self.calls <= 2:
                return "{not json"
            return '{"transactions":[{"amount":500,"currency":"INR","item":"beer","category":"Drinks","type":"expense"}]}'

    provider = FlakyProvider()
    providers = {"flaky": provider}
    pcfgs = {"flaky": _PcfgStub("flaky")}
    scheduler = _StubScheduler(["flaky", "flaky"])

    class _CfgRepair:
        class _Og:
            label_attempts_per_input = 2
            providers = []
            provider_priority = ()
        class _V:
            retry_invalid_with_stricter_prompt = True
            max_repair_attempts = 1
        output_generation = _Og()
        validation = _V()

    # Stub provider_priority to point at flaky (so repair uses it).
    cfg = _CfgRepair()
    # Make _highest_priority_provider find "flaky".
    cfg.output_generation.providers = [_PcfgStub("flaky")]

    inp, best, all_outcomes = orch._process_one_input(
        "500 beer",
        providers_by_name=providers,
        pcfgs_by_name=pcfgs,
        scheduler=scheduler,
        priority_map={"flaky": 0},
        cfg=cfg,
        parser_candidates=[],
    )
    assert best is not None
    assert best.is_repair_attempt is True
    assert len(all_outcomes) == 3   # 2 initial + 1 repair
```

- [ ] **Step 4: Run the orchestrator tests**

```bash
cd "C:/work/llm training" && pytest tests/test_label_orchestrator.py -v
```
Expected: all pass.

- [ ] **Step 5: Full suite stays green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```

- [ ] **Step 6: Do NOT commit.**

---

### Task 9: Flag contract update + main() dispatch + help text

**Files:**
- Modify: `scripts/05_generate_distillation_data.py` — delete `phase=="label"` block; update `main()`; refresh flag help
- Modify: `tests/test_stage5_flag_contract.py` — delete the "label exits 2" test; add "label succeeds" test

Spec reference: §7.

- [ ] **Step 1: Delete the obsolete `phase=="label"` test in `tests/test_stage5_flag_contract.py`**

Find and DELETE the test `test_provider_config_with_multi_provider_and_phase_label_exits_2`. Keep the `eval` and `all` tests; those phases still exit 2.

- [ ] **Step 2: Add the new "label succeeds" test**

Append to `tests/test_stage5_flag_contract.py`:

```python
def test_provider_config_with_multi_provider_and_phase_label_succeeds(tmp_path):
    """Headline Slice 3 capability: --phase label --multi-provider runs."""
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "fake_labels.jsonl"
    assert fixture_path.exists()
    # Use the first input from the fixture so the FakeProvider has a labeled match.
    first_row = json.loads(fixture_path.read_text(encoding="utf-8").splitlines()[0])
    input_text = first_row["input"]

    cfg_data = {
        "version": 1,
        "input_generation": {"enabled": False, "providers": []},
        "output_generation": {
            "enabled": True,
            "label_attempts_per_input": 1,
            "providers": [
                {"name": "fake_a", "type": "fake", "weight": 1, "threads": 1,
                 "fixture_labels": str(fixture_path)},
            ],
        },
        "validation": {"schema": True, "semantic_validator": True,
                       "reject_invalid": True,
                       "retry_invalid_with_stricter_prompt": False,
                       "max_repair_attempts": 0},
        "rate_limits": {"global_max_workers": 1, "write_flush_every": 5},
    }
    cfg_path = tmp_path / "label_test_providers.json"
    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")
    (tmp_path / "inputs_raw.jsonl").write_text(
        json.dumps({"input": input_text}) + "\n", encoding="utf-8",
    )

    result = run_cli(
        "--provider-config", str(cfg_path),
        "--multi-provider", "--phase", "label",
        env={"DISTILL_DIR_OVERRIDE": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    train_path = tmp_path / "train.jsonl"
    assert train_path.exists()
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["_source"] == "multi_provider_label"
    assert rows[0]["_provider"] == "fake_a"
```

Add `import json` to the file's imports if it's not already there.

- [ ] **Step 3: Run, expect failures (label is still rejected by contract)**

```bash
cd "C:/work/llm training" && pytest tests/test_stage5_flag_contract.py -v
```
Expected: the new test fails because the contract still errors on `--phase label`.

- [ ] **Step 4: Delete the `phase=="label"` block in `_enforce_flag_contract`**

In `scripts/05_generate_distillation_data.py`, find `_enforce_flag_contract`. Find and DELETE this block:

```python
        if args.phase == "label":
            parser.error(
                "--multi-provider --phase label is reserved for Slice 3 "
                "(validator-gated output labeling)."
            )
```

The `eval` and `all` blocks stay.

- [ ] **Step 5: Update `main()` dispatch**

Find `main()`. The current `if args.multi_provider:` block looks like:

```python
if args.multi_provider:
    assert args.phase == "inputs"
    assert args.provider_config is not None
    phase_inputs_multi_provider(args)
    logging.info("Stage 5 done.")
    return 0
```

Replace with:

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

- [ ] **Step 6: Refresh flag help text**

In `parse_args`, find the `--provider-config` and `--multi-provider` help strings. Update them:

```python
    p.add_argument("--provider-config", type=Path, default=None,
                   help="Path to a multi-provider generation config JSON. "
                        "Use with --dry-run-quota to inspect the config, or with "
                        "--multi-provider --phase inputs or --phase label to run "
                        "real generation.")
    # --dry-run-quota help unchanged.
    p.add_argument("--multi-provider", action="store_true",
                   help="Run multi-provider execution. Supports --phase inputs "
                        "(real input generation) and --phase label (validator-gated "
                        "output labeling). --phase eval is legacy-only — omit "
                        "--multi-provider for eval. --phase all exits via parser.error.")
```

- [ ] **Step 7: Run flag contract tests**

```bash
cd "C:/work/llm training" && pytest tests/test_stage5_flag_contract.py -v
```
Expected: all pass, including the new `_phase_label_succeeds` test.

- [ ] **Step 8: Full suite stays green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```

- [ ] **Step 9: Do NOT commit.**

---

### Task 10: End-to-end subprocess tests

**Files:**
- Create: `tests/test_multi_provider_labels.py`

Spec reference: §8.1.

- [ ] **Step 1: Write the subprocess tests**

`tests/test_multi_provider_labels.py`:
```python
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
```

- [ ] **Step 2: Run the new tests**

```bash
cd "C:/work/llm training" && pytest tests/test_multi_provider_labels.py -v
```
Expected: 6 passed. Subprocess tests, ~5-15s each.

- [ ] **Step 3: Full suite stays green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```

- [ ] **Step 4: Do NOT commit.**

---

### Task 11: Configs + docs

**Files:**
- Create: `configs/smoke_local_teacher_providers.json`
- Modify: `configs/smoke_real_providers.json` — add `output_generation` block
- Modify: `docs/provider_config.md` — append Slice 3 paragraph

Spec reference: §9.

- [ ] **Step 1: Create `configs/smoke_local_teacher_providers.json`**

```json
{
  "version": 1,
  "input_generation": {"enabled": false, "providers": []},
  "output_generation": {
    "enabled": true,
    "label_attempts_per_input": 2,
    "selection_policy": "first_valid_then_score",
    "providers": [
      {"name": "local_teacher", "type": "local_teacher",
       "model": "models/teacher/adapters",
       "weight": 70, "threads": 1, "max_tokens": 384},
      {"name": "gemini_flash", "type": "gemini", "model": "gemini-2.5-flash",
       "weight": 30, "threads": 4, "temperature": 0.0, "max_tokens": 512,
       "structured_output": true}
    ],
    "provider_priority": ["local_teacher", "gemini_flash"]
  },
  "validation": {"schema": true, "semantic_validator": true,
                 "reject_invalid": true,
                 "retry_invalid_with_stricter_prompt": true,
                 "max_repair_attempts": 1},
  "rate_limits": {"global_max_workers": 4, "write_flush_every": 5}
}
```

- [ ] **Step 2: Update `configs/smoke_real_providers.json`**

Read the current file. Replace the `output_generation` block (currently `enabled: false`) with:

```json
"output_generation": {
  "enabled": true,
  "label_attempts_per_input": 2,
  "selection_policy": "first_valid_then_score",
  "providers": [
    {"name": "deepseek_v4_pro", "type": "deepseek", "model": "deepseek-chat",
     "weight": 50, "threads": 4, "temperature": 0.0, "max_tokens": 512},
    {"name": "gemini_flash", "type": "gemini", "model": "gemini-2.5-flash",
     "weight": 50, "threads": 4, "temperature": 0.0, "max_tokens": 512,
     "structured_output": true}
  ],
  "provider_priority": ["gemini_flash", "deepseek_v4_pro"]
}
```

The other sections stay unchanged.

- [ ] **Step 3: Verify both configs load**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
from generation_config import load_generation_config
cfg1 = load_generation_config('configs/smoke_real_providers.json')
cfg2 = load_generation_config('configs/smoke_local_teacher_providers.json')
print('smoke_real output enabled?', cfg1.output_generation.enabled)
print('smoke_local output enabled?', cfg2.output_generation.enabled)
"
```
Expected:
```
smoke_real output enabled? True
smoke_local output enabled? True
```

- [ ] **Step 4: Verify `example_providers.json` is still safe**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
from generation_config import load_generation_config
cfg = load_generation_config('configs/example_providers.json')
assert cfg.input_generation.enabled is False, 'example input should be disabled'
assert cfg.output_generation.enabled is False, 'example output should be disabled'
print('example_providers.json: still safe (both phases disabled)')
"
```
Expected: `example_providers.json: still safe (both phases disabled)`.

NOTE: If your Slice 2 commit left `output_generation.enabled` as `true` on `example_providers.json`, change it to `false` for safety-by-default. Run this command to check first:
```bash
grep -A1 '"output_generation"' configs/example_providers.json | head -5
```
If you see `"enabled": true`, flip it to `false`.

- [ ] **Step 5: Update `docs/provider_config.md`**

Open `docs/provider_config.md`. Find the section titled `## What loads and runs` (Slice 2 added this heading). REPLACE the body (everything under `## What loads and runs` until the next `## ...` heading) with:

```markdown
**Slice 1 (configuration + dry-run):**
- Any `type: fake` provider with proper fixture paths.
- Any `type: deepseek` / `gemini` / `local_teacher` provider — these load without SDK imports. `--dry-run-quota` works for all of them.

**Slice 2 (real input generation):**
- `type: deepseek` and `type: gemini` providers run real API calls when invoked via
  `--phase inputs --provider-config <path> --multi-provider`. Requires `DEEPSEEK_API_KEY`
  and/or `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) env vars.

**Slice 3 (real output labeling):**
- `--phase label --provider-config <path> --multi-provider` runs the validator-gated
  multi-provider labeling loop.
- `type: local_teacher` providers require `model` (path to the adapter directory).
- Gemini's `structured_output: true` activates response_schema + response_mime_type
  for label generation only. Input generation ignores it.
- `validation.retry_invalid_with_stricter_prompt: true` plus `max_repair_attempts: N`
  enables the repair loop: when all initial attempts fail, re-prompt the highest-
  priority provider with a stricter repair prompt up to N times.
- `--phase eval` is legacy-only — omit `--multi-provider` for eval. `--phase all`
  exits via `parser.error`.
```

- [ ] **Step 6: Full suite stays green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```

- [ ] **Step 7: Do NOT commit.**

---

### Task 12: README delta

**Files:**
- Modify: `README.md`

Spec reference: §11.

- [ ] **Step 1: Append the Slice 3 subsection**

Open `README.md`. Find the `### Multi-provider input generation (Slice 2)` heading. After that subsection, APPEND:

```markdown
### Multi-provider output labeling (Slice 3)

Slice 3 makes `--phase label --provider-config <path> --multi-provider` actually run: real DeepSeek + Gemini label generation, a `LocalTeacherProvider` wrapping the fp16 Unsloth backend, validator-gated candidate scoring, and an optional repair-on-failure retry loop.

```bash
# Real API smoke (DeepSeek + Gemini)
export DEEPSEEK_API_KEY=sk-...
export GOOGLE_API_KEY=...     # or GEMINI_API_KEY
python scripts/05_generate_distillation_data.py \
    --phase label \
    --provider-config configs/smoke_real_providers.json \
    --multi-provider \
    --limit 10
```

For each input the orchestrator generates `label_attempts_per_input` candidates across configured providers, hard-rejects any with validator errors, scores survivors (clean validator pass is base 80; +10 count match, +5 no warnings, +5 priority bonus; max 100), and picks the highest-scoring. If all candidates fail and repair is enabled (`validation.retry_invalid_with_stricter_prompt: true` plus `max_repair_attempts > 0`), the orchestrator re-prompts the highest-priority provider with a stricter repair prompt containing the parsed amount candidates and validator failure summary.

Accepted rows in `data/distill/train.jsonl` carry `_source: "multi_provider_label"`, `_provider`, `_model`, `_validation_score`, `_attempts` metadata. Failed inputs land in `data/distill/failed.jsonl` with `reason ∈ {all_candidates_failed, repair_exhausted, provider_error}` and an `attempts[]` array of per-provider diagnostics.

Resume-safe: re-running skips inputs already in `train.jsonl` and (unless `--retry-failed`) inputs already in `failed.jsonl`. Use `--retry-validation-failed` to re-attempt only rows whose attempts include a validator-class failure.

`--phase eval` is legacy-only — omit `--multi-provider` for eval. `--phase all` exits via `parser.error`.

For local-teacher labeling (requires GPU + trained adapter at `models/teacher/adapters`), use `configs/smoke_local_teacher_providers.json`.
```

- [ ] **Step 2: Verify prior-slice README sections are intact**

```bash
cd "C:/work/llm training" && grep -E "Validator gate \(Phase 2\)|validator-derived aggregates|Dev dependencies|Multi-provider scaffolding \(Slice 1\)|Multi-provider input generation \(Slice 2\)|Multi-provider output labeling \(Slice 3\)" README.md
```
Expected: six matches. If any is missing, STOP — that section was lost.

- [ ] **Step 3: Sanity-check Slice 3 heading exists**

```bash
cd "C:/work/llm training" && python -c "
content = open('README.md', encoding='utf-8').read()
print('Slice 3 heading present:', 'Multi-provider output labeling (Slice 3)' in content)
"
```
Expected: `Slice 3 heading present: True`.

- [ ] **Step 4: Do NOT commit. Single commit lands at Task 14.**

---

### Task 13: Manual smokes + verify clean staging area

**Files:** none modified.

Spec reference: §11.

- [ ] **Step 1: Smoke 1 — Slice 2 input phase regression**

```bash
cd "C:/work/llm training"
DISTILL_DIR_OVERRIDE=/tmp/s3_smoke \
python scripts/05_generate_distillation_data.py \
    --phase inputs \
    --provider-config configs/test_providers.json \
    --multi-provider 2>&1 | tail -5
```
Expected: completes with exit 0, prints "Stage 5 done." Verify file:
```bash
ls -la /tmp/s3_smoke/inputs_raw.jsonl && wc -l /tmp/s3_smoke/inputs_raw.jsonl
```
Expected: 10 lines (target_inputs from test_providers.json).

- [ ] **Step 2: Smoke 2 — Slice 3 label phase against fake providers**

```bash
cd "C:/work/llm training"
DISTILL_DIR_OVERRIDE=/tmp/s3_smoke \
python scripts/05_generate_distillation_data.py \
    --phase label \
    --provider-config configs/test_providers.json \
    --multi-provider \
    --limit 5 2>&1 | tail -10
```
Expected: completes with exit 0. Check files:
```bash
ls -la /tmp/s3_smoke/ && \
  echo "train.jsonl rows: $(wc -l < /tmp/s3_smoke/train.jsonl 2>/dev/null || echo 0)" && \
  echo "failed.jsonl rows: $(wc -l < /tmp/s3_smoke/failed.jsonl 2>/dev/null || echo 0)"
```
Expected: total of `train.jsonl + failed.jsonl` rows = 5. Successful rows in `train.jsonl` have `_source: "multi_provider_label"`.

- [ ] **Step 3: Smoke 3 — Dry-run validates real-provider config (no API call)**

```bash
cd "C:/work/llm training"
python scripts/05_generate_distillation_data.py \
    --provider-config configs/smoke_real_providers.json \
    --dry-run-quota 2>&1 | tail -20
```
Expected: prints multi-provider dry-run summary, exit 0.

- [ ] **Step 4: Verify clean staging area**

```bash
cd "C:/work/llm training" && git status --short
```
Look at the list. The following MUST NOT appear staged:
- `data/distill/*.jsonl` (any modifications to data files)
- `reports/*`
- `.coverage`
- `htmlcov/*`

(They may appear as unstaged modifications or untracked — that's fine, just not staged. Note: Slice 1+2 baseline already shows `data/distill/*` as modified working tree; that's pre-existing.)

- [ ] **Step 5: Full suite final run**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: all tests pass.

- [ ] **Step 6: Coverage final check**

```bash
cd "C:/work/llm training" && pytest tests/ --cov=label_selection --cov=llm_providers --cov-report=term 2>&1 | tail -10
```
Expected:
- `label_selection.py`: ≥ 95%
- `llm_providers.py`: ≥ 80%

- [ ] **Step 7: Do NOT commit yet. Final commit is Task 14.**

---

### Task 14: Final commit

**Files:** none modified — this is the commit step.

- [ ] **Step 1: Stage the Slice 3 files**

CRITICAL: do NOT stage `data/distill/*`, `reports/*`, `.coverage`, `htmlcov/*`. Stage only source + tests + configs + docs.

```bash
cd "C:/work/llm training"

git add \
    scripts/_lib.py \
    scripts/label_selection.py \
    scripts/llm_providers.py \
    scripts/05_generate_distillation_data.py \
    configs/example_providers.json \
    configs/smoke_real_providers.json \
    configs/smoke_local_teacher_providers.json \
    docs/provider_config.md \
    tests/conftest.py \
    tests/test_candidate_scoring.py \
    tests/test_label_providers.py \
    tests/test_label_orchestrator.py \
    tests/test_multi_provider_labels.py \
    tests/test_stage5_flag_contract.py \
    tests/test_real_providers.py \
    tests/test_fake_provider.py \
    tests/fixtures/fake_labels_with_failures.jsonl \
    README.md

git status
```

Verify the staged section shows ONLY the listed files. The pre-existing `data/distill/*` modifications and `.coverage` / `reports/` should remain unstaged.

If `tests/test_fake_provider.py` shows no changes (the test we may have edited in Task 6 might not need an edit if the existing test was already structured that way), drop it from the add command.

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
Slice 3: validator-gated multi-provider output labeling

Makes --phase label --provider-config <path> --multi-provider actually
run with real DeepSeek + Gemini API calls and a real LocalTeacherProvider
wrapping the fp16 Unsloth backend. Per-input candidate generation, hard
rejection of any error-severity validator failure, score-based best-pick
across providers (max 100), and an optional repair-on-failure retry loop.

Adds:
- scripts/label_selection.py — pure scoring + best-pick + repair-prompt
  (CandidateOutcome dataclass; hard-reject any error-severity validator
  failure; tie-break score → priority rank → provider name)
- DeepSeekProvider.generate_label + _call_api_messages (OpenAI messages API)
- GeminiProvider.generate_label + _call_api_label + _label_schema_for_gemini
  (structured-output for labels only, enums sourced from _lib)
- LocalTeacherProvider — lazy backend load, threading.Lock-serialized,
  requires cfg.model = adapter directory path
- _lib.build_teacher_fp16_backend + _FP16Backend (additive; heavy imports
  inside the function bodies)
- phase_label_multi_provider — ThreadPoolExecutor + as_completed, main-thread
  JSONL writer, never-raises _try_one_attempt, repair-on-failure loop
- Stage 5 flag matrix: --multi-provider --phase label runs
  (eval/all still error; dry-run still wins)
- configs/smoke_local_teacher_providers.json (opt-in local-teacher smoke)
- configs/smoke_real_providers.json — output_generation block (opt-in)

train.jsonl accepted rows carry _source/provider/model/score/attempts metadata.
failed.jsonl rows carry attempts[] with per-provider diagnostics.

Out of scope (Slice 4): per-provider metrics, cost/latency tracking,
ensemble amount-voting.

Spec: docs/superpowers/specs/2026-05-16-multi-provider-generation-slice3-design.md
EOF
)"
```

- [ ] **Step 3: Verify the commit**

```bash
cd "C:/work/llm training" && git log --oneline -5 && git status
```
Expected: new commit on top of the Slice 3 spec/plan/amendment commits. Working tree shows only pre-existing data file modifications and untracked artifacts.

- [ ] **Step 4: Report**

Surface to the user:
- The new commit SHA.
- The full test pass count.
- Coverage percentages for `label_selection.py` and `llm_providers.py`.
- Confirmation that smokes 1+2+3 succeeded.
- Suggested next step: opt-in real-API smoke (`configs/smoke_real_providers.json`), or Slice 4 planning (metrics).
