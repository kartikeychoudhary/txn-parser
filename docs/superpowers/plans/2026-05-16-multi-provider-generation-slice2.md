# Multi-Provider Generation — Slice 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--phase inputs --provider-config <path> --multi-provider` actually run with real DeepSeek + Gemini API calls — threaded, deduped, resume-safe, with provider metadata in `inputs_raw.jsonl`. Legacy Stage 5 paths stay byte-identical when `--multi-provider` is absent.

**Architecture:** Adds `scripts/_retry.py` (backoff utility) and real `DeepSeekProvider` / `GeminiProvider` classes in `scripts/llm_providers.py` (lazy SDK imports — top-level imports stay SDK-free). Adds `phase_inputs_multi_provider` in `scripts/05_generate_distillation_data.py` using a `ThreadPoolExecutor` + `queue.Queue` writer model (workers produce, main thread writes — no JSONL contention). Moves `clean_input_line` to `scripts/_lib.py` with a bug-fix regex that no longer corrupts digit-led natural inputs. Spec at `docs/superpowers/specs/2026-05-16-multi-provider-generation-slice2-design.md` is the source of truth.

**Tech Stack:** Python 3.11+, stdlib (`dataclasses`, `pathlib`, `threading`, `queue`, `concurrent.futures`, `argparse`, `json`, `re`, `math`, `time`, `random`, `os`, `logging`), new deps `openai>=1.50` + `google-genai` (floor pinned by Task 0). Tests use `pytest` + `pytest-cov` (already in `requirements-dev.txt`). Existing `tests/conftest.py` puts `scripts/` on `sys.path`.

**File map:**

| File | Purpose |
|---|---|
| `scripts/_retry.py` (new) | `retry_with_backoff` — exponential backoff + jitter + injectable sleep |
| `scripts/_lib.py` (modify) | +`clean_input_line` (with safer regex), +`normalize_input`, +`_LINE_PREFIX_RE` |
| `scripts/llm_providers.py` (modify) | +`DeepSeekProvider`, +`GeminiProvider`, +`_parse_input_lines`, +`_is_retryable_gemini_error`; update `create_provider` factory |
| `scripts/05_generate_distillation_data.py` (modify) | +`phase_inputs_multi_provider` + worker helpers; rename `_enforce_slice1_flag_contract` → `_enforce_flag_contract` with Slice 2 phase rules; honor `DISTILL_DIR_OVERRIDE`; update `read_inputs_jsonl` malformed-row policy; delete in-file `clean_input_line` and `_LINE_PREFIX_RE` (now in `_lib.py`) — keep a re-export |
| `configs/example_providers.json` (modify) | `enabled: false`, `target_inputs: 10`, safe-by-default |
| `configs/smoke_real_providers.json` (new) | Opt-in real-API smoke (`target_inputs: 100`, 50/50 deepseek/gemini) |
| `docs/provider_config.md` (modify) | Update "What loads successfully" — Slice 2 makes deepseek/gemini runnable, not just configurable |
| `tests/test_retry.py` (new) | 6 tests; pure unit |
| `tests/test_clean_input_line.py` (new) | Digit-led regression suite + bullet/numbering tests |
| `tests/test_real_providers.py` (new) | DeepSeek + Gemini construction, retry behavior, omit-None kwargs |
| `tests/test_multi_provider_inputs.py` (new) | End-to-end via subprocess + FakeProvider |
| `tests/test_stage5_flag_contract.py` (modify) | Update rows; `run_cli` accepts `env=` |
| `requirements-train.txt` (modify) | +`openai>=1.50`, +`google-genai>=<pinned-at-task0>` |
| `README.md` (modify) | "Slice 2: Multi-provider input generation" subsection |

**Commit policy:** Single commit at Task 13. Every preceding task says **do NOT commit** explicitly.

**Prior-slice tests must stay green throughout:** 177 from Slice 1 + slice baseline. Any task that breaks them is a regression, not progress.

---

### Task 0: Preflight — verify prerequisites and install SDKs

**Files:** none modified.

Spec reference: §10, §11 Task 0.

This task fails fast if Slice 1 isn't merged or branched-off, deps don't install, or SDK surface differs from the spec.

- [ ] **Step 1: Confirm branch**

```bash
cd "C:/work/llm training" && git branch --show-current
```
Expected: `feat/multi-provider-slice2`. If not, STOP and report — this plan assumes that branch.

- [ ] **Step 2: Verify Slice 1 + validator-slice imports**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "from _lib import validate_example, serialize_validation_result, parse_amounts; print('validator slice ok')"
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "from llm_providers import FakeProvider, UnimplementedProvider, create_provider, ProviderError, LLMProvider; print('Slice 1 ok')"
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "from generation_config import load_generation_config, ConfigError; print('config ok')"
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "from generation_orchestrator import allocate_quota, RoundRobinScheduler, render_dry_run; print('orchestrator ok')"
```
Expected: four lines, each printing `<name> ok`.

- [ ] **Step 3: Prior-slice tests green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: `<N> passed`. If anything fails, STOP and report.

- [ ] **Step 4: Install new SDK deps**

```bash
pip install 'openai>=1.50' google-genai
```
Expected: both packages install without errors.

- [ ] **Step 5: Verify SDK surface matches the spec**

```bash
python -c "from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError; print('openai ok')"
python -c "from google import genai; from google.genai import types, errors; print('genai ok')"
```
Expected: both prints succeed. If `google.genai.errors` doesn't exist or `types.GenerateContentConfig` is missing, STOP — spec assumes the new Gen AI SDK; verify the package is `google-genai`, not `google-generativeai`.

- [ ] **Step 6: Capture installed `google-genai` version for floor pin**

```bash
python -c "import importlib.metadata as m; print(m.version('google-genai'))"
```
Record the output (e.g., `0.4.2`). The Task 10 `requirements-train.txt` edit pins `google-genai>=<this-version>` as the floor.

- [ ] **Step 7: Do NOT commit. Preflight has no diffs.**

---

### Task 1: `scripts/_retry.py` + tests

**Files:**
- Create: `scripts/_retry.py`
- Create: `tests/test_retry.py`

Spec reference: §4.

- [ ] **Step 1: Write the failing tests**

`tests/test_retry.py`:
```python
import pytest
from _retry import retry_with_backoff


class TransientError(Exception):
    pass


class PermanentError(Exception):
    pass


def test_succeeds_on_first_try():
    calls = []
    def ok():
        calls.append(1)
        return "ok"
    result = retry_with_backoff(
        ok,
        retryable=(TransientError,),
        sleep=lambda _: None,
    )
    assert result == "ok"
    assert len(calls) == 1


def test_retries_then_succeeds():
    state = {"attempts": 0}
    def flaky():
        state["attempts"] += 1
        if state["attempts"] < 3:
            raise TransientError("nope")
        return "ok"
    result = retry_with_backoff(
        flaky,
        retryable=(TransientError,),
        max_retries=3,
        sleep=lambda _: None,
    )
    assert result == "ok"
    assert state["attempts"] == 3


def test_exhausts_retries_and_reraises():
    def always_fails():
        raise TransientError("persistent")
    with pytest.raises(TransientError):
        retry_with_backoff(
            always_fails,
            retryable=(TransientError,),
            max_retries=2,
            sleep=lambda _: None,
        )


def test_non_retryable_exception_propagates_immediately():
    state = {"calls": 0}
    def fails():
        state["calls"] += 1
        raise PermanentError("auth")
    with pytest.raises(PermanentError):
        retry_with_backoff(
            fails,
            retryable=(TransientError,),
            max_retries=5,
            sleep=lambda _: None,
        )
    # Permanent exception is not in `retryable` tuple, so no retries happened.
    assert state["calls"] == 1


def test_is_retryable_predicate_filters():
    """Used by Gemini: ClientError is retryable iff status in {429,5xx}."""
    class StatusError(Exception):
        def __init__(self, status):
            super().__init__(f"status={status}")
            self.status_code = status

    state = {"calls": 0}
    def call_with_400():
        state["calls"] += 1
        raise StatusError(400)

    with pytest.raises(StatusError):
        retry_with_backoff(
            call_with_400,
            retryable=(StatusError,),
            is_retryable=lambda e: e.status_code in {429, 500, 502, 503, 504},
            max_retries=5,
            sleep=lambda _: None,
        )
    # 400 fails the predicate, so we bail after the first call.
    assert state["calls"] == 1


def test_sleep_durations_grow_exponentially():
    sleeps = []
    def fail():
        raise TransientError("x")
    with pytest.raises(TransientError):
        retry_with_backoff(
            fail,
            retryable=(TransientError,),
            max_retries=3,
            base_delay=1.0,
            jitter=0.0,
            sleep=sleeps.append,
        )
    # 3 retries → 3 sleeps. Each at least 1, 2, 4.
    assert len(sleeps) == 3
    assert sleeps[0] >= 1.0
    assert sleeps[1] >= 2.0
    assert sleeps[2] >= 4.0
```

- [ ] **Step 2: Run, expect ModuleNotFoundError**

```bash
cd "C:/work/llm training" && pytest tests/test_retry.py -v
```
Expected: `ModuleNotFoundError: No module named '_retry'`.

- [ ] **Step 3: Create `scripts/_retry.py`**

```python
"""Exponential-backoff retry with jitter. Pure stdlib.

Used by DeepSeekProvider, GeminiProvider, and any future provider that
talks to an external API. Slice 4 will likely add observability hooks
here (per-attempt latency, attempt counter) — keep the surface small now.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, Type, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    retryable: tuple[Type[BaseException], ...],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.5,
    is_retryable: Callable[[BaseException], bool] | None = None,
    logger_name: str = "retry",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `fn`; on a retryable exception, sleep and retry up to max_retries.

    Retry predicate:
      - Exception type must be in `retryable` tuple, AND
      - If `is_retryable` is provided, it must return True for the exception.

    Sleep duration: min(max_delay, base_delay * 2**attempt) + uniform(0, jitter).
    On final failure: re-raises the original exception unchanged.
    """
    logger = logging.getLogger(logger_name)
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retryable as e:
            if is_retryable is not None and not is_retryable(e):
                raise   # non-retryable subclass; bail immediately
            last_exc = e
            if attempt >= max_retries:
                break
            delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, jitter)
            logger.warning(
                "Attempt %d/%d failed: %s. Sleeping %.1fs before retry.",
                attempt + 1, max_retries + 1, e, delay,
            )
            sleep(delay)
    assert last_exc is not None
    raise last_exc
```

- [ ] **Step 4: Run tests, expect 6 passed**

```bash
cd "C:/work/llm training" && pytest tests/test_retry.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Full suite stays green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: prior count + 6.

- [ ] **Step 6: Do NOT commit.**

---

### Task 2: Move `clean_input_line` + add `normalize_input` to `_lib.py` (with safer regex)

**Files:**
- Modify: `scripts/_lib.py` — add new helpers
- Modify: `scripts/05_generate_distillation_data.py` — delete local definition, add re-export
- Create: `tests/test_clean_input_line.py`

Spec reference: §6.2. **Critical**: the new regex `r"^\s*(?:[-*]\s+|\d+[.)]\s+)"` is intentionally narrower than the old `r"^[\s\-\*\d]+[.)\s]+"`. Tests must catch any regression to the old behavior.

- [ ] **Step 1: Write the failing test file**

`tests/test_clean_input_line.py`:
```python
import pytest

from _lib import clean_input_line, normalize_input


@pytest.mark.parametrize("raw,expected", [
    # Numbered list — strip
    ("1. 500 rs on beer", "500 rs on beer"),
    ("2) 500 rs on beer", "500 rs on beer"),
    ("12. paid 200 for chai", "paid 200 for chai"),
    # Bullets — strip
    ("- 500 rs on beer", "500 rs on beer"),
    ("* 500 rs on beer", "500 rs on beer"),
    ("  - 500 rs on beer", "500 rs on beer"),
    # Digit-led NATURAL content — must NOT strip (regression for the old regex bug)
    ("500 beer", "500 beer"),
    ("200 chai", "200 chai"),
    ("1 lakh rent", "1 lakh rent"),
    ("500 rs on beer 50 rs on candy", "500 rs on beer 50 rs on candy"),
    ("12500 emi for laptop", "12500 emi for laptop"),
    # Quotes — strip
    ('"500 beer"', "500 beer"),
    ("'500 beer'", "500 beer"),
    # Junk — None
    ("", None),
    ("```", None),
    ("output: 500 beer", None),
    ("example: something", None),
    ("ab", None),                  # too short
    ("x" * 301, None),             # too long
])
def test_clean_input_line(raw, expected):
    assert clean_input_line(raw) == expected


def test_normalize_input_collapses_whitespace_and_lowercases():
    assert normalize_input("  500   Rs On BEER  ") == "500 rs on beer"


def test_normalize_input_idempotent():
    once = normalize_input("500 beer ")
    assert normalize_input(once) == once


def test_normalize_input_distinguishes_real_differences():
    assert normalize_input("500 beer") != normalize_input("500 chai")
```

- [ ] **Step 2: Run, expect failure (ImportError on `clean_input_line`/`normalize_input` from `_lib`)**

```bash
cd "C:/work/llm training" && pytest tests/test_clean_input_line.py -v
```
Expected: `ImportError: cannot import name 'clean_input_line' from '_lib'`.

- [ ] **Step 3: Add helpers to `scripts/_lib.py`**

Open `scripts/_lib.py`. Confirm `import re` is already at the top (it is — used by `extract_json`). Locate the Slice 1 re-export block at the bottom of the file (the comment line `# Re-exports — placed at the bottom of _lib.py so amount_parser/validator can`...). Insert the new helpers IMMEDIATELY BEFORE that re-export block — the re-export block must remain the last block in the file to avoid circular-import issues:

```python


# ---------------------------------------------------------------------------
# Input-line cleaning helpers (shared between legacy phase_inputs and the
# multi-provider _parse_input_lines). Migrated here in Slice 2 so both
# call sites use the same fixed regex.
# ---------------------------------------------------------------------------

# Strips leading bullets ("- ", "* ") and numbering ("1. ", "2) ").
# Deliberately narrow: must NOT match digit-led natural content like "500 beer".
# The previous regex r"^[\s\-\*\d]+[.)\s]+" had this bug.
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)")


def clean_input_line(line: str) -> str | None:
    """Strip leading bullets/numbering and surrounding whitespace.

    Strips:
      "- 500 beer"   -> "500 beer"
      "* 500 beer"   -> "500 beer"
      "1. 500 beer"  -> "500 beer"
      "2) 500 beer"  -> "500 beer"

    Does NOT strip digit-led natural content:
      "500 beer"     -> "500 beer"
      "1 lakh rent"  -> "1 lakh rent"

    Returns None for blank, fenced, or out-of-range inputs.
    """
    s = line.strip()
    if not s:
        return None
    if s.startswith("```") or s.lower().startswith("output:") or s.lower().startswith("example"):
        return None
    s = _LINE_PREFIX_RE.sub("", s).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if len(s) < 3 or len(s) > 300:
        return None
    return s


def normalize_input(text: str) -> str:
    """Canonical dedupe key. Lower-cased, whitespace-collapsed."""
    return " ".join(text.lower().split())
```

- [ ] **Step 4: Run new tests, expect 22 passed (19 parametrized + 3 normalize_input)**

```bash
cd "C:/work/llm training" && pytest tests/test_clean_input_line.py -v
```
Expected: 22 passed.

- [ ] **Step 5: Remove the in-file definition from `05_generate_distillation_data.py`**

Open `scripts/05_generate_distillation_data.py`. Find and DELETE these blocks:

```python
_LINE_PREFIX_RE = re.compile(r"^[\s\-\*\d]+[.)\s]+")


def clean_input_line(line: str) -> str | None:
    """Strip leading numbering/bullets and trailing whitespace; reject junk."""
    s = line.strip()
    if not s:
        return None
    if s.startswith("```") or s.lower().startswith("output:") or s.lower().startswith("example"):
        return None
    # Strip "1. ", "- ", "* ", "1) " style prefixes the model occasionally adds.
    s = _LINE_PREFIX_RE.sub("", s).strip()
    # Strip surrounding quotes.
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    if len(s) < 3 or len(s) > 300:
        return None
    return s
```

In the same file, locate the existing `from _lib import (...)` block (around line 46–54). Add `clean_input_line` to the imported names:

```python
from _lib import (  # noqa: E402
    BATCH_FOCUSES,
    build_messages,
    call_deepseek,
    clean_input_line,   # NEW in Slice 2 — moved from this file
    extract_json,
    load_jsonl,
    parse_amounts,
    serialize_validation_result,
    validate_example,
)
```

(Note: keep the existing import list — only ADD `clean_input_line`.)

- [ ] **Step 6: Sanity check — script still imports + works**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "import importlib; m = importlib.import_module('05_generate_distillation_data'); print(m.clean_input_line('- 500 beer'))"
```
Expected: prints `500 beer` (re-export works; legacy path can still call `clean_input_line`).

- [ ] **Step 7: Full suite green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: prior count + 22.

- [ ] **Step 8: Do NOT commit.**

---

### Task 3: Update `read_inputs_jsonl` malformed-row policy

**Files:**
- Modify: `scripts/05_generate_distillation_data.py` — replace `read_inputs_jsonl`

Spec reference: §6.1.

- [ ] **Step 1: Locate the existing `read_inputs_jsonl` in `05_generate_distillation_data.py`**

It looks like:
```python
def read_inputs_jsonl(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line)["input"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out
```

- [ ] **Step 2: Replace with the policy-compliant version**

```python
def read_inputs_jsonl(path: Path) -> list[str]:
    """Load existing input strings from a JSONL file.

    Malformed rows are skipped with a warning (per spec §6.1):
      - Bad JSON → bad_json counter
      - Non-object rows → bad_shape counter
      - Missing or non-string 'input' field → bad_shape counter
    """
    if not path.exists():
        return []
    out: list[str] = []
    bad_json = 0
    bad_shape = 0
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning("read_inputs_jsonl: %s:%d invalid JSON: %s",
                                path.name, line_no, e)
                bad_json += 1
                continue
            if not isinstance(obj, dict):
                logging.warning("read_inputs_jsonl: %s:%d row is not an object, skipping",
                                path.name, line_no)
                bad_shape += 1
                continue
            if "input" not in obj or not isinstance(obj["input"], str):
                logging.warning(
                    "read_inputs_jsonl: %s:%d missing or non-string 'input' field, skipping",
                    path.name, line_no,
                )
                bad_shape += 1
                continue
            out.append(obj["input"])
    if bad_json or bad_shape:
        logging.info(
            "read_inputs_jsonl: loaded %d inputs from %s; skipped bad_json=%d bad_shape=%d",
            len(out), path, bad_json, bad_shape,
        )
    return out
```

- [ ] **Step 3: Sanity check — script still imports + helper still callable on the existing data file**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
import importlib
m = importlib.import_module('05_generate_distillation_data')
from pathlib import Path
n = len(m.read_inputs_jsonl(Path('data/distill/inputs_raw.jsonl')))
print(f'read {n} inputs')
"
```
Expected: prints a number (e.g., `read 29890 inputs`) without errors. If the existing file has malformed rows, you may see warning logs — that's intended.

- [ ] **Step 4: Full suite green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: prior count, no regressions.

- [ ] **Step 5: Do NOT commit.**

---

### Task 4: `DeepSeekProvider` + `_parse_input_lines` + factory wiring

**Files:**
- Modify: `scripts/llm_providers.py` — add `DeepSeekProvider`, `_parse_input_lines`, update `create_provider`
- Create: `tests/test_real_providers.py` — DeepSeek half only (Gemini half lands in Task 5)

Spec reference: §3.3, §3.5, §3.6.

- [ ] **Step 1: Write the DeepSeek tests**

`tests/test_real_providers.py`:
```python
"""Tests for real provider implementations (DeepSeek + Gemini).

Strategy: SDK imports happen inside provider __init__, so the imports
are exercised by construction. SDK Client constructors are patched
(see _stub_openai_client / _stub_gemini_client fixtures) so no real
network I/O happens during construction. Behavior tests then patch
provider._call_api directly to control responses.
"""
import pytest

from llm_providers import (
    DeepSeekProvider,
    ProviderError,
    _parse_input_lines,
)


# ---- SDK constructor stubs ------------------------------------------------
# Reusable autouse fixtures keep test bodies focused on behavior, not on
# SDK construction plumbing.


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


@pytest.fixture(autouse=True)
def _patch_sdk_clients(monkeypatch):
    """Patch the SDK Client classes so provider __init__ never makes network calls."""
    import openai
    monkeypatch.setattr(openai, "OpenAI", _StubOpenAIClient)
    from google import genai
    monkeypatch.setattr(genai, "Client", _StubGeminiClient)


# ---- DeepSeek construction ------------------------------------------------

def test_deepseek_constructs_with_env_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    assert p.name == "ds"
    assert p.provider_type == "deepseek"
    assert p.model == "deepseek-chat"


def test_deepseek_constructs_with_explicit_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    p = DeepSeekProvider(name="ds", model="deepseek-chat", api_key="explicit-key")
    assert p.name == "ds"


def test_deepseek_missing_key_raises_provider_error(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ProviderError) as exc:
        DeepSeekProvider(name="ds", model="deepseek-chat")
    assert "DEEPSEEK_API_KEY" in str(exc.value)


def test_deepseek_does_not_fall_back_to_openai_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-be-used")
    with pytest.raises(ProviderError):
        DeepSeekProvider(name="ds", model="deepseek-chat")


# ---- DeepSeek generate_inputs ---------------------------------------------

def test_deepseek_generate_inputs_returns_parsed_lines(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    monkeypatch.setattr(p, "_call_api", lambda prompt: "500 beer\n200 chai\n100 samosa")
    result = p.generate_inputs("ignored prompt", n=3)
    assert result == ["500 beer", "200 chai", "100 samosa"]


def test_deepseek_generate_inputs_truncates_to_n(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    monkeypatch.setattr(p, "_call_api", lambda prompt: "a x\nb y\nc z\nd q\ne r")
    assert len(p.generate_inputs("ignored", n=3)) == 3


def test_deepseek_generate_inputs_zero_returns_empty(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    # _call_api should not even be invoked.
    monkeypatch.setattr(p, "_call_api",
                        lambda _: (_ for _ in ()).throw(AssertionError("should not call API")))
    assert p.generate_inputs("ignored", n=0) == []


def test_deepseek_generate_inputs_negative_n_raises(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    with pytest.raises(ProviderError):
        p.generate_inputs("ignored", n=-1)


def test_deepseek_generate_label_raises_notimplemented(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    with pytest.raises(NotImplementedError) as exc:
        p.generate_label("anything")
    assert "Slice 3" in str(exc.value)


# ---- DeepSeek retry behavior (using fake exception, NOT real SDK error) ---

class _FakeTransientError(Exception):
    pass


def test_deepseek_retries_on_transient(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat", max_retries=2)
    # Reassign retryable to our fake class so we don't depend on SDK exception constructor.
    p._retryable_excs = (_FakeTransientError,)
    p._sleep = lambda _: None

    state = {"calls": 0}
    def flaky(_prompt):
        state["calls"] += 1
        if state["calls"] < 2:
            raise _FakeTransientError("transient")
        return "ok line one"
    monkeypatch.setattr(p, "_call_api", flaky)

    result = p.generate_inputs("prompt", n=1)
    assert state["calls"] == 2
    assert result == ["ok line one"]


def test_deepseek_max_retries_exhausted_raises(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat", max_retries=1)
    p._retryable_excs = (_FakeTransientError,)
    p._sleep = lambda _: None

    state = {"calls": 0}
    def always_fail(_prompt):
        state["calls"] += 1
        raise _FakeTransientError("persistent")
    monkeypatch.setattr(p, "_call_api", always_fail)

    with pytest.raises(_FakeTransientError):
        p.generate_inputs("prompt", n=1)
    assert state["calls"] == 2   # 1 try + 1 retry


def test_deepseek_retryable_excs_is_populated_with_real_sdk_types(monkeypatch):
    """Verify __init__ wires the real SDK exception classes (just shape, no call)."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    p = DeepSeekProvider(name="ds", model="deepseek-chat")
    from openai import RateLimitError, APITimeoutError, APIConnectionError
    assert RateLimitError in p._retryable_excs
    assert APITimeoutError in p._retryable_excs
    assert APIConnectionError in p._retryable_excs


# ---- DeepSeek omits None kwargs --------------------------------------------

class _FakeMessage:
    content = "ok line"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResp:
    choices = [_FakeChoice()]


def test_deepseek_omits_temperature_and_max_tokens_when_none(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    captured = {}
    p = DeepSeekProvider(name="ds", model="deepseek-chat",
                         temperature=None, max_tokens=None)
    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResp()
    monkeypatch.setattr(p._client.chat.completions, "create", fake_create)
    p.generate_inputs("prompt", n=1)
    assert "temperature" not in captured
    assert "max_tokens" not in captured
    assert captured["model"] == "deepseek-chat"


def test_deepseek_passes_temperature_and_max_tokens_when_set(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    captured = {}
    p = DeepSeekProvider(name="ds", model="deepseek-chat",
                         temperature=0.7, max_tokens=500)
    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResp()
    monkeypatch.setattr(p._client.chat.completions, "create", fake_create)
    p.generate_inputs("prompt", n=1)
    assert captured["temperature"] == 0.7
    assert captured["max_tokens"] == 500


# ---- _parse_input_lines (direct unit) -------------------------------------

def test_parse_input_lines_cleans_dedupes_and_truncates():
    text = "1. 500 beer\n500 beer\n- 200 chai\n```"
    assert _parse_input_lines(text, expected=2) == ["500 beer", "200 chai"]


def test_parse_input_lines_handles_empty_input():
    assert _parse_input_lines("", expected=5) == []


def test_parse_input_lines_local_dedupe_is_case_insensitive():
    text = "500 Beer\n500 BEER\n200 chai"
    # First "500 Beer" wins; "500 BEER" is rejected as duplicate.
    result = _parse_input_lines(text, expected=10)
    assert result == ["500 Beer", "200 chai"]
```

- [ ] **Step 2: Run, expect ImportError on `DeepSeekProvider` / `_parse_input_lines`**

```bash
cd "C:/work/llm training" && pytest tests/test_real_providers.py -v
```
Expected: `ImportError: cannot import name 'DeepSeekProvider' from 'llm_providers'`.

- [ ] **Step 3: Add `_parse_input_lines` + `DeepSeekProvider` to `scripts/llm_providers.py`**

Open `scripts/llm_providers.py`. At the TOP of the file, add `os` and `time` to the imports (the file currently has `import json`, `import threading`, `from pathlib import Path`, `from typing import Protocol, runtime_checkable`):

```python
import os
import time
```

Then APPEND (after the existing `create_provider` function, but BEFORE you update `create_provider` in the next step):

```python
def _parse_input_lines(text: str, *, expected: int) -> list[str]:
    """Provider output → cleaned, locally-deduped input strings.

    Local dedupe catches the common case of one API call returning the
    same line twice. Global dedupe (against existing inputs_raw.jsonl +
    this run's accepted set) happens in the orchestrator.
    """
    from _lib import clean_input_line, normalize_input
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        cleaned = clean_input_line(raw)
        if cleaned is None:
            continue
        key = normalize_input(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= expected:
            break
    return out


class DeepSeekProvider:
    """Real DeepSeek provider via the openai SDK (OpenAI-compatible endpoint).

    Lazy SDK import: `openai` is imported inside __init__, NOT at module
    load, so `import llm_providers` stays SDK-free for tests/dry-run.
    """

    provider_type = "deepseek"

    def __init__(
        self,
        name: str,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        temperature: float | None = 1.0,
        max_tokens: int | None = 8000,
        max_retries: int = 3,
        timeout: int = 300,
    ) -> None:
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self._sleep = time.sleep   # test hook

        from openai import OpenAI                                       # lazy
        from openai import APITimeoutError, RateLimitError, APIConnectionError

        # Intentionally do not read OPENAI_API_KEY; DeepSeek has its own key.
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ProviderError(
                f"DeepSeekProvider {name!r}: DEEPSEEK_API_KEY not set"
            )
        self._client = OpenAI(api_key=key, base_url=base_url)
        self._retryable_excs = (APITimeoutError, RateLimitError, APIConnectionError)

    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        if n < 0:
            raise ProviderError(f"generate_inputs: n must be >= 0, got {n}")
        if n == 0:
            return []
        text = self._call_with_retry(prompt)
        return _parse_input_lines(text, expected=n)

    def generate_label(self, input_text: str) -> str:
        raise NotImplementedError(
            f"deepseek provider {self.name!r}: output labeling lands in Slice 3"
        )

    def _call_with_retry(self, prompt: str) -> str:
        from _retry import retry_with_backoff
        return retry_with_backoff(
            lambda: self._call_api(prompt),
            retryable=self._retryable_excs,
            max_retries=self.max_retries,
            logger_name=f"deepseek.{self.name}",
            sleep=self._sleep,
        )

    def _call_api(self, prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": self.timeout,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
```

- [ ] **Step 4: Update `create_provider` factory**

In `scripts/llm_providers.py`, find the existing `create_provider` function. The current Slice-1 body looks like:

```python
def create_provider(cfg) -> LLMProvider:
    if cfg.provider_type == "fake":
        return FakeProvider(...)
    if cfg.provider_type in {"deepseek", "gemini", "local_teacher"}:
        return UnimplementedProvider(
            name=cfg.name, provider_type=cfg.provider_type, model=cfg.model,
        )
    raise ValueError(...)
```

Replace the `if cfg.provider_type in {"deepseek", "gemini", "local_teacher"}` branch with:

```python
    if cfg.provider_type == "deepseek":
        return DeepSeekProvider(
            name=cfg.name,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_retries=cfg.max_retries,
        )
    if cfg.provider_type == "gemini":
        # GeminiProvider lands in Task 5. Until then, fall through to
        # UnimplementedProvider so configs parse but execution errors.
        return UnimplementedProvider(
            name=cfg.name, provider_type="gemini", model=cfg.model,
        )
    if cfg.provider_type == "local_teacher":
        return UnimplementedProvider(
            name=cfg.name, provider_type="local_teacher", model=cfg.model,
        )
```

Keep the final `raise ValueError(...)` for unknown types.

- [ ] **Step 5: Run DeepSeek tests, expect all 17 passed**

```bash
cd "C:/work/llm training" && pytest tests/test_real_providers.py -v
```
Expected: 17 passed. (4 construction + 5 generate_inputs/label + 3 retry-behavior + 2 omit-None kwargs + 3 _parse_input_lines = 17.)

- [ ] **Step 6: Verify Slice 1 SDK-leak guard test still passes**

```bash
cd "C:/work/llm training" && pytest tests/test_fake_provider.py::test_unimplemented_provider_has_no_sdk_imports -v
```
Expected: 1 passed. (UnimplementedProvider construction must not import openai.)

- [ ] **Step 7: Full suite green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: prior count + 17. No regressions.

- [ ] **Step 8: Do NOT commit.**

---

### Task 5: `GeminiProvider` + `_is_retryable_gemini_error` + factory

**Files:**
- Modify: `scripts/llm_providers.py` — add `GeminiProvider`, `_is_retryable_gemini_error`; update `create_provider` factory's gemini branch
- Modify: `tests/test_real_providers.py` — append Gemini tests

Spec reference: §3.4.

- [ ] **Step 1: Append Gemini tests to `tests/test_real_providers.py`**

```python
# ---- Gemini construction --------------------------------------------------

def test_gemini_constructs_with_google_api_key(monkeypatch):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash")
    assert p.name == "g"
    assert p.provider_type == "gemini"


def test_gemini_falls_back_to_gemini_api_key_env(monkeypatch):
    from llm_providers import GeminiProvider
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash")
    assert p.name == "g"


def test_gemini_missing_key_raises(monkeypatch):
    from llm_providers import GeminiProvider
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError) as exc:
        GeminiProvider(name="g", model="gemini-2.5-flash")
    assert "GOOGLE_API_KEY" in str(exc.value) or "GEMINI_API_KEY" in str(exc.value)


def test_gemini_generate_inputs_returns_parsed_lines(monkeypatch):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash")
    monkeypatch.setattr(p, "_call_api", lambda prompt: "500 beer\n200 chai")
    result = p.generate_inputs("ignored", n=2)
    assert result == ["500 beer", "200 chai"]


def test_gemini_generate_label_raises_notimplemented(monkeypatch):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash")
    with pytest.raises(NotImplementedError) as exc:
        p.generate_label("anything")
    assert "Slice 3" in str(exc.value)


# ---- _is_retryable_gemini_error predicate ---------------------------------

def test_is_retryable_gemini_error_retries_429_and_5xx(monkeypatch):
    from llm_providers import _is_retryable_gemini_error

    class E(Exception):
        def __init__(self, status):
            super().__init__()
            self.status_code = status

    for status in (429, 500, 502, 503, 504):
        assert _is_retryable_gemini_error(E(status)) is True


def test_is_retryable_gemini_error_does_not_retry_4xx(monkeypatch):
    from llm_providers import _is_retryable_gemini_error

    class E(Exception):
        def __init__(self, status):
            super().__init__()
            self.status_code = status

    for status in (400, 401, 403, 404):
        assert _is_retryable_gemini_error(E(status)) is False


def test_is_retryable_gemini_error_handles_string_status(monkeypatch):
    """SDK variation: status may be exposed as a string in some versions."""
    from llm_providers import _is_retryable_gemini_error

    class E(Exception):
        def __init__(self, status):
            super().__init__()
            self.status_code = status

    assert _is_retryable_gemini_error(E("429")) is True
    assert _is_retryable_gemini_error(E("400")) is False


def test_is_retryable_gemini_error_handles_no_status_attr():
    from llm_providers import _is_retryable_gemini_error
    assert _is_retryable_gemini_error(Exception("no status here")) is False


# ---- Gemini retry behavior (using fake exception class) -------------------

def test_gemini_retries_on_429_then_succeeds(monkeypatch):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash", max_retries=2)

    class FakeRetryable(Exception):
        def __init__(self, status):
            super().__init__()
            self.status_code = status

    p._retryable_excs = (FakeRetryable,)
    p._sleep = lambda _: None

    state = {"calls": 0}
    def flaky(_prompt):
        state["calls"] += 1
        if state["calls"] < 2:
            raise FakeRetryable(429)
        return "ok line"
    monkeypatch.setattr(p, "_call_api", flaky)
    result = p.generate_inputs("prompt", n=1)
    assert state["calls"] == 2
    assert result == ["ok line"]


def test_gemini_does_not_retry_400(monkeypatch):
    from llm_providers import GeminiProvider
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    p = GeminiProvider(name="g", model="gemini-2.5-flash", max_retries=5)

    class FakeRetryable(Exception):
        def __init__(self, status):
            super().__init__()
            self.status_code = status

    p._retryable_excs = (FakeRetryable,)
    p._sleep = lambda _: None

    state = {"calls": 0}
    def fails_with_400(_prompt):
        state["calls"] += 1
        raise FakeRetryable(400)
    monkeypatch.setattr(p, "_call_api", fails_with_400)

    with pytest.raises(FakeRetryable):
        p.generate_inputs("prompt", n=1)
    assert state["calls"] == 1   # NOT retried — predicate filters 400 out
```

- [ ] **Step 2: Run, expect ImportError on `GeminiProvider` / `_is_retryable_gemini_error`**

```bash
cd "C:/work/llm training" && pytest tests/test_real_providers.py -v -k "gemini or is_retryable"
```
Expected: failures with ImportError.

- [ ] **Step 3: Add `_is_retryable_gemini_error` + `GeminiProvider` to `scripts/llm_providers.py`**

Append to `scripts/llm_providers.py` (after `DeepSeekProvider`):

```python
def _is_retryable_gemini_error(exc: BaseException) -> bool:
    """Defensive against SDK variation: some versions expose status as int,
    others as string. Coerce to int; on failure, do not retry."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        status_int = int(status)
    except (TypeError, ValueError):
        return False
    return status_int in {429, 500, 502, 503, 504}


class GeminiProvider:
    """Real Gemini provider via the google-genai SDK.

    Lazy SDK import: `google.genai` is imported inside __init__, NOT at
    module load, so `import llm_providers` stays SDK-free.
    """

    provider_type = "gemini"

    def __init__(
        self,
        name: str,
        *,
        model: str,
        api_key: str | None = None,
        temperature: float | None = 1.0,
        max_tokens: int | None = 8000,
        max_retries: int = 3,
        structured_output: bool = False,   # ignored in Slice 2; Slice 3 wires it
    ) -> None:
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.structured_output = structured_output
        self._sleep = time.sleep

        from google import genai                                          # lazy
        from google.genai import errors as genai_errors

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ProviderError(
                f"GeminiProvider {name!r}: GOOGLE_API_KEY (or GEMINI_API_KEY) not set"
            )
        self._client = genai.Client(api_key=key)
        # Tuple is broad on purpose; the is_retryable predicate filters by status.
        self._retryable_excs = (genai_errors.APIError, genai_errors.ClientError)

    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        if n < 0:
            raise ProviderError(f"generate_inputs: n must be >= 0, got {n}")
        if n == 0:
            return []
        text = self._call_with_retry(prompt)
        return _parse_input_lines(text, expected=n)

    def generate_label(self, input_text: str) -> str:
        raise NotImplementedError(
            f"gemini provider {self.name!r}: output labeling lands in Slice 3"
        )

    def _call_with_retry(self, prompt: str) -> str:
        from _retry import retry_with_backoff
        return retry_with_backoff(
            lambda: self._call_api(prompt),
            retryable=self._retryable_excs,
            is_retryable=_is_retryable_gemini_error,
            max_retries=self.max_retries,
            logger_name=f"gemini.{self.name}",
            sleep=self._sleep,
        )

    def _call_api(self, prompt: str) -> str:
        from google.genai import types
        cfg_kwargs = {}
        if self.temperature is not None:
            cfg_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            cfg_kwargs["max_output_tokens"] = self.max_tokens
        config = types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return resp.text or ""
```

- [ ] **Step 4: Update `create_provider` to use the real GeminiProvider**

In `scripts/llm_providers.py`, find the `create_provider` function and update the gemini branch (it currently returns `UnimplementedProvider` for gemini after Task 4):

```python
    if cfg.provider_type == "gemini":
        return GeminiProvider(
            name=cfg.name,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_retries=cfg.max_retries,
            structured_output=cfg.structured_output,
        )
```

Leave the `local_teacher` branch returning `UnimplementedProvider`.

- [ ] **Step 5: Run Gemini tests, expect 12 passed**

```bash
cd "C:/work/llm training" && pytest tests/test_real_providers.py -v -k "gemini or is_retryable"
```
Expected: 12 passed.

- [ ] **Step 6: Full suite green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: prior count + 12 new (28 total in test_real_providers.py).

- [ ] **Step 7: Do NOT commit.**

---

### Task 6: `DISTILL_DIR_OVERRIDE` env var support

**Files:**
- Modify: `scripts/05_generate_distillation_data.py` — constants block

Spec reference: §6.3.

- [ ] **Step 1: Locate the constants block in `scripts/05_generate_distillation_data.py`**

Around line 56–66, the current block:
```python
REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_EVAL = REPO_ROOT / "data" / "clean" / "eval.jsonl"
DISTILL_DIR = REPO_ROOT / "data" / "distill"
INPUTS_FILE = DISTILL_DIR / "inputs_raw.jsonl"
TRAIN_FILE = DISTILL_DIR / "train.jsonl"
DISTILL_EVAL_FILE = DISTILL_DIR / "eval.jsonl"
FAILED_FILE = DISTILL_DIR / "failed.jsonl"
LOGS_DIR = REPO_ROOT / "logs"
TEACHER_ADAPTER_DIR = REPO_ROOT / "models" / "teacher" / "adapters"
TEACHER_GGUF_DIR = REPO_ROOT / "models" / "teacher" / "gguf"
```

Confirm `import os` is at the top of the file (it is — Slice 1 added it for `os.environ`).

- [ ] **Step 2: Replace `DISTILL_DIR` assignment to honor the override**

Update the constants block (only the `DISTILL_DIR` line changes; the other lines stay because they're derived AFTER):

```python
REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_EVAL = REPO_ROOT / "data" / "clean" / "eval.jsonl"

# DISTILL_DIR can be overridden by tests via env var. ALL derived paths
# (INPUTS_FILE, TRAIN_FILE, ...) must be computed AFTER this assignment.
DISTILL_DIR = Path(os.environ.get("DISTILL_DIR_OVERRIDE", REPO_ROOT / "data" / "distill"))
INPUTS_FILE = DISTILL_DIR / "inputs_raw.jsonl"
TRAIN_FILE = DISTILL_DIR / "train.jsonl"
DISTILL_EVAL_FILE = DISTILL_DIR / "eval.jsonl"
FAILED_FILE = DISTILL_DIR / "failed.jsonl"
LOGS_DIR = REPO_ROOT / "logs"
TEACHER_ADAPTER_DIR = REPO_ROOT / "models" / "teacher" / "adapters"
TEACHER_GGUF_DIR = REPO_ROOT / "models" / "teacher" / "gguf"
```

- [ ] **Step 3: Sanity check — script imports and honors the env var**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts DISTILL_DIR_OVERRIDE=/tmp/distill_test python -c "
import importlib
m = importlib.import_module('05_generate_distillation_data')
print('DISTILL_DIR:', m.DISTILL_DIR)
print('INPUTS_FILE:', m.INPUTS_FILE)
"
```
Expected:
```
DISTILL_DIR: /tmp/distill_test
INPUTS_FILE: /tmp/distill_test/inputs_raw.jsonl
```

- [ ] **Step 4: Verify default behavior unchanged**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
import importlib
m = importlib.import_module('05_generate_distillation_data')
print(str(m.DISTILL_DIR).endswith('data/distill') or str(m.DISTILL_DIR).endswith('data\\\\distill'))
"
```
Expected: prints `True`.

- [ ] **Step 5: Full suite green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: prior count, no regressions.

- [ ] **Step 6: Do NOT commit.**

---

### Task 7: `phase_inputs_multi_provider` + worker helpers

**Files:**
- Modify: `scripts/05_generate_distillation_data.py` — add helpers + new phase function

Spec reference: §5.

This task is the largest single block of new code. It's pure orchestration logic that's exercised end-to-end by Task 9's subprocess tests; no unit tests in this task.

- [ ] **Step 1: Add new imports to `scripts/05_generate_distillation_data.py`**

Near the top of the file (after the existing imports), confirm or add:

```python
import concurrent.futures
import math
import queue
import threading
```

These are stdlib; no requirements change. The existing file already has `import json`, `import logging`, `import os`, `import re`, `import sys`, `import time`, `from datetime import datetime`, `from pathlib import Path`, `from tqdm import tqdm`.

Also update the `from _lib import (...)` block (Task 2 added `clean_input_line`; this task adds `normalize_input`):

```python
from _lib import (  # noqa: E402
    BATCH_FOCUSES,
    build_messages,
    call_deepseek,
    clean_input_line,
    extract_json,
    load_jsonl,
    normalize_input,        # NEW in Slice 2
    parse_amounts,
    serialize_validation_result,
    validate_example,
)
```

- [ ] **Step 2: Add module-level constants for the multi-provider loop**

After the existing constants block but BEFORE `INPUT_GEN_PROMPT`, add:

```python
# Slice 2: multi-provider input generation tunables.
MAX_CONSECUTIVE_EMPTY_BATCHES = 3
MAX_CONSECUTIVE_PROVIDER_ERRORS = 3
MAX_CONSECUTIVE_DUPLICATES_BEFORE_GIVEUP = 200
```

- [ ] **Step 3: Add the worker helper + focus rotator near the bottom of the file (BEFORE `parse_args`)**

```python
# ---------------------------------------------------------------------------
# Multi-provider input generation (Slice 2). Concurrency model:
#   - One shared ThreadPoolExecutor, sized to min(sum(threads), global_max).
#   - For each provider with quota > 0, submit min(threads, max(1, ceil(quota/batch_size)))
#     worker tasks.
#   - Workers produce (input_text, provider_name, model_id) tuples to a
#     BOUNDED queue (prevents racing far ahead of the writer and wasting
#     API calls when writer has reached target). queue.put uses timeout +
#     stop_event re-check so executor shutdown cannot hang.
#   - Main thread is the SOLE writer to inputs_raw.jsonl.
#   - Completion: main sets stop_event when accepted_total >= pending.
#
# Quota is a soft scheduling hint, not a strict per-provider cap. See
# spec §5.2 for the rationale.
#
# Duplicate-stall guard is GLOBAL: a flood of duplicates from any one
# provider can stop the whole run. Slice 4 metrics may add per-provider
# health tracking; for Slice 2, global guard is acceptable.
# ---------------------------------------------------------------------------

_focus_counters: dict[str, int] = {}
_focus_lock = threading.Lock()


def _next_focus(provider_name: str) -> str:
    """Deterministic round-robin over BATCH_FOCUSES, keyed per provider."""
    with _focus_lock:
        i = _focus_counters.get(provider_name, 0)
        _focus_counters[provider_name] = i + 1
    return BATCH_FOCUSES[i % len(BATCH_FOCUSES)]


def _build_input_prompt(n: int, batch_focus: str) -> str:
    """Format the existing INPUT_GEN_PROMPT with the requested count + focus."""
    return INPUT_GEN_PROMPT.format(n=n, focus=batch_focus)


def _provider_worker(
    *,
    provider,                             # LLMProvider
    pcfg,                                 # ProviderConfig
    batch_size: int,
    results_queue: "queue.Queue",
    stop_event: threading.Event,
) -> None:
    """Run one provider's batches until stop_event is set or local caps hit.

    Maintains separate failure streaks:
      - empty_response_streak: parser returned 0 cleaned lines
      - provider_error_streak: exception bubbled through retries
    """
    empty_streak = 0
    error_streak = 0
    while not stop_event.is_set():
        if empty_streak >= MAX_CONSECUTIVE_EMPTY_BATCHES:
            logging.warning(
                "Provider %s: %d empty batches in a row, worker exiting early.",
                pcfg.name, empty_streak,
            )
            return
        if error_streak >= MAX_CONSECUTIVE_PROVIDER_ERRORS:
            logging.warning(
                "Provider %s: %d API errors in a row, worker exiting early.",
                pcfg.name, error_streak,
            )
            return
        prompt = _build_input_prompt(batch_size, batch_focus=_next_focus(pcfg.name))
        try:
            lines = provider.generate_inputs(prompt, batch_size)
        except Exception as e:    # noqa: BLE001 — surface any provider failure as an error streak
            logging.error("Provider %s batch failed: %s", pcfg.name, e)
            error_streak += 1
            continue
        if not lines:
            empty_streak += 1
            continue
        empty_streak = 0
        error_streak = 0
        model_id = getattr(provider, "model", None)
        for line in lines:
            if stop_event.is_set():
                return
            # Bounded queue: if writer is behind, wait briefly and retry,
            # but always re-check stop_event so executor shutdown can't hang.
            while not stop_event.is_set():
                try:
                    results_queue.put((line, provider.name, model_id), timeout=0.5)
                    break
                except queue.Full:
                    continue


def phase_inputs_multi_provider(args: argparse.Namespace) -> None:
    """Slice 2 multi-provider input-generation loop.

    Reads cfg.input_generation.{target_inputs, batch_size, providers}.
    On resume, dedupes against existing inputs_raw.jsonl and only
    generates the remaining `pending` inputs.
    """
    from generation_config import load_generation_config
    from generation_orchestrator import allocate_quota
    from llm_providers import create_provider

    cfg = load_generation_config(args.provider_config)
    ig = cfg.input_generation
    if not ig.enabled:
        logging.info("Phase inputs (multi-provider): input_generation.enabled=False — nothing to do.")
        return

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    existing_inputs = read_inputs_jsonl(INPUTS_FILE)
    existing_normalized = {normalize_input(s) for s in existing_inputs}
    existing_unique = len(existing_normalized)
    pending = max(0, ig.target_inputs - existing_unique)
    logging.info(
        "Phase inputs (multi-provider): existing_unique=%d target=%d pending=%d",
        existing_unique, ig.target_inputs, pending,
    )
    if pending == 0:
        logging.info("Already at target. Multi-provider phase complete.")
        return

    if args.n_inputs and args.n_inputs != ig.target_inputs:
        logging.warning(
            "--n-inputs=%d ignored in multi-provider mode; "
            "cfg.input_generation.target_inputs=%d is the source of truth.",
            args.n_inputs, ig.target_inputs,
        )

    providers = {p.name: create_provider(p) for p in ig.providers}
    quota = allocate_quota(pending, ig.providers)
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    stop_event = threading.Event()
    futures: list[concurrent.futures.Future] = []

    # Worker count per provider: min(threads, max(1, ceil(quota/batch_size))).
    # Providers with zero quota get zero workers (no calls made).
    total_workers = 0
    for pcfg in ig.providers:
        q = quota[pcfg.name]
        if q <= 0:
            continue
        n_workers = min(pcfg.threads, max(1, math.ceil(q / ig.batch_size)))
        total_workers += n_workers
    pool_size = min(total_workers, cfg.rate_limits.global_max_workers) if total_workers else 1

    # Bounded queue: prevents workers racing far ahead of the writer and
    # making wasted API calls when the writer has already reached target.
    # Size = batch_size * total_workers * 2 gives one full batch of slack
    # per worker before put() blocks.
    queue_max = max(1, ig.batch_size * max(1, total_workers) * 2)
    results_queue: "queue.Queue[tuple[str, str, object]]" = queue.Queue(maxsize=queue_max)

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as pool, \
         INPUTS_FILE.open("a", encoding="utf-8") as fout, \
         tqdm(total=pending, initial=0, desc="multi-provider inputs") as bar:

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
                ))

        accepted_total = 0
        consecutive_dupes = 0
        while accepted_total < pending:
            # Exit if all workers finished AND no more queued results.
            if all(f.done() for f in futures) and results_queue.empty():
                logging.warning(
                    "Multi-provider input loop: providers exhausted before "
                    "reaching target (accepted=%d, pending=%d).",
                    accepted_total, pending,
                )
                break
            try:
                input_text, provider_name, model_id = results_queue.get(timeout=1.0)
            except queue.Empty:
                # Timeout — workers may still be in flight. Do NOT increment
                # the dupe counter; just loop.
                continue
            normalized = normalize_input(input_text)
            if normalized in existing_normalized:
                consecutive_dupes += 1
                if consecutive_dupes >= MAX_CONSECUTIVE_DUPLICATES_BEFORE_GIVEUP:
                    logging.warning(
                        "Multi-provider input loop saw %d consecutive duplicates "
                        "without accepting a new input — providers exhausted unique "
                        "supply. Stopping with partial progress (accepted=%d / pending=%d).",
                        consecutive_dupes, accepted_total, pending,
                    )
                    stop_event.set()
                    break
                continue
            consecutive_dupes = 0
            existing_normalized.add(normalized)
            row = {
                "input": input_text,
                "_source": "synthetic_input",
                "_provider": provider_name,
                "_model": model_id,    # may be None for FakeProvider
                "_batch_id": batch_id,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            accepted_total += 1
            if accepted_total % cfg.rate_limits.write_flush_every == 0:
                fout.flush()
            bar.update(1)

        stop_event.set()
        fout.flush()

    logging.info(
        "Phase inputs (multi-provider) done. accepted=%d existing_unique=%d target=%d",
        accepted_total, existing_unique + accepted_total, ig.target_inputs,
    )
```

- [ ] **Step 4: Sanity check — script still imports**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
import importlib
m = importlib.import_module('05_generate_distillation_data')
print(m.phase_inputs_multi_provider.__name__)
print(m._provider_worker.__name__)
print(m._next_focus.__name__)
"
```
Expected: prints three function names.

- [ ] **Step 5: Full suite green (no new tests yet; Task 9 adds the end-to-end ones)**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: prior count, no regressions.

- [ ] **Step 6: Do NOT commit.**

---

### Task 8: Update flag contract + `main()` dispatch

**Files:**
- Modify: `scripts/05_generate_distillation_data.py` — rename and extend the enforcer; update `main()`
- Modify: `tests/test_stage5_flag_contract.py` — update rows, add `env` support to `run_cli`

Spec reference: §7.

- [ ] **Step 1: Modify `run_cli` in `tests/test_stage5_flag_contract.py` to accept `env`**

Open `tests/test_stage5_flag_contract.py`. Replace the existing `run_cli` helper with:

```python
def run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = {**os.environ}
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env=full_env,
    )
```

Add `import os` at the top of the file if not present (it may not be — check existing imports).

- [ ] **Step 2: Update the existing rows that previously asserted multi-provider always errored**

Find and DELETE these existing tests in `tests/test_stage5_flag_contract.py` (Slice 1 wrote them):

- `test_provider_config_with_multi_provider_exits_2` (was asserting "Slice 2 lands later")
- `test_provider_config_with_multi_provider_and_dry_run_exits_2` (asserting same)

Then APPEND these new rows:

```python
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
```

- [ ] **Step 3: Run the contract tests, expect failures**

```bash
cd "C:/work/llm training" && pytest tests/test_stage5_flag_contract.py -v
```
Expected: most existing tests still pass; the 5 new tests fail because the enforcer hasn't been updated yet.

- [ ] **Step 4: Rename + extend `_enforce_slice1_flag_contract` in `scripts/05_generate_distillation_data.py`**

Find the existing `_enforce_slice1_flag_contract` function. Replace it entirely with:

```python
def _enforce_flag_contract(
    args: argparse.Namespace, parser: argparse.ArgumentParser,
) -> None:
    """Multi-provider CLI contract — see spec §7.

    Every error path uses parser.error which exits with code 2.
    Dry-run wins: phase restrictions are skipped when --dry-run-quota is present.
    """
    # Standalone-flag errors.
    if args.multi_provider and not args.provider_config:
        parser.error("--multi-provider requires --provider-config.")
    if args.dry_run_quota and not args.provider_config:
        parser.error("--dry-run-quota requires --provider-config.")
    # --provider-config requires SOMETHING (either dry-run or multi-provider).
    if args.provider_config and not (args.dry_run_quota or args.multi_provider):
        parser.error(
            "--provider-config requires --dry-run-quota or --multi-provider."
        )
    # Multi-provider phase restrictions for Slice 2.
    # Dry-run wins: phase checks skipped when --dry-run-quota is present.
    if args.multi_provider and args.provider_config and not args.dry_run_quota:
        if args.phase == "label":
            parser.error(
                "--multi-provider --phase label is reserved for Slice 3 "
                "(validator-gated output labeling)."
            )
        if args.phase == "eval":
            parser.error(
                "--multi-provider --phase eval is not supported; "
                "use legacy mode (no --multi-provider) for eval."
            )
        if args.phase == "all":
            parser.error(
                "--multi-provider does not support --phase all in Slice 2; "
                "specify --phase inputs, or use legacy mode (no --multi-provider)."
            )
        # --phase inputs is the only allowed combo; falls through to main().
```

- [ ] **Step 5: Update `main()` in the same file**

Find the current `main()` function. It currently looks like:

```python
def main() -> int:
    parser, args = parse_args()
    _enforce_slice1_flag_contract(args, parser)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"05_generate_distillation_data_{ts}.log")

    if args.dry_run_quota:
        assert args.provider_config is not None
        return _run_dry_run(args.provider_config)

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Stage 5: phase=%s args=%s", args.phase, vars(args))

    if args.phase in ("all", "inputs"):
        phase_inputs(args)
    if args.phase in ("all", "label"):
        phase_label(args)
    if args.phase in ("all", "eval"):
        phase_eval(args)

    logging.info("Stage 5 done.")
    return 0
```

Replace it with:

```python
def main() -> int:
    parser, args = parse_args()
    _enforce_flag_contract(args, parser)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"05_generate_distillation_data_{ts}.log")

    if args.dry_run_quota:
        assert args.provider_config is not None
        return _run_dry_run(args.provider_config)

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Stage 5: phase=%s args=%s", args.phase, vars(args))

    if args.multi_provider:
        # Slice 2: only --phase inputs reaches here (others rejected by contract).
        assert args.phase == "inputs"
        assert args.provider_config is not None
        phase_inputs_multi_provider(args)
        logging.info("Stage 5 done.")
        return 0

    # Legacy path — unchanged.
    if args.phase in ("all", "inputs"):
        phase_inputs(args)
    if args.phase in ("all", "label"):
        phase_label(args)
    if args.phase in ("all", "eval"):
        phase_eval(args)

    logging.info("Stage 5 done.")
    return 0
```

- [ ] **Step 6: Update flag help text**

In `parse_args`, find the existing flag help strings (Slice 1 wrote them). Update:

```python
    p.add_argument("--provider-config", type=Path, default=None,
                   help="Path to a multi-provider generation config JSON. "
                        "Use with --dry-run-quota to inspect the config, or with "
                        "--multi-provider --phase inputs (Slice 2) to run real "
                        "input generation.")
    p.add_argument("--multi-provider", action="store_true",
                   help="Run multi-provider execution. Slice 2 supports "
                        "--phase inputs only; label and all exit via parser.error.")
    # --dry-run-quota help string stays unchanged from Slice 1.
```

- [ ] **Step 7: Run all flag-contract tests, expect all pass**

```bash
cd "C:/work/llm training" && pytest tests/test_stage5_flag_contract.py -v
```
Expected: all tests pass (prior surviving rows + 5 new rows).

- [ ] **Step 8: Full suite green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: prior count + (5 new - 2 deleted = +3 net) from flag-contract changes.

- [ ] **Step 9: Do NOT commit.**

---

### Task 9: End-to-end multi-provider input tests

**Files:**
- Create: `tests/test_multi_provider_inputs.py`

Spec reference: §8.1.

- [ ] **Step 1: Write the end-to-end test file**

`tests/test_multi_provider_inputs.py`:
```python
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
    # Either the per-worker streak guard or the global duplicate-stall guard
    # should have logged a warning.
    stderr_lower = result.stderr.lower()
    assert (
        "empty batches" in stderr_lower
        or "duplicate" in stderr_lower
        or "providers exhausted" in stderr_lower
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
    # File should not have been created since nothing was written.
    out_file = tmp_path / "inputs_raw.jsonl"
    assert not out_file.exists() or out_file.read_text(encoding="utf-8") == ""
```

- [ ] **Step 2: Run the new tests, expect 7 passed**

```bash
cd "C:/work/llm training" && pytest tests/test_multi_provider_inputs.py -v
```
Expected: 7 passed. (These are subprocess tests, so they'll be slower — ~5-30s each. The exhaustion test in particular may take ~15s because workers hit their empty-batch streak.)

- [ ] **Step 3: Full suite green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: prior count + 7.

- [ ] **Step 4: Do NOT commit.**

---

### Task 10: Update configs, docs, requirements

**Files:**
- Modify: `configs/example_providers.json` — `enabled: false`, `target_inputs: 10`
- Create: `configs/smoke_real_providers.json`
- Modify: `requirements-train.txt` — add the new deps
- Modify: `docs/provider_config.md` — update "What loads successfully"

Spec reference: §9, §10.

- [ ] **Step 1: Update `configs/example_providers.json`**

Read the current file. Change `input_generation.enabled` to `false` and `input_generation.target_inputs` to `10`. Other fields unchanged. The full file becomes:

```json
{
  "version": 1,
  "input_generation": {
    "enabled": false,
    "target_inputs": 10,
    "batch_size": 100,
    "dedupe": true,
    "providers": [
      {"name": "deepseek_v4_pro", "type": "deepseek", "model": "deepseek-chat",
       "weight": 50, "threads": 4, "max_retries": 3,
       "temperature": 1.0, "max_tokens": 8000},
      {"name": "gemini_flash", "type": "gemini", "model": "gemini-2.5-flash",
       "weight": 50, "threads": 4, "max_retries": 3,
       "temperature": 1.0, "max_tokens": 8000}
    ]
  },
  "output_generation": {
    "enabled": true,
    "label_attempts_per_input": 2,
    "selection_policy": "first_valid_then_score",
    "providers": [
      {"name": "local_teacher", "type": "local_teacher",
       "weight": 50, "threads": 1},
      {"name": "deepseek_v4_pro", "type": "deepseek", "model": "deepseek-chat",
       "weight": 30, "threads": 4, "temperature": 0.0, "max_tokens": 512},
      {"name": "gemini_flash", "type": "gemini", "model": "gemini-2.5-flash",
       "weight": 20, "threads": 4, "temperature": 0.0, "max_tokens": 512,
       "structured_output": true}
    ],
    "provider_priority": ["local_teacher", "gemini_flash", "deepseek_v4_pro"]
  },
  "validation": {"schema": true, "semantic_validator": true,
                 "reject_invalid": true,
                 "retry_invalid_with_stricter_prompt": true,
                 "max_repair_attempts": 1},
  "rate_limits": {"global_max_workers": 12, "write_flush_every": 50}
}
```

(Note: model ids are pinned to `"deepseek-chat"` and `"gemini-2.5-flash"` to make the file syntactically real, but `enabled: false` on input_generation prevents any actual call.)

- [ ] **Step 2: Create `configs/smoke_real_providers.json`**

```json
{
  "version": 1,
  "input_generation": {
    "enabled": true,
    "target_inputs": 100,
    "batch_size": 25,
    "dedupe": true,
    "providers": [
      {"name": "deepseek_v4_pro", "type": "deepseek", "model": "deepseek-chat",
       "weight": 50, "threads": 4, "max_retries": 3,
       "temperature": 1.0, "max_tokens": 4000},
      {"name": "gemini_flash", "type": "gemini", "model": "gemini-2.5-flash",
       "weight": 50, "threads": 4, "max_retries": 3,
       "temperature": 1.0, "max_tokens": 4000}
    ]
  },
  "output_generation": {"enabled": false, "providers": []},
  "validation": {"schema": true, "semantic_validator": true,
                 "reject_invalid": true,
                 "retry_invalid_with_stricter_prompt": false,
                 "max_repair_attempts": 0},
  "rate_limits": {"global_max_workers": 8, "write_flush_every": 25}
}
```

- [ ] **Step 3: Update `requirements-train.txt`**

Open `requirements-train.txt`. Append (or place near the other model/SDK deps):

```text
openai>=1.50
google-genai>=<VERSION>
```

Replace `<VERSION>` with the version captured in Task 0 Step 6 (e.g., `google-genai>=0.4.0` if `pip show google-genai` returned `0.4.2`).

- [ ] **Step 4: Update `docs/provider_config.md`**

Open `docs/provider_config.md`. Find the section "## What loads successfully in Slice 1" (this title was set by the Slice 1 plan). Replace its body with:

```markdown
## What loads and runs

**Slice 1 (configuration + dry-run):**
- Any `type: fake` provider with proper fixture paths.
- Any `type: deepseek` / `gemini` / `local_teacher` provider — these load without SDK imports. `--dry-run-quota` works for all of them.

**Slice 2 (real input generation):**
- `type: deepseek` and `type: gemini` providers run real API calls when invoked via
  `--phase inputs --provider-config <path> --multi-provider`. Requires `DEEPSEEK_API_KEY`
  and/or `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) env vars.
- `type: local_teacher` remains `UnimplementedProvider` until Slice 3.
- `--phase label` and `--phase all` exit via `parser.error` in Slice 2 (Slice 3 wires labeling).
```

- [ ] **Step 5: Sanity check — configs parse**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
from generation_config import load_generation_config
cfg1 = load_generation_config('configs/example_providers.json')
cfg2 = load_generation_config('configs/smoke_real_providers.json')
print('example: input enabled?', cfg1.input_generation.enabled)
print('smoke: input enabled?', cfg2.input_generation.enabled)
"
```
Expected:
```
example: input enabled? False
smoke: input enabled? True
```

- [ ] **Step 6: Full suite green**

```bash
cd "C:/work/llm training" && pytest tests/ -q 2>&1 | tail -3
```
Expected: prior count, no regressions. Note: the Slice 1 test `test_example_config_loads_and_dry_runs_without_sdk_imports` still passes because dry-run does not call `create_provider` — the SDK-import-leak guard remains valid.

- [ ] **Step 7: Do NOT commit.**

---

### Task 11: README update

**Files:**
- Modify: `README.md`

Spec reference: §10 step 9.

- [ ] **Step 1: Locate the "Multi-provider scaffolding (Slice 1)" subsection**

Open `README.md`. Find the `### Multi-provider scaffolding (Slice 1)` heading inside the `## Stage 5 — Teacher generates distillation data` section. After that subsection, APPEND a new subsection:

```markdown
### Multi-provider input generation (Slice 2)

Slice 2 makes `--phase inputs --provider-config <path> --multi-provider` actually run with real DeepSeek + Gemini API calls. Set the API keys, point at a config, and run:

```bash
export DEEPSEEK_API_KEY=sk-...
export GOOGLE_API_KEY=...     # or GEMINI_API_KEY
python scripts/05_generate_distillation_data.py \
    --phase inputs \
    --provider-config configs/smoke_real_providers.json \
    --multi-provider
```

The generation loop is threaded per-provider (each provider's `threads` field in the config), with global dedupe against the existing `data/distill/inputs_raw.jsonl` and resume safety (re-running picks up at the unique-input count and only generates the remaining `target_inputs`). New rows include provider metadata: `_provider`, `_model`, `_batch_id`.

Quota is a soft scheduling hint — workers stop when the global accepted-unique total reaches `target_inputs`, not when a per-provider quota fills. Provider distribution may drift from configured weights based on latency and duplicate rate.

`--phase label` and `--phase all` are reserved for Slice 3 (validator-gated output labeling).

For tests and CI, set `DISTILL_DIR_OVERRIDE=<tmp_path>` to redirect writes away from `data/distill/`. This is a dev/test-only knob and is not surfaced in `--help`.
```

- [ ] **Step 2: Verify prior-slice README sections are intact**

```bash
cd "C:/work/llm training" && grep -E "Validator gate \(Phase 2\)|validator-derived aggregates|Dev dependencies|Multi-provider scaffolding \(Slice 1\)" README.md
```
Expected: four matches. If any is missing, STOP — that section was lost.

- [ ] **Step 3: Do NOT commit yet.**

---

### Task 12: Final verification + Commit

**Files:** none modified.

Spec reference: §11, §12.

- [ ] **Step 1: Run the FULL test suite**

```bash
cd "C:/work/llm training" && pytest tests/ -v
```
Expected: prior count (177 baseline) + new tests from Tasks 1–9. Net adds: 6 (retry) + 22 (clean_input_line) + ~29 (real_providers DeepSeek+Gemini+predicate+_parse_input_lines) + 7 (multi-provider e2e) + 3 net flag-contract = ~67 new tests. All pass.

- [ ] **Step 2: Coverage check**

```bash
cd "C:/work/llm training" && pytest tests/ --cov=_retry --cov=llm_providers --cov-report=term
```
Expected:
- `_retry.py`: ≥ 95%
- `llm_providers.py`: ≥ 80% (UnimplementedProvider branches + real-provider _call_api SDK paths counted)

If under, identify missing lines and add 1–2 targeted tests.

- [ ] **Step 3: Run the human smokes (no API keys needed)**

```bash
cd "C:/work/llm training"

# Smoke 1: dry-run still works
python scripts/05_generate_distillation_data.py \
    --provider-config configs/test_providers.json \
    --dry-run-quota

# Smoke 2: multi-provider input generation against FakeProvider (no API spend)
DISTILL_DIR_OVERRIDE=/tmp/slice2_smoke \
python scripts/05_generate_distillation_data.py \
    --phase inputs \
    --provider-config configs/test_providers.json \
    --multi-provider

# Verify output
ls /tmp/slice2_smoke/
head -2 /tmp/slice2_smoke/inputs_raw.jsonl
```

Expected for Smoke 2: 10 rows in `/tmp/slice2_smoke/inputs_raw.jsonl`, each containing `"_source": "synthetic_input"`, `"_provider": "fake_a"` or `"fake_b"`, `"_model": null`, and a `_batch_id`.

- [ ] **Step 4: Run the phase-restriction smokes (confirm error paths)**

```bash
cd "C:/work/llm training"
python scripts/05_generate_distillation_data.py \
    --provider-config configs/test_providers.json \
    --multi-provider --phase label
# exit 2, stderr mentions "Slice 3"

python scripts/05_generate_distillation_data.py \
    --provider-config configs/test_providers.json \
    --multi-provider --phase all
# exit 2, stderr mentions "all"
```

- [ ] **Step 5: Confirm legacy `--help` still works and exposes the new flags + updated help text**

```bash
cd "C:/work/llm training" && python scripts/05_generate_distillation_data.py --help | grep -E "multi-provider|provider-config|dry-run-quota"
```
Expected: three lines (one per flag) with updated descriptions mentioning "Slice 2".

- [ ] **Step 6: Stage files**

CRITICAL: Do NOT stage any `data/distill/*` files, `reports/`, or `/tmp/slice2_smoke/` artifacts. Stage only source + tests + docs + configs.

```bash
cd "C:/work/llm training"
git add \
    scripts/_retry.py \
    scripts/_lib.py \
    scripts/llm_providers.py \
    scripts/05_generate_distillation_data.py \
    configs/example_providers.json \
    configs/smoke_real_providers.json \
    docs/provider_config.md \
    requirements-train.txt \
    tests/test_retry.py \
    tests/test_clean_input_line.py \
    tests/test_real_providers.py \
    tests/test_multi_provider_inputs.py \
    tests/test_stage5_flag_contract.py \
    README.md
git status
```

Verify the `git status` output shows only the listed files in the staged section. The pre-existing `data/distill/*` modifications and `.coverage` / `reports/` should remain unstaged.

- [ ] **Step 7: Commit**

```bash
cd "C:/work/llm training"
git commit -m "$(cat <<'EOF'
Slice 2: real multi-provider input generation

Makes --phase inputs --provider-config <path> --multi-provider actually
run with real DeepSeek + Gemini API calls. Threaded per-provider
execution with retry/backoff, global dedupe against existing
inputs_raw.jsonl, resume safety, and provider metadata in new rows
(_source, _provider, _model, _batch_id).

Adds:
- scripts/_retry.py — exponential backoff with jitter, injectable sleep,
  is_retryable predicate (used by Gemini for 429/5xx filter)
- DeepSeekProvider via openai SDK (OpenAI-compatible endpoint)
- GeminiProvider via google-genai SDK
- Both providers: lazy SDK imports keep `import llm_providers` SDK-free
- phase_inputs_multi_provider: ThreadPoolExecutor + queue.Queue writer
  model, soft-quota scheduling, duplicate-stall guard
- Moves clean_input_line + normalize_input to _lib.py with a BUG-FIX
  regex (old r"^[\s\-\*\d]+[.)\s]+" silently corrupted digit-led inputs;
  new r"^\s*(?:[-*]\s+|\d+[.)]\s+)" only strips real bullets/numbering)
- read_inputs_jsonl malformed-row policy (warn + skip + summary log)
- DISTILL_DIR_OVERRIDE env var for hermetic tests
- Stage 5 flag matrix: --multi-provider --phase inputs runs;
  label/eval/all still error; dry-run still wins

The legacy Stage 5 path is byte-identical when --multi-provider is absent
except for the clean_input_line regex fix, which is a deliberate
correctness improvement (the old regex was silently dropping leading
digits in natural inputs).

Out of scope (later slices): output labeling, local_teacher provider,
Gemini structured output, per-provider metrics.

Spec: docs/superpowers/specs/2026-05-16-multi-provider-generation-slice2-design.md
EOF
)"
```

- [ ] **Step 8: Verify the commit**

```bash
cd "C:/work/llm training" && git log --oneline -3 && git status
```
Expected: new commit on top of Slice 2 spec/plan/amendment commits. Working tree shows only pre-existing data file modifications and untracked artifacts.

- [ ] **Step 9: Report**

Surface to the user:
- The new commit SHA.
- The full test pass count.
- Coverage percentages for `_retry.py` and `llm_providers.py`.
- Confirmation that the smokes ran cleanly.
- Suggested next step: optional real-API smoke (`configs/smoke_real_providers.json`) with the user's own API keys, or proceed to Slice 3 planning.
