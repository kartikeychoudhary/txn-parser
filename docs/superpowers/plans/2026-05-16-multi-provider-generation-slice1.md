# Multi-Provider Generation — Slice 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-provider generation scaffolding (provider abstraction, config schema, dry-run, validator probe) to Stage 5 without changing any existing behavior — every invocation that doesn't pass `--provider-config` runs the legacy flow identically.

**Architecture:** Five new pure-stdlib Python modules under `scripts/` (`llm_providers.py`, `generation_config.py`, `generation_orchestrator.py`, `probe_validator.py`, plus modifications to `05_generate_distillation_data.py`). All real-provider implementations are deferred — Slice 1 ships `FakeProvider` (fixture-backed) and `UnimplementedProvider` (factory placeholder for deepseek/gemini/local_teacher). No SDK imports, no API calls, no GPU. Spec at `docs/superpowers/specs/2026-05-16-multi-provider-generation-slice1-design.md` is the source of truth.

**Tech Stack:** Python 3.11+, stdlib only (`dataclasses`, `pathlib`, `threading`, `json`, `re`, `math`, `itertools`, `argparse`). Tests use `pytest` + `pytest-cov` (already in `requirements-dev.txt` from the previous slice). Existing `tests/conftest.py` puts `scripts/` on `sys.path` so all new modules use flat-import style (`from llm_providers import ...`, `from _lib import ...`).

**File map:**

| File | Purpose |
|---|---|
| `tests/fixtures/fake_inputs.jsonl` (new) | 20 canned input strings for FakeProvider |
| `tests/fixtures/fake_labels.jsonl` (new) | 13 rows (10 valid + 3 raw_output failure simulations) |
| `tests/fixtures/probe_examples.jsonl` (new) | 11 hand-picked rows hitting every validator error code |
| `scripts/llm_providers.py` (new) | `LLMProvider` Protocol, `FakeProvider`, `UnimplementedProvider`, `create_provider`, `ProviderError` |
| `scripts/generation_config.py` (new) | Dataclasses + `load_generation_config` + `ConfigError` |
| `scripts/generation_orchestrator.py` (new) | `allocate_quota`, `RoundRobinScheduler`, `render_dry_run` |
| `scripts/probe_validator.py` (new) | Standalone CLI: JSONL pairs → validator report |
| `configs/test_providers.json` (new) | Fake-only minimal config for smokes |
| `configs/example_providers.json` (new) | Realistic config with unimplemented provider types |
| `scripts/05_generate_distillation_data.py` (modify) | New flags + `_enforce_slice1_flag_contract` + `_run_dry_run` |
| `docs/provider_config.md` (new) | Human-readable companion explaining the JSON config |
| `tests/test_fake_provider.py` (new) | FakeProvider behavior |
| `tests/test_generation_config.py` (new) | Config loader + validation |
| `tests/test_provider_scheduler.py` (new) | allocate_quota + RoundRobinScheduler + render_dry_run |
| `tests/test_probe_validator.py` (new) | probe CLI end-to-end |
| `tests/test_stage5_flag_contract.py` (new) | Subprocess-based CLI flag matrix |
| `README.md` (modify) | New "Multi-provider scaffolding (Slice 1)" subsection |

**Commit policy:** Single commit at the very end (Task 15). Every preceding task says **do NOT commit** explicitly. Rationale: spec §10 mandates a single additive commit on `feat/multi-provider-generation`.

---

### Task 1: Fixture files

**Files:**
- Create: `tests/fixtures/fake_inputs.jsonl`
- Create: `tests/fixtures/fake_labels.jsonl`
- Create: `tests/fixtures/probe_examples.jsonl`

Spec reference: §8.2.

These are static JSONL files used by tests in Tasks 2-14. Create them first so later tasks can reference them.

- [ ] **Step 1: Create `tests/fixtures/fake_inputs.jsonl`** (20 rows)

```jsonl
{"input": "500 rs on beer 50 rs on candy"}
{"input": "do sau rupay ka chai"}
{"input": "got 5000 salary today"}
{"input": "1.5k for shoes from myntra"}
{"input": "300 lunch 50 chai 200 uber back home"}
{"input": "500 on beer wait no 600 on beer"}
{"input": "umm paid 300 for that thing yesterday"}
{"input": "paanch hazaar rent and 200 chai"}
{"input": "fifty thousand for laptop"}
{"input": "rs 250 for petrol"}
{"input": "$50 coffee at starbucks"}
{"input": "char hazaar gym membership"}
{"input": "200/- on auto"}
{"input": "1 lakh for car downpayment"}
{"input": "received 200 cashback from paytm"}
{"input": "12500 emi for laptop"}
{"input": "paid 99 rs for netflix"}
{"input": "2 sau on samosa"}
{"input": "twenty five hundred on dinner"}
{"input": "doctor visit 800 rupees"}
```

- [ ] **Step 2: Create `tests/fixtures/fake_labels.jsonl`** (13 rows: 10 valid + 3 raw_output failure)

```jsonl
{"input": "500 rs on beer 50 rs on candy", "output": {"transactions": [{"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"}, {"amount": 50, "currency": "INR", "item": "candy", "category": "Food", "type": "expense"}]}}
{"input": "do sau rupay ka chai", "output": {"transactions": [{"amount": 200, "currency": "INR", "item": "chai", "category": "Drinks", "type": "expense"}]}}
{"input": "got 5000 salary today", "output": {"transactions": [{"amount": 5000, "currency": "INR", "item": "salary", "category": "Income", "type": "income"}]}}
{"input": "1.5k for shoes from myntra", "output": {"transactions": [{"amount": 1500, "currency": "INR", "item": "shoes", "category": "Shopping", "type": "expense"}]}}
{"input": "300 lunch 50 chai 200 uber back home", "output": {"transactions": [{"amount": 300, "currency": "INR", "item": "lunch", "category": "Food", "type": "expense"}, {"amount": 50, "currency": "INR", "item": "chai", "category": "Drinks", "type": "expense"}, {"amount": 200, "currency": "INR", "item": "uber ride", "category": "Transport", "type": "expense"}]}}
{"input": "500 on beer wait no 600 on beer", "output": {"transactions": [{"amount": 600, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"}]}}
{"input": "umm paid 300 for that thing yesterday", "output": {"transactions": [{"amount": 300, "currency": "INR", "item": "unspecified", "category": "Other", "type": "expense"}]}}
{"input": "paanch hazaar rent and 200 chai", "output": {"transactions": [{"amount": 5000, "currency": "INR", "item": "rent", "category": "Bills", "type": "expense"}, {"amount": 200, "currency": "INR", "item": "chai", "category": "Drinks", "type": "expense"}]}}
{"input": "fifty thousand for laptop", "output": {"transactions": [{"amount": 50000, "currency": "INR", "item": "laptop", "category": "Shopping", "type": "expense"}]}}
{"input": "$50 coffee at starbucks", "output": {"transactions": [{"amount": 50, "currency": "USD", "item": "coffee", "category": "Drinks", "type": "expense"}]}}
{"input": "bad json case", "raw_output": "{not json at all"}
{"input": "schema fail case", "raw_output": "{\"transactions\": [{\"amount\": 100}]}"}
{"input": "validator fail case", "raw_output": "{\"transactions\":[{\"amount\":999,\"currency\":\"INR\",\"item\":\"made up\",\"category\":\"Other\",\"type\":\"expense\"}]}"}
```

- [ ] **Step 3: Create `tests/fixtures/probe_examples.jsonl`** (11 rows hitting every validator code)

```jsonl
{"input": "500 beer", "output": {"transactions": [{"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"}]}}
{"input": "200 chai", "output": {"transactions": [{"amount": 200, "currency": "INR", "item": "chai", "category": "Drinks", "type": "expense"}]}}
{"input": "500 beer and 50 candy", "output": {"transactions": [{"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"}, {"amount": 50, "currency": "INR", "item": "candy", "category": "Food", "type": "expense"}]}}
{"input": "200 chai and 100 samosa", "output": {"transactions": [{"amount": 200, "currency": "INR", "item": "chai", "category": "Drinks", "type": "expense"}]}}
{"input": "just some words here", "output": {"transactions": [{"amount": 100, "currency": "INR", "item": "thing", "category": "Other", "type": "expense"}]}}
{"input": "600 beer", "output": {"transactions": [{"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"}]}}
{"input": "500 beer wait no 600 beer", "output": {"transactions": [{"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"}]}}
{"input": "$50 coffee", "output": {"transactions": [{"amount": 50, "currency": "INR", "item": "coffee", "category": "Drinks", "type": "expense"}]}}
{"input": "500 beer", "output": {"transactions": [{"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"}, {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"}]}}
{"input": "500 beer", "output": {"transactions": [{"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"}, {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"}, {"amount": 500, "currency": "INR", "item": "beer", "category": "Drinks", "type": "expense"}]}}
{"input": "500 beer", "output": {"transactions": [{"amount": 500, "currency": "INR", "item": "beer"}]}}
```

Row coverage (in order):
1. OK — clean single transaction.
2. OK — clean single transaction.
3. OK — warning only (TXN_COUNT_BELOW_CANDIDATES: 2 candidates, 1 txn — actually wait, this has 2 candidates and 2 txns, so this is just clean. Adjust if needed in test wiring.)
4. Warning only — TXN_COUNT_BELOW_CANDIDATES (2 candidates, 1 txn).
5. NO_AMOUNT_IN_INPUT — parser finds no amounts but txns emitted.
6. AMOUNT_NOT_IN_INPUT — model invented "500" not in input "600 beer".
7. SUPERSEDED_AMOUNT_USED — picks the corrected-out 500 instead of 600.
8. CURRENCY_MISMATCH — `$50` is USD, txn says INR.
9. SUSPICIOUS_DUPLICATE — two beer txns, input has one candidate.
10. TXN_COUNT_EXCEEDS_CANDIDATES — three txns, one candidate.
11. SCHEMA_INVALID — missing required fields.

- [ ] **Step 4: Verify the fixtures are valid JSONL**

Run:
```bash
cd "C:/work/llm training" && python -c "
import json
for path in ['tests/fixtures/fake_inputs.jsonl', 'tests/fixtures/fake_labels.jsonl', 'tests/fixtures/probe_examples.jsonl']:
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                json.loads(line)
    print(f'{path}: OK')
"
```

Expected: three lines each printing `<path>: OK`.

- [ ] **Step 5: Do NOT commit. Single commit lands at Task 15.**

---

### Task 2: `FakeProvider` scaffolding + initial tests

**Files:**
- Create: `scripts/llm_providers.py`
- Create: `tests/test_fake_provider.py`

Spec reference: §3.1, §3.2.

- [ ] **Step 1: Write the initial failing tests**

`tests/test_fake_provider.py`:
```python
import json
import threading
from pathlib import Path

import pytest

from llm_providers import FakeProvider, ProviderError, LLMProvider


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "fake_inputs.jsonl"
LABELS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "fake_labels.jsonl"


def test_fakeprovider_satisfies_protocol():
    p = FakeProvider(name="x", inputs_path=INPUTS_FIXTURE)
    assert p.name == "x"
    assert p.provider_type == "fake"
    assert hasattr(p, "generate_inputs")
    assert hasattr(p, "generate_label")


def test_provider_error_is_runtime_error():
    assert issubclass(ProviderError, RuntimeError)


def test_generate_inputs_returns_n_items():
    p = FakeProvider(name="x", inputs_path=INPUTS_FIXTURE)
    result = p.generate_inputs(prompt="ignored", n=3)
    assert len(result) == 3
    assert all(isinstance(s, str) for s in result)


def test_generate_inputs_wraps_around_modulo_fixture_length():
    """For [a,b,c], n=5 -> [a,b,c,a,b]; next call continues from c."""
    tiny_fixture = REPO_ROOT / "tests" / "fixtures" / "_tiny_inputs.jsonl"
    tiny_fixture.write_text(
        '{"input": "a"}\n{"input": "b"}\n{"input": "c"}\n',
        encoding="utf-8",
    )
    try:
        p = FakeProvider(name="x", inputs_path=tiny_fixture)
        first = p.generate_inputs(prompt="", n=5)
        assert first == ["a", "b", "c", "a", "b"]
        second = p.generate_inputs(prompt="", n=2)
        assert second == ["c", "a"]
    finally:
        tiny_fixture.unlink(missing_ok=True)


def test_loader_raises_provider_error_on_bad_json():
    bad_fixture = REPO_ROOT / "tests" / "fixtures" / "_bad.jsonl"
    bad_fixture.write_text('{"input": "ok"}\n{not json\n', encoding="utf-8")
    try:
        with pytest.raises(ProviderError) as exc:
            FakeProvider(name="x", inputs_path=bad_fixture)
        assert "line 2" in str(exc.value).lower() or "line 2" in str(exc.value)
    finally:
        bad_fixture.unlink(missing_ok=True)


def test_loader_raises_provider_error_on_missing_input_field():
    bad_fixture = REPO_ROOT / "tests" / "fixtures" / "_missing_field.jsonl"
    bad_fixture.write_text('{"input": "ok"}\n{"wrong_key": "value"}\n', encoding="utf-8")
    try:
        with pytest.raises(ProviderError) as exc:
            FakeProvider(name="x", inputs_path=bad_fixture)
        msg = str(exc.value)
        assert "input" in msg and ("line 2" in msg.lower() or "2" in msg)
    finally:
        bad_fixture.unlink(missing_ok=True)


def test_generate_inputs_is_thread_safe():
    p = FakeProvider(name="x", inputs_path=INPUTS_FIXTURE)
    collected = []
    lock = threading.Lock()

    def worker():
        items = p.generate_inputs(prompt="", n=10)
        with lock:
            collected.extend(items)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 8 threads * 10 items each = 80 total. Cursor must advance correctly under contention.
    assert len(collected) == 80
```

- [ ] **Step 2: Run the tests, expect ModuleNotFoundError**

Run: `cd "C:/work/llm training" && pytest tests/test_fake_provider.py -v`

Expected: `ModuleNotFoundError: No module named 'llm_providers'`.

- [ ] **Step 3: Create `scripts/llm_providers.py` with the scaffold + FakeProvider**

```python
"""Multi-provider abstraction for synthetic data + label generation.

Slice 1 ships:
  - LLMProvider Protocol (the public surface)
  - FakeProvider (fixture-backed; deterministic; used in tests + dry-runs)
  - UnimplementedProvider (factory placeholder for deepseek/gemini/local_teacher)
  - create_provider factory

No SDK imports. No real API calls. Real provider implementations land
in Slice 2 (input generation) and Slice 3 (output labeling).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal provider surface. All methods must be thread-safe."""
    name: str
    provider_type: str

    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        """Return up to n raw input strings, one per example."""
        ...

    def generate_label(self, input_text: str) -> str:
        """Return raw provider text. May be invalid JSON."""
        ...


class ProviderError(RuntimeError):
    """Raised for provider-level failures (bad fixture, API down, etc.).
    Distinct from validator/JSON failures."""


class FakeProvider:
    """In-process provider backed by JSONL fixture files.

    See spec §3.2 for fixture contract and behavior guarantees.
    """

    provider_type = "fake"

    def __init__(
        self,
        name: str,
        *,
        inputs_path: Path | None = None,
        labels_path: Path | None = None,
        seed: int = 0,
        miss_payload: str = "",
    ) -> None:
        self.name = name
        self._miss_payload = miss_payload
        self._seed = seed  # reserved; Slice 1 is cursor-deterministic.

        self._inputs: list[str] = []
        if inputs_path is not None:
            self._inputs = self._load_inputs(Path(inputs_path))

        self._labels: dict[str, str] = {}
        if labels_path is not None:
            self._labels = self._load_labels(Path(labels_path))

        self._cursor = 0
        self._cursor_lock = threading.Lock()

    @staticmethod
    def _load_inputs(path: Path) -> list[str]:
        rows: list[str] = []
        with path.open(encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ProviderError(
                        f"{path}: line {line_no}: invalid JSON: {e}"
                    ) from None
                if not isinstance(obj, dict) or "input" not in obj or not isinstance(obj["input"], str):
                    raise ProviderError(
                        f"{path}: line {line_no}: missing required 'input' string field"
                    )
                rows.append(obj["input"])
        return rows

    @staticmethod
    def _load_labels(path: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        with path.open(encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ProviderError(
                        f"{path}: line {line_no}: invalid JSON: {e}"
                    ) from None
                if not isinstance(obj, dict) or "input" not in obj or not isinstance(obj["input"], str):
                    raise ProviderError(
                        f"{path}: line {line_no}: missing required 'input' string field"
                    )
                if "raw_output" in obj:
                    out[obj["input"]] = str(obj["raw_output"])
                elif "output" in obj:
                    out[obj["input"]] = json.dumps(obj["output"], separators=(",", ":"))
                else:
                    raise ProviderError(
                        f"{path}: line {line_no}: row needs 'output' or 'raw_output'"
                    )
        return out

    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        if not self._inputs:
            return []
        with self._cursor_lock:
            result: list[str] = []
            for _ in range(n):
                result.append(self._inputs[self._cursor % len(self._inputs)])
                self._cursor += 1
            return result

    def generate_label(self, input_text: str) -> str:
        # _labels is immutable after __init__ — no lock needed.
        return self._labels.get(input_text, self._miss_payload)
```

- [ ] **Step 4: Run tests, expect all 7 pass**

Run: `cd "C:/work/llm training" && pytest tests/test_fake_provider.py -v`

Expected: 7 passed.

- [ ] **Step 5: Do NOT commit.**

---

### Task 3: `UnimplementedProvider` + `create_provider` factory + tests

**Files:**
- Modify: `scripts/llm_providers.py`
- Modify: `tests/test_fake_provider.py`

Spec reference: §3.3, §3.4.

- [ ] **Step 1: Add tests for UnimplementedProvider and factory**

Append to `tests/test_fake_provider.py`:
```python
def test_unimplemented_provider_constructs_but_methods_raise():
    from llm_providers import UnimplementedProvider

    p = UnimplementedProvider(name="ds", provider_type="deepseek", model="x")
    assert p.name == "ds"
    assert p.provider_type == "deepseek"
    assert p.model == "x"
    with pytest.raises(NotImplementedError) as exc:
        p.generate_inputs(prompt="x", n=1)
    assert "Slice 2" in str(exc.value)
    with pytest.raises(NotImplementedError) as exc:
        p.generate_label(input_text="x")
    assert "Slice 3" in str(exc.value)


def test_unimplemented_provider_has_no_sdk_imports():
    """UnimplementedProvider must not transitively import any external SDK."""
    import sys
    sdk_modules_before = {
        m for m in sys.modules
        if m.startswith(("openai", "google.generativeai", "google.genai", "unsloth"))
    }
    from llm_providers import UnimplementedProvider
    UnimplementedProvider(name="x", provider_type="gemini", model="y")
    sdk_modules_after = {
        m for m in sys.modules
        if m.startswith(("openai", "google.generativeai", "google.genai", "unsloth"))
    }
    assert sdk_modules_after == sdk_modules_before


def test_create_provider_returns_fake_for_fake_type():
    from llm_providers import FakeProvider, create_provider
    from generation_config import ProviderConfig  # noqa: forward import

    cfg = ProviderConfig(
        name="x", provider_type="fake",
        fixture_inputs=str(INPUTS_FIXTURE),
    )
    p = create_provider(cfg)
    assert isinstance(p, FakeProvider)
    assert p.name == "x"


def test_create_provider_returns_unimplemented_for_real_types():
    from llm_providers import UnimplementedProvider, create_provider
    from generation_config import ProviderConfig

    for t in ("deepseek", "gemini", "local_teacher"):
        cfg = ProviderConfig(name=f"x_{t}", provider_type=t, model="m")
        p = create_provider(cfg)
        assert isinstance(p, UnimplementedProvider)
        assert p.provider_type == t


def test_create_provider_raises_on_unknown_type():
    from llm_providers import create_provider
    from generation_config import ProviderConfig

    cfg = ProviderConfig(name="bad", provider_type="not_a_real_type")
    with pytest.raises(ValueError) as exc:
        create_provider(cfg)
    assert "not_a_real_type" in str(exc.value)
```

- [ ] **Step 2: Run tests, expect failures (no UnimplementedProvider, no create_provider, no ProviderConfig)**

Run: `cd "C:/work/llm training" && pytest tests/test_fake_provider.py -v`

Expected: 5 failures (ImportError on UnimplementedProvider / create_provider / ProviderConfig).

- [ ] **Step 3: Add UnimplementedProvider + create_provider to `scripts/llm_providers.py`**

Append to `scripts/llm_providers.py` AFTER the `FakeProvider` class:

```python
class UnimplementedProvider:
    """Placeholder for deepseek / gemini / local_teacher in Slice 1.

    Constructs without any SDK import so configs parse and dry-runs work.
    Any generation call raises NotImplementedError pointing at the slice
    where the real implementation lands.
    """

    def __init__(self, name: str, provider_type: str, model: str | None = None) -> None:
        self.name = name
        self.provider_type = provider_type
        self.model = model

    def generate_inputs(self, prompt: str, n: int) -> list[str]:
        raise NotImplementedError(
            f"{self.provider_type!r} provider {self.name!r}: real execution lands in Slice 2."
        )

    def generate_label(self, input_text: str) -> str:
        raise NotImplementedError(
            f"{self.provider_type!r} provider {self.name!r}: real execution lands in Slice 3."
        )


def create_provider(cfg) -> LLMProvider:
    """Map ProviderConfig.provider_type -> concrete provider.

    Only 'fake' is callable in Slice 1; other types return UnimplementedProvider
    so dry-run-quota and config parsing work with real-shaped configs.

    The loader (generation_config.py §4.5) stores resolved absolute paths in
    cfg.fixture_inputs / cfg.fixture_labels, so we wrap them in Path() directly.
    """
    if cfg.provider_type == "fake":
        return FakeProvider(
            name=cfg.name,
            inputs_path=Path(cfg.fixture_inputs) if cfg.fixture_inputs else None,
            labels_path=Path(cfg.fixture_labels) if cfg.fixture_labels else None,
            seed=cfg.seed or 0,
        )
    if cfg.provider_type in {"deepseek", "gemini", "local_teacher"}:
        return UnimplementedProvider(
            name=cfg.name, provider_type=cfg.provider_type, model=cfg.model,
        )
    raise ValueError(
        f"Unknown provider type {cfg.provider_type!r} for provider {cfg.name!r}"
    )
```

- [ ] **Step 4: Tests still fail because `generation_config.ProviderConfig` doesn't exist yet**

This is expected — Task 4 creates `generation_config.py`. The tests for `create_provider` will pass after Task 4. Skip running pytest right now; re-run after Task 4.

- [ ] **Step 5: Sanity-check llm_providers imports cleanly without generation_config**

Run:
```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "from llm_providers import FakeProvider, UnimplementedProvider, ProviderError, LLMProvider; print('ok')"
```

Expected: prints `ok`. (Don't import `create_provider` here — it accepts a `ProviderConfig` but doesn't import the module at load time, so import-only works.)

- [ ] **Step 6: Do NOT commit.**

---

### Task 4: `generation_config.py` dataclasses + bare loader

**Files:**
- Create: `scripts/generation_config.py`
- Create: `tests/test_generation_config.py`

Spec reference: §4.1, §4.2.

This task creates the dataclasses and a minimal loader that handles structure parsing only. Task 5 adds the validation rules.

- [ ] **Step 1: Write initial failing tests for the dataclass shape and basic load**

`tests/test_generation_config.py`:
```python
import json
from pathlib import Path

import pytest

from generation_config import (
    ConfigError,
    GenerationConfig,
    InputGenerationConfig,
    OutputGenerationConfig,
    ProviderConfig,
    RateLimitsConfig,
    ValidationGateConfig,
    load_generation_config,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _minimal_dict() -> dict:
    return {
        "version": 1,
        "input_generation": {
            "enabled": True,
            "target_inputs": 10,
            "batch_size": 5,
            "providers": [
                {"name": "fake_a", "type": "fake", "weight": 60, "threads": 1,
                 "fixture_inputs": str(FIXTURES / "fake_inputs.jsonl")},
            ],
        },
        "output_generation": {
            "enabled": True,
            "label_attempts_per_input": 1,
            "providers": [
                {"name": "fake_a", "type": "fake", "weight": 50, "threads": 1,
                 "fixture_labels": str(FIXTURES / "fake_labels.jsonl")},
            ],
        },
        "validation": {"schema": True, "semantic_validator": True,
                       "reject_invalid": True,
                       "retry_invalid_with_stricter_prompt": False,
                       "max_repair_attempts": 0},
        "rate_limits": {"global_max_workers": 4, "write_flush_every": 10},
    }


def test_providerconfig_is_frozen_dataclass():
    cfg = ProviderConfig(name="x", provider_type="fake")
    assert cfg.name == "x"
    assert cfg.provider_type == "fake"
    assert cfg.weight == 1  # default
    with pytest.raises(Exception):
        cfg.name = "changed"  # frozen


def test_load_minimal_valid_config(tmp_path):
    path = _write_config(tmp_path, _minimal_dict())
    cfg = load_generation_config(path)
    assert isinstance(cfg, GenerationConfig)
    assert cfg.version == 1
    assert cfg.input_generation.target_inputs == 10
    assert cfg.input_generation.providers[0].name == "fake_a"
    assert cfg.input_generation.providers[0].provider_type == "fake"
    assert cfg.source_path == str(path)


def test_load_default_version_is_1(tmp_path):
    data = _minimal_dict()
    del data["version"]
    path = _write_config(tmp_path, data)
    cfg = load_generation_config(path)
    assert cfg.version == 1


def test_config_error_is_value_error():
    assert issubclass(ConfigError, ValueError)
```

- [ ] **Step 2: Run, expect ModuleNotFoundError**

Run: `cd "C:/work/llm training" && pytest tests/test_generation_config.py -v`

Expected: `ModuleNotFoundError: No module named 'generation_config'`.

- [ ] **Step 3: Create `scripts/generation_config.py`**

```python
"""Multi-provider generation config — dataclasses + JSON loader.

Stdlib only. Loader raises ConfigError with field-path messages.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_VALID_PROVIDER_TYPES = {"fake", "deepseek", "gemini", "local_teacher"}
_VALID_SELECTION_POLICIES = {"first_valid_then_score"}
_KNOWN_PROVIDER_FIELDS = {
    "name", "type", "weight", "threads", "model", "temperature", "max_tokens",
    "max_retries", "structured_output", "seed", "fixture_inputs", "fixture_labels",
}
_KNOWN_TOP_LEVEL = {
    "version", "input_generation", "output_generation", "validation", "rate_limits",
}


class ConfigError(ValueError):
    """Raised with a field path in the message (e.g. 'input_generation.providers[1].weight')."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    provider_type: str
    weight: int = 1
    threads: int = 1
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int = 3
    structured_output: bool = False
    seed: int | None = None
    fixture_inputs: str | None = None
    fixture_labels: str | None = None


@dataclass(frozen=True)
class InputGenerationConfig:
    enabled: bool = True
    target_inputs: int = 0
    batch_size: int = 100
    dedupe: bool = True
    providers: tuple[ProviderConfig, ...] = ()


@dataclass(frozen=True)
class OutputGenerationConfig:
    enabled: bool = True
    label_attempts_per_input: int = 1
    selection_policy: str = "first_valid_then_score"
    providers: tuple[ProviderConfig, ...] = ()
    provider_priority: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationGateConfig:
    schema: bool = True
    semantic_validator: bool = True
    reject_invalid: bool = True
    retry_invalid_with_stricter_prompt: bool = False
    max_repair_attempts: int = 0


@dataclass(frozen=True)
class RateLimitsConfig:
    global_max_workers: int = 8
    write_flush_every: int = 50


@dataclass(frozen=True)
class GenerationConfig:
    version: int
    input_generation: InputGenerationConfig
    output_generation: OutputGenerationConfig
    validation: ValidationGateConfig
    rate_limits: RateLimitsConfig
    source_path: str | None = None


def load_generation_config(path: Path | str) -> GenerationConfig:
    """Parse JSON config into typed dataclasses. See spec §4."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path}: invalid JSON: {e}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level must be an object")

    version = raw.get("version", 1)
    if not isinstance(version, int) or version != 1:
        raise ConfigError(
            f"version: unsupported config version {version!r}, expected 1"
        )

    for key in raw.keys():
        if key not in _KNOWN_TOP_LEVEL:
            logger.warning("Unknown top-level key in %s: %r", path, key)

    config_dir = path.parent

    input_gen = _parse_input_generation(raw.get("input_generation", {}), config_dir)
    output_gen = _parse_output_generation(raw.get("output_generation", {}), config_dir)
    validation = _parse_validation_gate(raw.get("validation", {}))
    rate_limits = _parse_rate_limits(raw.get("rate_limits", {}))

    return GenerationConfig(
        version=version,
        input_generation=input_gen,
        output_generation=output_gen,
        validation=validation,
        rate_limits=rate_limits,
        source_path=str(path),
    )


def _parse_input_generation(d: Any, config_dir: Path) -> InputGenerationConfig:
    if not isinstance(d, dict):
        raise ConfigError("input_generation: must be an object")
    enabled = bool(d.get("enabled", True))
    target_inputs = d.get("target_inputs", 0)
    batch_size = d.get("batch_size", 100)
    dedupe = bool(d.get("dedupe", True))
    providers_raw = d.get("providers", [])
    providers = _parse_providers(
        providers_raw, "input_generation", config_dir, used_in_phase="input",
    )
    return InputGenerationConfig(
        enabled=enabled,
        target_inputs=int(target_inputs),
        batch_size=int(batch_size),
        dedupe=dedupe,
        providers=tuple(providers),
    )


def _parse_output_generation(d: Any, config_dir: Path) -> OutputGenerationConfig:
    if not isinstance(d, dict):
        raise ConfigError("output_generation: must be an object")
    enabled = bool(d.get("enabled", True))
    label_attempts = d.get("label_attempts_per_input", 1)
    selection_policy = d.get("selection_policy", "first_valid_then_score")
    providers_raw = d.get("providers", [])
    providers = _parse_providers(
        providers_raw, "output_generation", config_dir, used_in_phase="output",
    )
    priority = tuple(d.get("provider_priority", []) or [])
    return OutputGenerationConfig(
        enabled=enabled,
        label_attempts_per_input=int(label_attempts),
        selection_policy=str(selection_policy),
        providers=tuple(providers),
        provider_priority=priority,
    )


def _parse_validation_gate(d: Any) -> ValidationGateConfig:
    if not isinstance(d, dict):
        raise ConfigError("validation: must be an object")
    return ValidationGateConfig(
        schema=bool(d.get("schema", True)),
        semantic_validator=bool(d.get("semantic_validator", True)),
        reject_invalid=bool(d.get("reject_invalid", True)),
        retry_invalid_with_stricter_prompt=bool(d.get("retry_invalid_with_stricter_prompt", False)),
        max_repair_attempts=int(d.get("max_repair_attempts", 0)),
    )


def _parse_rate_limits(d: Any) -> RateLimitsConfig:
    if not isinstance(d, dict):
        raise ConfigError("rate_limits: must be an object")
    return RateLimitsConfig(
        global_max_workers=int(d.get("global_max_workers", 8)),
        write_flush_every=int(d.get("write_flush_every", 50)),
    )


def _parse_providers(
    raw: Any, phase_path: str, config_dir: Path, *, used_in_phase: str,
) -> list[ProviderConfig]:
    if not isinstance(raw, list):
        raise ConfigError(f"{phase_path}.providers: must be a list")
    out: list[ProviderConfig] = []
    for i, item in enumerate(raw):
        prefix = f"{phase_path}.providers[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{prefix}: must be an object")
        for key in item.keys():
            if key not in _KNOWN_PROVIDER_FIELDS:
                logger.warning("Unknown provider field at %s: %r", prefix, key)
        # JSON key 'type' -> Python attribute 'provider_type'
        ptype = item.get("type")
        if not isinstance(ptype, str):
            raise ConfigError(f"{prefix}.type: missing or not a string")
        fixture_inputs = item.get("fixture_inputs")
        if fixture_inputs is not None:
            fixture_inputs = _resolve_fixture(
                fixture_inputs, config_dir, f"{prefix}.fixture_inputs",
            )
        fixture_labels = item.get("fixture_labels")
        if fixture_labels is not None:
            fixture_labels = _resolve_fixture(
                fixture_labels, config_dir, f"{prefix}.fixture_labels",
            )
        out.append(ProviderConfig(
            name=str(item.get("name", "")),
            provider_type=ptype,
            weight=int(item.get("weight", 1)),
            threads=int(item.get("threads", 1)),
            model=item.get("model"),
            temperature=item.get("temperature"),
            max_tokens=item.get("max_tokens"),
            max_retries=int(item.get("max_retries", 3)),
            structured_output=bool(item.get("structured_output", False)),
            seed=item.get("seed"),
            fixture_inputs=fixture_inputs,
            fixture_labels=fixture_labels,
        ))
    return out


def _resolve_fixture(value: Any, config_dir: Path, field_path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field_path}: must be a string path")
    p = Path(value)
    if not p.is_absolute():
        p = (config_dir / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        raise ConfigError(f"{field_path}: fixture not found at {p}")
    return str(p)
```

- [ ] **Step 4: Run tests, expect 4 passed**

Run: `cd "C:/work/llm training" && pytest tests/test_generation_config.py -v`

Expected: 4 passed.

- [ ] **Step 5: Re-run Task 3's tests now that ProviderConfig exists**

Run: `cd "C:/work/llm training" && pytest tests/test_fake_provider.py -v`

Expected: all FakeProvider + create_provider tests pass.

- [ ] **Step 6: Do NOT commit.**

---

### Task 5: Full `generation_config.py` validation rules

**Files:**
- Modify: `scripts/generation_config.py`
- Modify: `tests/test_generation_config.py`

Spec reference: §4.3, §4.4, §4.5.

- [ ] **Step 1: Add failing tests for validation rules**

Append to `tests/test_generation_config.py`:
```python
def test_unknown_type_raises_config_error(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"][0]["type"] = "not_a_real_type"
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "not_a_real_type" in str(exc.value)


def test_duplicate_provider_name_in_phase_raises(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"].append(
        {"name": "fake_a", "type": "fake", "weight": 10, "threads": 1,
         "fixture_inputs": str(FIXTURES / "fake_inputs.jsonl")},
    )
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "fake_a" in str(exc.value)


def test_weight_zero_raises(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"][0]["weight"] = 0
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "weight" in str(exc.value)


def test_negative_max_tokens_raises(tmp_path):
    data = _minimal_dict()
    data["output_generation"]["providers"][0]["max_tokens"] = 0
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        load_generation_config(path)


def test_negative_temperature_raises(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"][0]["temperature"] = -0.5
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        load_generation_config(path)


def test_fake_provider_in_input_requires_fixture_inputs(tmp_path):
    data = _minimal_dict()
    del data["input_generation"]["providers"][0]["fixture_inputs"]
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "fixture_inputs" in str(exc.value)


def test_deepseek_provider_requires_model(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"][0] = {
        "name": "ds", "type": "deepseek", "weight": 10, "threads": 1,
    }
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "model" in str(exc.value)


def test_enabled_false_skips_required_provider_list(tmp_path):
    data = _minimal_dict()
    data["input_generation"] = {"enabled": False, "target_inputs": 0,
                                 "batch_size": 1, "providers": []}
    path = _write_config(tmp_path, data)
    cfg = load_generation_config(path)
    assert cfg.input_generation.enabled is False
    assert cfg.input_generation.providers == ()


def test_missing_input_providers_when_enabled_raises(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"] = []
    data["input_generation"]["target_inputs"] = 10
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "providers" in str(exc.value)


def test_missing_fixture_file_raises(tmp_path):
    data = _minimal_dict()
    data["input_generation"]["providers"][0]["fixture_inputs"] = "nonexistent.jsonl"
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "not found" in str(exc.value)


def test_unknown_top_level_key_warns(tmp_path, caplog):
    data = _minimal_dict()
    data["unknown_top_key"] = "x"
    path = _write_config(tmp_path, data)
    with caplog.at_level("WARNING"):
        load_generation_config(path)
    assert any("unknown_top_key" in r.message.lower() or "unknown_top_key" in r.message
               for r in caplog.records)


def test_provider_priority_must_reference_declared_provider(tmp_path):
    data = _minimal_dict()
    data["output_generation"]["provider_priority"] = ["fake_a", "ghost"]
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "ghost" in str(exc.value)


def test_version_2_raises(tmp_path):
    data = _minimal_dict()
    data["version"] = 2
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "version" in str(exc.value)


def test_unknown_selection_policy_raises(tmp_path):
    data = _minimal_dict()
    data["output_generation"]["selection_policy"] = "magic"
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError) as exc:
        load_generation_config(path)
    assert "selection_policy" in str(exc.value)


def test_fixture_path_resolved_relative_to_config_dir(tmp_path):
    """Place config in tmp_path/configs/, fixture path '../inputs.jsonl'
    must resolve to tmp_path/inputs.jsonl."""
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text('{"input":"x"}\n', encoding="utf-8")
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    labels = FIXTURES / "fake_labels.jsonl"
    data = _minimal_dict()
    data["input_generation"]["providers"][0]["fixture_inputs"] = "../inputs.jsonl"
    data["output_generation"]["providers"][0]["fixture_labels"] = str(labels)
    config_path = configs_dir / "providers.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    cfg = load_generation_config(config_path)
    resolved = cfg.input_generation.providers[0].fixture_inputs
    assert Path(resolved) == inputs.resolve()
```

- [ ] **Step 2: Run, expect most new tests fail**

Run: `cd "C:/work/llm training" && pytest tests/test_generation_config.py -v`

Expected: ~10+ new failures (validation rules not yet implemented).

- [ ] **Step 3: Add validation passes to `scripts/generation_config.py`**

Replace `load_generation_config` body with the version below (everything before `_parse_input_generation` stays the same; the loader gains a final `_validate(cfg)` call):

Edit the bottom of `load_generation_config` to add a final validation pass:

```python
def load_generation_config(path: Path | str) -> GenerationConfig:
    """Parse JSON config into typed dataclasses. See spec §4."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path}: invalid JSON: {e}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level must be an object")

    version = raw.get("version", 1)
    if not isinstance(version, int) or version != 1:
        raise ConfigError(
            f"version: unsupported config version {version!r}, expected 1"
        )

    for key in raw.keys():
        if key not in _KNOWN_TOP_LEVEL:
            logger.warning("Unknown top-level key in %s: %r", path, key)

    config_dir = path.parent

    input_gen = _parse_input_generation(raw.get("input_generation", {}), config_dir)
    output_gen = _parse_output_generation(raw.get("output_generation", {}), config_dir)
    validation = _parse_validation_gate(raw.get("validation", {}))
    rate_limits = _parse_rate_limits(raw.get("rate_limits", {}))

    cfg = GenerationConfig(
        version=version,
        input_generation=input_gen,
        output_generation=output_gen,
        validation=validation,
        rate_limits=rate_limits,
        source_path=str(path),
    )
    _validate(cfg)
    return cfg
```

Then add the `_validate` function at module bottom:

```python
def _validate(cfg: GenerationConfig) -> None:
    """Apply the rules in spec §4.3. Raises ConfigError on violation."""
    ig = cfg.input_generation
    if ig.target_inputs < 0:
        raise ConfigError("input_generation.target_inputs: must be >= 0")
    if ig.batch_size < 1:
        raise ConfigError("input_generation.batch_size: must be >= 1")
    if ig.enabled and ig.target_inputs >= 1 and not ig.providers:
        raise ConfigError(
            "input_generation.providers: must be non-empty when enabled=true and target_inputs>=1"
        )
    _validate_providers(ig.providers, "input_generation", used_in_phase="input")

    og = cfg.output_generation
    if og.label_attempts_per_input < 1:
        raise ConfigError("output_generation.label_attempts_per_input: must be >= 1")
    if og.selection_policy not in _VALID_SELECTION_POLICIES:
        raise ConfigError(
            f"output_generation.selection_policy: {og.selection_policy!r} "
            f"not in {sorted(_VALID_SELECTION_POLICIES)}"
        )
    if og.enabled and not og.providers:
        raise ConfigError(
            "output_generation.providers: must be non-empty when enabled=true"
        )
    _validate_providers(og.providers, "output_generation", used_in_phase="output")
    declared = {p.name for p in og.providers}
    for name in og.provider_priority:
        if name not in declared:
            raise ConfigError(
                f"output_generation.provider_priority: {name!r} not in declared providers"
            )

    if cfg.validation.max_repair_attempts < 0:
        raise ConfigError("validation.max_repair_attempts: must be >= 0")
    if cfg.rate_limits.global_max_workers < 1:
        raise ConfigError("rate_limits.global_max_workers: must be >= 1")
    if cfg.rate_limits.write_flush_every < 1:
        raise ConfigError("rate_limits.write_flush_every: must be >= 1")


def _validate_providers(
    providers: tuple[ProviderConfig, ...], phase_path: str, *, used_in_phase: str,
) -> None:
    seen_names: set[str] = set()
    for i, p in enumerate(providers):
        prefix = f"{phase_path}.providers[{i}]"
        if not p.name:
            raise ConfigError(f"{prefix}.name: must be non-empty")
        if p.name in seen_names:
            raise ConfigError(
                f"{prefix}.name: duplicate provider name {p.name!r} within {phase_path}"
            )
        seen_names.add(p.name)
        if p.provider_type not in _VALID_PROVIDER_TYPES:
            raise ConfigError(
                f"{prefix}.type: {p.provider_type!r} not in {sorted(_VALID_PROVIDER_TYPES)}"
            )
        if p.weight < 1:
            raise ConfigError(f"{prefix}.weight: must be >= 1")
        if p.threads < 1:
            raise ConfigError(f"{prefix}.threads: must be >= 1")
        if p.temperature is not None and p.temperature < 0:
            raise ConfigError(f"{prefix}.temperature: must be null or >= 0")
        if p.max_tokens is not None and p.max_tokens < 1:
            raise ConfigError(f"{prefix}.max_tokens: must be null or >= 1")
        if p.max_retries < 0:
            raise ConfigError(f"{prefix}.max_retries: must be >= 0")
        if p.provider_type == "fake":
            if used_in_phase == "input" and not p.fixture_inputs:
                raise ConfigError(
                    f"{prefix}: fake provider in input phase requires fixture_inputs"
                )
            if used_in_phase == "output" and not p.fixture_labels:
                raise ConfigError(
                    f"{prefix}: fake provider in output phase requires fixture_labels"
                )
        if p.provider_type in {"deepseek", "gemini"} and not p.model:
            raise ConfigError(f"{prefix}: {p.provider_type} provider requires model")
        if p.structured_output and p.provider_type != "gemini":
            logger.warning(
                "%s: structured_output=true on non-gemini provider %r is a no-op",
                prefix, p.name,
            )
```

- [ ] **Step 4: Run, expect all pass**

Run: `cd "C:/work/llm training" && pytest tests/test_generation_config.py -v`

Expected: all ~17 tests pass.

- [ ] **Step 5: Coverage check on generation_config.py**

Run: `cd "C:/work/llm training" && pytest tests/test_generation_config.py --cov=generation_config --cov-report=term-missing`

Expected: ≥ 90% coverage. If under, look at missing lines and add a fixture.

- [ ] **Step 6: Do NOT commit.**

---

### Task 6: `allocate_quota` in `generation_orchestrator.py`

**Files:**
- Create: `scripts/generation_orchestrator.py`
- Create: `tests/test_provider_scheduler.py`

Spec reference: §5.1.

- [ ] **Step 1: Write failing tests**

`tests/test_provider_scheduler.py`:
```python
import threading
from collections import Counter

import pytest

from generation_config import ProviderConfig
from generation_orchestrator import (
    RoundRobinScheduler,
    allocate_quota,
    render_dry_run,
)


def _provs(*pairs: tuple[str, int]) -> list[ProviderConfig]:
    return [ProviderConfig(name=n, provider_type="fake", weight=w) for n, w in pairs]


def test_allocate_quota_70_30_split():
    p = _provs(("deepseek", 70), ("gemini", 30))
    result = allocate_quota(100000, p)
    assert result == {"deepseek": 70000, "gemini": 30000}


def test_allocate_quota_60_40_small_total():
    p = _provs(("a", 60), ("b", 40))
    result = allocate_quota(10, p)
    assert result == {"a": 6, "b": 4}


def test_allocate_quota_tie_break_by_declaration_order():
    p = _provs(("a", 1), ("b", 1), ("c", 1))
    result = allocate_quota(10, p)
    # 10/3 = 3.33 each; floor gives 3 each (sum 9); leftover 1 by order -> a gets +1
    assert result == {"a": 4, "b": 3, "c": 3}
    assert sum(result.values()) == 10


def test_allocate_quota_zero_total():
    p = _provs(("a", 1), ("b", 1))
    assert allocate_quota(0, p) == {"a": 0, "b": 0}


def test_allocate_quota_single_provider_gets_all():
    p = _provs(("only", 1))
    assert allocate_quota(42, p) == {"only": 42}


def test_allocate_quota_empty_raises():
    with pytest.raises(ValueError):
        allocate_quota(10, [])


def test_allocate_quota_sums_to_total_random_weights():
    p = _provs(("a", 17), ("b", 41), ("c", 23), ("d", 5))
    for total in (1, 7, 100, 1234, 99999):
        result = allocate_quota(total, p)
        assert sum(result.values()) == total
        for v in result.values():
            assert v >= 0
```

- [ ] **Step 2: Run, expect ModuleNotFoundError**

Run: `cd "C:/work/llm training" && pytest tests/test_provider_scheduler.py -v`

Expected: `ModuleNotFoundError: No module named 'generation_orchestrator'`.

- [ ] **Step 3: Create `scripts/generation_orchestrator.py`**

```python
"""Multi-provider orchestration: quota allocation + weighted scheduler + dry-run rendering.

Pure stdlib. No I/O, no provider calls, no threading execution.
"""
from __future__ import annotations

import math
import threading
from typing import Sequence

from generation_config import GenerationConfig, ProviderConfig


def allocate_quota(
    total: int,
    providers: Sequence[ProviderConfig],
) -> dict[str, int]:
    """Allocate `total` across providers by weight (largest-remainder method).

    Guarantees:
      - sum(result.values()) == total
      - all values >= 0
      - deterministic given fixed provider order and weights

    Raises ValueError if providers is empty.
    """
    if not providers:
        raise ValueError("allocate_quota: providers must be non-empty")

    total_weight = sum(p.weight for p in providers)
    if total_weight <= 0:
        raise ValueError("allocate_quota: sum of weights must be positive")

    # Largest-remainder method.
    raw = [(p.name, total * p.weight / total_weight) for p in providers]
    floors = {name: int(value) for name, value in raw}
    leftover = total - sum(floors.values())
    # Remainders, with original index to break ties by declaration order.
    remainders = sorted(
        ((value - int(value), idx, name) for idx, (name, value) in enumerate(raw)),
        key=lambda x: (-x[0], x[1]),  # largest fractional remainder first; then by idx
    )
    for i in range(leftover):
        _, _, name = remainders[i]
        floors[name] += 1
    return floors
```

- [ ] **Step 4: Run, expect 7 passed**

Run: `cd "C:/work/llm training" && pytest tests/test_provider_scheduler.py -v -k allocate_quota`

Expected: 7 passed.

- [ ] **Step 5: Do NOT commit.**

---

### Task 7: `RoundRobinScheduler` (GCD-normalized, thread-safe)

**Files:**
- Modify: `scripts/generation_orchestrator.py`
- Modify: `tests/test_provider_scheduler.py`

Spec reference: §5.2.

- [ ] **Step 1: Append failing tests**

```python
def test_round_robin_cycle_length_normalized_by_gcd():
    p = _provs(("a", 60), ("b", 40))
    sched = RoundRobinScheduler(p)
    # gcd(60, 40) = 20; normalized to 3:2; cycle length 5.
    assert sched.cycle_length == 5


def test_round_robin_cycle_length_huge_weights_normalized():
    p = _provs(("a", 60000), ("b", 40000))
    sched = RoundRobinScheduler(p)
    assert sched.cycle_length == 5  # NOT 100000


def test_round_robin_distribution_over_full_cycles():
    p = _provs(("a", 3), ("b", 2))  # cycle length 5
    sched = RoundRobinScheduler(p)
    # 100 cycles = 500 calls; expect exactly 300 a's, 200 b's.
    counts = Counter(sched.next_provider() for _ in range(500))
    assert counts == {"a": 300, "b": 200}


def test_round_robin_distribution_over_full_cycles_50_30_20():
    p = _provs(("a", 50), ("b", 30), ("c", 20))  # gcd 10 -> 5:3:2 -> cycle 10
    sched = RoundRobinScheduler(p)
    assert sched.cycle_length == 10
    # 100 cycles = 1000 calls -> exactly 500/300/200
    counts = Counter(sched.next_provider() for _ in range(1000))
    assert counts == {"a": 500, "b": 300, "c": 200}


def test_round_robin_no_long_runs():
    """Interleaving should avoid blocky runs. Max run length should be small."""
    p = _provs(("a", 3), ("b", 2))
    sched = RoundRobinScheduler(p)
    seq = [sched.next_provider() for _ in range(20)]
    # Find longest consecutive run of same provider.
    max_run = 1
    run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    assert max_run <= 2, f"too-blocky sequence: {seq}"


def test_round_robin_reset():
    p = _provs(("a", 1), ("b", 1))
    sched = RoundRobinScheduler(p)
    first = [sched.next_provider() for _ in range(4)]
    sched.reset()
    second = [sched.next_provider() for _ in range(4)]
    assert first == second


def test_round_robin_thread_safe_no_lost_calls():
    """8 threads × 250 calls each = 2000 calls. Cycle 5 (3:2). Expect 1200 a, 800 b."""
    p = _provs(("a", 3), ("b", 2))
    sched = RoundRobinScheduler(p)
    counts = Counter()
    counts_lock = threading.Lock()

    def worker():
        local = []
        for _ in range(250):
            local.append(sched.next_provider())
        with counts_lock:
            counts.update(local)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counts == {"a": 1200, "b": 800}
```

- [ ] **Step 2: Run, watch failures**

Run: `cd "C:/work/llm training" && pytest tests/test_provider_scheduler.py -v -k round_robin`

Expected: failures (RoundRobinScheduler doesn't exist yet).

- [ ] **Step 3: Add `RoundRobinScheduler` to `scripts/generation_orchestrator.py`**

```python
class RoundRobinScheduler:
    """Deterministic weighted round-robin. Thread-safe.

    Weights are normalized by GCD before building the cycle, so [60000, 40000]
    behaves like [3, 2]. Bresenham-style interleaving keeps no long runs.

    Guarantees:
      - Over every full cycle, counts exactly match normalized weights.
      - Over partial cycles, counts deviate by at most one slot per provider.
      - next_provider() is protected by threading.Lock — concurrent callers
        receive serialized consecutive scheduler positions; cursor is never corrupted.
      - State is in-memory only; resets between process runs.
    """

    def __init__(self, providers: Sequence[ProviderConfig]) -> None:
        if not providers:
            raise ValueError("RoundRobinScheduler: providers must be non-empty")
        # Normalize by GCD.
        weights = [p.weight for p in providers]
        g = weights[0]
        for w in weights[1:]:
            g = math.gcd(g, w)
        normalized = [(p.name, w // g) for p, w in zip(providers, weights)]
        self._cycle: list[str] = _bresenham_interleave(normalized)
        self._cursor = 0
        self._lock = threading.Lock()

    @property
    def cycle_length(self) -> int:
        return len(self._cycle)

    def next_provider(self) -> str:
        with self._lock:
            name = self._cycle[self._cursor % len(self._cycle)]
            self._cursor += 1
            return name

    def reset(self) -> None:
        with self._lock:
            self._cursor = 0


def _bresenham_interleave(weighted: list[tuple[str, int]]) -> list[str]:
    """Build a length-sum(weights) sequence that interleaves names by weight.

    Uses a credit-accumulator scheme: at each slot, pick the provider whose
    (cumulative-credit / weight) ratio is smallest, breaking ties by
    declaration order. Equivalent to Bresenham's line-drawing applied to
    multi-line distribution.
    """
    total = sum(w for _, w in weighted)
    credits = {name: 0 for name, _ in weighted}
    weights = dict(weighted)
    order = [name for name, _ in weighted]
    out: list[str] = []
    for _ in range(total):
        # Pick the provider with the most "deserved" next slot.
        # Score = (credits + 1) / weight; smaller score = more deserving.
        best_name = order[0]
        best_score = (credits[best_name] + 1) / weights[best_name]
        for name in order[1:]:
            score = (credits[name] + 1) / weights[name]
            if score < best_score:
                best_score = score
                best_name = name
        out.append(best_name)
        credits[best_name] += 1
    return out
```

- [ ] **Step 4: Run scheduler tests, expect all pass**

Run: `cd "C:/work/llm training" && pytest tests/test_provider_scheduler.py -v -k round_robin`

Expected: all 7 pass.

- [ ] **Step 5: Do NOT commit.**

---

### Task 8: `render_dry_run`

**Files:**
- Modify: `scripts/generation_orchestrator.py`
- Modify: `tests/test_provider_scheduler.py`

Spec reference: §5.3.

- [ ] **Step 1: Append failing tests**

```python
def _make_minimal_cfg(tmp_path):
    """Build a minimal GenerationConfig for dry-run tests."""
    from generation_config import (
        GenerationConfig, InputGenerationConfig, OutputGenerationConfig,
        ValidationGateConfig, RateLimitsConfig,
    )
    p_input = ProviderConfig(
        name="fake_a", provider_type="fake", weight=60, threads=1,
        fixture_inputs="/tmp/fake.jsonl",
    )
    p_input2 = ProviderConfig(
        name="fake_b", provider_type="fake", weight=40, threads=1,
        fixture_inputs="/tmp/fake.jsonl",
    )
    p_output = ProviderConfig(
        name="fake_a", provider_type="fake", weight=50, threads=1,
        fixture_labels="/tmp/fake.jsonl",
    )
    p_output2 = ProviderConfig(
        name="fake_b", provider_type="fake", weight=50, threads=1,
        fixture_labels="/tmp/fake.jsonl",
    )
    return GenerationConfig(
        version=1,
        input_generation=InputGenerationConfig(
            enabled=True, target_inputs=10, batch_size=5, dedupe=True,
            providers=(p_input, p_input2),
        ),
        output_generation=OutputGenerationConfig(
            enabled=True, label_attempts_per_input=1,
            selection_policy="first_valid_then_score",
            providers=(p_output, p_output2),
        ),
        validation=ValidationGateConfig(),
        rate_limits=RateLimitsConfig(global_max_workers=4, write_flush_every=10),
        source_path=str(tmp_path / "providers.json"),
    )


def test_render_dry_run_contains_key_sections(tmp_path):
    cfg = _make_minimal_cfg(tmp_path)
    output = render_dry_run(cfg)
    assert "Multi-provider dry run" in output
    assert "Config:" in output
    assert "(version 1)" in output
    assert "Input generation: enabled" in output
    assert "target_inputs: 10" in output
    assert "Output generation: enabled" in output
    assert "selection_policy: first_valid_then_score" in output
    assert "scheduler preview (first 10):" in output
    assert "Validation gate:" in output
    assert "Execution limits:" in output
    assert "Execution: not started" in output


def test_render_dry_run_shows_quota_for_each_input_provider(tmp_path):
    cfg = _make_minimal_cfg(tmp_path)
    output = render_dry_run(cfg)
    # target=10, weights 60/40 -> 6/4
    assert "quota=6" in output
    assert "quota=4" in output


def test_render_dry_run_disabled_input_phase(tmp_path):
    from generation_config import InputGenerationConfig
    cfg = _make_minimal_cfg(tmp_path)
    cfg = type(cfg)(
        version=cfg.version,
        input_generation=InputGenerationConfig(enabled=False),
        output_generation=cfg.output_generation,
        validation=cfg.validation,
        rate_limits=cfg.rate_limits,
        source_path=cfg.source_path,
    )
    output = render_dry_run(cfg)
    assert "Input generation: disabled" in output
    assert "quota=" not in output  # no quota block at all
    assert "Output generation: enabled" in output


def test_render_dry_run_scheduler_preview_has_10_names(tmp_path):
    cfg = _make_minimal_cfg(tmp_path)
    output = render_dry_run(cfg)
    # Find the "scheduler preview (first 10):" line and verify 10 names follow.
    for line in output.splitlines():
        if "scheduler preview (first 10):" in line:
            after = line.split(":", 1)[1].strip()
            names = after.split()
            assert len(names) == 10
            assert all(n in {"fake_a", "fake_b"} for n in names)
            return
    raise AssertionError("scheduler preview line not found")
```

- [ ] **Step 2: Run, expect failures**

Run: `cd "C:/work/llm training" && pytest tests/test_provider_scheduler.py -v -k render_dry_run`

Expected: failures.

- [ ] **Step 3: Add `render_dry_run` to `scripts/generation_orchestrator.py`**

```python
def render_dry_run(cfg: GenerationConfig) -> str:
    """Human-readable summary of what the config would do.
    Pinned format — see spec §5.3. No execution, no side effects."""
    lines: list[str] = []
    lines.append("Multi-provider dry run")
    src = cfg.source_path or "<unknown>"
    lines.append(f"Config: {src}  (version {cfg.version})")
    lines.append("")

    # Input generation.
    if not cfg.input_generation.enabled:
        lines.append("Input generation: disabled")
    else:
        ig = cfg.input_generation
        lines.append("Input generation: enabled")
        lines.append(f"  target_inputs: {ig.target_inputs}")
        lines.append(f"  batch_size: {ig.batch_size}")
        lines.append(f"  dedupe: {ig.dedupe}")
        lines.append("  providers:")
        quotas = allocate_quota(ig.target_inputs, ig.providers) if ig.providers else {}
        for p in ig.providers:
            lines.append(
                f"    {p.name}  type={p.provider_type}  weight={p.weight}  "
                f"threads={p.threads}  quota={quotas.get(p.name, 0)}"
            )
    lines.append("")

    # Output generation.
    if not cfg.output_generation.enabled:
        lines.append("Output generation: disabled")
    else:
        og = cfg.output_generation
        lines.append("Output generation: enabled")
        lines.append(f"  label_attempts_per_input: {og.label_attempts_per_input}")
        lines.append(f"  selection_policy: {og.selection_policy}")
        priority = " ".join(og.provider_priority) if og.provider_priority else "(default declaration order)"
        lines.append(f"  provider_priority: {priority}")
        lines.append("  providers:")
        for p in og.providers:
            lines.append(
                f"    {p.name}  type={p.provider_type}  weight={p.weight}  threads={p.threads}"
            )
        if og.providers:
            sched = RoundRobinScheduler(og.providers)
            preview = " ".join(sched.next_provider() for _ in range(10))
            lines.append(f"  scheduler preview (first 10): {preview}")
    lines.append("")

    # Validation gate.
    v = cfg.validation
    lines.append("Validation gate:")
    lines.append(f"  schema: {v.schema}")
    lines.append(f"  semantic_validator: {v.semantic_validator}")
    lines.append(f"  reject_invalid: {v.reject_invalid}")
    lines.append(f"  retry_invalid_with_stricter_prompt: {v.retry_invalid_with_stricter_prompt}")
    lines.append(f"  max_repair_attempts: {v.max_repair_attempts}")
    lines.append("")

    # Execution limits.
    rl = cfg.rate_limits
    lines.append("Execution limits:")
    lines.append(f"  global_max_workers: {rl.global_max_workers}")
    lines.append(f"  write_flush_every: {rl.write_flush_every}")
    lines.append("")
    lines.append("Execution: not started. This was a dry run.")
    return "\n".join(lines)
```

- [ ] **Step 4: Run, expect pass**

Run: `cd "C:/work/llm training" && pytest tests/test_provider_scheduler.py -v`

Expected: all scheduler tests pass.

- [ ] **Step 5: Coverage check on generation_orchestrator.py**

Run: `cd "C:/work/llm training" && pytest tests/test_provider_scheduler.py --cov=generation_orchestrator --cov-report=term-missing`

Expected: ≥ 95%.

- [ ] **Step 6: Do NOT commit.**

---

### Task 9: Config files (`test_providers.json` + `example_providers.json`)

**Files:**
- Create: `configs/test_providers.json`
- Create: `configs/example_providers.json`

Spec reference: §4.6.

- [ ] **Step 1: Create `configs/` directory and `test_providers.json`**

```bash
mkdir -p "C:/work/llm training/configs"
```

`configs/test_providers.json`:
```json
{
  "version": 1,
  "input_generation": {
    "enabled": true,
    "target_inputs": 10,
    "batch_size": 5,
    "dedupe": true,
    "providers": [
      {"name": "fake_a", "type": "fake", "weight": 60, "threads": 1,
       "fixture_inputs": "../tests/fixtures/fake_inputs.jsonl"},
      {"name": "fake_b", "type": "fake", "weight": 40, "threads": 1,
       "fixture_inputs": "../tests/fixtures/fake_inputs.jsonl"}
    ]
  },
  "output_generation": {
    "enabled": true,
    "label_attempts_per_input": 1,
    "providers": [
      {"name": "fake_a", "type": "fake", "weight": 50, "threads": 1,
       "fixture_labels": "../tests/fixtures/fake_labels.jsonl"},
      {"name": "fake_b", "type": "fake", "weight": 50, "threads": 1,
       "fixture_labels": "../tests/fixtures/fake_labels.jsonl"}
    ]
  },
  "validation": {"schema": true, "semantic_validator": true,
                 "reject_invalid": true,
                 "retry_invalid_with_stricter_prompt": false,
                 "max_repair_attempts": 0},
  "rate_limits": {"global_max_workers": 4, "write_flush_every": 10}
}
```

- [ ] **Step 2: Create `configs/example_providers.json`** (realistic config with unimplemented types)

```json
{
  "version": 1,
  "input_generation": {
    "enabled": true,
    "target_inputs": 10000,
    "batch_size": 100,
    "dedupe": true,
    "providers": [
      {"name": "deepseek_v4_pro", "type": "deepseek", "model": "CONFIGURE_EXACT_MODEL_ID",
       "weight": 50, "threads": 4, "max_retries": 3,
       "temperature": 1.0, "max_tokens": 8000},
      {"name": "gemini_flash", "type": "gemini", "model": "CONFIGURE_EXACT_MODEL_ID",
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
      {"name": "deepseek_v4_pro", "type": "deepseek", "model": "CONFIGURE_EXACT_MODEL_ID",
       "weight": 30, "threads": 4, "temperature": 0.0, "max_tokens": 512},
      {"name": "gemini_flash", "type": "gemini", "model": "CONFIGURE_EXACT_MODEL_ID",
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

- [ ] **Step 3: Verify both configs load and dry-run renders without SDK imports**

Append this test to `tests/test_generation_config.py`:
```python
def test_test_providers_config_loads_and_dry_runs():
    from generation_orchestrator import render_dry_run
    config_path = REPO_ROOT / "configs" / "test_providers.json"
    cfg = load_generation_config(config_path)
    output = render_dry_run(cfg)
    assert "Multi-provider dry run" in output
    assert "fake_a" in output
    assert "fake_b" in output


def test_example_config_loads_and_dry_runs_without_sdk_imports():
    """The example config has deepseek/gemini providers but must load
    cleanly because they become UnimplementedProvider. No SDK imports."""
    import sys
    from generation_orchestrator import render_dry_run
    from llm_providers import UnimplementedProvider, create_provider

    sdk_before = {m for m in sys.modules if m.startswith(("openai", "google.generativeai", "google.genai", "unsloth"))}
    config_path = REPO_ROOT / "configs" / "example_providers.json"
    cfg = load_generation_config(config_path)
    output = render_dry_run(cfg)
    for p in cfg.input_generation.providers:
        provider = create_provider(p)
        if p.provider_type != "fake":
            assert isinstance(provider, UnimplementedProvider)
    sdk_after = {m for m in sys.modules if m.startswith(("openai", "google.generativeai", "google.genai", "unsloth"))}
    assert sdk_after == sdk_before, f"SDK imports leaked: {sdk_after - sdk_before}"
    assert "deepseek_v4_pro" in output
    assert "gemini_flash" in output
```

- [ ] **Step 4: Run, expect pass**

Run: `cd "C:/work/llm training" && pytest tests/test_generation_config.py -v -k "test_providers_config or example_config"`

Expected: 2 passed.

- [ ] **Step 5: Do NOT commit.**

---

### Task 10: `probe_validator.py` core (streaming reader + summary)

**Files:**
- Create: `scripts/probe_validator.py`
- Create: `tests/test_probe_validator.py`

Spec reference: §6.1–§6.5.

- [ ] **Step 1: Write failing tests for core probe behavior (CLI tests come in Task 11)**

`tests/test_probe_validator.py`:
```python
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
```

- [ ] **Step 2: Run, expect failures (script doesn't exist)**

Run: `cd "C:/work/llm training" && pytest tests/test_probe_validator.py -v`

Expected: failures with `[Errno 2] No such file or directory` on the script path.

- [ ] **Step 3: Create `scripts/probe_validator.py`**

```python
#!/usr/bin/env python3
"""Standalone validator probe.

Reads a JSONL of {input, output} pairs and reports how many pass the
semantic validator. No GPU, no API, no model load.

Usage:
    python scripts/probe_validator.py --input data/distill/train.jsonl
    python scripts/probe_validator.py --input <jsonl> --output reports/probe.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

# Flat-import setup — match the repo convention.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import validate_example, serialize_validation_result  # noqa: E402


def iter_jsonl_rows(path: Path) -> Iterator[tuple[int, str, dict | None, str | None]]:
    """Yield (line_no, raw, parsed_or_None, malformed_reason_or_None) for each non-empty line."""
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                yield line_no, raw.rstrip("\n"), json.loads(text), None
            except json.JSONDecodeError as e:
                yield line_no, raw.rstrip("\n"), None, f"json_parse_error: {e}"


def probe(
    rows: Iterator[tuple[int, str, dict | None, str | None]],
    *,
    mode: str = "strict",
    output_file=None,
) -> dict:
    """Walk rows, validate, accumulate counts. Optionally write per-row JSONL.

    Returns summary dict with keys:
      rows_scanned, malformed, ok, failed, failure_counts, warning_counts
    """
    rows_scanned = 0
    malformed = 0
    ok = 0
    failed = 0
    failure_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()

    for line_no, raw, obj, malformed_reason in rows:
        rows_scanned += 1
        if malformed_reason is not None:
            malformed += 1
            if output_file:
                output_file.write(json.dumps({
                    "line": line_no, "raw": raw,
                    "validator_ok": None, "validation": None,
                    "malformed": malformed_reason,
                }, ensure_ascii=False) + "\n")
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("input"), str) \
                or not isinstance(obj.get("output"), dict):
            malformed += 1
            if output_file:
                output_file.write(json.dumps({
                    "line": line_no, "raw": raw,
                    "validator_ok": None, "validation": None,
                    "malformed": "missing 'input' string or 'output' dict",
                }, ensure_ascii=False) + "\n")
            continue
        result = validate_example(obj["input"], obj["output"], mode=mode)
        # Severity-based classification — independent of result.ok.
        error_codes = [e.code for e in result.errors if e.severity == "error"]
        warning_codes = [e.code for e in result.errors if e.severity == "warning"]
        row_ok = len(error_codes) == 0
        if row_ok:
            ok += 1
        else:
            failed += 1
            for code in error_codes:
                failure_counts[code] += 1
        for code in warning_codes:
            warning_counts[code] += 1
        if output_file:
            output_file.write(json.dumps({
                "line": line_no,
                "input": obj["input"],
                "output": obj["output"],
                "validator_ok": row_ok,
                "validation": serialize_validation_result(result),
            }, ensure_ascii=False) + "\n")
    return {
        "rows_scanned": rows_scanned,
        "malformed": malformed,
        "ok": ok,
        "failed": failed,
        "failure_counts": failure_counts,
        "warning_counts": warning_counts,
    }


def render_summary(input_path: Path, summary: dict) -> str:
    lines: list[str] = []
    lines.append(f"Validator probe — {input_path}")
    lines.append(f"Rows scanned:      {summary['rows_scanned']}")
    lines.append(f"Malformed:         {summary['malformed']}")
    total_well_formed = summary["rows_scanned"] - summary["malformed"]
    if total_well_formed:
        ok_pct = summary["ok"] / total_well_formed * 100
        fail_pct = summary["failed"] / total_well_formed * 100
        lines.append(f"OK:                {summary['ok']}   ({ok_pct:.1f}%)")
        lines.append(f"Failed:            {summary['failed']}   ({fail_pct:.1f}%)")
    else:
        lines.append(f"OK:                {summary['ok']}")
        lines.append(f"Failed:            {summary['failed']}")
    if summary["failure_counts"]:
        lines.append("")
        lines.append("Failures by code:")
        for code, count in sorted(
            summary["failure_counts"].items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            lines.append(f"  {code:<32} {count}")
    if summary["warning_counts"]:
        lines.append("")
        lines.append("Warnings by code:")
        for code, count in sorted(
            summary["warning_counts"].items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            lines.append(f"  {code:<32} {count}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone validator probe.")
    p.add_argument("--input", type=Path, required=True,
                   help="JSONL of {input, output} pairs.")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional per-row report JSONL; parent dirs created automatically.")
    p.add_argument("--limit", type=int, default=0,
                   help="Probe only the first N rows. 0 = all.")
    p.add_argument("--mode", choices=("strict", "warn"), default="strict",
                   help="Validator mode. Summary classification is severity-based "
                        "regardless of mode (so 'warn' does not hide failures).")
    p.add_argument("--fail-on-malformed", action="store_true",
                   help="Exit 3 when any malformed row is encountered.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-row progress (no-op here; summary is always printed).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_file = args.output.open("w", encoding="utf-8")
    else:
        output_file = None

    try:
        rows = iter_jsonl_rows(args.input)
        if args.limit > 0:
            def truncate(it, n):
                for i, x in enumerate(it):
                    if i >= n:
                        break
                    yield x
            rows = truncate(rows, args.limit)
        summary = probe(rows, mode=args.mode, output_file=output_file)
    finally:
        if output_file is not None:
            output_file.close()

    print(render_summary(args.input, summary))

    if args.fail_on_malformed and summary["malformed"] > 0:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run probe tests**

Run: `cd "C:/work/llm training" && pytest tests/test_probe_validator.py -v`

Expected: all probe tests pass.

- [ ] **Step 5: Smoke the CLI manually**

Run:
```bash
cd "C:/work/llm training" && python scripts/probe_validator.py --input tests/fixtures/probe_examples.jsonl
```

Expected: prints summary with `Validator probe`, `Rows scanned: 11`, `Failures by code:` block listing the codes hit by the fixture.

- [ ] **Step 6: Do NOT commit.**

---

### Task 11: Stage 5 flag wiring + `_run_dry_run`

**Files:**
- Modify: `scripts/05_generate_distillation_data.py`

Spec reference: §7.

- [ ] **Step 1: Add new flags + change `parse_args` to return `(parser, namespace)`**

Open `scripts/05_generate_distillation_data.py`. Find the `parse_args` function. Change its signature and add the new flags AFTER `--retry-validation-failed`:

Replace the existing `def parse_args() -> argparse.Namespace:` signature and final `return` line, AND add the three new flags:

```python
def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    p = argparse.ArgumentParser(description="Generate distillation data (inputs + teacher labels).")
    # ... [keep all existing add_argument lines unchanged] ...
    # After the existing --retry-validation-failed flag, append:
    p.add_argument("--provider-config", type=Path, default=None,
                   help="Path to a multi-provider generation config JSON. "
                        "Slice 1: only --dry-run-quota is supported.")
    p.add_argument("--dry-run-quota", action="store_true",
                   help="With --provider-config: parse + validate config, "
                        "print quota allocations and scheduler preview, exit 0. "
                        "No generation runs.")
    p.add_argument("--multi-provider", action="store_true",
                   help="Reserved for Slice 2; in Slice 1 every combination "
                        "involving this flag exits via parser.error.")
    return p, p.parse_args()
```

(The detailed edit: read the file, locate the existing `def parse_args() -> argparse.Namespace:` line and its `return p.parse_args()` line. Update the function signature to return a 2-tuple, add the three new `p.add_argument(...)` calls just before the final `return`, and change the return to `return p, p.parse_args()`.)

- [ ] **Step 2: Add the contract enforcement and `_run_dry_run` helper**

Add these two helper functions ABOVE `def main()` (anywhere in the helpers section is fine, e.g., after `setup_logging`):

```python
def _enforce_slice1_flag_contract(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Slice 1 contract — see spec §7.3.
    Every error path uses parser.error which exits with code 2."""
    if args.multi_provider and not args.provider_config:
        parser.error("--multi-provider requires --provider-config.")
    if args.provider_config and args.multi_provider:
        parser.error(
            "Multi-provider execution lands in Slice 2; see "
            "docs/superpowers/specs/2026-05-16-multi-provider-generation-slice1-design.md. "
            "Slice 1 supports --provider-config only with --dry-run-quota; "
            "do not combine with --multi-provider."
        )
    if args.provider_config and not args.dry_run_quota:
        parser.error(
            "Slice 1: --provider-config requires --dry-run-quota. "
            "Real multi-provider execution lands in Slice 2."
        )
    if args.dry_run_quota and not args.provider_config:
        parser.error("--dry-run-quota requires --provider-config.")


def _run_dry_run(config_path: Path) -> int:
    """Load config + print render_dry_run output. Exit 0 on success, 2 on bad config."""
    from generation_config import ConfigError, load_generation_config
    from generation_orchestrator import render_dry_run

    try:
        cfg = load_generation_config(config_path)
    except ConfigError as e:
        logging.error("Invalid provider config: %s", e)
        print(f"Invalid provider config: {e}", file=sys.stderr)
        return 2

    logging.info("Loaded config from %s (version %d)", config_path, cfg.version)
    print(render_dry_run(cfg))
    return 0
```

- [ ] **Step 3: Update `main()` to call the new helpers**

Find the existing `def main() -> int:` and replace its first few lines. The current beginning is:
```python
def main() -> int:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"05_generate_distillation_data_{ts}.log")
    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Stage 5: phase=%s args=%s", args.phase, vars(args))
```

Change it to:
```python
def main() -> int:
    parser, args = parse_args()
    _enforce_slice1_flag_contract(args, parser)   # may parser.error and exit 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(LOGS_DIR / f"05_generate_distillation_data_{ts}.log")

    if args.dry_run_quota:
        assert args.provider_config is not None
        return _run_dry_run(args.provider_config)

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Stage 5: phase=%s args=%s", args.phase, vars(args))
```

- [ ] **Step 4: Sanity import**

Run:
```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "import importlib; m = importlib.import_module('05_generate_distillation_data'); print(m.parse_args.__name__)"
```

Expected: prints `parse_args` (no SyntaxError, no missing imports).

- [ ] **Step 5: Manual smoke — dry-run path**

Run:
```bash
cd "C:/work/llm training" && python scripts/05_generate_distillation_data.py --provider-config configs/test_providers.json --dry-run-quota
```

Expected: prints the multi-provider dry-run summary; exit 0.

- [ ] **Step 6: Manual smoke — each error path exits 2**

Run each and confirm exit code 2:
```bash
cd "C:/work/llm training"
python scripts/05_generate_distillation_data.py --provider-config configs/test_providers.json
python scripts/05_generate_distillation_data.py --dry-run-quota
python scripts/05_generate_distillation_data.py --multi-provider
python scripts/05_generate_distillation_data.py --provider-config configs/test_providers.json --multi-provider --dry-run-quota
```

Each should print an `argparse: error: ...` to stderr. (Check exit code with `echo $?` on bash or `$LASTEXITCODE` on PowerShell.)

- [ ] **Step 7: Full test suite stays green**

Run: `cd "C:/work/llm training" && pytest tests/ -v`

Expected: every prior-passing test still passes.

- [ ] **Step 8: Do NOT commit.**

---

### Task 12: `tests/test_stage5_flag_contract.py`

**Files:**
- Create: `tests/test_stage5_flag_contract.py`

Spec reference: §7.3, §8.1.

- [ ] **Step 1: Write the contract-matrix tests**

`tests/test_stage5_flag_contract.py`:
```python
"""Subprocess-based tests for the Stage 5 CLI flag matrix.

Uses subprocess.run rather than calling main() directly so the full
argparse path is exercised end-to-end and parser.error semantics
(exit code 2, message to stderr) are verified verbatim.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "05_generate_distillation_data.py"
CONFIG = REPO_ROOT / "configs" / "test_providers.json"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )


def test_help_works_without_provider_config():
    """Sanity: existing CLI exposes --help and imports cleanly."""
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


def test_provider_config_with_multi_provider_exits_2():
    result = run_cli("--provider-config", str(CONFIG), "--multi-provider")
    assert result.returncode == 2
    assert "Slice 2" in result.stderr


def test_provider_config_with_multi_provider_and_dry_run_exits_2():
    result = run_cli(
        "--provider-config", str(CONFIG),
        "--multi-provider", "--dry-run-quota",
    )
    assert result.returncode == 2
    assert "Slice 2" in result.stderr


def test_invalid_config_path_exits_2():
    result = run_cli("--provider-config", "definitely_not_real.json", "--dry-run-quota")
    assert result.returncode == 2
    assert "Invalid provider config" in result.stderr or "not found" in result.stderr.lower()
```

- [ ] **Step 2: Run, expect all pass**

Run: `cd "C:/work/llm training" && pytest tests/test_stage5_flag_contract.py -v`

Expected: 8 passed.

- [ ] **Step 3: Do NOT commit.**

---

### Task 13: `docs/provider_config.md` (human-readable config explainer)

**Files:**
- Create: `docs/provider_config.md`

Spec reference: §10 step 8.

- [ ] **Step 1: Write the doc**

`docs/provider_config.md`:
```markdown
# Multi-Provider Generation Config

The JSON config consumed by `scripts/05_generate_distillation_data.py --provider-config <path>` controls how Stage 5 generates synthetic inputs and teacher labels across one or more LLM providers. Slice 1 only supports `--dry-run-quota`; real execution lands in Slices 2/3.

See the spec at `docs/superpowers/specs/2026-05-16-multi-provider-generation-slice1-design.md` for the full schema.

## Top-level structure

```json
{
  "version": 1,
  "input_generation":  { ... },
  "output_generation": { ... },
  "validation":        { ... },
  "rate_limits":       { ... }
}
```

`version` must be `1` if present; missing is treated as `1`.

## `input_generation`

Controls how many synthetic input strings are produced and which providers contribute.

| Field | Type | Notes |
|---|---|---|
| `enabled` | bool | If false, the phase is skipped entirely. |
| `target_inputs` | int ≥ 0 | Total unique input strings to produce. |
| `batch_size` | int ≥ 1 | Inputs requested per API call. |
| `dedupe` | bool | Global dedupe across providers (Slice 2 behavior). |
| `providers` | list | Non-empty when `enabled=true` and `target_inputs≥1`. |

## `output_generation`

Controls how teacher labels are produced for each input.

| Field | Type | Notes |
|---|---|---|
| `enabled` | bool | If false, the phase is skipped entirely. |
| `label_attempts_per_input` | int ≥ 1 | How many candidate labels to generate per input. |
| `selection_policy` | str | Currently only `"first_valid_then_score"`. |
| `providers` | list | Non-empty when `enabled=true`. |
| `provider_priority` | list[str] | Tie-breaker order; each entry must reference a declared provider. |

## `ProviderConfig`

| Field | Type | Notes |
|---|---|---|
| `name` | str | Unique within its phase. |
| `type` | one of `fake`, `deepseek`, `gemini`, `local_teacher` | Slice 1 ships only `fake`. Others become `UnimplementedProvider`. |
| `weight` | int ≥ 1 | Used for quota allocation and round-robin scheduling. |
| `threads` | int ≥ 1 | Concurrency hint for Slice 2/3. |
| `model` | str | Required for `deepseek` and `gemini`. |
| `temperature` | null or float ≥ 0 | Optional. |
| `max_tokens` | null or int ≥ 1 | Optional. |
| `max_retries` | int ≥ 0 | Default 3. |
| `structured_output` | bool | Gemini only — uses JSON-schema response API. |
| `seed` | int or null | Reserved for forward compatibility. |
| `fixture_inputs` | str | Required for `type=fake` in input phase. Path relative to config dir. |
| `fixture_labels` | str | Required for `type=fake` in output phase. Path relative to config dir. |

## `validation`

| Field | Type | Notes |
|---|---|---|
| `schema` | bool | Run JSON-schema check (Slice 3 default true). |
| `semantic_validator` | bool | Run the validator from the prior slice. |
| `reject_invalid` | bool | Drop rows that fail validation (vs. flag-and-keep). |
| `retry_invalid_with_stricter_prompt` | bool | Try a stricter repair prompt on failure. |
| `max_repair_attempts` | int ≥ 0 | Cap on repair attempts. |

## `rate_limits`

| Field | Type | Notes |
|---|---|---|
| `global_max_workers` | int ≥ 1 | Cap across all providers. |
| `write_flush_every` | int ≥ 1 | Flush to JSONL every N accepted rows. |

## Fixture path resolution

Relative paths in `fixture_inputs` and `fixture_labels` resolve **relative to the directory containing the config file**. Absolute paths are used as-is. The loader validates the resolved path exists at load time.

## What loads successfully in Slice 1

- Any valid `type: fake` provider with proper fixture paths.
- Any `type: deepseek` / `gemini` / `local_teacher` provider — these become `UnimplementedProvider` and any actual generation call raises `NotImplementedError`. Dry-run-quota still works.

## Example configs

- `configs/test_providers.json` — fake-only, 10-input target, used by smoke tests.
- `configs/example_providers.json` — realistic shape with deepseek/gemini/local_teacher providers. Loads cleanly; cannot run in Slice 1.
```

- [ ] **Step 2: Do NOT commit.**

---

### Task 14: README update

**Files:**
- Modify: `README.md`

Spec reference: §10 step 9.

- [ ] **Step 1: Add a "Multi-provider scaffolding (Slice 1)" subsection**

Open `README.md`. Locate the Stage 5 section (heading: `## Stage 5 — Teacher generates distillation data`). After the existing "Validator gate (Phase 2)" paragraph (added by the prior slice), append:

```markdown
### Multi-provider scaffolding (Slice 1)

Stage 5 has a new optional path for multi-provider input/label generation, gated behind `--provider-config`. Slice 1 only supports dry-run config validation; real multi-provider execution lands in Slice 2/3.

```bash
# Validate a provider config and see quota allocations
python scripts/05_generate_distillation_data.py \
    --provider-config configs/test_providers.json \
    --dry-run-quota
```

The full config schema is documented in `docs/provider_config.md`. Two example configs ship:
- `configs/test_providers.json` — fake-only, used by smoke tests.
- `configs/example_providers.json` — realistic shape with DeepSeek / Gemini / local-teacher providers.

A standalone validator probe lets you inspect any JSONL of `(input, output)` pairs against the semantic validator:

```bash
python scripts/probe_validator.py \
    --input data/distill/train.jsonl \
    --output reports/validator_probe.jsonl
```

The probe reports per-code failure counts (e.g., AMOUNT_NOT_IN_INPUT, SUSPICIOUS_DUPLICATE) and writes an inspectable per-row report.

Without `--provider-config`, Stage 5 behaves exactly as before.
```

- [ ] **Step 2: Verify the prior slice's README content is still present**

Spec §10 step 9 says to verify the prior validator-slice content is still present. Check that these sections still exist:
- The "Validator gate (Phase 2)" paragraph in the Stage 5 section.
- The "Stage 4 also reports four validator-derived aggregates..." paragraph in the Stage 4 section.
- The "### Dev dependencies" subsection under Prerequisites.

Run a quick grep:
```bash
cd "C:/work/llm training" && grep -E "Validator gate \(Phase 2\)|validator-derived aggregates|Dev dependencies" README.md
```

Expected: three matches. If any is missing or wording has rotted, restore it from commit `1bf3a14`.

- [ ] **Step 3: Do NOT commit yet — final commit in Task 15.**

---

### Task 15: Final verification + Commit

**Files:**
- (No new files; this is the verification + commit task.)

Spec reference: §10 step 10, 11.

- [ ] **Step 1: Run the FULL test suite**

```bash
cd "C:/work/llm training" && pytest tests/ -v
```

Expected: every existing test (from previous slices) AND all new tests from this slice pass.

- [ ] **Step 2: Coverage check**

```bash
cd "C:/work/llm training" && pytest tests/ --cov=llm_providers --cov=generation_config --cov=generation_orchestrator --cov=probe_validator --cov-report=term
```

Expected:
- `llm_providers.py`: ≥ 85%
- `generation_config.py`: ≥ 90%
- `generation_orchestrator.py`: ≥ 95%
- `probe_validator.py`: ≥ 85%

If any is under, identify missing lines and add 1–2 targeted tests.

- [ ] **Step 3: Run the human smoke commands from spec §9**

```bash
cd "C:/work/llm training"

# 1. Dry-run quota on the test config
python scripts/05_generate_distillation_data.py \
    --provider-config configs/test_providers.json \
    --dry-run-quota

# 2. CLI help still works (legacy untouched)
python scripts/05_generate_distillation_data.py --help

# 3. Probe the validator against hand-picked fixture
python scripts/probe_validator.py --input tests/fixtures/probe_examples.jsonl

# 4. Probe against existing distillation training data (HIGH VALUE)
python scripts/probe_validator.py \
    --input data/distill/train.jsonl \
    --output reports/validator_probe_existing_train.jsonl
```

Capture the output of command 4 — that's the headline diagnostic this slice produces. Note the pass rate; it informs the decision point in spec §11.

- [ ] **Step 4: Stage the new files (be careful — do NOT stage `data/distill/*` or `reports/*`)**

```bash
cd "C:/work/llm training"

git add \
    scripts/llm_providers.py \
    scripts/generation_config.py \
    scripts/generation_orchestrator.py \
    scripts/probe_validator.py \
    scripts/05_generate_distillation_data.py \
    configs/test_providers.json \
    configs/example_providers.json \
    docs/provider_config.md \
    tests/test_fake_provider.py \
    tests/test_generation_config.py \
    tests/test_provider_scheduler.py \
    tests/test_probe_validator.py \
    tests/test_stage5_flag_contract.py \
    tests/fixtures/fake_inputs.jsonl \
    tests/fixtures/fake_labels.jsonl \
    tests/fixtures/probe_examples.jsonl \
    README.md

git status
```

Verify `git status` shows ONLY the listed files staged. The pre-existing `data/distill/*` modifications and `.coverage` / `reports/` should remain unstaged.

- [ ] **Step 5: Commit**

```bash
cd "C:/work/llm training"
git commit -m "$(cat <<'EOF'
Slice 1: multi-provider generation scaffolding + validator probe

Adds the provider abstraction (LLMProvider Protocol, FakeProvider,
UnimplementedProvider, create_provider), strict-validated config
dataclasses, GCD-normalized weighted round-robin scheduler, and the
render_dry_run summary. Stage 5 gets new flags (--provider-config,
--dry-run-quota, --multi-provider) gated by a strict contract: only
--provider-config + --dry-run-quota actually runs; every other combo
exits via parser.error with code 2.

Standalone scripts/probe_validator.py: feeds a JSONL of (input, output)
pairs through validate_example and reports per-code failure counts.
Useful for inspecting current data/distill/train.jsonl quality.

No SDK dependencies, no real API calls, no GPU/model loading.
Existing Stage 5 behavior unchanged when --provider-config is absent.

Spec: docs/superpowers/specs/2026-05-16-multi-provider-generation-slice1-design.md
EOF
)"
```

- [ ] **Step 6: Verify the commit**

```bash
cd "C:/work/llm training" && git log --oneline -3 && git status
```

Expected: the new commit on top, then the spec commits, then the prior slice commits. Working tree shows only pre-existing data file modifications and untracked report/coverage files.

- [ ] **Step 7: Report**

Surface to the user:
- The new commit SHA.
- The full test count (`pytest tests/ -q | tail -1`).
- The pass rate from probe command 4 (the headline number for Slice 2 planning).
- Coverage percentages.

This closes Slice 1. The decision point in spec §11 determines what slice we plan next.
