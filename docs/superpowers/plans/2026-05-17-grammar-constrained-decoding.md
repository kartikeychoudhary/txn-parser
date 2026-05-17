# Grammar-Constrained Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GBNF grammar-constrained decoding to every llama.cpp inference path (`04_eval.py::GgufBackend`, `predict_one.py`, `viewer/inference_worker.py`), default-on, with a `--no-grammar` opt-out. Default behavior of all GGUF inference changes from "free-form sampling" to "schema-constrained sampling derived from `_lib.CATEGORIES`/`TYPES`/`CURRENCIES`".

**Architecture:** A pure-stdlib `scripts/grammar.py` builds the GBNF string at runtime from `_lib` constants and lazily compiles a cached `LlamaGrammar` on first use. Each call site gains an opt-out flag (`--no-grammar`) and a one-line log/output indicating the grammar state. `TransformersBackend`, `viewer/server.js`, and API-provider paths (DeepSeek/Gemini) are untouched.

**Tech Stack:** Python 3.11+, stdlib only (`re` isn't used; the builder is plain string formatting). `llama-cpp-python` is already a dependency (`requirements-eval.txt`); `LlamaGrammar` lives in `llama_cpp`. Existing pytest harness.

**Spec:** `docs/superpowers/specs/2026-05-17-grammar-constrained-decoding-design.md` (commit `f5636f7`).

**Branch:** `feat/grammar-decoding` (already checked out, branched from `main` HEAD after the multi-provider PR merged).

**Test baseline at start:** Whatever `pytest tests/ -q` reports on this branch. Target at end of plan: **baseline + ~20 new tests** (15 grammar + 5 wiring). If `llama-cpp-python` is not installed in the test environment, 2 of the 15 grammar tests skip via `pytest.importorskip`, yielding +18. Treat exact counts as smoke signals, not hard blockers.

**Single commit at landing.** Same convention as previous slices: spec lives on the branch as its own commit; the implementation lands as one additive commit at Task 8. Files in `data/distill/*`, `eval_results/*`, `.coverage`, and `reports/` are NEVER staged.

---

### Task 0: Preflight

**Files:** none modified.

- [ ] **Step 1: Confirm branch and HEAD**

```bash
cd "C:/work/llm training" && git rev-parse --abbrev-ref HEAD && git log --oneline -3
```
Expected:
```
feat/grammar-decoding
f5636f7 Spec: grammar-constrained decoding (validator-roadmap Slice 2)
<whatever was on main before>
```

- [ ] **Step 2: Confirm test baseline**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Record the count. Subsequent tasks reference "baseline" as this number. (Expected somewhere around 334, but could drift if main moved.)

- [ ] **Step 3: Confirm `_lib` enum constants are present and non-empty**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
from _lib import CATEGORIES, TYPES, CURRENCIES
assert CATEGORIES and TYPES and CURRENCIES, 'enums must be non-empty'
print('CATEGORIES:', len(CATEGORIES), '->', CATEGORIES)
print('TYPES     :', TYPES)
print('CURRENCIES:', CURRENCIES)
"
```
Expected: prints all three lists. If any list is empty, STOP — the grammar builder's load-time check will fail.

- [ ] **Step 4: Confirm `viewer/inference_worker.py` already has `scripts/` on `sys.path`**

```bash
cd "C:/work/llm training" && grep -n "sys.path.insert.*scripts" viewer/inference_worker.py
```
Expected: one match around line 25 — `sys.path.insert(0, str(REPO_ROOT / "scripts"))`. If missing, STOP and flag — Task 5 assumes this is in place.

- [ ] **Step 5: Confirm `predict_one.py` already imports `llama_cpp` lazily**

```bash
cd "C:/work/llm training" && grep -n "from llama_cpp import Llama" scripts/predict_one.py
```
Expected: one match around line 29 inside `main()` (not at module top).

---

### Task 1: `scripts/grammar.py` — pure builder + lazy loader + tests

**Files:**
- Create: `scripts/grammar.py`
- Create: `tests/test_grammar.py`

Spec reference: §4.

This task is TDD: write the failing tests, then make them pass. **Critical:** before locking the expected strings in the escape tests, verify them by running `repr(_gbnf_literal(...))` against the actual implementation. The escape semantics are subtle (Python string + GBNF string both have backslash conventions), and a transcribed expectation may be wrong.

- [ ] **Step 1: Create `tests/test_grammar.py` with the test skeleton**

Create `tests/test_grammar.py` with this EXACT content. Note: the escape-test assertions are written as the intended Python literal values; if Step 4 verification shows the implementation produces something different, fix the implementation first (do NOT just edit the assertions — the GBNF terminal we want is unambiguous: `"\"INR\""`).

```python
"""Unit tests for scripts/grammar.py.

Covers:
- _gbnf_literal escape semantics
- build_label_grammar presence + uniqueness checks (pure stdlib, no llama_cpp)
- load_label_grammar caching + loud-failure behavior (requires llama_cpp;
  uses pytest.importorskip)
- module load-time validation (raises if _lib enums are empty)
"""
import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _json_key_terminal(key: str) -> str:
    """Return the GBNF terminal string that emits the JSON key plus colon,
    e.g. _json_key_terminal('amount') == '"\\"amount\\":"'."""
    return f'"\\"{key}\\":"'


# ---------------------------------------------------------------------------
# _gbnf_literal
# ---------------------------------------------------------------------------


def test_gbnf_literal_wraps_json_string_quotes():
    """_gbnf_literal('INR') must produce the GBNF terminal "\"INR\"".

    Python repr of that terminal is '"\\"INR\\""'.
    """
    from grammar import _gbnf_literal
    assert _gbnf_literal("INR") == '"\\"INR\\""'


def test_gbnf_literal_escapes_backslash():
    """_gbnf_literal('a\\b') must produce GBNF terminal "\"a\\\\b\"".

    That is: in the GBNF text we want a literal backslash twice (so the
    GBNF parser reads one backslash). In Python repr, that's '"\\"a\\\\\\\\b\\""'.
    """
    from grammar import _gbnf_literal
    assert _gbnf_literal("a\\b") == '"\\"a\\\\\\\\b\\""'


def test_gbnf_literal_escapes_embedded_quote():
    """_gbnf_literal('he said \"hi\"') must produce GBNF terminal
    "\"he said \\\"hi\\\"\"" (each embedded quote becomes \\\")."""
    from grammar import _gbnf_literal
    assert _gbnf_literal('he said "hi"') == '"\\"he said \\\\\\"hi\\\\\\"\\""'


# ---------------------------------------------------------------------------
# build_label_grammar — presence
# ---------------------------------------------------------------------------


def test_build_label_grammar_contains_every_category():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import CATEGORIES
    g = build_label_grammar()
    for cat in CATEGORIES:
        assert _gbnf_literal(cat) in g, f"missing category literal for {cat!r}"


def test_build_label_grammar_contains_every_type():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import TYPES
    g = build_label_grammar()
    for t in TYPES:
        assert _gbnf_literal(t) in g, f"missing type literal for {t!r}"


def test_build_label_grammar_contains_every_currency():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import CURRENCIES
    g = build_label_grammar()
    for c in CURRENCIES:
        assert _gbnf_literal(c) in g, f"missing currency literal for {c!r}"


def test_build_label_grammar_has_required_top_level_shape():
    from grammar import build_label_grammar
    g = build_label_grammar()
    assert 'root ::= "{"' in g
    assert _json_key_terminal("transactions") in g


def test_build_label_grammar_has_required_transaction_keys():
    from grammar import build_label_grammar
    g = build_label_grammar()
    for key in ("amount", "currency", "item", "category", "type"):
        assert _json_key_terminal(key) in g, f"missing key terminal for {key!r}"


def test_build_label_grammar_uses_crlf_safe_whitespace():
    from grammar import build_label_grammar
    g = build_label_grammar()
    # `ws` rule must include all four whitespace chars including \r.
    assert "[ \\t\\n\\r]*" in g


# ---------------------------------------------------------------------------
# build_label_grammar — enum uniqueness
# ---------------------------------------------------------------------------


def _rule_body(grammar: str, name: str) -> str:
    """Return text from `<name> ::=` until the next `<other> ::=` line.

    Handles multi-line rule bodies (e.g., `category ::= "A" | "B"\n           | "C"`).
    """
    lines = grammar.splitlines()
    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith(f"{name} ::="))
    body = [lines[start]]
    for l in lines[start + 1:]:
        # New rule starts when a line begins with `<word> ::=` at column 0
        # (no leading whitespace).
        if l and not l.startswith((" ", "\t")) and "::=" in l:
            break
        body.append(l)
    return "\n".join(body)


def test_category_enum_has_no_duplicate_alternatives():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import CATEGORIES
    body = _rule_body(build_label_grammar(), "category")
    for cat in CATEGORIES:
        assert body.count(_gbnf_literal(cat)) == 1, \
            f"category {cat!r} appears != 1 time in its rule body"


def test_type_enum_has_no_duplicate_alternatives():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import TYPES
    body = _rule_body(build_label_grammar(), "type")
    for t in TYPES:
        assert body.count(_gbnf_literal(t)) == 1, \
            f"type {t!r} appears != 1 time in its rule body"


def test_currency_enum_has_no_duplicate_alternatives():
    from grammar import build_label_grammar, _gbnf_literal
    from _lib import CURRENCIES
    body = _rule_body(build_label_grammar(), "currency")
    for c in CURRENCIES:
        assert body.count(_gbnf_literal(c)) == 1, \
            f"currency {c!r} appears != 1 time in its rule body"


# ---------------------------------------------------------------------------
# module load-time validation
# ---------------------------------------------------------------------------


def test_module_raises_on_empty_enum(monkeypatch):
    """If _lib.CATEGORIES ever becomes empty, importing grammar must fail loud."""
    import _lib
    monkeypatch.setattr(_lib, "CATEGORIES", [])
    sys.modules.pop("grammar", None)
    try:
        with pytest.raises(ValueError, match="non-empty"):
            importlib.import_module("grammar")
    finally:
        sys.modules.pop("grammar", None)


# ---------------------------------------------------------------------------
# load_label_grammar — requires llama_cpp
# ---------------------------------------------------------------------------


def test_load_label_grammar_compiles_and_caches(monkeypatch):
    pytest.importorskip("llama_cpp")
    import grammar
    monkeypatch.setattr(grammar, "_LABEL_GRAMMAR", None)
    g1 = grammar.load_label_grammar()
    g2 = grammar.load_label_grammar()
    assert g1 is g2   # cache returns same object


def test_load_label_grammar_failure_is_loud(monkeypatch):
    """Skipped when llama_cpp is missing: this test patches the real
    LlamaGrammar class to force a compile failure, which requires the
    real module to exist."""
    pytest.importorskip("llama_cpp")
    import grammar
    monkeypatch.setattr(grammar, "_LABEL_GRAMMAR", None)

    def _explode(_s):
        raise ValueError("forced compile failure")

    import llama_cpp
    monkeypatch.setattr(llama_cpp.LlamaGrammar, "from_string", _explode)
    with pytest.raises(RuntimeError, match="--no-grammar"):
        grammar.load_label_grammar()
```

- [ ] **Step 2: Run the tests — expect ImportError**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_grammar.py -v 2>&1 | tail -20
```
Expected: every test ERRORS with `ModuleNotFoundError: No module named 'grammar'`.

- [ ] **Step 3: Create `scripts/grammar.py` with this EXACT content**

```python
"""GBNF grammar for transaction-parser output, derived at runtime from
_lib.CATEGORIES/TYPES/CURRENCIES.

Pure-stdlib at import time. The lazy `load_label_grammar()` is the only
function that pulls in `llama_cpp`; it caches the compiled grammar
module-level so it's built exactly once per process.

Spec: docs/superpowers/specs/2026-05-17-grammar-constrained-decoding-design.md
"""
from __future__ import annotations

from _lib import CATEGORIES, TYPES, CURRENCIES


if not CATEGORIES or not TYPES or not CURRENCIES:
    raise ValueError(
        "grammar.py: _lib enums (CATEGORIES, TYPES, CURRENCIES) must be non-empty"
    )


def _gbnf_literal(s: str) -> str:
    """Build a GBNF terminal string that emits the JSON token `"<s>"`.

    Escaping order matters: backslash first (so we don't escape the
    backslashes we add for quotes), then double-quote.

    Returns a Python string. When that string is rendered into the GBNF
    text, it produces a terminal like `"\\"INR\\""`, which is how GBNF
    spells the literal three-char sequence `"INR"`.
    """
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return '"\\"' + escaped + '\\""'


def _enum_rule(name: str, values: list[str]) -> str:
    """Return a GBNF line like:
        category ::= "\\"Food & Drinks\\"" | "\\"Groceries\\"" | ...
    """
    alts = " | ".join(_gbnf_literal(v) for v in values)
    return f"{name} ::= {alts}"


def build_label_grammar() -> str:
    """Return the full GBNF text. Pure string building."""
    # Standard JSON-string and JSON-number rules, adapted from llama.cpp's
    # grammars/json.gbnf reference set.
    return "\n".join([
        'root ::= "{" ws "\\"transactions\\":" ws "[" ws transactions ws "]" ws "}"',
        'transactions ::= transaction (ws "," ws transaction)* | ""',
        '',
        'transaction ::= "{"',
        '    ws "\\"amount\\":" ws number ws ","',
        '    ws "\\"currency\\":" ws currency ws ","',
        '    ws "\\"item\\":" ws string ws ","',
        '    ws "\\"category\\":" ws category ws ","',
        '    ws "\\"type\\":" ws type',
        '    ws "}"',
        '',
        _enum_rule("currency", list(CURRENCIES)),
        _enum_rule("type", list(TYPES)),
        _enum_rule("category", list(CATEGORIES)),
        '',
        'string ::= "\\"" char* "\\""',
        'char ::= [^"\\\\] | "\\\\" ["\\\\/bfnrt] | "\\\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]',
        'number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
        'ws ::= [ \\t\\n\\r]*',
        '',
    ])


_LABEL_GRAMMAR = None    # cached LlamaGrammar instance


def load_label_grammar():
    """Return the cached LlamaGrammar. Lazy import of `llama_cpp`.

    Failure to import or compile raises RuntimeError with a hint about
    --no-grammar so the operator knows the escape hatch exists. No silent
    fallback — that would defeat the entire point of grammar-on-by-default.
    """
    global _LABEL_GRAMMAR
    if _LABEL_GRAMMAR is not None:
        return _LABEL_GRAMMAR
    try:
        from llama_cpp import LlamaGrammar
        _LABEL_GRAMMAR = LlamaGrammar.from_string(build_label_grammar())
    except Exception as e:    # noqa: BLE001
        raise RuntimeError(
            f"Failed to load label grammar: {e}. "
            "Use --no-grammar to run unconstrained."
        ) from e
    return _LABEL_GRAMMAR
```

- [ ] **Step 4: Verify the escape outputs match the test expectations**

Before running the test suite, sanity-check the escape semantics by hand. The test assertions are written as Python repr of intended values; the implementation must produce the SAME Python strings. Run:

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
from grammar import _gbnf_literal, build_label_grammar
print('repr(_gbnf_literal(\"INR\"))                ->', repr(_gbnf_literal('INR')))
print('repr(_gbnf_literal(\"a\\\\\\\\b\"))             ->', repr(_gbnf_literal('a\\\\b')))
print('repr(_gbnf_literal(quoted))            ->', repr(_gbnf_literal('he said \"hi\"')))
print()
print('--- Full grammar ---')
print(build_label_grammar())
"
```

Expected (look for these EXACT Python repr strings):
```
repr(_gbnf_literal("INR"))                -> '"\\"INR\\""'
repr(_gbnf_literal("a\\b"))             -> '"\\"a\\\\\\\\b\\""'
repr(_gbnf_literal(quoted))            -> '"\\"he said \\\\\\"hi\\\\\\"\\""'
```

And the full grammar should look like (categories interpolated; this is the GBNF text, not Python repr):
```
root ::= "{" ws "\"transactions\":" ws "[" ws transactions ws "]" ws "}"
transactions ::= transaction (ws "," ws transaction)* | ""

transaction ::= "{"
    ws "\"amount\":" ws number ws ","
    ws "\"currency\":" ws currency ws ","
    ws "\"item\":" ws string ws ","
    ws "\"category\":" ws category ws ","
    ws "\"type\":" ws type
    ws "}"

currency ::= "\"INR\"" | "\"USD\""
type ::= "\"expense\"" | "\"income\""
category ::= "\"Food & Drinks\"" | "\"Groceries\"" | "\"Travel\"" | ...
...
```

If `repr(...)` doesn't match the test expectation: **do not weaken the assertion**. The intended GBNF terminal is unambiguous — for `"INR"` it must be the 7-character GBNF sequence `"\"INR\""` (quote, escaped-quote, I, N, R, escaped-quote, quote). Verify the intended GBNF terminal visually first, then fix whichever side is wrong (implementation OR the test's Python literal) so both reflect that GBNF target. Python escape semantics are confusing; the GBNF text is the invariant.

- [ ] **Step 5: Run the tests — expect all pass**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_grammar.py -v 2>&1 | tail -25
```
Expected: 15 passed (3 escape + 6 presence + 3 uniqueness + 1 empty-enum + 2 compile/cache). If `pytest.importorskip("llama_cpp")` skips the last 2 because `llama-cpp-python` isn't installed in this dev environment, that's expected; 13 pass + 2 skip is fine.

- [ ] **Step 6: Verify `import grammar` doesn't pull in `llama_cpp`**

```bash
cd "C:/work/llm training" && PYTHONPATH=scripts python -c "
import sys, grammar
loaded = [m for m in sys.modules if m.startswith(('llama_cpp', 'torch', 'unsloth', 'transformers'))]
assert not loaded, f'leaked: {loaded}'
print('import grammar: clean (no llama_cpp pulled in at import time)')
"
```
Expected: `import grammar: clean (no llama_cpp pulled in at import time)`.

- [ ] **Step 7: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: baseline + 15 passed (or +13 if 2 skipped).

- [ ] **Step 8: Do NOT commit.** Single commit lands at Task 8.

---

### Task 2: `tests/test_eval_grammar_wiring.py` + `GgufBackend` modification

**Files:**
- Create: `tests/test_eval_grammar_wiring.py`
- Modify: `scripts/04_eval.py` (`GgufBackend` class only)

Spec reference: §5.1, §6.2.

This is TDD: write the wiring tests first; they require `GgufBackend` to accept `use_grammar` and pass `grammar=` into `create_chat_completion`. The current `GgufBackend` has neither.

- [ ] **Step 1: Create `tests/test_eval_grammar_wiring.py` with this EXACT content**

```python
"""GgufBackend wiring tests: verify the grammar plumbing without real
llama_cpp or any real GGUF file. We inject a fake `llama_cpp` module
into sys.modules so tests run anywhere.
"""
import sys
import types
from importlib import import_module
from pathlib import Path

import pytest


class _FakeLlama:
    def __init__(self, *args, **kwargs):
        # Tolerant of any signature shift in real Llama; we only care
        # that GgufBackend can construct one.
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "choices": [{"message": {"content": '{"transactions":[]}'}}],
            "usage": {"completion_tokens": 3},
        }


class _FakeGrammar:
    """Stand-in for llama_cpp.LlamaGrammar. Provides a from_string
    classmethod so any code path that compiles grammar (instead of
    monkeypatching grammar.load_label_grammar) still works."""

    @classmethod
    def from_string(cls, _grammar: str):
        return cls()


def _inject_fake_llama_cpp(monkeypatch):
    fake = types.SimpleNamespace(Llama=_FakeLlama, LlamaGrammar=_FakeGrammar)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)


def _eval_module():
    """Import the stage-5-eval module by its file-path name.

    The script is named `04_eval.py` (leading digit), so a normal
    `import 04_eval` is illegal. importlib.import_module works because
    we add scripts/ to sys.path via the test scaffolding.
    """
    SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return import_module("04_eval")


def test_gguf_backend_with_grammar_true_loads_grammar(monkeypatch):
    """use_grammar=True calls load_label_grammar once during init."""
    _inject_fake_llama_cpp(monkeypatch)
    import grammar
    sentinel = object()
    monkeypatch.setattr(grammar, "load_label_grammar", lambda: sentinel)
    eval_mod = _eval_module()
    backend = eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                                    n_gpu_layers=0, use_grammar=True)
    assert backend._grammar is sentinel


def test_gguf_backend_with_grammar_false_skips_load(monkeypatch):
    """use_grammar=False does NOT call load_label_grammar."""
    _inject_fake_llama_cpp(monkeypatch)
    import grammar
    called = []
    monkeypatch.setattr(grammar, "load_label_grammar",
                        lambda: called.append(1) or _FakeGrammar())
    eval_mod = _eval_module()
    backend = eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                                    n_gpu_layers=0, use_grammar=False)
    assert called == []
    assert backend._grammar is None


def test_gguf_backend_grammar_load_failure_is_loud(monkeypatch):
    """Default use_grammar=True must raise if grammar load fails."""
    _inject_fake_llama_cpp(monkeypatch)
    import grammar

    def _raises():
        raise RuntimeError("forced grammar load failure")

    monkeypatch.setattr(grammar, "load_label_grammar", _raises)
    eval_mod = _eval_module()
    with pytest.raises(RuntimeError, match="grammar"):
        eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                              n_gpu_layers=0, use_grammar=True)


def test_gguf_backend_batch_infer_passes_grammar(monkeypatch):
    """When grammar is enabled, create_chat_completion receives grammar=<obj>."""
    _inject_fake_llama_cpp(monkeypatch)
    import grammar
    sentinel = object()
    monkeypatch.setattr(grammar, "load_label_grammar", lambda: sentinel)
    eval_mod = _eval_module()
    backend = eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                                    n_gpu_layers=0, use_grammar=True)
    backend.batch_infer(["500 beer"], max_tokens=128)
    assert backend.llm.calls[0]["grammar"] is sentinel


def test_gguf_backend_batch_infer_omits_grammar_when_disabled(monkeypatch):
    """When grammar is disabled, the grammar kwarg must NOT be in the call."""
    _inject_fake_llama_cpp(monkeypatch)
    eval_mod = _eval_module()
    backend = eval_mod.GgufBackend(Path("fake.gguf"), n_ctx=512,
                                    n_gpu_layers=0, use_grammar=False)
    backend.batch_infer(["500 beer"], max_tokens=128)
    assert "grammar" not in backend.llm.calls[0]
```

- [ ] **Step 2: Run the new tests — expect failures**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_eval_grammar_wiring.py -v 2>&1 | tail -20
```
Expected: 5 tests fail (or 4 fail + 1 pass — the disabled case may already work by accident). Failures complain about `GgufBackend.__init__()` getting an unexpected keyword argument `use_grammar`.

- [ ] **Step 3: Modify `GgufBackend.__init__` to accept `use_grammar`**

In `scripts/04_eval.py`, find `class GgufBackend:` (around line 66). The current `__init__` signature is:

```python
    def __init__(self, gguf_path: Path, *, n_ctx: int, n_gpu_layers: int) -> None:
```

Change it to:

```python
    def __init__(
        self, gguf_path: Path, *, n_ctx: int, n_gpu_layers: int,
        use_grammar: bool = True,
    ) -> None:
```

After the existing `self.llm = Llama(...)` block (around line 101), and before the method ends, ADD:

```python
        self._grammar = None
        if use_grammar:
            from grammar import load_label_grammar
            self._grammar = load_label_grammar()
```

(Be careful with indentation — match the surrounding code.)

- [ ] **Step 4: Modify `GgufBackend.batch_infer` to pass grammar conditionally**

The current `batch_infer` method (around line 103) is:

```python
    def batch_infer(self, inputs: list[str], max_tokens: int) -> list[tuple[str, float]]:
        out = []
        for s in inputs:
            t0 = time.perf_counter()
            resp = self.llm.create_chat_completion(
                messages=build_messages(s),
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            text = resp["choices"][0]["message"]["content"] or ""
            out.append((text, latency_ms))
        return out
```

Replace with:

```python
    def batch_infer(self, inputs: list[str], max_tokens: int) -> list[tuple[str, float]]:
        out = []
        for s in inputs:
            t0 = time.perf_counter()
            kwargs = dict(
                messages=build_messages(s),
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
            if self._grammar is not None:
                kwargs["grammar"] = self._grammar
            resp = self.llm.create_chat_completion(**kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            text = resp["choices"][0]["message"]["content"] or ""
            out.append((text, latency_ms))
        return out
```

- [ ] **Step 5: Run the wiring tests — expect all pass**

```bash
cd "C:/work/llm training" && python -m pytest tests/test_eval_grammar_wiring.py -v 2>&1 | tail -15
```
Expected: 5 passed.

- [ ] **Step 6: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: baseline + 20 (or +18 if 2 grammar tests skipped due to missing llama-cpp-python).

- [ ] **Step 7: Do NOT commit.**

---

### Task 3: `04_eval.py` CLI wiring + per-row `_grammar` field

**Files:**
- Modify: `scripts/04_eval.py` (`parse_args` + `resolve_backend` + `main` + per-row write)

Spec reference: §5.1.

No new tests in this task; the changes are CLI/log/output formatting.

- [ ] **Step 1: Add `--no-grammar` to `parse_args`**

Find `def parse_args() -> argparse.Namespace:` (around line 381). After the last `p.add_argument(...)` and before `return p.parse_args()`, ADD:

```python
    p.add_argument("--no-grammar", action="store_true",
                   help="Disable GBNF grammar-constrained decoding for GGUF inference. "
                        "TransformersBackend ignores this flag (no grammar surface).")
```

- [ ] **Step 2: Thread `use_grammar` into every `GgufBackend` construction**

First find every call site:

```bash
cd "C:/work/llm training" && grep -n "GgufBackend(" scripts/04_eval.py
```

Expected: two matches inside `resolve_backend` (one for the `.gguf` file branch, one for the directory-glob branch). If the grep finds more or fewer call sites than expected, update ALL of them — the rule is every `GgufBackend(...)` construction in `04_eval.py` gets the `use_grammar` kwarg.

Replace each `GgufBackend(...)` construction with:

```python
GgufBackend(
    path,    # or `ggufs[0]` in the directory-glob branch
    n_ctx=args.n_ctx,
    n_gpu_layers=args.n_gpu_layers,
    use_grammar=not args.no_grammar,
)
```

Keep the existing `TransformersBackend(path, max_seq_length=args.n_ctx)` call unchanged — `TransformersBackend` has no grammar.

- [ ] **Step 3: Log the grammar state at startup**

Find `main()` (around line 405). After the line `logging.info("Evaluating %s on %s", model_name, args.eval_file)`, ADD:

```python
    if isinstance(backend, GgufBackend):
        logging.info("Grammar (GGUF backend): %s",
                     "enabled" if not args.no_grammar else "disabled")
```

This only logs for GGUF runs; TransformersBackend runs stay quiet.

**If `04_eval.py` has a `--all` / multi-backend loop:** the current file (as of `f5636f7`) has a single `backend = resolve_backend(...)` call in `main()`, so the log line above is correct as written. If a future change adds a per-backend loop, the log line must move INSIDE that loop so each constructed GGUF backend emits its own state line. Verify by re-running `grep -n "resolve_backend\|backend = " scripts/04_eval.py` after this change.

- [ ] **Step 4: Add `_grammar` to each per-row JSONL output**

Find the row-write block (around line 489):

```python
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

Modify the `row` dict to include `_grammar`:

```python
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
                        "_grammar": isinstance(backend, GgufBackend) and not args.no_grammar,
                        **val_fields,
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
```

(For TransformersBackend rows, `_grammar` is `False` — accurate since the transformers path has no grammar.)

- [ ] **Step 5: Verify Python syntax**

```bash
cd "C:/work/llm training" && python -c "
import ast, pathlib
ast.parse(pathlib.Path('scripts/04_eval.py').read_text(encoding='utf-8'))
print('parses cleanly')
"
```
Expected: `parses cleanly`.

- [ ] **Step 6: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: same count as Task 2 (no new tests; existing behavior preserved for `--no-grammar` users; default-on path requires real `llama_cpp` which the test env may not have — but wiring tests with the fake module still cover the default path).

- [ ] **Step 7: Do NOT commit.**

---

### Task 4: `predict_one.py` CLI wiring

**Files:**
- Modify: `scripts/predict_one.py`

Spec reference: §5.2.

No new tests; smoke-tested manually in Task 7.

- [ ] **Step 1: Add `--no-grammar` flag and a `GRAMMAR:` header line**

Open `scripts/predict_one.py`. The current `main()` (around line 19) has:

```python
def main() -> int:
    p = argparse.ArgumentParser(description="Predict one input against a GGUF.")
    p.add_argument("--model", required=True, help="Path to a .gguf file.")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--n-gpu-layers", type=int, default=-1,
                   help="-1 = all on GPU, 0 = CPU only.")
    p.add_argument("input", help="The transcribed transaction string to parse.")
    args = p.parse_args()

    from llama_cpp import Llama  # lazy import
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        verbose=False,
        seed=42,
    )
    resp = llm.create_chat_completion(
        messages=build_messages(args.input),
        temperature=0.0, top_p=1.0,
        max_tokens=args.max_tokens,
    )
    raw = resp["choices"][0]["message"]["content"] or ""
    parsed = extract_json(raw)

    print(f"MODEL  : {args.model}")
    print(f"INPUT  : {args.input}")
    print(f"RAW    : {raw}")
    print(f"PARSED : {json.dumps(parsed, indent=2, ensure_ascii=False) if parsed else '(not valid JSON)'}")
    return 0
```

**Preserve existing behavior.** Do NOT wholesale-replace `main()`; make four minimal patches in place:

1. **Add the `--no-grammar` argument** to the existing argparse block. Place it next to the other flags:

```python
    p.add_argument("--no-grammar", action="store_true",
                   help="Disable GBNF grammar-constrained decoding.")
```

2. **Convert the `create_chat_completion` call to use a kwargs dict.** Find the existing call:

```python
    resp = llm.create_chat_completion(
        messages=build_messages(args.input),
        temperature=0.0, top_p=1.0,
        max_tokens=args.max_tokens,
    )
```

Replace with:

```python
    kwargs = dict(
        messages=build_messages(args.input),
        temperature=0.0, top_p=1.0,
        max_tokens=args.max_tokens,
    )
    if not args.no_grammar:
        from grammar import load_label_grammar
        kwargs["grammar"] = load_label_grammar()
    resp = llm.create_chat_completion(**kwargs)
```

3. **Add a `GRAMMAR:` header line** before the existing `MODEL:` print. Find:

```python
    print(f"MODEL  : {args.model}")
```

Insert above it:

```python
    print(f"GRAMMAR: {'off' if args.no_grammar else 'on'}")
```

4. **Leave everything else (other flags, `Llama(...)` construction, `extract_json`, the other print lines) exactly as it is.** Verify with `git diff scripts/predict_one.py` that no behavior was deleted — only the four additions above appear.

- [ ] **Step 2: Verify syntax + `--help` still works**

```bash
cd "C:/work/llm training" && python -c "
import ast, pathlib
ast.parse(pathlib.Path('scripts/predict_one.py').read_text(encoding='utf-8'))
print('parses cleanly')
"
cd "C:/work/llm training" && python scripts/predict_one.py --help 2>&1 | head -15
```
Expected: `parses cleanly`, then `--help` output includes a `--no-grammar` line.

- [ ] **Step 3: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: same count as Task 3.

- [ ] **Step 4: Do NOT commit.**

---

### Task 5: `viewer/inference_worker.py` CLI wiring

**Files:**
- Modify: `viewer/inference_worker.py`

Spec reference: §5.3.

No new tests; smoke-tested in Task 7.

- [ ] **Step 1: Make three minimal patches — do NOT replace the whole `main()` body**

Open `viewer/inference_worker.py`. Make these three discrete edits in place; leave everything else (logging helpers, model-path validation, send/receive protocol, error handling) exactly as it is.

**Patch A — add the `--no-grammar` argument** to the existing argparse block:

```python
    parser.add_argument("--no-grammar", action="store_true",
                        help="Disable GBNF grammar-constrained decoding.")
```

**Patch B — load grammar after the `Llama(...)` construction.** Find the existing line:

```python
    log(f"loaded in {time.perf_counter() - t0:.1f}s")
    send({"ready": True, "model": model_path.name})
```

INSERT between these two lines:

```python

    grammar_obj = None
    if not args.no_grammar:
        from grammar import load_label_grammar
        grammar_obj = load_label_grammar()
    log(f"grammar: {'enabled' if grammar_obj is not None else 'disabled'}")
```

The result is: `loaded in ...` → blank line → grammar load + log → `send({"ready"...})`.

**Patch C — thread grammar into the request loop (Step 2 below).**

Do not delete or rewrite the model-path validation, the `loading` log line, the lazy SDK imports, or anything else. Use `git diff viewer/inference_worker.py` after the patches to confirm only the three additions above appear.

- [ ] **Step 2: Thread grammar into each request**

The current request loop has:

```python
        try:
            t_start = time.perf_counter()
            resp = llm.create_chat_completion(
                messages=build_messages(input_text),
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
            latency_ms = (time.perf_counter() - t_start) * 1000.0
```

Change it to:

```python
        try:
            t_start = time.perf_counter()
            kwargs = dict(
                messages=build_messages(input_text),
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
            if grammar_obj is not None:
                kwargs["grammar"] = grammar_obj
            resp = llm.create_chat_completion(**kwargs)
            latency_ms = (time.perf_counter() - t_start) * 1000.0
```

- [ ] **Step 3: Verify syntax + `--help`**

```bash
cd "C:/work/llm training" && python -c "
import ast, pathlib
ast.parse(pathlib.Path('viewer/inference_worker.py').read_text(encoding='utf-8'))
print('parses cleanly')
"
cd "C:/work/llm training" && python viewer/inference_worker.py --help 2>&1 | head -15
```
Expected: `parses cleanly`, then `--help` output includes `--no-grammar`.

- [ ] **Step 4: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: same count as Task 4.

- [ ] **Step 5: Do NOT commit.**

---

### Task 6: README one-line note

**Files:**
- Modify: `README.md`

Spec reference: §8.

- [ ] **Step 1: Find the Stage 4 or Stage 7 section**

```bash
cd "C:/work/llm training" && grep -nE "^## Stage 4|^## Stage 7|^### " README.md | head -20
```
Pick the place that reads most natural — typically just before or after the Stage 4 eval section, since grammar applies most visibly there.

- [ ] **Step 2: Add a one-line subsection**

Add this paragraph in an appropriate location (before or after the Stage 4 description, or at the bottom of the Stage 7 description — engineer's call):

```markdown
### GBNF grammar (default-on for GGUF inference)

By default, every GGUF-based inference path (Stage 4 eval, Stage 7
playground, and `scripts/predict_one.py`) constrains output via a GBNF
grammar derived from the validator's enum constants
(`_lib.CATEGORIES`/`TYPES`/`CURRENCIES`). Pass `--no-grammar` to disable
for baseline comparisons. The grammar is rebuilt from `_lib` at every
process start, so adding a category never drifts. See
`scripts/grammar.py` and the design at
`docs/superpowers/specs/2026-05-17-grammar-constrained-decoding-design.md`.
```

If the README structure prefers an inline mention (not a subsection), a single sentence works:

> *By default, GGUF inference (Stages 4, 7, and `predict_one.py`) constrains output via a GBNF grammar derived from `_lib`'s enum constants. Pass `--no-grammar` to disable.*

Either form is acceptable. Pick the shape that fits the surrounding README style.

- [ ] **Step 3: Verify the README is syntactically valid Markdown**

```bash
cd "C:/work/llm training" && python -c "
content = open('README.md', encoding='utf-8').read()
assert 'no-grammar' in content, '--no-grammar mention is missing from README'
print('README contains --no-grammar mention')
"
```
Expected: prints the confirmation line.

- [ ] **Step 4: Full suite stays green**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: same count.

- [ ] **Step 5: Do NOT commit.**

---

### Task 7: Manual smokes + clean staging

**Files:** none modified.

Spec reference: §7.

These smokes require `llama-cpp-python` with a working CUDA build AND a real trained student GGUF at `models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf`. If your dev environment lacks either, document this in the Task 8 commit message and skip smokes 1–5; rely on the unit tests for coverage.

**CLI flag note:** `scripts/04_eval.py` takes `--model` (not `--gguf`). Confirmed by `python scripts/04_eval.py --help`. Smoke commands below use the verified flag.

- [ ] **Step 1: Smoke 1 — `predict_one.py` with grammar (default)**

```bash
cd "C:/work/llm training" && python scripts/predict_one.py \
    --model models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf \
    "500 rs on beer 50 rs on candy" 2>&1 | tail -10
```
Expected: exits 0; output includes `GRAMMAR: on`; `RAW` is a single `{"transactions":[...]}` object; `PARSED` is non-null with two transactions.

- [ ] **Step 2: Smoke 2 — `predict_one.py --no-grammar`**

```bash
cd "C:/work/llm training" && python scripts/predict_one.py --no-grammar \
    --model models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf \
    "500 rs on beer 50 rs on candy" 2>&1 | tail -10
```
Expected: exits 0; output includes `GRAMMAR: off`. The actual content may still be valid JSON — that's not a failure, just a baseline comparison.

- [ ] **Step 3: Smoke 3 — `04_eval.py` with grammar (limit to a few rows for speed)**

```bash
cd "C:/work/llm training" && python scripts/04_eval.py \
    --model models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf \
    --limit 10 2>&1 | tail -15
```
Expected: log line `Grammar (GGUF backend): enabled`; `eval_results/<name>.jsonl` rows include `"_grammar": true`; `JSON valid:` percentage is at or near 100%.

- [ ] **Step 4: Smoke 4 — `04_eval.py --no-grammar`**

```bash
cd "C:/work/llm training" && python scripts/04_eval.py \
    --model models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf \
    --limit 10 --no-grammar 2>&1 | tail -15
```
Expected: log line `Grammar (GGUF backend): disabled`; rows include `"_grammar": false`; `JSON valid:` percentage is the current unconstrained baseline for this model on these 10 rows. Compare against Smoke 3 to feel the lift.

- [ ] **Step 5: Smoke 5 — `inference_worker.py` startup log**

```bash
cd "C:/work/llm training" && python viewer/inference_worker.py \
    --model models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf < /dev/null 2>&1 | head -8
```
Expected: stderr includes `[worker] grammar: enabled`. Then re-run with `--no-grammar`:

```bash
cd "C:/work/llm training" && python viewer/inference_worker.py --no-grammar \
    --model models/student/gguf/gemma3_text-fixed.Q5_K_M.gguf < /dev/null 2>&1 | head -8
```
Expected: stderr includes `[worker] grammar: disabled`.

This smoke only verifies startup/logging — stdin EOF shuts the worker down immediately after the ready/log lines.

- [ ] **Step 6: Verify clean staging area**

```bash
cd "C:/work/llm training" && git status --short
```
Expected list of staged-able items: `scripts/grammar.py` (NEW), `scripts/04_eval.py` (M), `scripts/predict_one.py` (M), `viewer/inference_worker.py` (M), `tests/test_grammar.py` (NEW), `tests/test_eval_grammar_wiring.py` (NEW), `README.md` (M).

These MUST NOT appear staged at Task 8:
- `data/distill/*` modifications
- `eval_results/*` files (from smokes 3+4)
- `.coverage`, `htmlcov/`, `reports/`

(They may show as unstaged/untracked — that's fine.)

- [ ] **Step 7: Final full suite + coverage**

```bash
cd "C:/work/llm training" && python -m pytest tests/ -q 2>&1 | tail -3
```
Expected: baseline + 20 passed (or +18 if 2 compile-step tests skipped).

```bash
cd "C:/work/llm training" && python -m pytest tests/ --cov=grammar --cov-report=term 2>&1 | tail -10
```
Expected: `grammar.py` coverage ≥ 90%. If lower, identify which branch is uncovered (likely the `RuntimeError` re-raise or the empty-enum check) and add a small unit test.

- [ ] **Step 8: Do NOT commit yet. Final commit lands at Task 8.**

---

### Task 8: Final commit

**Files:** none modified — this is the commit step.

- [ ] **Step 1: Stage the slice files (and nothing else)**

```bash
cd "C:/work/llm training" && git add \
    scripts/grammar.py \
    scripts/04_eval.py \
    scripts/predict_one.py \
    viewer/inference_worker.py \
    tests/test_grammar.py \
    tests/test_eval_grammar_wiring.py \
    README.md
git status --short
```

Verify the `git status --short` output shows ONLY the listed files staged. Untracked artifacts (`.coverage`, `reports/`, `eval_results/*` from smokes) and pre-existing `data/distill/*` modifications must remain unstaged.

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
Slice 2 (validator roadmap): GBNF grammar-constrained decoding for GGUF inference

Adds default-on grammar enforcement to every llama.cpp inference path:
- scripts/04_eval.py::GgufBackend
- scripts/predict_one.py
- viewer/inference_worker.py

The grammar is rebuilt from _lib.CATEGORIES/TYPES/CURRENCIES at every
process start (single source of truth shared with the validator and the
Gemini structured-output schema). --no-grammar disables it for baseline
comparisons.

Adds:
- scripts/grammar.py — pure-stdlib build_label_grammar() + lazy
  load_label_grammar() with a module-level LlamaGrammar cache.
  _gbnf_literal escapes backslashes and quotes for safe enum inlining.
  Module load raises ValueError if any _lib enum is empty.
  load_label_grammar raises RuntimeError("Use --no-grammar...") on
  compile/import failure — no silent fallback.

Grammar shape:
- Single top-level {"transactions": [...]}. The grammar has no valid
  continuation after the closing brace, so llama.cpp terminates
  constrained generation there. Existing max_tokens and extract_json
  remain as defensive backups.
- Strict enums for currency / type / category from _lib.
- Free-form item string, JSON number for amount.
- Empty transactions array allowed.
- Fixed field order: amount, currency, item, category, type.
- ws includes \r for Windows CRLF safety.

Invariants pinned:
- TransformersBackend untouched (no grammar surface).
- viewer/server.js unchanged; playground inherits grammar-on from the
  worker. To run unconstrained, invoke inference_worker.py manually with
  --no-grammar.
- Sampling params (temperature, top_p, max_tokens) unchanged. Only the
  grammar kwarg is added to create_chat_completion.
- 04_eval.py emits a startup log line for the GGUF backend and a
  per-row _grammar field in eval_results JSONL.

Tests: 15 in test_grammar.py (3 escape + 6 presence + 3 uniqueness +
1 empty-enum + 2 compile/cache) + 5 in test_eval_grammar_wiring.py
(GgufBackend plumbing via injected fake llama_cpp module). Compile-step
tests use pytest.importorskip("llama_cpp") so the suite runs even
without llama-cpp-python installed.

Out of scope (deferred): schema changes, training-loop changes,
API-provider grammar (Gemini already has structured_output; DeepSeek
doesn't support GBNF), TransformersBackend grammar, per-model variation,
server.js changes, eval-set expansion (Slice 3 of validator roadmap).

Single additive commit on feat/grammar-decoding from main HEAD.

Spec: docs/superpowers/specs/2026-05-17-grammar-constrained-decoding-design.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify the commit**

```bash
cd "C:/work/llm training" && git log --oneline -5 && git status --short
```
Expected: new commit on top of `f5636f7` (spec). Working tree shows only pre-existing unstaged items.

- [ ] **Step 4: Report**

Surface to the user:
- The new commit SHA.
- Final test pass count (record exact number; baseline + 20 expected, treat ±2 as smoke signal — and 2 compile-step tests skip silently if llama-cpp-python isn't installed).
- `grammar.py` coverage.
- Which smokes ran (3/3 unit + 5/5 manual? or 3/3 unit + 0/5 manual if no GPU/GGUF available?).
- Smoke 3 vs Smoke 4 JSON-valid lift (if smokes ran).
- Suggested next step: open a PR against `main` for this slice, or proceed to validator-roadmap Slice 3 (expanded eval set + per-slice metrics).
